"""Per-gene, per-perturbation, and per-(perturbation, TF) evaluators.

All evaluators consume any model in this package — they only call
``model.apply(params, x_tf, lib) → (mean, conc)``. Reductions over cells
into per-perturbation accumulators happen on the host (numpy) to keep
the jit'd region pure and avoid materialising large (n_cells, n_genes)
arrays in JAX.
"""

import warnings
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from sklearn.metrics import average_precision_score, precision_recall_curve


class PerPertNLL(NamedTuple):
    """Mean NB-NLL per (perturbation, gene)."""

    nll: np.ndarray  # (n_perts, n_genes) mean NLL/cell
    perts: np.ndarray  # (n_perts,) np.unique-sorted perturbation labels
    counts: np.ndarray  # (n_perts,) cells per perturbation


class PerPertMoments(NamedTuple):
    """Per-(perturbation, gene) raw moments needed for NB-based diagnostics.

    Rnum_{p,g} = Σ_{i∈p} (y_{i,g} − μ_{i,g})
    Vsum_{p,g} = Σ_{i∈p} V_{i,g},   V = μ + μ²/conc
    """

    Rnum: np.ndarray  # (n_perts, n_genes) Σ residuals
    Vsum: np.ndarray  # (n_perts, n_genes) Σ NB variances
    conc: np.ndarray  # (n_genes,) per-gene NB concentration (frozen across cells)
    perts: np.ndarray  # (n_perts,) np.unique-sorted labels
    counts: np.ndarray  # (n_perts,) cells per perturbation


class TFZScores(NamedTuple):
    """Per-(perturbation, TF) signed z-scores from a regulon aggregation."""

    Z: np.ndarray  # (n_perts, n_tfs) — ~N(0,1) under H0
    Z_emp: np.ndarray | None  # (n_perts, n_tfs) robust per-TF recalibration, or None
    n_targets: np.ndarray  # (n_tfs,) targets per TF in Amask_tf


class PRCurve(NamedTuple):
    """Precision-recall curve over a flattened (n_perts, n_tfs) score matrix."""

    precision: np.ndarray
    recall: np.ndarray
    thresholds: np.ndarray
    auprc: float
    prevalence: float
    n_positives: int
    n_total: int


# ── per-gene NLL ──────────────────────────────────────────────────────────────


def _make_per_gene_nll_step(model, nll_fn):
    @jax.jit
    def step(params, x_tf, y_raw, lib):
        mean, conc = model.apply({"params": params}, x_tf, lib)
        return nll_fn(mean, conc, y_raw)  # (B, n_genes)

    return step


def per_gene_nll(model, params, X_tf, Y_raw, lib, batch_size, nll_fn):
    """Mean NLL/cell per gene over the dataset. Returns (n_genes,).

    ``nll_fn`` selects the likelihood — e.g. ``nb_nll`` or ``poisson_nll`` (to
    score a model trained with a Poisson loss). No default — pass it
    explicitly.
    """
    step = _make_per_gene_nll_step(model, nll_fn)
    n, n_genes = X_tf.shape[0], Y_raw.shape[1]
    acc = np.zeros(n_genes, dtype=np.float64)
    for s in range(0, n, batch_size):
        nll = step(
            params,
            jnp.asarray(X_tf[s : s + batch_size]),
            jnp.asarray(Y_raw[s : s + batch_size]),
            jnp.asarray(lib[s : s + batch_size]),
        )
        acc += np.asarray(nll.sum(0))
    return acc / n


def per_cell_nll(model, params, X_tf, Y_raw, lib, batch_size, nll_fn):
    """Mean NLL/gene per cell over the dataset, aligned row-for-row with ``X_tf``.

    Returns ``(n_cells,)``.  This is the per-cell reduction complementary to
    ``per_gene_nll`` — the vector ``fit_dro`` consumes as its ``baseline`` (e.g.
    the NLL of a W=0 null model), so ``baseline[i]`` corresponds to cell ``i``.
    ``nll_fn`` selects the likelihood — no default, pass it explicitly.
    """
    step = _make_per_gene_nll_step(model, nll_fn)
    n = X_tf.shape[0]
    out = np.empty(n, dtype=np.float64)
    for s in range(0, n, batch_size):
        sl = slice(s, s + batch_size)
        nll = step(
            params,
            jnp.asarray(X_tf[sl]),
            jnp.asarray(Y_raw[sl]),
            jnp.asarray(lib[sl]),
        )
        out[sl] = np.asarray(nll.mean(axis=1))
    return out


def per_perturbation_nll(
    model, params, X_tf, Y_raw, lib, pert_labels, batch_size, nll_fn
) -> PerPertNLL:
    """Mean NLL per (perturbation, gene). Returns ``PerPertNLL``.

    ``nll_fn`` selects the likelihood — no default, pass it explicitly.
    """
    step = _make_per_gene_nll_step(model, nll_fn)
    pert_labels = np.asarray(pert_labels)
    unique_perts, pert_idx = np.unique(pert_labels, return_inverse=True)
    n_perts, n_genes = len(unique_perts), Y_raw.shape[1]

    nll_sum = np.zeros((n_perts, n_genes), dtype=np.float64)
    counts = np.zeros(n_perts, dtype=np.int64)
    n = X_tf.shape[0]
    for s in range(0, n, batch_size):
        sl = slice(s, s + batch_size)
        nll = np.asarray(
            step(
                params,
                jnp.asarray(X_tf[sl]),
                jnp.asarray(Y_raw[sl]),
                jnp.asarray(lib[sl]),
            )
        )
        pi = pert_idx[sl]
        np.add.at(nll_sum, pi, nll)
        np.add.at(counts, pi, 1)

    nll_matrix = nll_sum / np.maximum(counts, 1)[:, None]
    return PerPertNLL(nll=nll_matrix, perts=unique_perts, counts=counts)


# ── per-(perturbation, gene) NB residual / variance moments ───────────────────


def _make_moments_step(model):
    @jax.jit
    def step(params, x_tf, y_raw, lib):
        mean, conc = model.apply({"params": params}, x_tf, lib)
        residual = y_raw - mean  # (B, n_genes)
        V = mean + mean**2 / conc  # broadcasts; conc shape (n_genes,)
        return residual, V, conc

    return step


def per_perturbation_moments(
    model, params, X_tf, Y_raw, lib, pert_labels, *, batch_size: int = 512
) -> PerPertMoments:
    """Accumulate NB residual sums and variance sums per (perturbation, gene).

    Returns ``PerPertMoments``.  ``conc`` is reported once per gene since it
    does not vary across cells or perturbations for any model in this package.
    """
    step = _make_moments_step(model)
    pert_labels = np.asarray(pert_labels)
    unique_perts, pert_idx = np.unique(pert_labels, return_inverse=True)
    n_perts, n_genes = len(unique_perts), Y_raw.shape[1]

    Rnum = np.zeros((n_perts, n_genes), dtype=np.float64)
    Vsum = np.zeros((n_perts, n_genes), dtype=np.float64)
    counts = np.zeros(n_perts, dtype=np.int64)
    conc: np.ndarray | None = None
    n = X_tf.shape[0]
    for s in range(0, n, batch_size):
        sl = slice(s, s + batch_size)
        resid_j, V_j, conc_j = step(
            params,
            jnp.asarray(X_tf[sl]),
            jnp.asarray(Y_raw[sl]),
            jnp.asarray(lib[sl]),
        )
        pi = pert_idx[sl]
        np.add.at(Rnum, pi, np.asarray(resid_j, dtype=np.float64))
        np.add.at(Vsum, pi, np.asarray(V_j, dtype=np.float64))
        np.add.at(counts, pi, 1)
        if conc is None:
            conc = np.asarray(conc_j)

    return PerPertMoments(
        Rnum=Rnum,
        Vsum=Vsum,
        conc=conc,
        perts=unique_perts,
        counts=counts,
    )


# ── per-(perturbation, TF) z-scores ───────────────────────────────────────────


def mask_perturbed_gene(moments: PerPertMoments, var_names) -> PerPertMoments:
    """Zero the perturbed gene's own moments per perturbation.

    The directly-perturbed gene ``g*(p)`` carries a large NB residual that is a
    *direct* knockdown effect, not a regulatory response — if ``g*(p)`` falls in
    a TF's regulon it leaks into that TF's z-score.  Because the regulon
    aggregation in ``tf_zscores`` is linear in the moments, zeroing
    ``Rnum[p, g*]`` and ``Vsum[p, g*]`` is exactly equivalent to dropping row
    ``g*`` from a per-perturbation ``W_signed`` — without materialising a 3D
    weight tensor.

    Parameters
    ----------
    moments : PerPertMoments
        Output of ``per_perturbation_moments(...)``.  ``moments.perts`` are
        expected to be gene symbols on the same axis as ``var_names``.
    var_names : iterable[str]
        Gene symbols defining the ``(n_genes,)`` axis of ``moments.Rnum`` /
        ``moments.Vsum``.  Matching is exact (case-sensitive) — fold case
        upstream the same way the regulon mask does.

    Returns
    -------
    PerPertMoments
        A copy with the perturbed-gene entries zeroed.  Perturbations whose
        label is not a gene in ``var_names`` (e.g. controls) are left untouched.
    """
    gene_idx = {g: i for i, g in enumerate(var_names)}
    cols = np.array([gene_idx.get(p, -1) for p in moments.perts], dtype=np.int64)
    rows = np.where(cols >= 0)[0]
    Rnum, Vsum = moments.Rnum.copy(), moments.Vsum.copy()
    Rnum[rows, cols[rows]] = 0.0
    Vsum[rows, cols[rows]] = 0.0
    return moments._replace(Rnum=Rnum, Vsum=Vsum)


def tf_zscores(
    moments: PerPertMoments, W_signed, Amask_tf, *, recalibrate: bool = True
) -> TFZScores:
    """Aggregate per-gene NB residuals over each TF's regulon into signed z-scores.

    For each (perturbation p, TF t):

        Num_{p,t} = Σ_g  W_signed_{g,t} · Rnum_{p,g}
        Den_{p,t} = Σ_g  W_signed_{g,t}² · Vsum_{p,g}
        Z_{p,t}   = Num / √Den           # ~ N(0, 1) under H0 ("regulon behaves as predicted")
        Z_emp     = (Z − median_p) / MAD_p          # robust per-TF recalibration

    Parameters
    ----------
    moments : PerPertMoments
        Output of ``per_perturbation_moments(...)``.

      : array (n_genes, n_tfs)
        Signed weights for the regulon aggregation.  Sign-only:
        ``np.sign(W·Amask_tf)``.  Magnitude-weighted: ``W·Amask_tf``.  The
        contract is ``W_signed[g, t] == 0`` wherever ``Amask_tf[g, t] == 0``.
    Amask_tf : array (n_genes, n_tfs)  {0, 1}
        Used only to count targets per TF for NaN handling.
    recalibrate : bool, default True
        Compute ``Z_emp`` via per-TF (median, MAD) robust standardisation.

    Returns
    -------
    TFZScores
    """
    W_signed = np.asarray(W_signed)
    Amask = np.asarray(Amask_tf)
    Num = moments.Rnum @ W_signed  # (n_perts, n_tfs)
    Den = moments.Vsum @ (W_signed**2)  # (n_perts, n_tfs)
    n_targets = (Amask > 0).sum(0).astype(np.int64)  # (n_tfs,)

    with np.errstate(divide="ignore", invalid="ignore"):
        Z = Num / np.sqrt(Den)
    Z = np.where(Den > 0, Z, np.nan)
    Z[:, n_targets == 0] = np.nan

    Z_emp = None
    if recalibrate:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
            med = np.nanmedian(Z, axis=0)
            mad = median_abs_deviation(Z, axis=0, scale="normal", nan_policy="omit")
        mad = np.where(mad > 1e-8, mad, np.nan)
        Z_emp = (Z - med[None, :]) / mad[None, :]

    return TFZScores(Z=Z, Z_emp=Z_emp, n_targets=n_targets)


def tf_score_frame(z: TFZScores, perts, tf_genes, *, dropna: bool = True) -> pd.DataFrame:
    """Melt a ``TFZScores`` into a long (perturbation, TF) DataFrame.

    ``z.Z`` / ``z.Z_emp`` are (n_perts, n_tfs) aligned to ``(perts, tf_genes)``;
    this is the row-for-row long view (one row per cell, C-order) used for
    inspection and for joining against per-(perturbation, TF) ground truth.

    Columns: ``perturbation, TF, Zscore, Zscore_emp, n_targets, abs_Zscore,
    abs_Zscore_emp``. With ``dropna=True`` (default) rows where ``Zscore`` is
    NaN (TF with no targets, or zero NB variance) are dropped — matching the
    notebook's prior behaviour.
    """
    perts = np.asarray(perts)
    tf_genes = np.asarray(tf_genes)
    n_perts, n_tfs = z.Z.shape
    if (len(perts), len(tf_genes)) != (n_perts, n_tfs):
        raise ValueError(
            f"axis mismatch: z.Z is {z.Z.shape} but got "
            f"{len(perts)} perts x {len(tf_genes)} tf_genes"
        )
    df = pd.DataFrame(
        {
            "perturbation": np.repeat(perts, n_tfs),
            "TF": np.tile(tf_genes, n_perts),
            "Zscore": z.Z.ravel().astype(np.float32),
            "n_targets": np.tile(z.n_targets, n_perts),
        }
    ).assign(abs_Zscore=lambda d: d["Zscore"].abs())
    if z.Z_emp is not None:
        df["Zscore_emp"] = z.Z_emp.ravel().astype(np.float32)
        df["abs_Zscore_emp"] = df["Zscore_emp"].abs()
    return df.dropna(subset=["Zscore"]).reset_index(drop=True) if dropna else df


# ── PR curves against metabolic GT labels ─────────────────────────────────────


def pr_curve(scores, labels, coverage=None) -> PRCurve:
    """Flatten (n_perts, n_tfs) scores/labels and compute a PR curve.

    ``scores`` should be higher for stronger evidence of TF response (e.g.
    ``np.abs(tf_zscores(...).Z_emp)``). ``coverage``, if given, restricts the
    evaluation to cells where it is True — see ``build_gt_labels`` for why
    this masking is load-bearing. Non-finite scores are dropped silently.

    ``prevalence`` is the positive rate over the evaluated cells and serves
    as the random-baseline AUPRC.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=bool).ravel()
    if coverage is not None:
        m = np.asarray(coverage, dtype=bool).ravel()
        scores, labels = scores[m], labels[m]
    finite = np.isfinite(scores)
    scores, labels = scores[finite], labels[finite]

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    auprc = float(average_precision_score(labels, scores))
    return PRCurve(
        precision=precision,
        recall=recall,
        thresholds=thresholds,
        auprc=auprc,
        prevalence=float(labels.mean()),
        n_positives=int(labels.sum()),
        n_total=int(labels.size),
    )
