"""Metrics for held-out perturbation prediction.

The core is :func:`profile_metrics`: given three ``(n_genes,)`` mean expression
profiles -- observed (``mu_gt``), predicted (``mu_pred``) and control
(``mu_control``) -- it returns scalar MSE and squared-Pearson metrics on both
the mean profile and the log fold-change vs control (``lfc = mu - mu_control``),
each computed over all genes and over the top-K DEGs (largest ``|lfc_gt|``).

It is pure and framework-agnostic, so the same function serves test-time
reporting and in-training validation; the jax↔host boundary sits at the
predicted mean profile.

:class:`PerturbationPredictionMetrics` is a thin wrapper that reduces
per-perturbation cell arrays to those mean profiles and tabulates the result.
"""

import numpy as np
import pandas as pd

# profile_metrics lives in the cellbox package (single source of truth, shared
# with CellBoxEstimator.validate_perturbations); re-exported here for the API.
from legacy.cellbox.metrics import profile_metrics


class PerturbationPredictionMetrics:
    """Tabulate :func:`profile_metrics` over a set of held-out perturbations.

    Parameters
    ----------
    n_top_degs:
        Number of top DEGs (by observed ``|mu_gt - mu_control|``) used for the
        ``*_top_degs`` metrics.
    """

    def __init__(self, n_top_degs: int = 20):
        self.n_top_degs = n_top_degs

    def compute_metrics(self, X_gt0, X_gt, X_pred, targets=None, mu_baseline=None):
        """Per-perturbation scalar metrics, one row per target.

        Parameters
        ----------
        X_gt0:
            (n_cells0, n_genes) observed control cells.
        X_gt, X_pred:
            Lists of (n_cells, n_genes) arrays, aligned with each other and with
            ``targets``.
        targets:
            Optional perturbation names; defaults to integer positions.
        mu_baseline:
            (n_genes,) per-gene mean over training perturbations, used to
            compute R²_i.  When None, ``r2`` is NaN.
        """
        if len(X_gt) != len(X_pred):
            raise ValueError(f"X_gt ({len(X_gt)}) and X_pred ({len(X_pred)}) length mismatch")
        if targets is None:
            targets = list(range(len(X_gt)))
        mu_control = np.asarray(X_gt0).mean(0)

        rows = []
        for target, Xg, Xp in zip(targets, X_gt, X_pred):
            Xg, Xp = np.asarray(Xg), np.asarray(Xp)
            mu_gt = Xg.mean(0)
            lfc_gt = mu_gt - mu_control
            rows.append(
                {
                    "target": target,
                    "n_cells_gt": Xg.shape[0],
                    "n_cells_pred": Xp.shape[0],
                    "lfc_norm": float(np.linalg.norm(lfc_gt)),
                    **profile_metrics(mu_gt, Xp.mean(0), mu_control, self.n_top_degs, mu_baseline),
                }
            )
        return pd.DataFrame(rows).set_index("target")

    def compute_gene_metrics(self, X_gt, X_pred, mu_baseline, var_names=None):
        """Per-gene R², one row per gene.

        R²_g = 1 - Σ_i(mu_pred_{g,i} - mu_gt_{g,i})² / Σ_i(mu_baseline_g - mu_gt_{g,i})²
        """
        MU_gt = np.stack([np.asarray(Xg).mean(0) for Xg in X_gt])  # (n_perts, n_genes)
        MU_pred = np.stack([np.asarray(Xp).mean(0) for Xp in X_pred])  # (n_perts, n_genes)
        mu_baseline = np.asarray(mu_baseline)  # (n_genes,)

        ss_res = np.sum((MU_pred - MU_gt) ** 2, axis=0)
        ss_tot = np.sum((mu_baseline[None, :] - MU_gt) ** 2, axis=0)
        r2 = np.where(ss_tot != 0, 1 - ss_res / ss_tot, np.nan)

        df = pd.DataFrame({"r2": r2}, index=var_names)
        df.index.name = "gene"
        return df

    def from_result(self, result, mu_baseline=None):
        """Compute metrics from a ``run()`` result dict.

        Returns a dict with keys:
        - ``perturbation_centric_metrics``: DataFrame, one row per perturbation.
        - ``gene_centric_metrics``: DataFrame, one row per gene (None when mu_baseline is absent).
        - ``overall_metrics``: dict, mean of each numeric column in perturbation_centric_metrics.
        """
        df = self.compute_metrics(
            result["X_gt0"],
            result["X_gt"],
            result["X_pred"],
            targets=result["test_targets"],
            mu_baseline=mu_baseline,
        )
        df["model_name"] = result["model_name"]
        df = df.reset_index()

        gene_df = None
        if mu_baseline is not None:
            gene_df = self.compute_gene_metrics(
                result["X_gt"],
                result["X_pred"],
                mu_baseline,
                var_names=result.get("var_names"),
            )

        return {
            "perturbation_centric_metrics": df,
            "gene_centric_metrics": gene_df,
            "overall_metrics": df.mean(numeric_only=True).to_dict(),
        }
