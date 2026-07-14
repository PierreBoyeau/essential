"""Minibatch training on NB-NLL: plain MLE (``fit``) and group-DRO (``fit_dro``).

Both loops work with any nn.Module whose ``__call__(x_tf, lib)`` returns
``(mean, concentration)`` — i.e. all models in this package.

``fit`` minimises the mean NB-NLL over the whole training set (empirical risk).
``fit_dro`` minimises the *worst-group* risk over perturbations via the
exponentiated-weights group-DRO scheme (Sagawa et al., 2020): a running weight
``q_c`` per perturbation is upweighted by ``exp(eta_q * loss_c)`` and the model
descends the ``q``-weighted loss.

When a per-cell ``baseline`` is supplied to ``fit_dro``, the per-perturbation
objective becomes the *regret* ``loss_c - baseline_c`` (e.g. against a W=0 null
fit on the whole training set).  Because the baseline is constant in θ, this
reshapes the ``q`` weighting only — the gradient is unchanged given ``q``.

Both return ``(final_state, history)``.  When a validation set is supplied,
history additionally carries per-evaluation ``validation_nll_avg`` (mean NLL over
the complete validation set) and ``validation_nll_worst`` (worst-case mean NLL
over the validation perturbations — its own group set, independent of train).
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from tqdm import tqdm

from .eval import per_perturbation_nll


def _validation_metrics(model, params, X_tf, Y_raw, lib, pert_labels, batch_size, nll_fn):
    """(avg, worst) mean NLL for a validation set, under the given ``nll_fn``.

    ``avg`` is the cell-weighted mean NLL over the complete set (same scale as
    the training loss); ``worst`` is the largest per-perturbation mean NLL.
    Both are computed from ``eval.per_perturbation_nll`` so the model contract is
    identical to every other evaluator in this package.
    """
    ppn = per_perturbation_nll(
        model, params, X_tf, Y_raw, lib, pert_labels, batch_size=batch_size, nll_fn=nll_fn
    )
    per_pert_mean = ppn.nll.mean(axis=1)  # (n_perts,) mean NLL/cell·gene per perturbation
    total = int(ppn.counts.sum())
    avg = float((per_pert_mean * ppn.counts).sum() / max(total, 1))
    worst = float(per_pert_mean.max())
    return avg, worst


# ── plain MLE ─────────────────────────────────────────────────────────────────


def _make_step(model, nll_fn):
    @jax.jit
    def step(state, x_tf, y_raw, lib):
        def loss_fn(params):
            mean, conc = model.apply({"params": params}, x_tf, lib)
            return nll_fn(mean, conc, y_raw).mean()

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss

    return step


def fit(
    model,
    X_tf,
    Y_raw,
    lib,
    *,
    n_steps: int,
    batch_size: int,
    lr: float,
    nll_fn,
    X_tf_val=None,
    Y_raw_val=None,
    lib_val=None,
    val_pert_labels=None,
    eval_every_n_steps: int = 50,
    eval_batch_size: int = 512,
    key=None,
    tqdm_: bool = True,
):
    """Train ``model`` by minibatch Adam on mean NLL (empirical risk).

    Each of the ``n_steps`` iterations draws one minibatch of ``batch_size``
    cells (a reshuffled sweep through the whole training set, reshuffling once
    exhausted) and takes one Adam step on its mean NLL.  Both parameter init
    and the shuffling are derived from a single ``key`` (default ``PRNGKey(0)``);
    pass a different key to vary either.

    ``nll_fn`` selects the likelihood — any ``(mean, conc, y_raw) -> (B,
    n_genes)`` callable, e.g. ``nb_nll`` or ``poisson_nll`` (``conc`` is
    ignored). No default is provided — pass it explicitly to avoid silently
    training under the wrong likelihood. Pass the same ``nll_fn`` to whichever
    eval helper scores this fit, since nothing here enforces that pairing.

    If a validation set (``X_tf_val, Y_raw_val, lib_val, val_pert_labels``) is
    supplied, validation metrics are computed every ``eval_every_n_steps`` steps
    (and on the final step) using ``eval_batch_size`` for the evaluation passes.

    Returns ``(final_state, history)`` where ``history`` always has
    ``'train_nll'`` (one entry per step) and, when a validation set is given,
    ``'eval_step'``, ``'validation_nll_avg'`` and ``'validation_nll_worst'``
    (one entry per evaluation, aligned by the step index in ``'eval_step'``).
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    init_key, shuffle_key = jax.random.split(key, 2)

    init_x = jnp.asarray(X_tf[:4])
    init_lib = jnp.asarray(lib[:4])
    params = model.init(init_key, init_x, init_lib)["params"]
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(lr),
    )
    step = _make_step(model, nll_fn)

    has_val = X_tf_val is not None
    if has_val and val_pert_labels is None:
        raise ValueError("val_pert_labels is required when a validation set is provided")

    shuffle_seed = int(shuffle_key[0]) & 0x7FFFFFFF
    rng = np.random.default_rng(shuffle_seed)
    history: dict[str, list] = {"train_nll": []}
    if has_val:
        history["eval_step"] = []
        history["validation_nll_avg"] = []
        history["validation_nll_worst"] = []

    n = X_tf.shape[0]
    perm = rng.permutation(n)
    pos = 0
    iterator = tqdm(range(n_steps)) if tqdm_ else range(n_steps)
    for t in iterator:
        if pos + batch_size > n:
            perm = rng.permutation(n)
            pos = 0
        b = perm[pos : pos + batch_size]
        pos += batch_size
        state, loss = step(
            state,
            jnp.asarray(X_tf[b]),
            jnp.asarray(Y_raw[b]),
            jnp.asarray(lib[b]),
        )
        history["train_nll"].append(float(loss))

        if has_val and ((t + 1) % eval_every_n_steps == 0 or t == n_steps - 1):
            avg, worst = _validation_metrics(
                model,
                state.params,
                X_tf_val,
                Y_raw_val,
                lib_val,
                val_pert_labels,
                batch_size=eval_batch_size,
                nll_fn=nll_fn,
            )
            history["eval_step"].append(t)
            history["validation_nll_avg"].append(avg)
            history["validation_nll_worst"].append(worst)

    return state, history


# ── group-DRO ───────────────────────────────────────────────────────────────


def _make_dro_step(model, eta_q: float, nll_fn):
    """One DRO step: update the group weights ``q`` then descend the weighted objective.

    ``Xb, Yb, libb`` are stacked per-condition minibatches with leading axis
    ``C`` (number of conditions): shapes ``(C, B, n_tf)``, ``(C, B, n_genes)``,
    ``(C, B)``.  ``base_c`` is the per-condition baseline mean NLL ``(C,)`` for
    this minibatch (zeros ⇒ plain worst-group loss).  ``logits`` are normalised
    log-weights ``log q`` carried across steps.  The ``q`` update uses the
    current-θ per-condition objective ``o_c = l_c - base_c`` (stopped gradient)
    exactly as in the algorithm, so a single backward pass suffices.
    """

    @jax.jit
    def step(state, logits, Xb, Yb, libb, base_c):
        def loss_fn(params):
            def one(x, y, l):
                mean, conc = model.apply({"params": params}, x, l)
                return nll_fn(mean, conc, y).mean()

            lc = jax.vmap(one)(Xb, Yb, libb)  # (C,) per-condition mean NLL
            oc = lc - base_c  # per-condition objective (regret when base_c != 0)
            logits_new = jax.nn.log_softmax(logits + eta_q * jax.lax.stop_gradient(oc))
            q = jnp.exp(logits_new)  # q constant w.r.t. params (stop_gradient above)
            L = jnp.sum(q * oc)  # base_c const in θ ⇒ same grad as Σ q·lc
            return L, (lc, oc, logits_new)

        (_, (lc, oc, logits_new)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        return state.apply_gradients(grads=grads), logits_new, lc, oc

    return step


def fit_dro(
    model,
    X_tf,
    Y_raw,
    lib,
    pert_labels,
    *,
    n_steps: int,
    batch_size: int,
    lr: float,
    eta_q: float,
    nll_fn,
    baseline=None,
    X_tf_val=None,
    Y_raw_val=None,
    lib_val=None,
    val_pert_labels=None,
    eval_every_n_steps: int = 50,
    eval_batch_size: int = 512,
    key=None,
    tqdm_: bool = True,
):
    """Group-DRO training over perturbations (worst-group risk).

    Each of the ``n_steps`` iterations:

    1. draw one minibatch of ``batch_size`` cells **with replacement** from every
       condition (so conditions smaller than ``batch_size`` are handled);
    2. compute the per-condition mean NB-NLL ``l_c`` at the current θ and the
       objective ``o_c = l_c - base_c`` (``base_c`` is the minibatch-matched
       baseline; ``o_c = l_c`` when no ``baseline`` is given);
    3. update the running weights ``q_c ← q_c · exp(eta_q · o_c)`` (normalised);
    4. take one Adam step on the ``q``-weighted objective ``Σ_c q_c o_c``.

    The full per-condition objective is computed every step (conditions are
    never sampled at random).  ``key`` (default ``PRNGKey(0)``) seeds both
    parameter init and minibatch sampling.

    ``baseline`` is an optional per-cell mean-NLL vector ``(n_cells,)`` aligned
    row-for-row with ``X_tf`` — e.g. ``eval.per_cell_nll`` of a W=0 null fit on
    the whole training set.  When given, the per-perturbation objective becomes
    the regret ``l_c - base_c``, where ``base_c`` averages the baseline over the
    same sampled minibatch (a paired, low-variance difference).  This reweights
    ``q`` only; the gradient given ``q`` is unchanged.

    Memory note: the stacked target batch is ``(C, B, n_genes)``; with many
    conditions keep ``batch_size`` modest.

    ``nll_fn`` selects the likelihood used for ``l_c`` — see ``fit`` for the
    contract (e.g. ``nb_nll``, or ``poisson_nll`` which ignores ``conc``). No
    default — pass it explicitly. Pass the same ``nll_fn`` to whichever eval
    helper scores this fit.

    Returns ``(final_state, history)``.  ``history`` always carries per-step
    ``'train_nll_avg'`` / ``'train_nll_worst'`` (mean / max ``l_c``) and
    ``'train_regret_avg'`` / ``'train_regret_worst'`` (mean / max ``o_c``; equal
    to the NLL entries when no ``baseline`` is given).  When a validation set is
    supplied, every ``eval_every_n_steps`` (and on the final step) it also
    records ``'eval_step'``, ``'validation_nll_avg'``, ``'validation_nll_worst'``
    and the group weights ``'q'`` at that step.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    init_key, sample_key = jax.random.split(key, 2)

    pert_labels = np.asarray(pert_labels)
    groups, group_idx = np.unique(pert_labels, return_inverse=True)
    C = len(groups)
    group_members = [np.where(group_idx == c)[0] for c in range(C)]

    if baseline is not None:
        baseline = np.asarray(baseline, dtype=np.float32)
        if baseline.shape != (X_tf.shape[0],):
            raise ValueError(
                f"baseline must be a per-cell vector of shape {(X_tf.shape[0],)}, "
                f"got {baseline.shape}"
            )

    init_x = jnp.asarray(X_tf[:4])
    init_lib = jnp.asarray(lib[:4])
    params = model.init(init_key, init_x, init_lib)["params"]
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(lr),
    )
    step = _make_dro_step(model, eta_q, nll_fn)

    has_val = X_tf_val is not None
    if has_val and val_pert_labels is None:
        raise ValueError("val_pert_labels is required when a validation set is provided")

    sample_seed = int(sample_key[0]) & 0x7FFFFFFF
    rng = np.random.default_rng(sample_seed)
    logits = jnp.zeros(C)  # uniform log-weights

    history: dict[str, list] = {
        "train_nll_avg": [],
        "train_nll_worst": [],
        "train_regret_avg": [],
        "train_regret_worst": [],
    }
    if has_val:
        history["eval_step"] = []
        history["validation_nll_avg"] = []
        history["validation_nll_worst"] = []
        history["q"] = []

    iterator = tqdm(range(n_steps)) if tqdm_ else range(n_steps)
    for t in iterator:
        idx = np.stack(
            [rng.choice(group_members[c], size=batch_size, replace=True) for c in range(C)]
        )  # (C, B)
        base_c = (
            np.zeros(C, dtype=np.float32)
            if baseline is None
            else baseline[idx].mean(axis=1)  # (C,) matched to the sampled cells
        )
        state, logits, lc, oc = step(
            state,
            logits,
            jnp.asarray(X_tf[idx]),
            jnp.asarray(Y_raw[idx]),
            jnp.asarray(lib[idx]),
            jnp.asarray(base_c),
        )
        lc_np, oc_np = np.asarray(lc), np.asarray(oc)
        history["train_nll_avg"].append(float(lc_np.mean()))
        history["train_nll_worst"].append(float(lc_np.max()))
        history["train_regret_avg"].append(float(oc_np.mean()))
        history["train_regret_worst"].append(float(oc_np.max()))

        if has_val and ((t + 1) % eval_every_n_steps == 0 or t == n_steps - 1):
            avg, worst = _validation_metrics(
                model,
                state.params,
                X_tf_val,
                Y_raw_val,
                lib_val,
                val_pert_labels,
                batch_size=eval_batch_size,
                nll_fn=nll_fn,
            )
            history["eval_step"].append(t)
            history["validation_nll_avg"].append(avg)
            history["validation_nll_worst"].append(worst)
            history["q"].append(np.asarray(jnp.exp(logits)))

    return state, history
