"""
Kernel Deviation from Stationarity (KDS) utilities.

Reference: Lorch et al., "Causal Modeling with Stationary Diffusions", AISTATS 2024.

Three estimators are available for the KDS U-statistic:
  - "linear"     (default): pairs x[:N//2] with x[N//2:], O(N) evaluations
  - "u-statistic": full N×N matrix excluding diagonal, O(N²) — memory-intensive
  - "v-statistic": full N×N matrix including diagonal, O(N²) — memory-intensive
"""

from functools import partial

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


def rbf_kernel(x, y, bandwidth=1.0):
    """Gaussian RBF kernel k(x, y) = exp(-||x-y||^2 / (2 * bw^2))."""
    return jnp.exp(-jnp.sum(jnp.square(x - y)) / (2.0 * bandwidth**2))


def median_bandwidth(X):
    """Median heuristic: bw = sqrt(median(||x_i - x_j||^2) / 2).

    Parameters
    ----------
    X : (N, d)

    Returns
    -------
    scalar jnp array
    """
    i, j = jnp.triu_indices(X.shape[0], k=1)
    sq_dists = jnp.sum(jnp.square(X[i] - X[j]), axis=-1)
    return jnp.sqrt(jnp.median(sq_dists) / 2.0)


# ---------------------------------------------------------------------------
# Hessian diagonal helper
# ---------------------------------------------------------------------------


def _hess_diag(f, x):
    """Diagonal of the Hessian of scalar f at x, without forming the full matrix.

    Uses d forward-over-reverse JVP calls via vmap:
        (H @ e_i)[i]  =  H[i, i]

    Memory: O(d) per JVP call (vs O(d^2) for jax.hessian).
    Compute: O(d^2) — same asymptotic order as the full Hessian, but the
             (d, d) intermediate is never materialised as a single array.

    Parameters
    ----------
    f : scalar-valued callable
    x : (d,) array

    Returns
    -------
    (d,) array  diag(H_f(x))
    """
    grad_f = jax.grad(f)
    eye = jnp.eye(x.shape[0])
    # vmap over basis vectors: tangents[i] = H @ e_i  (i-th column of H)
    _, tangents = jax.vmap(lambda v: jax.jvp(grad_f, (x,), (v,)))(eye)
    return jnp.diag(tangents)  # tangents[i, i] = H[i, i]


# ---------------------------------------------------------------------------
# Single-pair operator: L_x L_y k(x, y)
# ---------------------------------------------------------------------------


def compute_LxLy_k(x, y, f_x, c_x, f_y, c_y, bandwidth=1.0):
    """
    Analytically computes L_x L_y k(x, y) in O(d) time and memory.
    This entirely avoids nested jax.grad and jax.jvp calls which cause OOMs.
    """
    bw2 = bandwidth**2
    r = x - y
    r_sq = jnp.square(r)
    kv = jnp.exp(-jnp.sum(r_sq) / (2.0 * bw2))

    # Precompute common analytical terms
    r_sq_bw2 = r_sq / bw2
    r_sq_bw2_m1 = r_sq_bw2 - 1.0

    F = jnp.dot(f_y, r)
    S = jnp.sum(c_y**2 * r_sq_bw2_m1)

    # 1. Analytical grad_x (Ly_k)
    grad_A = (kv / bw2) * (f_y - (F / bw2) * r)
    grad_B = 0.5 * (kv / bw2**2) * r * (2.0 * c_y**2 - S)
    grad_x = grad_A + grad_B

    # 2. Analytical diag(H_x (Ly_k))
    diag_H_A = (kv / bw2**2) * (-2.0 * f_y * r + F * r_sq_bw2_m1)
    diag_H_B = 0.5 * (kv / bw2**2) * (c_y**2 * (2.0 - 4.0 * r_sq_bw2) + S * r_sq_bw2_m1)
    diag_H_x = diag_H_A + diag_H_B

    # 3. Combine with L_x operator
    drift_term = jnp.dot(f_x, grad_x)
    diff_term = 0.5 * jnp.dot(c_x**2, diag_H_x)

    return drift_term + diff_term


# ---------------------------------------------------------------------------
# Batched KDS estimator
# ---------------------------------------------------------------------------


def compute_kds_batch(X, F, C, bandwidth=1.0, estimator="linear"):
    """KDS estimator over a batch of N data points.

    Parameters
    ----------
    X        : (N, d)  data samples
    F        : (N, d)  drift f evaluated at each sample
    C        : (N, d)  diagonal of sigma evaluated at each sample
    bandwidth: float or jnp scalar
    estimator: str  "linear" (default), "u-statistic", or "v-statistic"

    Returns
    -------
    scalar  KDS estimate
    """
    N = X.shape[0]
    _pair = partial(compute_LxLy_k, bandwidth=bandwidth)

    if estimator == "linear":
        N_trim = N - (N % 2)
        half = N_trim // 2
        vals = jax.vmap(_pair)(
            X[:half],
            X[half:N_trim],
            F[:half],
            C[:half],
            F[half:N_trim],
            C[half:N_trim],
        )
        return jnp.mean(vals)

    elif estimator == "u-statistic":
        vmap_y = jax.vmap(_pair, in_axes=(None, 0, None, None, 0, 0))
        vmap_xy = jax.vmap(vmap_y, in_axes=(0, None, 0, 0, None, None))
        KDS_matrix = vmap_xy(X, X, F, C, F, C)
        mask = 1.0 - jnp.eye(N)
        return jnp.sum(KDS_matrix * mask) / (N * (N - 1))

    elif estimator == "v-statistic":
        vmap_y = jax.vmap(_pair, in_axes=(None, 0, None, None, 0, 0))
        vmap_xy = jax.vmap(vmap_y, in_axes=(0, None, 0, 0, None, None))
        return jnp.mean(vmap_xy(X, X, F, C, F, C))

    else:
        raise ValueError(
            f"Unknown estimator '{estimator}'. " "Options: 'linear', 'u-statistic', 'v-statistic'."
        )


# ---------------------------------------------------------------------------
# Top-level: KDS from an SDE model + dataset
# ---------------------------------------------------------------------------


def compute_kds(sde_model, params, X, u=None, bandwidth=1.0, estimator="linear"):
    """Compute the KDS loss for a dataset X under intervention u.

    Parameters
    ----------
    sde_model : SDE (flax.linen.Module)
    params    : dict  flax parameter dict
    X         : (N, d)  data samples
    u         : int or None  intervention target (None = observational)
    bandwidth : float or jnp scalar
    estimator : str  "linear" (default), "u-statistic", or "v-statistic"

    Returns
    -------
    scalar  KDS estimate
    """
    F, c, _ = sde_model.apply({"params": params}, X, u, method=sde_model.forward)
    # c is (d,) and constant across x; broadcast to (N, d)
    C = jnp.broadcast_to(c[None], (X.shape[0], c.shape[0]))
    return compute_kds_batch(X, F, C, bandwidth=bandwidth, estimator=estimator)
