"""Prediction metrics on mean expression profiles.

The core is :func:`profile_metrics`: given three ``(n_genes,)`` mean expression
profiles -- observed (``mu_gt``), predicted (``mu_pred``) and control
(``mu_control``) -- it returns scalar MSE and Pearson metrics on both the mean
profile and the log fold-change vs control (``lfc = mu - mu_control``), each
over all genes and over the top-K DEGs (largest ``|lfc_gt|``).

It is pure and framework-agnostic, so the same function serves test-time
reporting (``perturbation_prediction_metrics.PerturbationPredictionMetrics``)
and in-training validation (``CellBoxEstimator.validate_perturbations``); the
jax↔host boundary sits at the predicted mean profile.
"""

import numpy as np
import scipy.stats as stats


def _safe_pearson_r(a, b):
    """Signed Pearson correlation, NaN if undefined (e.g. constant input)."""
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    r, _ = stats.pearsonr(a, b)
    return float(r)


def _safe_pearson_r2(a, b):
    """Squared Pearson correlation, NaN if undefined (e.g. constant input)."""
    r = _safe_pearson_r(a, b)
    return r**2 if not np.isnan(r) else np.nan


def profile_metrics(mu_gt, mu_pred, mu_control, n_top_degs=20, mu_baseline=None):
    """Scalar prediction metrics from three ``(n_genes,)`` mean profiles.

    Returns mean-profile and log-fold-change (vs control) MSE and Pearson
    metrics, over all genes and over the top-K DEGs (largest ``|lfc_gt|``).

    If ``mu_baseline`` is provided (per-gene mean over training perturbations),
    also returns ``r2``: R²_i = 1 - Σ_g(mu_pred-mu_gt)² / Σ_g(mu_baseline-mu_gt)².
    """
    lfc_gt = mu_gt - mu_control
    lfc_pred = mu_pred - mu_control
    deg = np.argsort(np.abs(lfc_gt))[::-1][:n_top_degs]

    def mse(a, b):
        return float(np.mean((a - b) ** 2))

    out = {
        "mu_mse": mse(mu_pred, mu_gt),
        "mu_pearson": _safe_pearson_r2(mu_pred, mu_gt),
        "lfc_mse": mse(lfc_pred, lfc_gt),
        "lfc_pearson_r": _safe_pearson_r(lfc_pred, lfc_gt),
        "lfc_pearson": _safe_pearson_r2(lfc_pred, lfc_gt),
        "mu_mse_top_degs": mse(mu_pred[deg], mu_gt[deg]),
        "mu_pearson_top_degs": _safe_pearson_r2(mu_pred[deg], mu_gt[deg]),
        "lfc_mse_top_degs": mse(lfc_pred[deg], lfc_gt[deg]),
        "lfc_pearson_top_degs": _safe_pearson_r2(lfc_pred[deg], lfc_gt[deg]),
    }
    if mu_baseline is not None:
        mu_baseline = np.asarray(mu_baseline)
        ss_res = float(np.sum((mu_pred - mu_gt) ** 2))
        ss_tot = float(np.sum((mu_baseline - mu_gt) ** 2))
        out["r2"] = float(1 - ss_res / ss_tot) if ss_tot != 0 else np.nan
    return out
