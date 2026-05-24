"""Prediction metrics on mean expression profiles.

The core is :func:`profile_metrics`: given three ``(n_genes,)`` mean expression
profiles -- observed (``mu_gt``), predicted (``mu_pred``) and control
(``mu_control``) -- it returns scalar MSE and squared-Pearson metrics on both
the mean profile and the log fold-change vs control (``lfc = mu - mu_control``),
each over all genes and over the top-K DEGs (largest ``|lfc_gt|``).

It is pure and framework-agnostic, so the same function serves test-time
reporting (``perturbation_prediction_metrics.PerturbationPredictionMetrics``)
and in-training validation (``CellBoxEstimator.validate_perturbations``); the
jax↔host boundary sits at the predicted mean profile.
"""

import numpy as np
import scipy.stats as stats


def _safe_pearson_r2(a, b):
    """Squared Pearson correlation, NaN if undefined (e.g. constant input)."""
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    r, _ = stats.pearsonr(a, b)
    return r**2


def profile_metrics(mu_gt, mu_pred, mu_control, n_top_degs=20):
    """Scalar prediction metrics from three ``(n_genes,)`` mean profiles.

    Returns mean-profile and log-fold-change (vs control) MSE and squared-Pearson,
    over all genes and over the top-K DEGs (largest ``|mu_gt - mu_control|``).
    """
    lfc_gt = mu_gt - mu_control
    lfc_pred = mu_pred - mu_control
    deg = np.argsort(np.abs(lfc_gt))[::-1][:n_top_degs]

    def mse(a, b):
        return float(np.mean((a - b) ** 2))

    return {
        "mu_mse": mse(mu_pred, mu_gt),
        "mu_pearson": _safe_pearson_r2(mu_pred, mu_gt),
        "lfc_mse": mse(lfc_pred, lfc_gt),
        "lfc_pearson": _safe_pearson_r2(lfc_pred, lfc_gt),
        "mu_mse_top_degs": mse(mu_pred[deg], mu_gt[deg]),
        "mu_pearson_top_degs": _safe_pearson_r2(mu_pred[deg], mu_gt[deg]),
        "lfc_mse_top_degs": mse(lfc_pred[deg], lfc_gt[deg]),
        "lfc_pearson_top_degs": _safe_pearson_r2(lfc_pred[deg], lfc_gt[deg]),
    }
