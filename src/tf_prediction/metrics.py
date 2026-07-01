"""Prediction metrics on mean expression profiles.

profile_metrics: given three (n_genes,) mean expression profiles -- observed
(mu_gt), predicted (mu_pred) and control (mu_control) -- returns:

  lfc_mse                  MSE on LFC (mu - mu_control) over all genes
  lfc_pearson_r            Pearson r on LFC over all genes
  lfc_mse_top{k}_degs      MSE on top-K DEG subset (K in _TOP_K)
  lfc_pearson_r_top{k}_degs Pearson r on top-K DEG subset
  auroc_top{k}_degs        AUROC for ranking top-K true DEGs by |lfc_pred|
  r2 (optional)            1 - SS_res/SS_tot vs a training-mean baseline

Note: mu_mse == lfc_mse and mu_pearson_r == lfc_pearson_r always (MSE and
Pearson r are both shift-invariant), so the mu_* variants are omitted.

DEG subsets are selected by largest |lfc_gt|. K values are set by _TOP_K.
"""

import numpy as np
import scipy.stats as stats
from sklearn.metrics import roc_auc_score

_TOP_K = (20, 100)


def _safe_pearson_r(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    r, _ = stats.pearsonr(a, b)
    return float(r)


def _auroc(lfc_gt, lfc_pred, k):
    y_true = np.zeros(len(lfc_gt))
    y_true[np.argsort(np.abs(lfc_gt))[::-1][:k]] = 1.0
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return np.nan
    return float(roc_auc_score(y_true, np.abs(lfc_pred)))


def profile_metrics(mu_gt, mu_pred, mu_control, mu_baseline=None):
    """Scalar prediction metrics from three (n_genes,) mean profiles."""
    lfc_gt = mu_gt - mu_control
    lfc_pred = mu_pred - mu_control

    def mse(a, b):
        return float(np.mean((a - b) ** 2))

    out = {
        "lfc_mse": mse(lfc_pred, lfc_gt),
        "lfc_pearson_r": _safe_pearson_r(lfc_pred, lfc_gt),
    }

    for k in _TOP_K:
        deg = np.argsort(np.abs(lfc_gt))[::-1][:k]
        out[f"lfc_mse_top{k}_degs"] = mse(lfc_pred[deg], lfc_gt[deg])
        out[f"lfc_pearson_r_top{k}_degs"] = _safe_pearson_r(lfc_pred[deg], lfc_gt[deg])
        out[f"auroc_top{k}_degs"] = _auroc(lfc_gt, lfc_pred, k)

    if mu_baseline is not None:
        mu_baseline = np.asarray(mu_baseline)
        ss_res = float(np.sum((mu_pred - mu_gt) ** 2))
        ss_tot = float(np.sum((mu_baseline - mu_gt) ** 2))
        out["r2"] = float(1 - ss_res / ss_tot) if ss_tot != 0 else np.nan

    return out
