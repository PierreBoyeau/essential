import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import os
import subprocess
import hashlib
import json

def get_hash(config):
    """Generate hash from config, excluding paths.

    Args:
        config: ConfigDict containing experiment configuration

    Returns:
        str: 8-character hash of the configuration
    """
    config_dict_copy = config.to_dict()
    # Exclude paths from hash to ensure reproducibility
    hash_dict = {
        k: v
        for k, v in config_dict_copy.items()
        if k not in ["output_path"] and not (k == "processing" and "adata_path" in str(v))
    }
    # Also exclude adata_path from processing if it exists
    if "processing" in hash_dict and isinstance(hash_dict["processing"], dict):
        hash_dict["processing"] = {
            k: v for k, v in hash_dict["processing"].items() if k != "adata_path"
        }

    str_config = json.dumps(hash_dict, sort_keys=True)
    return hashlib.sha256(str_config.encode()).hexdigest()[:8]


def get_git_hash():
    """Get the current git commit hash."""
    try:
        file_dir = os.path.dirname(os.path.abspath(__file__))
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=file_dir)
            .decode("ascii")
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "not_a_git_repo"


def compute_metrics(
    results_df,
    gt_col="is_evidence",
    score_col="score",
    decision_cols=None,
):
    """
    Compute precision–recall metrics and optional decision-threshold metrics.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Must contain columns `gt_col` and `score_col`. If `decision_cols` is
        provided, each corresponding column should be boolean decisions (0/1).
    gt_col : str, default "is_evidence"
        Name of the ground-truth boolean column.
    score_col : str, default "score"
        Name of the predicted score column.
    decision_cols : list[str] or None, default None
        Optional list of decision columns. For each column name, the suffix
        after the first underscore is used to label metrics (e.g.,
        `decision_0p90` -> type `0p90`).

    Returns
    -------
    dict
        Dictionary containing `pr_auc`, `ap`, `ntotal` and, if `decision_cols`
        is given, `precision_*`, `recall_*`, `ndetections_*`, and
        `ntruepos_*`.
    """
    base_metrics = compute_pr_metrics(
        y_true=results_df[gt_col],
        y_score=results_df[score_col],
    )
    metrics_res = {
        "pr_auc": base_metrics["auc"],
        "ap": base_metrics["ap"],
        "ntotal": results_df.shape[0],
    }

    if decision_cols is not None:
        for decision_col in decision_cols:
            decision_type = decision_col.split("_")[1]
            decision_metrics = compute_pr_metrics(
                y_true=results_df[gt_col],
                y_score=results_df[score_col],
                y_decision=results_df[decision_col],
            )
            metrics_res[f"precision_{decision_type}"] = decision_metrics.get("precision", 0.0)
            metrics_res[f"recall_{decision_type}"] = decision_metrics.get("recall", 0.0)
            metrics_res[f"ndetections_{decision_type}"] = (results_df[decision_col]).sum()
            metrics_res[f"ntruepos_{decision_type}"] = (
                results_df[gt_col] & results_df[decision_col]
            ).sum()
    return metrics_res


def compute_pr_metrics(y_true, y_score, y_decision=None, return_pandas=False):
    """
    Compute precision–recall summary metrics, optionally at a decision threshold.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary ground-truth labels (0/1).
    y_score : array-like of shape (n_samples,)
        Continuous scores; larger implies more likely positive.
    y_decision : array-like of shape (n_samples,), optional
        Binary decisions (0/1). If provided, include precision and recall at
        this operating point.

    Returns
    -------
    dict
        Keys:
        - 'auc': area under the precision–recall curve
        - 'ap': average precision
        - 'precision' and 'recall' if y_decision is provided
    """
    if y_decision is not None:
        decision_precision = metrics.precision_score(y_true, y_decision, zero_division=0)
        decision_recall = metrics.recall_score(y_true, y_decision, zero_division=0)

    prec, rec, _ = metrics.precision_recall_curve(y_true, y_score)
    result = {
        "auc": metrics.auc(rec, prec),
        "ap": metrics.average_precision_score(y_true, y_score),
    }
    if y_decision is not None:
        result.update({"precision": decision_precision, "recall": decision_recall})
    if return_pandas:
        return pd.Series(result)
    return result

