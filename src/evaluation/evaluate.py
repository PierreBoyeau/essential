"""Evaluate perturbation predictions against held-out test cells.

Reads adata_pred (output of predict.py), adata_test, and adata_control.
For each perturbation, computes LFC-based metrics (MSE, Pearson R², top-DEG
variants) and, when adata_train is provided, the R² relative to the training
mean predictor baseline.

Outputs written to config.outputs.metrics_dir:
    perturbation_metrics.csv   -- one row per perturbation
    overall_metrics.csv        -- mean of each numeric metric
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

sys.path.insert(0, "/workspace/src")
from cellbox.metrics import profile_metrics


def _as_dense(X):
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


def _get_layer(adata, layer=None):
    X = adata.layers[layer] if layer else adata.X
    return _as_dense(X)


def _train_mean_profile(adata_train, perturbation_col, layer=None):
    profiles = []
    for t in adata_train.obs[perturbation_col].unique():
        mask = np.asarray(adata_train.obs[perturbation_col] == t)
        profiles.append(_get_layer(adata_train[mask], layer).mean(0))
    return np.mean(profiles, axis=0)


def run(config):
    model_cfg = getattr(config, "model", None)
    layer = getattr(model_cfg, "layer", None)
    # layer_eval: space for ground-truth metrics; defaults to layer.
    # For NB models (layer="raw"), set layer_eval="log1p" so GT matches
    # the log-CP10K space that predict_steady_state returns.
    layer_eval = getattr(model_cfg, "layer_eval", None) or layer

    adata_pred = sc.read_h5ad(config.outputs.adata_pred)
    adata_test = sc.read_h5ad(config.adata_test)
    adata_control = sc.read_h5ad(config.adata_control)

    mu_control = _get_layer(adata_control, layer_eval).mean(0)

    mu_baseline = None
    if config.adata_train:
        adata_train = sc.read_h5ad(config.adata_train)
        mu_baseline = _train_mean_profile(adata_train, config.perturbation_col, layer_eval)

    perturbations = (
        adata_pred.obs[config.perturbation_col].value_counts().loc[lambda x: x >= 1].index
    )

    var_names = adata_control.var_names
    rows, lfc_pred_rows, lfc_gt_rows = [], [], []
    for pert in perturbations:
        mask_pred = np.asarray(adata_pred.obs[config.perturbation_col] == pert)
        mask_gt = np.asarray(adata_test.obs[config.perturbation_col] == pert)
        if not mask_gt.any():
            continue
        mu_pred = _as_dense(adata_pred[mask_pred].X).mean(0)
        mu_gt = _get_layer(adata_test[mask_gt], layer_eval).mean(0)
        row_info = {
            "perturbation": pert,
            "n_cells_gt": int(mask_gt.sum()),
            "n_cells_pred": int(mask_pred.sum()),
            **profile_metrics(mu_gt, mu_pred, mu_control, mu_baseline=mu_baseline),
        }
        print(pert, row_info["n_cells_gt"], row_info["lfc_pearson_r"])
        rows.append(row_info)
        lfc_pred_rows.append(pd.Series(mu_pred - mu_control, index=var_names, name=pert))
        lfc_gt_rows.append(pd.Series(mu_gt - mu_control, index=var_names, name=pert))

    df = pd.DataFrame(rows).set_index("perturbation")
    overall = df.mean(numeric_only=True).rename("mean").to_frame()
    lfc_pred_df = pd.DataFrame(lfc_pred_rows)
    lfc_gt_df = pd.DataFrame(lfc_gt_rows)

    out_dir = Path(config.outputs.metrics_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "perturbation_metrics.csv")
    overall.to_csv(out_dir / "overall_metrics.csv", index=False)
    lfc_pred_df.to_csv(out_dir / "lfc_pred.csv")
    lfc_gt_df.to_csv(out_dir / "lfc_gt.csv")

    print(overall.to_string())
    return df, overall


if __name__ == "__main__":
    from absl import app, flags
    from ml_collections import config_flags

    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file("config", None, "Path to config file")

    def main(_):
        run(FLAGS.config)

    app.run(main)
