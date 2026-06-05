import os

import numpy as np
import pandas as pd
from perturbation_prediction_metrics import PerturbationPredictionMetrics
from predictors import build

from data import _as_dense, prepare_data


def _train_mean_profile(adata_train, perturbation_col):
    """Mean of per-perturbation mean profiles over all training conditions."""
    profiles = []
    for t in adata_train.obs[perturbation_col].unique():
        mask = np.asarray(adata_train.obs[perturbation_col] == t)
        profiles.append(_as_dense(adata_train[mask].X).mean(0))
    return np.mean(profiles, axis=0)


def _save_metrics(metrics, output_path):
    os.makedirs(output_path, exist_ok=True)
    metrics["perturbation_centric_metrics"].to_csv(
        os.path.join(output_path, "perturbation_centric_metrics.csv"), index=False
    )
    if metrics["gene_centric_metrics"] is not None:
        metrics["gene_centric_metrics"].to_csv(
            os.path.join(output_path, "gene_centric_metrics.csv")
        )
    pd.DataFrame([metrics["overall_metrics"]]).to_csv(
        os.path.join(output_path, "overall_metrics.csv"), index=False
    )


def run(config) -> dict:
    """Prepare data, train ``config.model_name``, predict, and return metrics."""
    adata_train, adata_test, adata_control, Amask = prepare_data(config)
    mu_baseline = _train_mean_profile(adata_train, config.perturbation_col)
    pred = build(config.model_name, config, adata_train, Amask)
    pred.fit()
    result = pred.collect_predictions(adata_test, adata_control, config.perturbation_col)
    result["model_name"] = config.tag
    metrics = PerturbationPredictionMetrics().from_result(result, mu_baseline=mu_baseline)
    print(metrics["overall_metrics"])
    _save_metrics(metrics, config.output_path)
    return metrics


if __name__ == "__main__":
    from absl import app, flags
    from ml_collections import config_flags

    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file("config", "config.py", "Path to config file")

    def main(_):
        run(FLAGS.config)

    app.run(main)
