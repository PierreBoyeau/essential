import pandas as pd
from perturbation_prediction_metrics import PerturbationPredictionMetrics
from predictors import build

from data import prepare_data


def run(config) -> pd.DataFrame:
    """Prepare data, train ``config.model_name``, predict, and return metrics."""
    adata_train, adata_test, adata_control, Amask = prepare_data(config)
    pred = build(config.model_name, config, adata_train, Amask)
    pred.fit()
    result = pred.collect_predictions(adata_test, adata_control, config.perturbation_col)
    result["model_name"] = config.tag
    metrics = PerturbationPredictionMetrics().from_result(result, output_path=config.output_path)
    print(metrics.mean(numeric_only=True))
    return metrics


if __name__ == "__main__":
    from absl import app, flags
    from ml_collections import config_flags

    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file("config", "config.py", "Path to config file")

    def main(_):
        run(FLAGS.config)

    app.run(main)
