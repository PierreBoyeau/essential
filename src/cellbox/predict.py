import numpy as np
import pandas as pd
import scanpy as sc

from cellbox import CellBoxEstimator


def run(config):
    adata_control, _ = CellBoxEstimator.prepare_data(
        adata_path=config.adata_control,
        min_library_size=config.prepare_data.min_library_size,
        perturbation_col=config.perturbation_col,
        normalization=config.prepare_data.normalization,
    )
    adata_test = sc.read_h5ad(config.adata_test)
    model = CellBoxEstimator.load(config.outputs.checkpoint_dir)

    perturbations = (
        adata_test.obs[config.perturbation_col].value_counts().loc[lambda x: x >= 1].index
    )
    perturbations = perturbations[perturbations.isin(model.adata.var_names)]

    all_preds, obs_rows = [], []
    for pert in perturbations:
        pred = model.predict(adata_control, perturbation=pert)
        all_preds.append(pred)
        obs_rows.extend([{config.perturbation_col: pert}] * len(pred))

    adata_pred = sc.AnnData(
        np.vstack(all_preds),
        obs=pd.DataFrame(obs_rows),
        var=adata_control.var,
    )
    adata_pred.write_h5ad(config.outputs.adata_pred)


if __name__ == "__main__":
    from absl import app, flags
    from ml_collections import config_flags

    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file("config", None, "Path to config file")

    def main(_):
        run(FLAGS.config)

    app.run(main)
