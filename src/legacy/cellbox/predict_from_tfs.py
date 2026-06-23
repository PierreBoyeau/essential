from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

from . import CellBoxEstimator


def run(config):
    output_dir = Path(config.outputs.output_dir)
    adata_test = sc.read_h5ad(config.adata_test)
    model = CellBoxEstimator.load(output_dir / "checkpoint")

    model.n_rollout_val = 1
    model._jit_predict = model._make_predict_step()

    perturbations = (
        adata_test.obs[config.perturbation_col].value_counts().loc[lambda x: x >= 1].index
    )
    perturbations = perturbations[perturbations.isin(model.adata.var_names)]

    all_preds, obs_rows = [], []
    for pert in perturbations:
        mask = np.asarray(adata_test.obs[config.perturbation_col] == pert)
        pred = model.predict(adata_test[mask], perturbation=pert)
        all_preds.append(pred)
        obs_rows.extend([{config.perturbation_col: pert}] * len(pred))

    adata_pred = sc.AnnData(
        np.vstack(all_preds),
        obs=pd.DataFrame(obs_rows),
        var=adata_test.var,
    )
    out_path = output_dir / "adata_pred_from_tfs.h5ad"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata_pred.write_h5ad(out_path)


if __name__ == "__main__":
    from absl import app, flags
    from ml_collections import config_flags

    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file("config", None, "Path to config file")

    def main(_):
        run(FLAGS.config)

    app.run(main)
