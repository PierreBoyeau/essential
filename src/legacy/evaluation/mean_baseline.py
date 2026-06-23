"""Mean-expression baseline for perturbation prediction.

Predicts the per-gene mean over all training cells as a constant response
for every test perturbation.  Produces an adata_pred.h5ad with the same
structure as predict.py so evaluate.py can consume it without modification.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

sys.path.insert(0, "/workspace/src")


def _get_layer(adata, layer=None):
    X = adata.layers[layer] if layer else adata.X
    return X.toarray() if sparse.issparse(X) else np.asarray(X, dtype=np.float32)


def run(config):
    model_cfg = getattr(config, "model", None)
    layer = getattr(model_cfg, "layer", None)
    layer_eval = getattr(model_cfg, "layer_eval", None) or layer

    adata_train = sc.read_h5ad(config.adata_train)
    adata_test = sc.read_h5ad(config.adata_test)
    adata_control = sc.read_h5ad(config.adata_control)

    mu = _get_layer(adata_train, layer_eval).mean(0)

    perturbations = (
        adata_test.obs[config.perturbation_col].value_counts().loc[lambda x: x >= 1].index
    )

    X = np.vstack([mu[None]] * len(perturbations))
    obs = pd.DataFrame({config.perturbation_col: list(perturbations)})

    adata_pred = sc.AnnData(X, obs=obs, var=adata_control.var)

    out_path = Path(config.outputs.output_dir) / config.outputs.adata_pred_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata_pred.write_h5ad(out_path)
    print(f"wrote {adata_pred.shape} → {out_path}")


if __name__ == "__main__":
    from absl import app, flags
    from ml_collections import config_flags

    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file("config", None, "Path to config file")

    def main(_):
        run(FLAGS.config)

    app.run(main)
