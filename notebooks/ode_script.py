"""ODE estimation script using ml-collections for configuration management.

This script fits ODE models to perturbation sequencing data using the ODEstimator class.
Configurations are managed via ml-collections config files, with support for command-line
overrides of any parameter.

Usage:
    python ode_script.py --config=configs/models/dynamic_cellbox.py --output_path=outputs/
    
    # With overrides:
    python ode_script.py --config=configs/models/dynamic_cellbox.py \
        --config.processing.consolidated_cluster=CD4 \
        --config.training.n_epochs=500 \
        --output_path=outputs/
"""

import scanpy as sc
from essential.ode import ODEstimator
from ml_collections import config_flags
from absl import app, flags
import hashlib
import os
import json
import numpy as np


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Path to config file", lock_config=False)
flags.DEFINE_string("output_path", None, "Output path for results")
flags.mark_flag_as_required("config")
flags.mark_flag_as_required("output_path")


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


def main(_):
    """Main execution function."""
    config = FLAGS.config
    config.output_path = FLAGS.output_path


    config_hash = get_hash(config)
    folder_path = os.path.join(config.output_path, config_hash)
    os.makedirs(folder_path, exist_ok=True)

    # data processing
    adata = sc.read_h5ad(config.processing.adata_path)
    if config.processing.rt_bc != "all":
        adata = adata[adata.obs["rt_bc"] == config.processing.rt_bc].copy()
    if config.processing.consolidated_cluster != "all":
        adata = adata[
            adata.obs["consolidated_cluster"] == config.processing.consolidated_cluster
        ].copy()

    # model fitting
    estimator = ODEstimator(adata, **config.estimator.to_dict())
    estimator.fit(**config.training.to_dict())

    a_mat = estimator.get_interaction_matrix()
    config_to_save = config.to_dict()
    config_to_save["config_hash"] = config_hash
    config_path = os.path.join(folder_path, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_to_save, f, indent=4)

    # Save interaction matrix
    output_file = os.path.join(folder_path, "Amat.npz")
    np.savez_compressed(output_file, matrix=a_mat.values, genes=a_mat.columns.to_numpy())


if __name__ == "__main__":
    app.run(main)
