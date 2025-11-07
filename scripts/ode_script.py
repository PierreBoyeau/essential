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

from essential.gpu_utils import select_best_gpus

select_best_gpus(n_gpus=1)

import scanpy as sc
from essential.ode import ODEstimator
from essential.utils import get_hash, compute_topk_precision_metrics
from ml_collections import config_flags
from absl import app, flags
import os
import json
import numpy as np
import optax


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Path to config file", lock_config=False)
flags.DEFINE_string("output_path", None, "Output path for results")
flags.mark_flag_as_required("config")
flags.mark_flag_as_required("output_path")


def main(_):
    """Main execution function."""
    config = FLAGS.config
    config.output_path = FLAGS.output_path
    experiment_tag = config.tag

    print(config)

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
    sc.pp.filter_genes(adata, min_cells=10)

    # model fitting
    ODEstimator.process_data(adata, latent_obsm_key=config.processing.latent_obsm_key)
    estimator = ODEstimator(adata, **config.estimator.to_dict())
    if config.do_lr_optimization:
        best_params = estimator.find_best_lr(**config.lr_optimization_params.to_dict())
        optimizer = optax.sgd(**best_params["best_params"])
        config.training.optimizer = optimizer
    estimator.fit(**config.training.to_dict())

    a_mat = estimator.get_interaction_matrix()

    # Process interaction matrix before computing metrics
    delta = config.estimator.get("delta", 0.125)
    processed_a_mat = ODEstimator.process_interaction_matrix(
        a_mat, return_square=False, delta=delta
    )

    config_path = os.path.join(folder_path, "config.json")
    with open(config_path, "w") as f:
        f.write(config.to_json_best_effort(indent=4))

    # Save interaction matrix
    output_file = os.path.join(folder_path, "Amat.npz")
    np.savez_compressed(output_file, matrix=a_mat.values, genes=a_mat.columns.to_numpy())

    # processed_a_mat_ = processed_a_mat.loc[lambda x: x["regulator_gene"].isin(targetted_tfs)]
    topk_precision_df = compute_topk_precision_metrics(processed_a_mat, experiment_tag)
    topk_precision_df.to_csv(os.path.join(folder_path, "topk_precision.csv"), index=False)

    summary_df = (
        topk_precision_df.groupby(["model", "type"])
        .apply(lambda x: x.query("is_evidence == True").shape[0])
        .sort_values(ascending=False)
    )
    history_df = estimator.epoch_history_df
    history_df.to_csv(os.path.join(folder_path, "history.csv"), index=False)
    step_history_df = estimator.step_history_df
    step_history_df.to_csv(os.path.join(folder_path, "step_history.csv"), index=False)
    print("--------------------------------")
    print("Experiment completed successfully")
    print("Summary of results:")
    print(summary_df)
    print("Config:")
    print(config)
    print("Results saved to: ", folder_path)
    print("--------------------------------")


if __name__ == "__main__":
    app.run(main)
