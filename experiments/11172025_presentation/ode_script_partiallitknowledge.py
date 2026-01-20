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
from essential.utils import load_regulondb_full
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
flags.DEFINE_string("heldout_targets", None, "Path to JSON file with heldout target genes")
flags.DEFINE_float("perc_targets_in_training", None, "Percentage of targets to include in training")
flags.DEFINE_integer("random_seed", None, "Random seed")
flags.mark_flag_as_required("config")
flags.mark_flag_as_required("output_path")
flags.mark_flag_as_required("heldout_targets")
flags.mark_flag_as_required("perc_targets_in_training")
flags.mark_flag_as_required("random_seed")


def build_aweight(
    adata: sc.AnnData, heldout_targets: list[str], perc_targets_in_training: float, random_seed: int
):
    ref_db = load_regulondb_full()
    ref_db_ = ref_db.set_index("tf_promoter")

    targets_in_literature_ = ref_db_["target_gene"].unique()
    targets_in_literature_ = [t.lower() for t in targets_in_literature_]
    var_names_ = [v.lower() for v in adata.var_names]
    covered_targets_ = np.intersect1d(targets_in_literature_, var_names_)

    heldout_targets_ = [t.lower() for t in heldout_targets]
    valid_targets = np.setdiff1d(covered_targets_, heldout_targets_)
    print("Total targets in literature: ", len(targets_in_literature_))
    print("Covered targets: ", len(covered_targets_))
    print("Heldout targets: ", len(heldout_targets_))
    print("Targets that can be used for training: ", len(valid_targets))
    print(
        f"Expected number of targets in training: ({perc_targets_in_training*100}% of {len(valid_targets)})",
        int(len(valid_targets) * perc_targets_in_training),
    )

    if perc_targets_in_training < 1.0:
        print("Sampling targets for training...")
        np.random.seed(random_seed)
        valid_targets = np.random.choice(
            valid_targets, size=int(len(valid_targets) * perc_targets_in_training), replace=False
        )
        print("Sampled targets for training: ", len(valid_targets))

        ref_db_ = ref_db_[ref_db_["target_gene"].str.lower().isin(valid_targets)]

    print("--------------------------------")
    print("Number of targets in training: ", len(ref_db_["target_gene"].unique()))

    Aweight = np.ones((adata.n_vars, adata.n_vars), dtype=np.float32)
    confidence_to_weight = {
        "S": 0.1,
        "C": 0.0,
        "W": 0.5,
        np.nan: 1.0,
    }

    for i, target_gene in enumerate(adata.var_names):
        for j, regulator_gene in enumerate(adata.var_names):
            target_gene_ = target_gene.lower()
            regulator_gene_ = regulator_gene.lower()
            key_ = f"{regulator_gene_}_{target_gene_}"
            if key_ in ref_db_.index:
                confidence_level = ref_db_.loc[key_, "confidenceLevel"]
                Aweight[i, j] = confidence_to_weight[confidence_level]
    return Aweight


def main(_):
    """Main execution function."""
    config = FLAGS.config
    config.output_path = FLAGS.output_path
    experiment_tag = config.tag
    heldout_targets_file = FLAGS.heldout_targets
    perc_targets_in_training = FLAGS.perc_targets_in_training
    random_seed = FLAGS.random_seed

    with open(heldout_targets_file, "r") as f:
        heldout_targets = json.load(f)

    config.heldout_targets = heldout_targets
    config.perc_targets_in_training = perc_targets_in_training
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
    aweight = build_aweight(adata, heldout_targets, perc_targets_in_training, random_seed)
    config.estimator.model_kwargs.Aweight = aweight
    estimator = ODEstimator(adata, **config.estimator.to_dict())
    if config.do_lr_optimization:
        best_params = estimator.find_best_lr(**config.lr_optimization_params.to_dict())
        optimizer = optax.sgd(**best_params["best_params"])
        config.training.optimizer = optimizer
    estimator.fit(**config.training.to_dict())

    a_mat = estimator.get_interaction_matrix()
    delta = config.estimator.get("delta", 0.125)
    processed_a_mat = ODEstimator.process_interaction_matrix(
        a_mat, return_square=False, delta=delta
    )
    config_path = os.path.join(folder_path, "config.json")
    with open(config_path, "w") as f:
        f.write(config.to_json_best_effort(indent=4))
    output_file = os.path.join(folder_path, "Amat.npz")
    np.savez_compressed(output_file, matrix=a_mat.values, genes=a_mat.columns.to_numpy())
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
