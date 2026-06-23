import argparse
import hashlib
import json
import os

import numpy as np
import scanpy as sc

from essential.baselines import MarginalEstimator
from essential.utils import compute_topk_precision_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adata_path",
        type=str,
        default="/workspace/data/250516_TF_perturbseq/250516_TF_perturbseq.annotated.h5ad",
    )
    parser.add_argument("--preprocess_mode", type=str, default="normalized_concentration")
    parser.add_argument("--rt_bc", type=str, default="all")
    parser.add_argument("--consolidated_cluster", type=str, default="all")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--tag", type=str, default=None, help="A tag for the experiment.")
    return parser.parse_args()


def save_config(kwargs, output_path, tag=None):
    if tag is not None:
        kwargs["tag"] = tag
    with open(output_path, "w") as f:
        json.dump(kwargs, f, indent=4)
    return tag


def get_hash(kwargs):
    hash_kwargs = {k: v for k, v in kwargs.items() if k not in ["adata_path", "output_path", "tag"]}
    str_kwargs = json.dumps(hash_kwargs, sort_keys=True)
    tag = hashlib.sha256(str_kwargs.encode()).hexdigest()[:8]
    return tag


def main():
    args = parse_args()
    kwargs = vars(args)
    config_hash = get_hash(kwargs)
    folder_path = os.path.join(kwargs["output_path"], config_hash)
    os.makedirs(folder_path, exist_ok=True)

    adata = sc.read_h5ad(kwargs["adata_path"])
    if kwargs["rt_bc"] != "all":
        adata = adata[adata.obs["rt_bc"] == kwargs["rt_bc"]].copy()
    if kwargs["consolidated_cluster"] != "all":
        adata = adata[adata.obs["consolidated_cluster"] == kwargs["consolidated_cluster"]].copy()

    sc.pp.filter_genes(adata, min_cells=10)

    estimator = MarginalEstimator(adata, preprocess_mode=kwargs["preprocess_mode"])
    estimator.fit()
    scores = estimator.get_interaction_matrix(return_square=False, delta=0.1)

    experiment_tag = kwargs.get("tag") or config_hash
    hash_ = save_config(kwargs, os.path.join(folder_path, "config.json"), experiment_tag)
    output_file = os.path.join(folder_path, f"Amat.npz")
    np.savez_compressed(
        output_file,
        matrix=scores.values,
        genes=scores.index.to_numpy(),
        tfs=scores.columns.to_numpy(),
    )

    targetted_tfs = adata.obs["consensus_target"].unique()
    targetted_tfs = [t.lower() for t in targetted_tfs if t in adata.var_names]
    processed_a_mat_ = scores.loc[lambda x: x["regulator_gene"].isin(targetted_tfs)]

    topk_precision_df = compute_topk_precision_metrics(processed_a_mat_, experiment_tag)
    topk_precision_df.to_csv(os.path.join(folder_path, "topk_precision.csv"), index=False)


if __name__ == "__main__":
    main()
