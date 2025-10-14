import scanpy as sc
import argparse
import hashlib
import os
import json
import numpy as np


from essential.baselines import MarginalEstimator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adata_path",
        type=str,
        default="../data/250516_TF_perturbseq/250516_TF_perturbseq.annotated.h5ad",
    )
    parser.add_argument(
        "--preprocess_mode", type=str, default="normalized_concentration"
    )
    parser.add_argument("--rt_bc", type=str, default="all")
    parser.add_argument("--consolidated_cluster", type=str, default="all")
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()


def save_config(kwargs, output_path, tag=None):
    if tag is not None:
        kwargs["tag"] = tag
    with open(output_path, "w") as f:
        json.dump(kwargs, f, indent=4)
    return tag


def get_hash(kwargs):
    hash_kwargs = {
        k: v for k, v in kwargs.items() if k not in ["adata_path", "output_path"]
    }
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
        adata = adata[
            adata.obs["consolidated_cluster"] == kwargs["consolidated_cluster"]
        ].copy()

    estimator = MarginalEstimator(
        adata, preprocess_mode=kwargs["preprocess_mode"]
    )
    estimator.fit()
    scores, tf_knockdowns = estimator.get_interaction_matrix()
    hash_ = save_config(kwargs, os.path.join(folder_path, "config.json"), config_hash)
    output_file = os.path.join(folder_path, f"Amat.npz")
    np.savez_compressed(
        output_file, matrix=scores, genes=adata.var_names.to_numpy(), tfs=tf_knockdowns
    )


if __name__ == "__main__":
    main()
