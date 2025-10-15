import scanpy as sc
from essential.ode import ODEstimator
import argparse
import hashlib
import pandas as pd
import os
import json
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adata_path",
        type=str,
        default="../data/250516_TF_perturbseq/250516_TF_perturbseq.annotated.h5ad",
    )
    parser.add_argument("--preprocess_mode", type=str, default="normalized_concentration")
    parser.add_argument("--model_class", type=str, default="steady_state_forcing")
    parser.add_argument("--lambda_prior", type=float, default=1e0)
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
    hash_kwargs = {k: v for k, v in kwargs.items() if k not in ["adata_path", "output_path"]}
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

    model_kwargs = {"lambda_prior": kwargs["lambda_prior"]}
    estimator = ODEstimator(
        adata,
        expression_type=kwargs["preprocess_mode"],
        model_kwargs=model_kwargs,
        model_class=kwargs["model_class"],
    )
    estimator.fit(learning_rate=1e-2, n_iter=2500)
    a_mat = estimator.get_interaction_matrix()
    hash_ = save_config(kwargs, os.path.join(folder_path, "config.json"), config_hash)
    output_file = os.path.join(folder_path, f"Amat.npz")
    np.savez_compressed(output_file, matrix=a_mat.values, genes=a_mat.columns.to_numpy())


if __name__ == "__main__":
    main()
