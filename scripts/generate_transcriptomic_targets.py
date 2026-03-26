import argparse
import yaml
import os
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances


def compute_pairwise(df1, df2=None, metric="euclidean"):
    if metric == "euclidean":
        metric_func = euclidean_distances
    else:
        raise ValueError(f"Metric {metric} not supported")

    if df2 is None:
        pairwise_d = metric_func(df1)
        pairwise_d = pd.DataFrame(pairwise_d, index=df1.index, columns=df1.index)
    else:
        pairwise_d = metric_func(df1, df2)
        pairwise_d = pd.DataFrame(pairwise_d, index=df1.index, columns=df2.index)
    return pairwise_d


def to_long_no_diagonal(df):
    return (
        df.stack()
        .reset_index()
        .rename(columns={"level_0": "gene1", "level_1": "gene2", 0: "distance"})
        .loc[lambda x: x["gene1"] != x["gene2"]]
        .assign(
            gene_pair=lambda x: np.where(
                x["gene1"] < x["gene2"],
                x["gene1"] + "_" + x["gene2"],
                x["gene2"] + "_" + x["gene1"],
            ),
        )
        .drop_duplicates(subset=["gene_pair"], keep="first")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_distances", required=True)
    parser.add_argument("--out_kernel", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    task_config = config["tasks"][args.task]
    if "transcriptomic_adata" not in task_config:
        raise ValueError(f"No transcriptomic_adata specified for task {args.task}")

    t_config = task_config["transcriptomic_adata"]
    adata_path = t_config["path"]
    perturbation_key = t_config["perturbation_key"]
    cp10k_layer = t_config.get("cp10k_layer", "cp10k")
    raw_layer = t_config.get("raw_layer", "counts")

    target_genes = pd.read_csv(task_config["genes_csv"])["gene"].tolist()

    print(f"Loading AnnData from {adata_path}...")
    adata = sc.read_h5ad(adata_path)

    if cp10k_layer not in adata.layers:
        print(f"Layer {cp10k_layer} not found. Normalizing from {raw_layer}...")
        if raw_layer not in adata.layers and raw_layer != "X":
            # Fallback to X
            adata.layers[cp10k_layer] = sc.pp.normalize_total(adata, target_sum=1e4, inplace=False)[
                "X"
            ]
        else:
            adata.X = adata.layers[raw_layer]
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            adata.layers[cp10k_layer] = adata.X.copy()

    transcript_df = []
    gene_names = []

    print("Computing mean expression profiles...")
    valid_genes = [g for g in target_genes if g in adata.obs[perturbation_key].values]

    for gene in valid_genes:
        adata_gene = adata[adata.obs[perturbation_key] == gene]
        if adata_gene.shape[0] > 0:
            X_gene = adata_gene.layers[cp10k_layer]
            if hasattr(X_gene, "toarray"):
                X_gene = X_gene.toarray()

            gene_names.append(gene)
            transcript_df.append(X_gene.mean(axis=0))

    transcript_df = pd.DataFrame(transcript_df, index=gene_names)

    print("Running PCA...")
    transcript_df_pca = PCA(n_components=50).fit_transform(transcript_df)
    transcript_df_pca_ = pd.DataFrame(transcript_df_pca, index=gene_names)

    print("Computing pairwise distances and kernel...")
    transcript_pairwise_pc = compute_pairwise(transcript_df_pca_, metric="euclidean")
    distance_pc = to_long_no_diagonal(transcript_pairwise_pc).rename(
        columns={"distance": "distance_pc"}
    )

    # Compute the transcriptomic kernel using the linear kernel
    K_transcript = transcript_df_pca_ @ transcript_df_pca_.T

    print("Saving outputs...")
    os.makedirs(os.path.dirname(args.out_distances), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_kernel), exist_ok=True)

    distance_pc.to_csv(args.out_distances, index=False)
    K_transcript.to_pickle(args.out_kernel)
    print("Done!")


if __name__ == "__main__":
    main()
