import argparse
import yaml
import os
import sys
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.essential.stats import MMDTestJax


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

    # Filter AnnData to target genes
    valid_genes = [g for g in target_genes if g in adata.obs[perturbation_key].values]

    print("Extracting layer and computing global PCA...")
    # Get indices of valid genes
    valid_idx = adata.obs[perturbation_key].isin(valid_genes)
    adata_filtered = adata[valid_idx].copy()

    X_filtered = adata_filtered.layers[cp10k_layer]
    if hasattr(X_filtered, "toarray"):
        X_filtered = X_filtered.toarray()

    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X_filtered)

    # Map genes to their PCA coordinates
    gene_to_pca = {}
    obs_genes = adata_filtered.obs[perturbation_key].values

    print("Subsampling to max 50 cells per gene...")
    np.random.seed(42)
    for gene in valid_genes:
        idx = np.where(obs_genes == gene)[0]
        if len(idx) == 0:
            continue
        if len(idx) > 50:
            idx = np.random.choice(idx, 50, replace=False)
        gene_to_pca[gene] = X_pca[idx]

    # Global sigma heuristic
    print("Computing global sigma heuristic...")
    all_sampled_pca = np.concatenate(list(gene_to_pca.values()), axis=0)

    # Use median heuristic on a subset to avoid O(N^2) explosion
    if all_sampled_pca.shape[0] > 2000:
        idx_sigma = np.random.choice(all_sampled_pca.shape[0], 2000, replace=False)
        Z_sigma = all_sampled_pca[idx_sigma]
    else:
        Z_sigma = all_sampled_pca

    dists = ((Z_sigma[:, None] - Z_sigma[None, :]) ** 2).sum(-1)
    upper_tri = dists[np.triu_indices_from(dists, k=1)]
    median_sq_dist = np.median(upper_tri)
    global_sigma = np.sqrt(median_sq_dist) if median_sq_dist > 0 else 1.0

    print(f"Global sigma estimated as: {global_sigma}")

    print("Computing pairwise MMD distances...")
    mmd_test = MMDTestJax(kernel_type="rbf", sigma=global_sigma, max_n=50)

    genes = list(gene_to_pca.keys())
    n_genes = len(genes)

    # distance_matrix = np.zeros((n_genes, n_genes))
    #
    # # Compute MMD for all unique pairs
    # for i in tqdm(range(n_genes)):
    #     for j in range(i + 1, n_genes):
    #         X1 = gene_to_pca[genes[i]]
    #         X2 = gene_to_pca[genes[j]]
    #         dist = mmd_test.compute_mmd(X1, X2)
    #         distance_matrix[i, j] = dist
    #         distance_matrix[j, i] = dist

    print("Building X_all with padding...")
    import jax.numpy as jnp

    max_n = 50
    X_all_list = []
    counts_list = []
    for gene in genes:
        X_g = gene_to_pca[gene]
        n_samples = X_g.shape[0]
        counts_list.append(n_samples)
        # Pad to max_n
        if n_samples < max_n:
            pad_width = ((0, max_n - n_samples), (0, 0))
            X_g_padded = np.pad(X_g, pad_width, mode="constant")
        else:
            X_g_padded = X_g
        X_all_list.append(X_g_padded)
    X_all = jnp.array(X_all_list)
    counts = jnp.array(counts_list)

    print("Computing distance matrix using compute_distance_matrix...")
    distance_matrix_jax = mmd_test.compute_distance_matrix(X_all, counts, batch_size=100)
    distance_matrix = np.array(distance_matrix_jax)

    # distance_matrix diagonal is 0 by definition of unbiased estimator
    distance_df = pd.DataFrame(distance_matrix, index=genes, columns=genes)

    print("Computing RBF Kernel...")
    # K = exp(-gamma * D) using gamma = 1 / (mean(D) + 1e-8)
    # Clip negative values to 0 for distance (unbiased estimator can produce small negatives)
    D_clipped = np.clip(distance_matrix, 0, None)
    gamma = 1.0 / (np.mean(D_clipped) + 1e-8)
    kernel_matrix = np.exp(-gamma * D_clipped)

    kernel_df = pd.DataFrame(kernel_matrix, index=genes, columns=genes)

    distance_long = to_long_no_diagonal(distance_df).rename(columns={"distance": "distance_mmd"})

    print("Saving outputs...")
    os.makedirs(os.path.dirname(args.out_distances), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_kernel), exist_ok=True)

    distance_long.to_csv(args.out_distances, index=False)
    kernel_df.to_pickle(args.out_kernel)
    print("Done!")


if __name__ == "__main__":
    main()
