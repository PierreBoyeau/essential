import argparse
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_centered_kernel(K):
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def compute_cka(K, L):
    K_centered = compute_centered_kernel(K)
    L_centered = compute_centered_kernel(L)

    K_norm = np.linalg.norm(K_centered, "fro")
    L_norm = np.linalg.norm(L_centered, "fro")

    if K_norm == 0 or L_norm == 0:
        return 0.0

    return np.trace(K_centered @ L_centered) / (K_norm * L_norm)


def compute_cosine_similarity(K, L):
    K_norm = np.linalg.norm(K, "fro")
    L_norm = np.linalg.norm(L, "fro")

    if K_norm == 0 or L_norm == 0:
        return 0.0

    return np.trace(K @ L) / (K_norm * L_norm)


def compute_spearman_off_diagonal(K, L):
    # Extract upper triangle indices, excluding diagonal
    n = K.shape[0]
    iu = np.triu_indices(n, k=1)

    k_off_diag = K[iu]
    l_off_diag = L[iu]

    # Check for constant arrays to avoid NaNs
    if np.all(k_off_diag == k_off_diag[0]) or np.all(l_off_diag == l_off_diag[0]):
        return 0.0

    corr, _ = spearmanr(k_off_diag, l_off_diag)
    return corr


def compute_distance_matrix(K: pd.DataFrame) -> pd.DataFrame:
    k_diag = np.diag(K)
    dist_squared = k_diag[:, None] + k_diag[None, :] - 2 * K
    dist_squared = np.clip(dist_squared, 0, None)
    D = np.sqrt(dist_squared)
    return D


def process_and_align(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    df1.index = df1.index.astype(str)
    df1.columns = df1.columns.astype(str)
    df2.index = df2.index.astype(str)
    df2.columns = df2.columns.astype(str)
    common_genes = np.intersect1d(df1.index, df2.index)
    df1_aligned = df1.loc[common_genes, common_genes].values
    df2_aligned = df2.loc[common_genes, common_genes].values
    return df1_aligned, df2_aligned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_kernel", required=True)
    parser.add_argument("--target_kernel", required=True)
    parser.add_argument("--out_metrics", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    # Load kernels
    if args.pred_kernel.endswith(".csv"):
        pred_df = pd.read_csv(args.pred_kernel, index_col=0)
    else:
        pred_df = pd.read_pickle(args.pred_kernel)

    if args.target_kernel.endswith(".csv"):
        target_df = pd.read_csv(args.target_kernel, index_col=0)
    else:
        target_df = pd.read_pickle(args.target_kernel)

    # compute distance matrices
    pred_dist = compute_distance_matrix(pred_df)
    target_dist = compute_distance_matrix(target_df)

    # Convert string indices to consistent format if necessary
    pred_df.index = pred_df.index.astype(str)
    pred_df.columns = pred_df.columns.astype(str)
    target_df.index = target_df.index.astype(str)
    target_df.columns = target_df.columns.astype(str)

    pred_sub, target_sub = process_and_align(pred_df, target_df)
    cka = compute_cka(pred_sub, target_sub)
    cos_sim = compute_cosine_similarity(pred_sub, target_sub)
    spearman = compute_spearman_off_diagonal(pred_sub, target_sub)

    pred_dist_sub, target_dist_sub = process_and_align(pred_dist, target_dist)
    dist_corr = compute_spearman_off_diagonal(pred_dist_sub, target_dist_sub)
    metrics = {
        "kernel_cka": float(cka),
        "kernel_cosine_similarity": float(cos_sim),
        "kernel_spearman_off_diagonal": float(spearman) if not np.isnan(spearman) else 0.0,
        "distance_spearman_off_diagonal": float(dist_corr) if not np.isnan(dist_corr) else 0.0,
        "n_genes": pred_sub.shape[0],
        "tag": args.tag,
    }

    os.makedirs(os.path.dirname(args.out_metrics), exist_ok=True)
    with open(args.out_metrics, "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
