import argparse
import json
import os
import numpy as np
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.evaluation.kernel_evaluation import (
    compute_distance_matrix,
    process_and_align,
    compute_cka,
    compute_cosine_similarity,
    compute_spearman_off_diagonal,
    wide_to_long,
)

QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_kernel", required=True)
    parser.add_argument("--target_kernel", required=True)
    parser.add_argument("--out_metrics", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    # Load kernels
    pred_kernel = pd.read_pickle(args.pred_kernel)
    target_kernel = pd.read_pickle(args.target_kernel)

    # compute distance matrices
    pred_dist = compute_distance_matrix(pred_kernel)
    target_dist = compute_distance_matrix(target_kernel)

    # kernel metrics
    pred_sub, target_sub = process_and_align(pred_kernel, target_kernel)
    cka = compute_cka(pred_sub, target_sub)
    cos_sim = compute_cosine_similarity(pred_sub, target_sub)
    spearman = compute_spearman_off_diagonal(pred_sub.values, target_sub.values)

    # distance metrics
    pred_dist_sub, target_dist_sub = process_and_align(pred_dist, target_dist)
    dist_corr = compute_spearman_off_diagonal(pred_dist_sub.values, target_dist_sub.values)

    metrics = {
        "kernel_cka": float(cka),
        "kernel_cosine_similarity": float(cos_sim),
        "kernel_spearman_off_diagonal": float(spearman) if not np.isnan(spearman) else 0.0,
        "distance_spearman_off_diagonal": float(dist_corr) if not np.isnan(dist_corr) else 0.0,
        "n_genes": pred_sub.shape[0],
        "tag": args.tag,
    }

    # metric: percentile of distances in target for smallest elements in pred
    pred_dist_sub_long = wide_to_long(
        pred_dist_sub,
        "distance_pred",
        "gene1",
        "gene2",
        remove_diagonal=True,
        remove_lower_triangle=True,
    )
    target_dist_sub_long = wide_to_long(
        target_dist_sub,
        "distance_target",
        "gene1",
        "gene2",
        remove_diagonal=True,
        remove_lower_triangle=True,
    )
    joint_long = pd.merge(
        pred_dist_sub_long, target_dist_sub_long, on=["gene1", "gene2"], how="inner"
    )
    joint_long = joint_long.sample(
        frac=1.0, replace=False
    )  # Shuffle to break sorting ties randomly. When many distances are zero, a stable sort could cause a single gene to dominate the top-k pairs.
    target_dist_med = np.quantile(target_dist_sub_long["distance_target"], q=0.5)
    for k in [50, 100, 500, 1000]:
        joint_long_k = joint_long.sort_values(by="distance_pred").head(k)
        target_dist_med_topk_from_pred = np.quantile(joint_long_k["distance_target"], q=0.5)
        metrics[f"target_dist_median_of_top_{k}_pred_pairs"] = target_dist_med_topk_from_pred
        metrics[f"target_dist_median_ratio_of_top_{k}_pred_pairs_to_global"] = (
            target_dist_med_topk_from_pred / target_dist_med
        )

    os.makedirs(os.path.dirname(args.out_metrics), exist_ok=True)
    with open(args.out_metrics, "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
