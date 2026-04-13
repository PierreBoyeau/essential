import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.stats as stats

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.evaluation.kernel_evaluation import (
    compute_spearman_off_diagonal,
    process_and_align,
    wide_to_long,
)


def evaluate_distances(pred_dist, target_dist, tag="tag"):
    # distance metrics
    pred_dist_sub, target_dist_sub = process_and_align(pred_dist, target_dist)
    metrics = {
        "n_genes": pred_dist_sub.shape[0],
        "tag": tag,
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

    joint_long_sorted = joint_long.sort_values(by="distance_pred")

    rank_pred = stats.rankdata(joint_long_sorted["distance_pred"], method="ordinal")
    rank_target = stats.rankdata(joint_long_sorted["distance_target"], method="ordinal")
    metrics["distance_spearman_off_diagonal"] = stats.pearsonr(rank_pred, rank_target)[0]

    n_q25 = int(len(joint_long_sorted) * 0.25)
    subset_q25 = joint_long_sorted.iloc[:n_q25]
    rank_pred_q25 = stats.rankdata(subset_q25["distance_pred"], method="ordinal")
    rank_target_q25 = stats.rankdata(subset_q25["distance_target"], method="ordinal")
    metrics["distance_spearman_off_diagonal_q25"] = stats.pearsonr(rank_pred_q25, rank_target_q25)[
        0
    ]

    n_q10 = int(len(joint_long_sorted) * 0.10)
    subset_q10 = joint_long_sorted.iloc[:n_q10]
    rank_pred_q10 = stats.rankdata(subset_q10["distance_pred"], method="ordinal")
    rank_target_q10 = stats.rankdata(subset_q10["distance_target"], method="ordinal")
    metrics["distance_spearman_off_diagonal_q10"] = stats.pearsonr(rank_pred_q10, rank_target_q10)[
        0
    ]

    if not joint_long.empty:
        target_dist_med = np.quantile(target_dist_sub_long["distance_target"], q=0.5).item()
        for k in [50, 100, 500, 1000, 5000, 10000]:
            if k <= len(joint_long):
                joint_long_k = joint_long.sort_values(by="distance_pred").head(k)
                target_dist_med_topk_from_pred = np.quantile(
                    joint_long_k["distance_target"], q=0.5
                ).item()
                metrics[f"target_dist_median_of_top_{k}_pred_pairs"] = (
                    target_dist_med_topk_from_pred
                )
                if target_dist_med > 0:
                    metrics[f"target_dist_median_ratio_of_top_{k}_pred_pairs_to_global"] = (
                        target_dist_med_topk_from_pred / target_dist_med
                    )
                else:
                    metrics[f"target_dist_median_ratio_of_top_{k}_pred_pairs_to_global"] = 0.0

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_distance", required=True)
    parser.add_argument("--target_distance", required=True)
    parser.add_argument("--out_metrics", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    # Load distances
    pred_dist = pd.read_pickle(args.pred_distance)
    target_dist = pd.read_pickle(args.target_distance)

    metrics = evaluate_distances(pred_dist, target_dist, tag=args.tag)

    os.makedirs(os.path.dirname(args.out_metrics), exist_ok=True)
    with open(args.out_metrics, "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
