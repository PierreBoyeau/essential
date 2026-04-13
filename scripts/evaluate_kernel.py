import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.essential.legacy.benchmark.evaluation.kernel_evaluation import (
    compute_cka,
    compute_cosine_similarity,
    compute_distance_matrix,
    compute_spearman_off_diagonal,
    process_and_align,
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

    metrics = {
        "kernel_cka": float(cka),
        "kernel_cosine_similarity": float(cos_sim),
        "kernel_spearman_off_diagonal": float(spearman) if not np.isnan(spearman) else 0.0,
        "n_genes": pred_sub.shape[0],
        "tag": args.tag,
    }

    os.makedirs(os.path.dirname(args.out_metrics), exist_ok=True)
    with open(args.out_metrics, "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
