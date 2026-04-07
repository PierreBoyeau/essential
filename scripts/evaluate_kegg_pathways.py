import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd


def build_pathway_dict(kegg_data):
    pathway_to_genes = defaultdict(list)
    for entry in kegg_data:
        gene = entry["query_gene"]
        for path_id, path_name in entry.get("pathways", {}).items():
            pathway_to_genes[path_name].append(gene)
    return pathway_to_genes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance_matrix", required=True)
    parser.add_argument("--kegg_json", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    dist_df = pd.read_pickle(args.distance_matrix)

    with open(args.kegg_json, "r") as f:
        kegg_data = json.load(f)

    pathway_to_genes = build_pathway_dict(kegg_data)

    mat = dist_df.values
    i_upper, j_upper = np.triu_indices_from(mat, k=1)
    global_avg_dist = mat[i_upper, j_upper].mean()

    results = []
    available_genes = set(dist_df.index)

    for pathway, genes in pathway_to_genes.items():
        valid_genes = [g for g in genes if g in available_genes]

        if len(valid_genes) < 2:
            continue

        sub_dist = dist_df.loc[valid_genes, valid_genes].values
        i_sub, j_sub = np.triu_indices_from(sub_dist, k=1)
        pathway_avg_dist = sub_dist[i_sub, j_sub].mean()

        distance_ratio = pathway_avg_dist / global_avg_dist if global_avg_dist > 0 else np.nan

        results.append(
            {
                "pathway": pathway,
                "n_genes": len(valid_genes),
                "pathway_avg_dist": pathway_avg_dist,
                "distance_ratio": distance_ratio,
            }
        )

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        results_df = results_df.sort_values("distance_ratio")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    results_df.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
