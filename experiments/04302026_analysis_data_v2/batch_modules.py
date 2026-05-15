"""
Run PathwayDiscontinuity on all modules that have surprises and save figures.
Figures are organized into subfolders by activity_prediction value.

Usage (from experiments/04302026_analysis_data_v2/):
    python batch_modules.py
"""

import os
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    load_adata,
    load_fitness_data,
    run_module_analysis,
    save_module_outputs,
)

MODULE_INFO_PATH = "../04152026_kegg/module_info.csv"
MODULE_PREDICTION_PATH = "../04152026_kegg/module_prediction.json"
FIGURES_BASE = "figures"
MIN_GENES = 2

ADATA_PATH = "/workspace/data/de122_lce75/adata_de122_lce75_merged.h5ad"
ADATA_CASE_PATH = "/workspace/data/de122_lce75/adata_de122_lce75_merged.case.h5ad"
EQUIV_RESULTS_PATH = "module_equivalence_results.csv"
REPRESENTATION_OBSM_KEY = "scvi_latent"
PERTURBATION_OBS_KEY = "target"
CONTROL_PERTURBATION_KEY = "nontargeting"
LOGNORMALIZED_LAYER = "lognormalized"
GLOBAL_SIGMA = 5.0
MODE = "mmd_stat"
THRESHOLD = 0.15
POINT_SIZE = 2.0
UMAP_DPI = 500
N_GENES = 5
HEATMAP_FIGSIZE = (10, 4)
HEATMAP_RANGE = (-3, 3)


def main():
    print("Loading data...")
    adata = load_adata(adata_path=ADATA_PATH)
    adata_case = load_adata(adata_path=ADATA_CASE_PATH)
    fitness_data = load_fitness_data()

    module_info = pd.read_csv(MODULE_INFO_PATH)
    module_prediction = pd.read_json(MODULE_PREDICTION_PATH)
    module_df = (
        module_info.merge(module_prediction, on="module_id", how="left").query(
            "n_genes >= @MIN_GENES"
        )
        # .query("module_has_surprises == True")
        .sort_values("n_equivalences", ascending=False)
    )

    print(f"Processing {len(module_df)} modules with surprises")

    all_eq_dfs = []
    for _, row in module_df.iterrows():
        module_id = row["module_id"]
        activity = str(row.get("activity_prediction", "unknown")).replace(" ", "_")
        save_dir = os.path.join(FIGURES_BASE, activity, module_id)

        print(f"  {module_id}  [{activity}]  n_equiv={row['n_equivalences']:.0f}")
        try:
            g, results = run_module_analysis(
                adata,
                module_id,
                representation_obsm_key=REPRESENTATION_OBSM_KEY,
                perturbation_obs_key=PERTURBATION_OBS_KEY,
                global_sigma=GLOBAL_SIGMA,
                mode=MODE,
                threshold=THRESHOLD,
                control_perturbation_key=CONTROL_PERTURBATION_KEY,
            )
            _, results = save_module_outputs(
                adata,
                fitness_data,
                g,
                results,
                save_dir,
                point_size=POINT_SIZE,
                umap_dpi=UMAP_DPI,
                heatmap_n_genes=N_GENES,
                heatmap_figsize=HEATMAP_FIGSIZE,
                heatmap_range=HEATMAP_RANGE,
                adata_case=adata_case,
                layer=LOGNORMALIZED_LAYER,
            )
            eq_df = results["equiv_df"]
            all_eq_dfs.append(eq_df.assign(module_id=module_id))
        except Exception as e:
            print(f"    FAILED: {e}")

    all_eq_dfs = pd.concat(all_eq_dfs, ignore_index=True)
    all_eq_dfs.to_csv(EQUIV_RESULTS_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
