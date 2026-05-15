"""
Pathway discontinuity analysis for multiple KEGG modules.

Usage (from experiments/04302026_analysis_data_v2/):
    python analyze_modules.py
"""

import os
import sys

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
plt.rcParams["svg.fonttype"] = "none"

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    load_adata,
    load_fitness_data,
    run_module_analysis,
    save_module_outputs,
)

ADATA_PATH = "/workspace/data/de122_lce75/adata_de122_lce75_merged.h5ad"
ADATA_CASE_PATH = "/workspace/data/de122_lce75/adata_de122_lce75_merged.case.h5ad"
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

configs = [
    {
        "module_id": "eco_M00049",
        "save_dir": "figures/figures_atp",
        "renamer": {
            "C00130": "IMP",
            "C03794": "SAMP",
            "C00020": "AMP",
            "C00008": "ADP",
            "C00002": "ATP",
        },
    },
    {
        "module_id": "eco_M00120",
        "save_dir": "figures/figures_coa",
        "renamer": {
            "C00864": "PAN",
            "C03492": "PPAN",
            "C04352": "PPAN-Cys",
            "C01134": "PTE-P",
            "C00882": "dCoA",
            "C00010": "CoA",
        },
    },
    {
        "module_id": "eco_M00125",
        "save_dir": "figures/figures_fad",
        "renamer": {
            "C00044": "GTP",
            "C01304": "DAPRP",
            "C01268": "APRU-P",
            "C04454": "APRit-P",
            "C04732": "ARU",
            "C00199": "Ru5P",
            "C15556": "DHBP",
            "C04332": "DMRL",
            "C00255": "RBF",
            "C00061": "FMN",
            "C00016": "FAD",
        },
    },
    {
        "module_id": "eco_M00115",
        "save_dir": "figures/figures_nad",
        "renamer": {
            "C00049": "Asp",
            "C05840": "IminAsp",
            "C03722": "PDC",
            "C01185": "NaMN",
            "C00857": "DeamNAD",
            "C00003": "NAD",
        },
    },
    {
        "module_id": "eco_M00001",
        "save_dir": "figures/figures_glycolysis",
        "renamer": {
            "C00267": "Glc",
            "C00668": "G6P",
            "C00085": "F6P",
            "C00354": "FBP",
            "C00111": "DHAP",
            "C00118": "GAP",
            "C00236": "1,3-BPG",
            "C00197": "3PG",
            "C00631": "2PG",
            "C00074": "PEP",
            "C00022": "Pyr",
        },
    },
    {
        "module_id": "eco_M00060",
        "save_dir": "figures/figures_lpx",
        "renamer": {
            "C00043": "UDP-GlcNAc",
            "C04738": "UDP-acyl-GlcNAc",
            "C06022": "UDP-acyl-GlcN",
            "C04652": "UDP-diacyl-GlcN",
            "C04824": "diacyl-GlcN-1P",
            "C04932": "4acyl-diGlcN-1P",
            "C04919": "LipIVA",
            "C06024": "KDO-LipIVA",
            "C06025": "KDO2-LipIVA",
            "C06251": "Lau-KDO2-LipIVA",
            "C06026": "KDO2-LipA",
        },
    },
    {
        "module_id": "eco_M00063",
        "save_dir": "figures/figures_kds",
        "renamer": {
            "C00199": "D-Rib5P",
            "C01112": "D-Ara5P",
            "C04478": "KDO8P",
            "C01187": "KDO",
            "C04121": "CMP-KDO",
        },
    },
    {
        "module_id": "eco_M00121",
        "save_dir": "figures/figures_heme",
        "renamer": {
            "C00025": "Glu",
            "C02987": "Glu-tRNA",
            "C03741": "GSA",
            "C00430": "ALA",
            "C00931": "PBG",
            "C01024": "HMB",
            "C01051": "Urogen III",
            "C03263": "Coprogen III",
            "C01079": "Protogen IX",
            "C02191": "Proto IX",
            "C00032": "Heme",
        },
    },
]


def main():
    print("Loading data...")
    adata = load_adata(adata_path=ADATA_PATH)
    adata_case = load_adata(adata_path=ADATA_CASE_PATH)
    fitness_data = load_fitness_data()

    for cfg in configs:
        module_id = cfg["module_id"]
        save_dir = cfg["save_dir"]
        print(f"Running analysis for {module_id}...")
        g, results = run_module_analysis(
            adata,
            module_id,
            representation_obsm_key=REPRESENTATION_OBSM_KEY,
            perturbation_obs_key=PERTURBATION_OBS_KEY,
            renamer=cfg.get("renamer"),
            global_sigma=GLOBAL_SIGMA,
            mode=MODE,
            threshold=THRESHOLD,
            control_perturbation_key=CONTROL_PERTURBATION_KEY,
        )
        print(f"Saving figures to {save_dir}/")
        save_module_outputs(
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

    print("Done.")


if __name__ == "__main__":
    main()
