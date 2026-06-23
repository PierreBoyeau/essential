# %%
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import optlang
import pandas as pd
import plotnine as gg
from cobra.flux_analysis import moma
from cobra_models import get_model_components_df, load_ecoli_rich_medium_model
from joblib import Parallel, delayed
from tqdm import tqdm

metabolites_df, reactions_df, genes_df = get_model_components_df()

# %%
save_dir = "/workspace/experiments/01232026_fba/data"

# %%
# Load model once in main process to get WT solution
model_main = load_ecoli_rich_medium_model()
sol_wt_main = model_main.optimize()
growth_wt_main = sol_wt_main.objective_value
# %%

# Global variable for the worker
_worker_model = None


def get_worker_model():
    """Lazy loader for the model in the worker process."""
    global _worker_model
    if _worker_model is None:
        _worker_model = load_ecoli_rich_medium_model()
        _worker_model.solver = "glpk"
    return _worker_model


def process_gene(gene_id, sol_wt, growth_wt, gene_name):
    # Retrieve the model for this worker (loads it if first time)
    model = get_worker_model()

    result = {
        "fba_fluxes": None,
        "fba_growth_ratio": None,
        "moma_fluxes": None,
        "gene_name": gene_name,
    }

    # We use a context to revert changes for the NEXT run in this same worker
    with model:
        # No try/except block as requested, we let errors bubble up or crash the worker (joblib handles this)
        model.genes.get_by_id(gene_id).knock_out()

        # 1. Run FBA
        model.solver = "glpk"
        sol_fba = model.optimize()

        if sol_fba.status == "optimal":
            fba_growth = sol_fba.objective_value
            f_fluxes = sol_fba.fluxes
            f_fluxes.name = gene_name
            result["fba_fluxes"] = f_fluxes
            result["fba_growth_ratio"] = {
                "gene_name": gene_name,
                "growth_ratio": fba_growth / growth_wt,
            }
        else:
            result["fba_growth_ratio"] = {"gene_name": gene_name, "growth_ratio": np.nan}

        # 2. Run MOMA
        # Using linear=True (GLPK) for stability
        # Passing the FULL solution object as required
        sol_moma = moma(model, solution=sol_wt, linear=True)

        if sol_moma.status == "optimal":
            m_fluxes = sol_moma.fluxes
            m_fluxes.name = gene_name
            result["moma_fluxes"] = m_fluxes

    return result


genes_df.to_csv(os.path.join(save_dir, "iJO1366_genes.csv"))

# results = Parallel(n_jobs=8)(
#     delayed(process_gene)(gid, sol_wt_main, growth_wt_main, genes_df.loc[gid, "name"])
#     for gid in tqdm(genes_df.index)
# )

# # Unpack results
# fba_fluxes = [r["fba_fluxes"] for r in results if r["fba_fluxes"] is not None]
# fba_growth_ratios = [r["fba_growth_ratio"] for r in results if r["fba_growth_ratio"] is not None]
# moma_fluxes = [r["moma_fluxes"] for r in results if r["moma_fluxes"] is not None]

# # Combine results
# fba_growth_ratios_df = pd.DataFrame(fba_growth_ratios).set_index("gene_name")
# fba_fluxes_df = pd.DataFrame(fba_fluxes)
# moma_fluxes_df = pd.DataFrame(moma_fluxes)

# # %%
# fba_growth_ratios_df.to_csv(os.path.join(save_dir, "fba_growth_ratios.csv"))
# fba_fluxes_df.to_csv(os.path.join(save_dir, "fba_fluxes.csv"))
# moma_fluxes_df.to_csv(os.path.join(save_dir, "moma_fluxes.csv"))
# # %%
