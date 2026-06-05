# %%
import anndata as ad
import numpy as np
import pandas as pd

adata = ad.read_h5ad("/workspace/data/de122_lce75/adata_de122_lce75_merged.h5ad")
lfc = pd.read_csv(
    "/workspace/experiments/05142026_metabolome/rapp_avg_lfc_by_gene.csv", index_col=0
)

# %%
zeros = pd.Series(0.0, index=lfc.columns)
metabolite_fc = pd.DataFrame(
    [lfc.loc[t] if t in lfc.index else zeros for t in adata.obs["target"]],
    index=adata.obs_names,
)

adata.obsm["metabolite_fc"] = metabolite_fc

# %%
adata.write_h5ad("adata_de122_lce75_merged.metabolome.h5ad")

# %%
