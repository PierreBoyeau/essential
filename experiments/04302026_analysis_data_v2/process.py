import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import scanpy as sc
import scvi

adata_path1 = "/workspace/data/Nov2025_DE122_genomescale_EZRDM_Glu_newpipeline_preprocessed/Nov2025_DE122_genomescale_EZRDM_Glu_newpipeline_preprocessed.h5ad"
adata_path2 = "/workspace/data/260309_lce75_genomescale_ezrdm_glu_preprocessed.h5ad"
adata1 = sc.read_h5ad(adata_path1)
adata1.obs["experiment"] = "de122"
adata2 = sc.read_h5ad(adata_path2)
adata2.obs["experiment"] = "lce75"

adata = sc.concat([adata1, adata2])
adata_ = adata.copy()
TARGET_KEY = "target"
adata_ = adata_[~adata_.obs[TARGET_KEY].isna()].copy()


adata_.X = adata_.layers["reads"]
sc.pp.normalize_total(adata_)
sc.pp.log1p(adata_)
sc.pp.pca(adata_, n_comps=50)
sc.pp.neighbors(adata_, use_rep="X_pca")
sc.tl.umap(adata_)
adata_.obsm["X_umap_lognormalized"] = adata_.obsm["X_umap"].copy()
adata_.layers["lognormalized"] = adata_.X.copy()

scvi.model.SCVI.setup_anndata(
    adata_, categorical_covariate_keys=["rt_bc", "experiment"], layer="reads"
)
model = scvi.model.SCVI(adata_)
model.train()

latent = model.get_latent_representation()
adata_.obsm["scvi_latent"] = latent
sc.pp.neighbors(adata_, use_rep="scvi_latent")
sc.tl.umap(adata_)
sc.tl.leiden(adata_, resolution=0.1, key_added="leiden_0.1")
sc.tl.leiden(adata_, resolution=0.5, key_added="leiden_0.5")
sc.tl.leiden(adata_, resolution=1.0, key_added="leiden_1.0")
adata_.obsm["X_umap_scvi"] = adata_.obsm["X_umap"].copy()

adata_.write_h5ad("adata_de122_lce75_merged.h5ad")
