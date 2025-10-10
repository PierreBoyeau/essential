import scanpy as sc
import jax
from essentiality.ode import ODEstimator

def test_ode_model():
    adata = sc.read_h5ad("/workspace/data/250516_TF_perturbseq/250516_TF_perturbseq.annotated.h5ad")
    adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata, target_sum=1)
    adata.layers["concentration"] = adata.X.copy()
    adata.X = adata.layers["counts"].copy()
    # sc.pp.log1p(adata)

    adata_ = adata
    sc.pp.filter_genes(adata_, min_cells=10)

    with jax.disable_jit():
        ode_model = ODEstimator(adata_, preprocess_mode="concentration", model_kwargs={"lambda_prior": 1.5e-7})
        ode_model.fit(learning_rate=1e-2, n_iter=10)