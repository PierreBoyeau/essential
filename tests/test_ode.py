import scanpy as sc
import jax
from essential.ode import ODEstimator
from essential.simulator import ODESimulator


def test_cellbox_model():
    adata = sc.read_h5ad("/workspace/data/250516_TF_perturbseq/250516_TF_perturbseq.annotated.h5ad")
    adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata, target_sum=1)
    adata.layers["concentration"] = adata.X.copy()
    adata.X = adata.layers["counts"].copy()
    # sc.pp.log1p(adata)

    adata_ = adata
    sc.pp.filter_genes(adata_, min_cells=10)

    # with jax.disable_jit():
    ode_model = ODEstimator(
        adata_,
        expression_type="concentration",
        model_kwargs={"lambda_prior": 1.5e-7},
        model_class="dynamic_cellbox",
        pairing_strategy="nn",
    )
    ode_model.fit(learning_rate=1e-2, n_epochs=1, log_every_n_steps=10)


def test_steady_state_decay_model():
    adata = sc.read_h5ad("/workspace/data/250516_TF_perturbseq/250516_TF_perturbseq.annotated.h5ad")
    adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata, target_sum=1)
    adata.layers["concentration"] = adata.X.copy()
    adata.X = adata.layers["counts"].copy()
    # sc.pp.log1p(adata)

    adata_ = adata
    sc.pp.filter_genes(adata_, min_cells=10)

    # with jax.disable_jit():
    ode_model = ODEstimator(
        adata_,
        expression_type="concentration",
        model_kwargs={"lambda_prior": 1.5e-7},
        model_class="steady_state_decay",
    )
    ode_model.fit(learning_rate=1e-2, n_epochs=10, log_every_n_steps=5)

    ode_model = ODEstimator(
        adata_,
        expression_type="concentration",
        model_kwargs={"lambda_prior": 1.5e-7},
        model_class="steady_state_decay",
    )
    ode_model.fit(learning_rate=1e-2, n_epochs=2, log_every_n_steps=1, batch_size=100)


def test_simulator():
    simulator = ODESimulator(
        n_obs=1000,
        n_genes=100,
        n_perturbed=10,
        sparsity=0.1,
        model_class="dynamic_cellbox",
    )
    x0 = ode_model.X[:10]
    u = ode_model.U[:10]
    x_pred = simulator.simulate(x0, u)
    assert x_pred.shape == (10, adata_.shape[1], 1)
