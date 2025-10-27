import scanpy as sc
import jax
import numpy as np
from essential.ode import ODEstimator
from essential.simulator import ODESimulator


def test_cellbox_model():
    adata = sc.read_h5ad("/workspace/data/250516_TF_perturbseq/250516_TF_perturbseq.annotated.h5ad")
    adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata, target_sum=1)
    adata.layers["concentration"] = adata.X.copy()
    adata.X = adata.layers["counts"].copy()

    adata_ = adata
    sc.pp.filter_genes(adata_, min_cells=10)

    ode_model = ODEstimator(
        adata_,
        expression_type="concentration",
        model_kwargs={"lambda_prior": 1.5e-7},
        model_class="dynamic_cellbox",
        pairing_strategy="nn",
    )
    ode_model.fit(learning_rate=1e-2, n_epochs=1, log_every_n_steps=10)


def test_cellboxlowdim_model():
    adata = sc.read_h5ad("/workspace/data/250516_TF_perturbseq/250516_TF_perturbseq.annotated.h5ad")
    adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata, target_sum=1)
    adata.layers["concentration"] = adata.X.copy()
    adata.X = adata.layers["counts"].copy()

    adata_ = adata
    sc.pp.filter_genes(adata_, min_cells=10)

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
        n_genes=100,
        n_perturbed=10,
        sparsity=0.1,
        model_class="dynamic_cellbox",
    )
    adata = simulator.simulate(1000, t=1.0, fraction_control=0.1)
    assert adata.shape == (1000, 100)
    assert "consensus_target" in adata.obs.columns
    assert "rt_bc" in adata.obs.columns
    assert "counts" in adata.layers
    assert "nontargeting" in adata.obs["consensus_target"].unique()
    n_control = (adata.obs["consensus_target"] == "nontargeting").sum()
    assert n_control == 100
    assert adata.obs["consensus_target"].nunique() <= 11


def test_estimator_with_simulator():
    simulator = ODESimulator(
        n_genes=100,
        n_perturbed=10,
        sparsity=0.1,
        model_class="dynamic_cellbox",
    )
    adata = simulator.simulate(1000, t=1.0, fraction_control=0.1)
    estimator = ODEstimator(
        adata, model_class="dynamic_cellbox", pairing_strategy="nn", subset_treated=True
    )
    estimator.fit(n_epochs=2)
    assert estimator.epoch_history_df is not None
    assert len(estimator.epoch_history_df) > 0


def test_estimator_with_simulator_exact_pairing():
    simulator = ODESimulator(
        n_genes=100,
        n_perturbed=10,
        sparsity=0.1,
        model_class="dynamic_cellbox",
    )
    adata = simulator.simulate(1000, t=1.0, fraction_control=0.1, pairing_strategy="exact")
    assert "x0" in adata.obsm
    assert adata.obsm["x0"].shape == adata.X.shape
    estimator = ODEstimator(
        adata,
        model_class="dynamic_cellbox",
        pairing_strategy="exact",
        subset_treated=True,
    )
    estimator.fit(n_epochs=2, batch_size=100)
    assert estimator.epoch_history_df is not None
    assert len(estimator.epoch_history_df) > 0
