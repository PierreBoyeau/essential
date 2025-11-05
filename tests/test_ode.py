import scanpy as sc
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from essential.ode import ODEstimator
from essential.simulator import ODESimulator


@pytest.fixture
def synthetic_adata():
    n_obs = 1000
    n_vars = 100
    X = np.random.uniform(0, 1, size=(n_obs, n_vars))
    var_names = [f"gene{i+1}" for i in range(n_vars)]
    obs_data = {
        "rt_bc": np.random.choice(["batch_1", "batch_2"], size=n_obs),
        "consensus_target": np.random.choice(
            [f"gene{i+1}" for i in range(10)] + ["nontargeting"], size=n_obs
        ),
    }
    obs = pd.DataFrame(obs_data)
    var = pd.DataFrame(index=var_names)
    adata = AnnData(X, obs=obs, var=var)
    adata.layers["counts"] = adata.X.copy()
    return adata


def test_cellbox_model(synthetic_adata):
    adata_ = synthetic_adata
    sc.pp.filter_genes(adata_, min_cells=10)

    adata_.obsm["latent_rep"] = adata_.X
    ODEstimator.process_data(adata_, latent_obsm_key="latent_rep", K=5)

    ode_model = ODEstimator(
        adata_,
        expression_type="none",
        model_kwargs={"lambda_prior": 1.5e-7, "mode": "dynamic", "Amask": None},
        model_class="linear",
        pairing_strategy="nn",
    )
    ode_model.fit(
        learning_rate=1e-2, n_epochs=1, log_every_n_steps=10, batch_size=100, batch_size_eval=5
    )


def test_cellbox_variations(synthetic_adata):
    adata_ = synthetic_adata
    sc.pp.filter_genes(adata_, min_cells=10)

    model_kwargs = {
        "adata": adata_,
        "pairing_strategy": "nn",
        "expression_type": "none",
        "model_kwargs": {
            "lambda_prior": 1.5e-7,
            "mode": "dynamic",
        },
    }
    fit_kwargs = {
        "learning_rate": 1e-2,
        "n_epochs": 1,
        "log_every_n_steps": 10,
        "batch_size": 100,
        "batch_size_eval": 5,
    }
    adata_.obsm["latent_rep"] = adata_.X
    ODEstimator.process_data(adata_, latent_obsm_key="latent_rep", K=5)

    for model_class in [
        "linear",
        "linearhardko",
        "linearhardkozeroorder",
        "linearzeroorder",
        "sigmoidhardkozeroorder",
        "linearhardmultiplicative",
        "linearmultiplicative",
        "lineardecay",
    ]:
        ode_model = ODEstimator(model_class=model_class, **model_kwargs)
        ode_model.fit(**fit_kwargs)


def test_cellbox_masked(synthetic_adata):
    adata_ = synthetic_adata
    sc.pp.filter_genes(adata_, min_cells=10)

    fit_kwargs = {
        "learning_rate": 1e-2,
        "n_epochs": 1,
        "log_every_n_steps": 10,
        "batch_size": 100,
        "batch_size_eval": 5,
    }
    adata_.obsm["latent_rep"] = adata_.X
    ODEstimator.process_data(adata_, latent_obsm_key="latent_rep", K=5)
    Amask = np.random.rand(adata_.n_vars, adata_.n_vars) >= 0.5
    Amask = jnp.array(Amask.astype(np.float32))

    model_kwargs = {
        "adata": adata_,
        "pairing_strategy": "nn",
        "expression_type": "none",
        "model_kwargs": {
            "lambda_prior": 1.5e-7,
            "mode": "dynamic",
            "Amask": Amask,
        },
    }

    for model_class in [
        "linear",
        "linearhardko",
        "linearhardkozeroorder",
        "linearzeroorder",
        "sigmoidhardkozeroorder",
        "linearhardmultiplicative",
        "linearmultiplicative",
        "lineardecay",
    ]:
        ode_model = ODEstimator(model_class=model_class, **model_kwargs)
        ode_model.fit(**fit_kwargs)


def test_cellboxlowdim_model(synthetic_adata):
    adata_ = synthetic_adata
    sc.pp.filter_genes(adata_, min_cells=10)

    adata_.obsm["latent_rep"] = adata_.X
    ODEstimator.process_data(adata_, latent_obsm_key="latent_rep", K=5)

    ode_model = ODEstimator(
        adata_,
        expression_type="none",
        model_kwargs={"lambda_prior": 1.5e-7, "n_latent": 64, "mode": "dynamic"},
        model_class="linearlowdim2",
        pairing_strategy="nn",
    )
    ode_model.fit(
        learning_rate=1e-2, n_epochs=1, log_every_n_steps=10, batch_size=100, batch_size_eval=5
    )


def test_steady_state_decay_model(synthetic_adata):
    adata_ = synthetic_adata
    sc.pp.filter_genes(adata_, min_cells=10)

    # with jax.disable_jit():
    ode_model = ODEstimator(
        adata_,
        expression_type="none",
        model_kwargs={"lambda_prior": 1.5e-7, "mode": "steady"},
        model_class="lineardecay",
        pairing_strategy=None,
    )
    ode_model.fit(
        learning_rate=1e-2, n_epochs=10, log_every_n_steps=5, batch_size=100, batch_size_eval=5
    )


def test_simulator():
    simulator = ODESimulator(
        n_genes=100,
        n_perturbed=10,
        sparsity=0.1,
        model_class="linear",
        model_kwargs={"mode": "dynamic", "lambda_prior": 0.0},
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


def test_estimator_with_simulator_exact_pairing():
    simulator = ODESimulator(
        n_genes=100,
        n_perturbed=10,
        sparsity=0.1,
        model_class="linear",
        model_kwargs={"mode": "dynamic", "lambda_prior": 0.0},
    )
    adata = simulator.simulate(1000, t=1.0, fraction_control=0.1)
    assert "x0" in adata.obsm
    assert adata.obsm["x0"].shape == adata.X.shape
    estimator = ODEstimator(
        adata,
        model_class="linear",
        pairing_strategy="exact",
        subset_treated=True,
        expression_type="none",
        model_kwargs={"mode": "dynamic", "lambda_prior": 0.0},
    )
    estimator.fit(n_epochs=2, batch_size=100, batch_size_eval=5)
    assert estimator.epoch_history_df is not None
    assert len(estimator.epoch_history_df) > 0
