import scanpy as sc
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
import optax

from essential.neural_ode import NeuralODEEstimator


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
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    return adata


def test_neural_ode_model(synthetic_adata):
    adata_ = synthetic_adata
    sc.pp.filter_genes(adata_, min_cells=10)

    adata_.obsm["latent_rep"] = adata_.X
    NeuralODEEstimator.process_data(adata_, latent_obsm_key="latent_rep", K=5)

    ode_model = NeuralODEEstimator(
        adata_,
        expression_type="none",
        model_kwargs={"lambda_prior": 1.5e-7, "Amask": None},
        model_class="sigmoid2_ode",
        pairing_strategy="nn",
    )
    ode_model.fit(
        learning_rate=1e-2, n_epochs=1, log_every_n_steps=10, batch_size=100, batch_size_eval=5
    )


def test_neural_ode_masked(synthetic_adata):
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
    NeuralODEEstimator.process_data(adata_, latent_obsm_key="latent_rep", K=5)
    Amask = np.random.rand(adata_.n_vars, adata_.n_vars) >= 0.5
    Amask = jnp.array(Amask.astype(np.float32))

    model_kwargs = {
        "adata": adata_,
        "pairing_strategy": "nn",
        "expression_type": "none",
        "model_kwargs": {
            "lambda_prior": 1.5e-7,
            "Amask": Amask,
        },
    }

    ode_model = NeuralODEEstimator(model_class="sigmoid2_ode", **model_kwargs)
    ode_model.fit(**fit_kwargs)
