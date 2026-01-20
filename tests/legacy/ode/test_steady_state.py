import scanpy as sc
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
import optax

from essential.steady_state import SteadyStateEstimator


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


def test_steady_state_model(synthetic_adata):
    adata_ = synthetic_adata
    sc.pp.filter_genes(adata_, min_cells=10)

    ss_model = SteadyStateEstimator(
        adata_,
        expression_type="none",
        model_kwargs={"lambda_prior": 1.5e-7, "Amask": None},
        model_class="hardsigmoid2_steady_state",
    )
    ss_model.fit(
        learning_rate=1e-2, n_epochs=1, log_every_n_steps=10, batch_size=100, batch_size_eval=5
    )

    n_genes = adata_.n_vars
    u = np.zeros(n_genes)
    # single KO
    u[0] = 1
    preds = ss_model.predict(adata_, u, batch_size=100)
    assert preds.shape == (adata_.n_obs, n_genes)

    u[:3] = 1
    preds = ss_model.predict(adata_, u, batch_size=100)
    assert preds.shape == (adata_.n_obs, n_genes)


def test_steady_state_embedding_model(synthetic_adata):
    adata_ = synthetic_adata
    sc.pp.filter_genes(adata_, min_cells=10)
    n_genes = adata_.n_vars
    d_embedding = 10
    embeddings = np.random.normal(0, 1, size=(n_genes, d_embedding))
    embeddings = jnp.array(embeddings.astype(np.float32))

    ss_model = SteadyStateEstimator(
        adata_,
        expression_type="none",
        model_kwargs={
            "lambda_prior": 1.5e-7,
            "embeddings": embeddings,
            "embedding_dim": d_embedding,
        },
        model_class="hardsigmoid2_embedding_steady_state",
    )
    ss_model.fit(
        learning_rate=1e-2, n_epochs=1, log_every_n_steps=10, batch_size=100, batch_size_eval=5
    )


def test_steady_state_masked(synthetic_adata):
    adata_ = synthetic_adata
    sc.pp.filter_genes(adata_, min_cells=10)

    fit_kwargs = {
        "learning_rate": 1e-2,
        "n_epochs": 1,
        "log_every_n_steps": 10,
        "batch_size": 100,
        "batch_size_eval": 5,
    }

    Amask = np.random.rand(adata_.n_vars, adata_.n_vars) >= 0.5
    Amask = jnp.array(Amask.astype(np.float32))

    model_kwargs = {
        "adata": adata_,
        "expression_type": "none",
        "model_kwargs": {
            "lambda_prior": 1.5e-7,
            "Amask": Amask,
        },
    }

    ss_model = SteadyStateEstimator(model_class="hardsigmoid2_steady_state", **model_kwargs)
    ss_model.fit(**fit_kwargs)
