import scanpy as sc
import numpy as np
import jax.numpy as jnp
import pandas as pd
import pytest
from anndata import AnnData
from essential.flow import FlowMatchingEstimator


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


def test_flow_pairing():
    x0 = np.random.random((10, 2))
    x1 = np.random.random((20, 2))
    x0 = jnp.array(x0)
    x1 = jnp.array(x1)
    match_indices = FlowMatchingEstimator._ot_pairing(x0, x1)
    assert match_indices.shape == (20,)
    x0_matched = x0[match_indices]
    assert x0_matched.shape == (20, 2)
    assert x1.shape == (20, 2)


def test_flow_model(synthetic_adata):
    adata_ = synthetic_adata
    sc.pp.filter_genes(adata_, min_cells=10)

    flow_model = FlowMatchingEstimator(
        adata_,
        expression_type="none",
        model_kwargs={"lambda_prior": 1.5e-7, "mode": "dynamic", "Amask": None},
        model_class="sigmoid2_flow",
    )

    flow_model.fit(
        learning_rate=1e-2, n_epochs=2, log_every_n_steps=10, batch_size=100, batch_size_eval=50
    )

    assert flow_model.epoch_history_df is not None
    assert len(flow_model.epoch_history_df) > 0
