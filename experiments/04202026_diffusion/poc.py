from essential.gpu_utils import select_best_gpus

select_best_gpus()

import jax
import jax.numpy as jnp
import numpy as np
import optax
import scanpy as sc
from dataloader import PerturbationDataLoader
from sde import SDE
from tqdm import tqdm
from utils import compute_kds, median_bandwidth

# ---------------------------------------------------------------------------
# Synthetic AnnData
# ---------------------------------------------------------------------------


def make_synthetic_adata(n: int = 1010, d: int = 100, seed: int = 0) -> sc.AnnData:
    """Create a synthetic AnnData.

    Parameters
    ----------
    n : total number of observations (split as evenly as possible across groups)
    d : number of features / genes
    seed : numpy RNG seed

    Returns
    -------
    AnnData with:
      .X              float32 (n, d),  N(0, I) samples
      .var_names      ["g0", ..., "g{d-1}"]
      .obs["gene"]    perturbation label; "ctrl" for u=-1, "gk" for u=k
    """
    rng = np.random.default_rng(seed)

    var_names = [f"g{k}" for k in range(d)]

    # 101 groups: u in {-1, 0, ..., d-1}
    u_values = list(range(-1, d))
    n_groups = len(u_values)  # 101
    n_per_group = n // n_groups  # 10 (with 1010 obs this is exact)

    gene_labels = []
    for u in u_values:
        label = "ctrl" if u == -1 else f"g{u}"
        gene_labels.extend([label] * n_per_group)

    # Assign any remaining observations to the control group
    remainder = n - len(gene_labels)
    gene_labels.extend(["ctrl"] * remainder)

    X = rng.standard_normal((len(gene_labels), d)).astype(np.float32)

    adata = sc.AnnData(X=X, obs={"gene": gene_labels})
    adata.var_names = var_names
    return adata


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    adata: sc.AnnData,
    n_steps: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
    log_every: int = 50,
):
    """Minimize the KDS loss over interventional minibatches.

    Parameters
    ----------
    adata      : AnnData with .X and .obs["gene"]
    n_steps    : number of gradient update steps
    batch_size : observations per minibatch
    lr         : Adam learning rate
    seed       : JAX PRNGKey seed for parameter initialisation
    log_every  : print loss every this many steps

    Returns
    -------
    params : dict   final Flax parameter dict
    losses : list   scalar KDS loss recorded at each step
    """
    d = adata.n_vars
    sde = SDE(n_vars=d)
    loader = PerturbationDataLoader(adata, batch_size=batch_size, seed=seed)
    optimizer = optax.adam(lr)

    # Initialise parameters with a dummy forward pass
    key = jax.random.PRNGKey(seed)
    dummy_x = jnp.zeros((batch_size, d))
    params = sde.init(key, dummy_x, None, method=sde.forward)["params"]
    opt_state = optimizer.init(params)

    def loss_fn(params, X, u, bandwidth):
        return compute_kds(sde, params, X, u=u, bandwidth=bandwidth)

    losses = []
    for step_idx in tqdm(range(n_steps)):
        u, Xu = loader.sample()
        bw = median_bandwidth(Xu)  # data-adaptive; stays a jnp scalar

        loss, grads = jax.value_and_grad(loss_fn)(params, Xu, u, bw)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        losses.append(float(loss))
        if step_idx % log_every == 0:
            print(f"step {step_idx:4d} | KDS = {loss:.6f}")

    return params, losses


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Synthetic AnnData ===")
    adata = make_synthetic_adata(n=1010, d=100)
    print(adata)
    print(adata.obs["gene"].value_counts().sort_index().head(10), "\n...")

    print("\n=== Training ===")
    params, losses = train(adata, n_steps=200, batch_size=64, lr=1e-3)

    print(f"\nInitial KDS : {losses[0]:.6f}")
    print(f"Final   KDS : {losses[-1]:.6f}")
