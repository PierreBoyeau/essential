"""
Perturbation dataloader for stationary SDE causal models.

Usage
-----
    import scanpy as sc
    from essential.configs.base import get_config
    from dataloader import PerturbationDataLoader

    config = get_config()
    adata  = sc.read_h5ad(config.processing.adata_path)

    loader = PerturbationDataLoader(adata, batch_size=256)

    for u, Xu in loader:          # infinite iterator
        loss = compute_kds(model, params, Xu, u=u)
        ...

    # Or draw a single batch:
    u, Xu = loader.sample()
"""

import jax.numpy as jnp
import numpy as np
import scipy.sparse


class PerturbationDataLoader:
    """Minibatch sampler over interventional groups in an AnnData object.

    At each call, a perturbation group is drawn uniformly at random, and
    `batch_size` observations are sampled from that group with replacement.

    Intervention index u
    --------------------
    If `adata.obs["gene"]` contains a gene name that is also in
    `adata.var_names`, u is set to its integer position in var_names
    (0-indexed).  Otherwise u is set to None, meaning no intervention
    shift is applied (observational / non-targeting control regime).

    Parameters
    ----------
    adata      : AnnData  input data (adata.X used as features)
    batch_size : int      number of observations per minibatch
    seed       : int      numpy random seed
    """

    def __init__(self, adata, batch_size: int = 256, seed: int = 0):
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

        var_name_to_idx = {name: i for i, name in enumerate(adata.var_names)}

        # Build per-perturbation data arrays
        self._groups: dict[str, tuple[int, np.ndarray]] = {}

        for gene_name in adata.obs["gene"].unique():
            # None signals the observational regime (no shift applied in SDE.forward).
            # Falling back to 0 would conflate control with intervention on variable 0.
            u = var_name_to_idx.get(gene_name, None)

            # Boolean mask → integer positions, safe for both dense and sparse X
            mask = (adata.obs["gene"] == gene_name).values
            X_group = adata.X[mask]
            # Handle sparse matrices (common after sc.pp.normalize_total)
            if scipy.sparse.issparse(X_group):
                X_group = X_group.toarray()
            self._groups[gene_name] = (u, np.asarray(X_group, dtype=np.float32))

        self._keys = list(self._groups.keys())
        self.n_perturbations = len(self._keys)
        self.n_vars = adata.n_vars

    # ------------------------------------------------------------------

    def sample(self) -> tuple[int, jnp.ndarray]:
        """Draw one (u, Xu) pair.

        Returns
        -------
        u  : int            intervention index
        Xu : (batch_size, n_vars)  jnp array of sampled observations
        """
        key = self._keys[self.rng.integers(self.n_perturbations)]
        u, X = self._groups[key]
        idx = self.rng.integers(len(X), size=self.batch_size)
        return u, jnp.array(X[idx])

    def sample_group(self, gene_name: str) -> tuple[int, jnp.ndarray]:
        """Draw a minibatch from a specific perturbation by name."""
        if gene_name not in self._groups:
            raise KeyError(f"Perturbation '{gene_name}' not found.")
        u, X = self._groups[gene_name]
        idx = self.rng.integers(len(X), size=self.batch_size)
        return u, jnp.array(X[idx])

    def sample_dataset(self, n: int) -> jnp.ndarray:
        """Sample n rows uniformly from the full dataset (all groups pooled).

        Used to compute a global bandwidth estimate before training.

        Parameters
        ----------
        n : int
            Number of rows to sample (with replacement if n exceeds dataset size).

        Returns
        -------
        (n, n_vars) jnp array
        """
        all_X = np.concatenate([X for _, X in self._groups.values()], axis=0)
        replace = n > len(all_X)
        idx = (
            self.rng.integers(len(all_X), size=n)
            if replace
            else self.rng.choice(len(all_X), size=n, replace=False)
        )
        return jnp.array(all_X[idx])

    def iter_all_groups(self) -> list[tuple[str, int, jnp.ndarray]]:
        """Return one full minibatch per perturbation group.

        Useful for evaluation (not training).

        Yields
        ------
        (gene_name, u, Xu)
        """
        for key in self._keys:
            u, X = self._groups[key]
            idx = self.rng.integers(len(X), size=self.batch_size)
            yield key, u, jnp.array(X[idx])

    # ------------------------------------------------------------------
    # Python iterator protocol (infinite)

    def __iter__(self):
        while True:
            yield self.sample()

    def __len__(self) -> int:
        return self.n_perturbations

    def __repr__(self) -> str:
        return (
            f"PerturbationDataLoader("
            f"n_perturbations={self.n_perturbations}, "
            f"n_vars={self.n_vars}, "
            f"batch_size={self.batch_size})"
        )
