"""
Depreciated model; do not use without testing.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import issparse

from .trainer import Trainer


def compute_nns_in_latent(adata, latent_obsm_key, condition_key, condition0, K=5):
    if latent_obsm_key not in adata.obsm:
        raise ValueError(f"Latent representation '{latent_obsm_key}' not found in adata.obsm.")
    z = adata.obsm[latent_obsm_key]

    condition_mask = (adata.obs[condition_key] == condition0).values
    z0 = z[condition_mask]
    indices_in_z = np.where(condition_mask)[0]

    nn_algo = NearestNeighbors(n_neighbors=K, algorithm="auto")
    nn_algo.fit(z0)
    nns_in_z0 = nn_algo.kneighbors(z, return_distance=False)
    nns = indices_in_z[nns_in_z0]
    return nns


class NeuralODEEstimator(Trainer):
    def __init__(
        self,
        adata: sc.AnnData,
        model_class,
        expression_type,
        pairing_strategy,
        model_kwargs=None,
    ):
        if pairing_strategy not in ["nn", "exact"]:
            raise ValueError(f"Invalid pairing strategy: {pairing_strategy}")
        self.pairing_strategy = pairing_strategy

        # Initialize Trainer
        super().__init__(
            adata=adata,
            model_class=model_class,
            expression_type=expression_type,
            model_kwargs=model_kwargs,
        )

    def _prepare_data(self):
        # obtain expression data
        Xs = self.adata.X
        if issparse(Xs):
            Xs = Xs.toarray()
        self.X = Xs
        self.n_genes = self.X.shape[1]

        # create n-cells x n_genes matrix
        U = np.zeros((self.X.shape[0], self.n_genes))
        for cell_idx in range(self.X.shape[0]):
            pert_name = self.adata.obs[self.perturbation_col].iloc[cell_idx]
            if pert_name in self.adata.var_names:
                gene_idx = self.adata.var_names.get_loc(pert_name)
                U[cell_idx, gene_idx] = 1
        self.U = jnp.array(U)

        # compute NNs
        if self.pairing_strategy == "nn":
            if "nns" not in self.adata.obsm:
                raise ValueError(
                    "For 'nn' pairing, pre-computed nearest neighbors must be in adata.obsm['nns']. "
                    "Make sure `process_data` was called before fitting the model."
                )
            print("NNs found in adata.obsm, using them...")
            self.nn_index = self.adata.obsm["nns"]

            nn_index_flat = self.nn_index.flatten()
            perturbation_of_nns = self.adata.obs[self.perturbation_col].values[nn_index_flat]
            if not np.all(perturbation_of_nns == self.control_key):
                raise ValueError("NNs are not all control cells. Please recompute them.")
        else:
            raise ValueError(f"Invalid pairing strategy: {self.pairing_strategy}")

        self.x_full_ = jnp.array(self.X)

        mask_ = (self.adata.obs[self.perturbation_col] != self.control_key).values
        self.adata = self.adata[mask_].copy()
        self.X = self.X[mask_]
        self.U = self.U[mask_]
        if self.pairing_strategy == "nn":
            self.nn_index = self.nn_index[mask_]
        elif self.pairing_strategy == "exact":
            self.X0 = self.X0[mask_]

        print("shape of X: ", self.X.shape)

        self.x_ = jnp.array(self.X)
        if self.X0 is not None:
            self.x0_ = jnp.array(self.X0)
        self.u_ = jnp.array(self.U)
        if self.pairing_strategy == "nn":
            self.nn_index_ = jnp.array(self.nn_index)

    @staticmethod
    def process_data(adata, latent_obsm_key, K=5):
        adata.X = adata.layers["counts"].copy()
        sc.pp.normalize_total(adata, target_sum=1)
        adata.layers["concentration"] = adata.X.copy()
        adata.X = adata.layers["counts"].copy()
        nns = compute_nns_in_latent(
            adata,
            latent_obsm_key=latent_obsm_key,
            condition_key="consensus_target",
            condition0="nontargeting",
            K=K,
        )
        adata.obsm["nns"] = nns
        adata.uns["n_cells_with_nns"] = nns.shape[0]
        return adata

    def get_dataloader(self, batch_size, key, split):
        """
        Yields batch dictionaries for ODE training/eval.
        Batch dict keys: x0, xt, t, u
        """
        if split == "train":
            indices = self.train_indices
        elif split == "val":
            indices = self.val_indices
        else:
            raise ValueError(f"Invalid split: {split}")

        if indices is None or len(indices) == 0:
            return

        n_obs = len(indices)
        n_batches = n_obs // batch_size

        # Shuffle if training
        if split == "train":
            if n_batches == 0:
                raise ValueError(
                    "batch_size is larger than number of training observations; drop-last leaves zero batches"
                )
            key, perm_key = jax.random.split(key)
            perm_indices = jax.random.permutation(perm_key, indices)
        else:
            if n_batches == 0:
                return
            perm_indices = indices

        for j in range(n_batches):
            start = j * batch_size
            end = start + batch_size
            batch_indices = perm_indices[start:end]

            x_batch = self.x_[batch_indices]
            u_batch = self.u_[batch_indices]

            key, pairing_key = jax.random.split(key)
            x0_batch = self._get_x0_batch(batch_indices, pairing_key)

            yield {"x0": x0_batch, "xt": x_batch, "t": jnp.ones((x_batch.shape[0],)), "u": u_batch}

    def _get_x0_batch(self, batch_indices, pairing_key):
        if self.pairing_strategy == "nn":
            nn_batch = self.nn_index_[batch_indices]
            x0_batch = self._sample_from_neighbors(nn_batch, self.x_full_, pairing_key)
        elif self.pairing_strategy == "exact":
            x0_batch = self.x0_[batch_indices]
        else:
            raise ValueError(f"Invalid pairing strategy: {self.pairing_strategy}")
        return x0_batch

    @staticmethod
    def _sample_from_neighbors(knn, x_neighbors, random_key):
        """
        Select one random neighbor per row given a k-NN index.

        Parameters
        ----------
        knn : jnp.ndarray, shape (B, K), int
            For each row b in x, `knn[b, :]` contains K neighbor row indices into x.
        x_neighbors : jnp.ndarray, shape (N, D)
            Feature vectors of neighbors, indexable by `knn`.
        random_key : jax.random.PRNGKey
            Key used to sample one neighbor index per row.

        Returns
        -------
        jnp.ndarray, shape (B, D)
            `x0_batch` where row b equals `x[knn[b, r]]` with r ~ Uniform{0, ..., K-1}.
        """
        batch_size = knn.shape[0]
        n_neighbors = knn.shape[1]
        rdm_neighbor_idx = jax.random.randint(random_key, (batch_size,), 0, n_neighbors)
        arange_batch = jnp.arange(batch_size)
        rdm_neighbor = knn[arange_batch, rdm_neighbor_idx]
        x0_batch = x_neighbors[rdm_neighbor]
        return x0_batch

    def _cleanup_after_fit(self):
        """
        Clean up temporary attributes specific to NeuralODEEstimator.
        """
        del self.x_
        if hasattr(self, "x0_"):
            del self.x0_
        del self.u_
        if hasattr(self, "nn_index_"):
            del self.nn_index_
        if hasattr(self, "x_full_"):
            del self.x_full_
