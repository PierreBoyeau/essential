import jax
import jax.numpy as jnp
import pandas as pd
from ott.geometry import pointcloud
from ott.solvers import linear
from .trainer import Trainer
from scipy.sparse import issparse
import numpy as np


class FlowMatchingEstimator(Trainer):
    def _prepare_data(self):
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

        X_perturbed = self.X[U.sum(axis=1) > 0]
        X_control = self.X[U.sum(axis=1) == 0]
        U_perturbed = U[U.sum(axis=1) > 0]

        self.X_perturbed = jnp.array(X_perturbed)
        self.X_control = jnp.array(X_control)
        self.U_perturbed = jnp.array(U_perturbed)

    def split_train_val(self, train_size, random_seed=0):
        np_obs = self.X_perturbed.shape[0]
        nc_obs = self.X_control.shape[0]
        split_key = jax.random.PRNGKey(random_seed)
        indices_perturbed = jax.random.permutation(split_key, np_obs)
        indices_control = jax.random.permutation(split_key, nc_obs)
        train_idx_int_perturbed = int(np_obs * train_size)
        train_idx_int_control = int(nc_obs * train_size)
        self.train_indices_perturbed = indices_perturbed[:train_idx_int_perturbed]
        self.val_indices_perturbed = indices_perturbed[train_idx_int_perturbed:]
        self.train_indices_control = indices_control[:train_idx_int_control]
        self.val_indices_control = indices_control[train_idx_int_control:]

        # Set base class indices to enable validation loop in Trainer.fit
        self.train_indices = self.train_indices_perturbed
        self.val_indices = self.val_indices_perturbed
        print(
            f"Data split: {len(self.train_indices)} train, {len(self.val_indices)} val (perturbed)"
        )

    @staticmethod
    @jax.jit
    def _ot_pairing(x0: jnp.ndarray, x1: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        geom = pointcloud.PointCloud(x0, x1)
        ot_solve = linear.solve(geom).matrix
        match_indices = jnp.argmax(ot_solve, axis=0)
        return match_indices

    def get_dataloader(self, batch_size, key, split):
        """
        Yields batch dictionaries for Flow Matching training/eval.
        Batch dict keys: x0, x1 (target), t, u
        """
        if split == "train":
            p_indices = self.train_indices_perturbed
            c_indices = self.train_indices_control
        elif split == "val":
            p_indices = self.val_indices_perturbed
            c_indices = self.val_indices_control
        else:
            raise ValueError(f"Invalid split: {split}")

        n_obs = len(p_indices)
        n_control = len(c_indices)
        n_batches = n_obs // batch_size

        if split == "train":
            if n_batches == 0:
                raise ValueError(
                    "batch_size is larger than number of training observations; drop-last leaves zero batches"
                )
            key, c_perm_key, p_perm_key = jax.random.split(key, 3)
            p_perm_indices = jax.random.permutation(p_perm_key, p_indices)
            c_perm_indices = jax.random.permutation(c_perm_key, c_indices)
        else:
            if n_batches == 0:
                return
            p_perm_indices = p_indices
            c_perm_indices = c_indices

        for j in range(n_batches):
            start = j * batch_size
            end = start + batch_size
            p_batch_indices = p_perm_indices[start:end]

            # Cycle through control indices
            c_start = (j * batch_size) % n_control
            c_indices_cyclic = jnp.arange(batch_size) + c_start
            c_indices_cyclic = c_indices_cyclic % n_control
            c_batch_indices = c_perm_indices[c_indices_cyclic]

            key, t_key = jax.random.split(key)
            t = jax.random.uniform(t_key, shape=(batch_size,))

            x0 = self.X_control[c_batch_indices]
            x1 = self.X_perturbed[p_batch_indices]
            u = self.U_perturbed[p_batch_indices]

            x0_match_indices = self._ot_pairing(x0, x1)
            # dists = jnp.sum((x1[:, None, :] - x0[None, :, :]) ** 2, axis=-1)
            # x0_match_indices = jnp.argmin(dists, axis=1)

            x0_matched = x0[x0_match_indices]
            yield {"x0": x0_matched, "x1": x1, "t": t, "u": u}

    def _cleanup_after_fit(self):
        del self.X_perturbed
        del self.X_control
        del self.U_perturbed
