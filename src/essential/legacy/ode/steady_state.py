import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse

from .trainer import Trainer


def _predict_batch_steady_state(model, params, x, u, n_steps=100):
    return jax.vmap(
        lambda x, u: model.apply(
            {"params": params}, x, u, method="predict_steady_state", n_steps=n_steps
        ),
        in_axes=(0, None),
    )(x, u)


class SteadyStateEstimator(Trainer):
    def __init__(
        self,
        adata: sc.AnnData,
        model_class,
        expression_type,
        model_kwargs=None,
        perturbation_col="consensus_target",
        control_key="nontargeting",
    ):
        # Initialize Trainer
        super().__init__(
            adata=adata,
            model_class=model_class,
            expression_type=expression_type,
            model_kwargs=model_kwargs,
            perturbation_col=perturbation_col,
            control_key=control_key,
        )

    def _prepare_data(self):
        Xs = self.adata.X
        if issparse(Xs):
            Xs = Xs.toarray()
        self.X = Xs
        self.n_genes = self.X.shape[1]

        U = np.zeros((self.X.shape[0], self.n_genes))
        for cell_idx in range(self.X.shape[0]):
            pert_name = self.adata.obs[self.perturbation_col].iloc[cell_idx]
            if pert_name in self.adata.var_names:
                gene_idx = self.adata.var_names.get_loc(pert_name)
                U[cell_idx, gene_idx] = 1
        self.U = jnp.array(U)

        print("shape of X: ", self.X.shape)

        self.x_ = jnp.array(self.X)
        self.u_ = jnp.array(self.U)

    def get_dataloader(self, batch_size, key, split):
        """
        Yields batch dictionaries for steady-state training/eval.
        Batch dict keys: xt, u
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

            yield {"xt": x_batch, "u": u_batch}

    def _cleanup_after_fit(self):
        """
        Clean up temporary attributes specific to SteadyStateEstimator.
        """
        del self.x_
        del self.u_

    def predict(self, adata, u, batch_size=1000, n_steps=100):
        if self.state is None:
            raise RuntimeError("Model has not been trained yet. Please call .fit() first.")

        x0s = adata.X
        is_sparse_input = issparse(x0s)
        n_obs = x0s.shape[0]

        u_ = jnp.array(u)
        if u_.shape != (self.n_genes,):
            raise ValueError(f"u must be a gene mask of shape ({self.n_genes},), got {u_.shape}")

        # Ensure efficient JIT compilation
        # Jitting here causes problems with Aweight; OK otherwise
        predict_fn = _predict_batch_steady_state

        preds = []
        for i in range(0, n_obs, batch_size):
            if is_sparse_input:
                x_batch = x0s[i : i + batch_size].toarray()
            else:
                x_batch = x0s[i : i + batch_size]

            x_batch = jnp.array(x_batch)
            preds.append(predict_fn(self.model, self.state.params, x_batch, u_, n_steps))

        return jnp.concatenate(preds, axis=0)
