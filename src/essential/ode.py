import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm
from flax.training import train_state
import pandas as pd
import scanpy as sc
from scvi.model import SCVI
from sklearn.neighbors import NearestNeighbors
import time


from .models import MODEL_REGISTRY


def compute_nns_in_latent(adata, batch_key, condition_key, condition0, K=5):
    SCVI.setup_anndata(adata, batch_key=batch_key, layer="counts")
    model = SCVI(adata)
    model.train()
    z = model.get_latent_representation()
    z0 = z[(adata.obs[condition_key] == condition0).values]

    nn_algo = NearestNeighbors(n_neighbors=K, algorithm="auto")
    nn_algo.fit(z0)
    nns = nn_algo.kneighbors(z, return_distance=False)
    return nns


class ODEstimator:
    def __init__(
        self,
        adata: sc.AnnData,
        model_class="steady_state_forcing",
        expression_type="normalized_concentration",
        pairing_strategy=None,
        subset_treated=False,
        model_kwargs=None,
    ):
        self.adata = adata.copy()
        self.preprocess_mode = expression_type
        self.pairing_strategy = pairing_strategy
        self.subset_treated = subset_treated

        self.X = None
        self.U = None
        self.gene_perturbations = None
        self.perturbation2geneidx = None
        self.n_genes = None
        self.n_pertubations = None
        self.pertubation_col = "consensus_target"
        self.batch_key = "rt_bc"
        self.control_key = "nontargeting"
        self.nns = None
        self._prepare_data()
        self._print_dataset_statistics(self.U)

        if model_kwargs is None:
            model_kwargs = {}
        model_kwargs["lambda_prior"] = model_kwargs.get("lambda_prior", 1e0)
        if isinstance(model_class, str):
            model_class = MODEL_REGISTRY[model_class]
        self.model = model_class(
            n_genes=self.n_genes,
            n_tfs=self.n_pertubations,
            tf2gene_indicators=self.perturbation2geneidx,
            **model_kwargs,
        )
        self.random_key = jax.random.PRNGKey(0)
        self.state = None
        self.epoch_history_df = None
        self.step_history_df = None

    def _prepare_data(self):
        # obtain expression data
        self.adata.X = self.adata.layers["counts"].copy()
        sc.pp.filter_genes(self.adata, min_cells=10)
        sc.pp.normalize_total(self.adata, target_sum=1)
        self.X = self.adata.X.toarray()
        if self.preprocess_mode == "normalized_concentration":
            xmax = np.quantile(self.X, 0.95, axis=0)
            self.X = self.X / xmax
        elif self.preprocess_mode == "concentration":
            pass
        else:
            raise ValueError("Invalid preprocess mode")

        # main perturbations to gene variables
        self.gene_perturbations = self.adata.obs[self.pertubation_col].unique()
        self.gene_perturbations = [t for t in self.gene_perturbations if t in self.adata.var_names]
        self.n_pertubations = len(self.gene_perturbations)
        self.n_genes = self.X.shape[1]
        perturbation2geneidx = np.zeros((self.n_genes, self.n_pertubations))
        for pert_idx, pert_name in enumerate(self.gene_perturbations):
            gene_idx = self.adata.var_names.get_loc(pert_name)
            perturbation2geneidx[gene_idx, pert_idx] = 1
        self.perturbation2geneidx = jnp.array(perturbation2geneidx)

        # create n-cells x n_perturbations matrix
        U = []
        for cell_idx in range(self.X.shape[0]):
            pert_name = self.adata.obs[self.pertubation_col].iloc[cell_idx]
            if pert_name in self.gene_perturbations:
                res_ = np.zeros((self.n_pertubations,))
                pert_idx = self.gene_perturbations.index(pert_name)
                res_[pert_idx] = 1
                U.append(res_)
            else:
                U.append(np.zeros((self.n_pertubations,)))
        self.U = jnp.array(U)

        # compute NNs
        if "nns" not in self.adata.obsm:
            if self.pairing_strategy is not None:
                raise ValueError(f"Invalid pairing strategy: {self.pairing_strategy}")
            elif self.pairing_strategy == "nn":
                print("NNs not found in adata.obsm, computing them...")
                self.nn_index = compute_nns_in_latent(
                    self.adata,
                    batch_key=self.batch_key,
                    condition_key=self.pertubation_col,
                    condition0=self.control_key,
                )
            else:
                raise ValueError(f"Invalid pairing strategy: {self.pairing_strategy}")
        else:
            print("NNs found in adata.obsm, using them...")
            self.nn_index = self.adata.obsm["nns"]

        if self.subset_treated:
            mask_ = (self.adata.obs[self.pertubation_col] != self.control_key).values
            self.adata = self.adata[mask_].copy()
            self.X = self.adata.X.toarray()
            self.U = self.U[mask_]
            self.nn_index = self.nn_index[mask_]

    def _print_dataset_statistics(self, u_):
        n_perturbed = (u_.sum(axis=1) > 0).sum()
        n_total = u_.shape[0]
        print(f"\nDataset info:")
        print(f"  Total cells: {n_total}")
        print(f"  Perturbed cells: {n_perturbed} ({100*n_perturbed/n_total:.1f}%)")
        print(
            f"  Control cells: {n_total - n_perturbed} ({100*(n_total - n_perturbed)/n_total:.1f}%)"
        )
        print(f"  Number of TFs: {self.n_pertubations}")
        print(f"  Number of genes: {self.n_genes}\n")

    @staticmethod
    @jax.jit
    def _eval_step(state, x0, xt, t, u):
        variables = {"params": state.params}
        loss_dict = state.apply_fn(variables, x0, xt, t, u)
        return loss_dict

    @staticmethod
    @jax.jit
    def _train_step(state, x0, xt, t, u):
        def loss_fn(params):
            variables = {"params": params}
            loss_dict = state.apply_fn(variables, x0, xt, t, u)
            return loss_dict["loss"], loss_dict

        (_, loss_dict), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss_dict, grads

    def _run_epoch(self, state, key, indices, batch_size, is_training):
        epoch_losses = []
        step_history_for_epoch = []
        grads = None

        n_obs = len(indices)
        n_batches = n_obs // batch_size

        if is_training:
            if n_batches == 0:
                raise ValueError(
                    "batch_size is larger than number of training observations; drop-last leaves zero batches"
                )
            key, perm_key = jax.random.split(key)
            perm_indices = jax.random.permutation(perm_key, indices)
        else:
            if n_batches == 0:
                return state, {}, [], None, key
            perm_indices = indices

        for j in range(n_batches):
            start = j * batch_size
            end = start + batch_size
            batch_indices = perm_indices[start:end]

            x_batch = self.x_[batch_indices]
            u_batch = self.u_[batch_indices]
            nn_batch = self.nn_index_[batch_indices]

            key, neighbor_key = jax.random.split(key)
            x0_batch = self.get_x0(nn_batch, self.x_, neighbor_key)

            if is_training:
                state, loss_dict, grads = self._train_step(
                    state,
                    x0=x0_batch,
                    xt=x_batch,
                    t=jnp.ones((x_batch.shape[0],)),
                    u=u_batch,
                )
                step_history_for_epoch.append(loss_dict)
            else:
                loss_dict = self._eval_step(
                    state,
                    x0=x0_batch,
                    xt=x_batch,
                    t=jnp.ones((x_batch.shape[0],)),
                    u=u_batch,
                )
            epoch_losses.append(loss_dict)

        avg_loss_dict = {}
        if epoch_losses:
            avg_loss_dict = {
                k: jnp.mean(jnp.array([d[k] for d in epoch_losses])) for k in epoch_losses[0]
            }

        return state, avg_loss_dict, step_history_for_epoch, grads, key

    def _log_gradient_norms(self, name, grads):
        grad_norms = {k: float(jnp.linalg.norm(v)) for k, v in grads.items()}
        print(f"\n{name} Gradient norms:")
        for param_name, norm in grad_norms.items():
            print(f"  {param_name}: {norm:.6e}")

    def _log_model_diagnostics(self, name, loss_dict):
        if "cross_terms_mean" in loss_dict:
            print(f"\n{name} Model diagnostics:")
            print(f"  cross_terms_mean: {float(loss_dict['cross_terms_mean']):.6e}")
            print(f"  perturb_term_mean: {float(loss_dict['perturb_term_mean']):.6e}")
            print(f"  u_active_fraction: {float(loss_dict['u_active_fraction']):.4f}")
            if "perturbation_decay_mean" in loss_dict:
                print(
                    f"  perturbation_decay_mean: {float(loss_dict['perturbation_decay_mean']):.6e}"
                )

    def fit(
        self,
        learning_rate=1e-3,
        n_epochs=5000,
        batch_size=4096,
        train_size=0.9,
        early_stopping_patience=20,
        early_stopping_metric="loss",
        log_every_n_steps=100,
        optimizer=None,
    ):
        self.x_ = jnp.array(self.X)
        self.u_ = jnp.array(self.U)
        batch_size_eval = 128
        self.nn_index_ = jnp.array(self.nn_index)
        n_obs = self.x_.shape[0]
        key = self.random_key

        # Train/validation split
        split_key = jax.random.PRNGKey(0)
        indices = jax.random.permutation(split_key, n_obs)
        train_idx_int = int(n_obs * train_size)
        train_indices = indices[:train_idx_int]
        val_indices = indices[train_idx_int:]

        dummy_x0 = jnp.zeros((4, self.n_genes))
        dummy_x = jnp.zeros((4, self.n_genes))
        dummy_t = jnp.ones((4,))
        dummy_u = jnp.zeros((4, self.n_pertubations))
        params = self.model.init(jax.random.PRNGKey(0), dummy_x0, dummy_x, dummy_t, dummy_u)[
            "params"
        ]

        if optimizer is None:
            optimizer = optax.adam(learning_rate=learning_rate)
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=params, tx=optimizer
        )

        train_epoch_history = []
        train_step_history = []
        val_epoch_history = []
        pbar = tqdm(range(n_epochs))
        hasnt_improved_counter = 0
        best_loss = 1e10

        for i in pbar:
            # Training
            self.state, avg_train_loss_dict, train_steps, grads, key = self._run_epoch(
                self.state, key, train_indices, batch_size=batch_size, is_training=True
            )
            train_step_history.extend(train_steps)
            train_epoch_history.append(avg_train_loss_dict)

            # Log gradients intermittently
            if log_every_n_steps > 0 and i % log_every_n_steps == 0 and grads is not None:
                step_name = f"Epoch {i}"
                self._log_gradient_norms(step_name, grads)
                self._log_model_diagnostics(step_name, avg_train_loss_dict)

            # Validation
            _, avg_val_loss_dict, _, _, key = self._run_epoch(
                self.state, key, val_indices, batch_size=batch_size_eval, is_training=False
            )
            if avg_val_loss_dict:
                val_epoch_history.append(avg_val_loss_dict)

            pbar_postfix = {f"train_{k}": f"{v.item():.2E}" for k, v in avg_train_loss_dict.items()}
            if avg_val_loss_dict:
                pbar_postfix.update(
                    {f"val_{k}": f"{v.item():.2E}" for k, v in avg_val_loss_dict.items()}
                )
            pbar.set_postfix(pbar_postfix)

            metric_to_check = (
                avg_val_loss_dict.get(early_stopping_metric)
                if avg_val_loss_dict
                else avg_train_loss_dict[early_stopping_metric]
            )

            if metric_to_check < best_loss:
                best_loss = metric_to_check
                hasnt_improved_counter = 0
            else:
                hasnt_improved_counter += 1
            if hasnt_improved_counter > early_stopping_patience:
                break

        train_df = pd.DataFrame(train_epoch_history).add_prefix("train_")
        val_df = pd.DataFrame(val_epoch_history).add_prefix("val_")
        self.epoch_history_df = pd.concat([train_df, val_df], axis=1).astype(float)
        self.step_history_df = pd.DataFrame(train_step_history).astype(float)

        # Clean up temporary attributes
        del self.x_
        del self.u_
        del self.nn_index_

    @staticmethod
    def get_x0(knn, x_neighbors, random_key):
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

    def get_interaction_matrix(self, return_square=True, delta=None):
        """Extract the learned gene-gene interaction matrix from the trained model.

        This method retrieves the interaction matrix (Amat) that represents regulatory
        relationships between genes. The matrix can be returned in square form or as a
        long-format DataFrame with additional filtering options.

        Parameters
        ----------
        return_square : bool, default=True
            If True, returns a square DataFrame with genes as both rows and columns.
            If False, returns a long-format DataFrame with one row per gene pair.
        delta : float, optional
            Threshold for filtering interactions by absolute score. Only used when
            return_square=False. If provided, only interactions with score > delta
            are returned.

        Returns
        -------
        pd.DataFrame
            If return_square=True:
                Square DataFrame with shape (n_genes, n_genes), where rows and columns
                are indexed by gene names from adata.var_names. Values represent the
                signed interaction strength.
            If return_square=False:
                Long-format DataFrame with columns:
                - 'target_gene': lowercase target gene name
                - 'regulator_gene': lowercase regulator gene name
                - 'signed_score': signed interaction strength
                - 'score': absolute interaction strength

        Raises
        ------
        RuntimeError
            If the model has not been trained yet (state is None). Call .fit() first.

        Examples
        --------
        >>> # Get square interaction matrix
        >>> Amat = model.get_interaction_matrix(return_square=True)

        >>> # Get filtered long-format interactions
        >>> interactions = model.get_interaction_matrix(return_square=False, delta=0.1)
        """
        if self.state is None:
            raise RuntimeError("Model has not been trained yet. Please call .fit() first.")

        processed_Amat = self.model.apply({"params": self.state.params}, method=self.model.get_Amat)
        Amat_ = pd.DataFrame(
            processed_Amat, index=self.adata.var_names, columns=self.adata.var_names
        )
        if return_square:
            return Amat_

        Amat_unstack = (
            Amat_.unstack()
            .to_frame("signed_score")
            .reset_index()
            # .rename(columns={"level_0": "target_gene", "level_1": "regulator_gene"})
            .rename(columns={"level_0": "regulator_gene", "level_1": "target_gene"})
            .assign(
                target_gene_=lambda x: x["target_gene"],
                regulator_gene_=lambda x: x["regulator_gene"],
                target_gene=lambda x: x["target_gene"].str.lower(),
                regulator_gene=lambda x: x["regulator_gene"].str.lower(),
                score=lambda x: np.abs(x["signed_score"].values),
            )
        )
        if delta is not None:
            Amat_unstack.loc[:, "decision"] = Amat_unstack["score"] > delta
        return Amat_unstack
