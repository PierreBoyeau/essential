import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm
from flax.training import train_state
import pandas as pd
import scanpy as sc

from .models import MODEL_REGISTRY


class ODEstimator:
    def __init__(
        self,
        adata: sc.AnnData,
        model_class="steady_state_forcing",
        preprocess_mode="normalized_concentration",
        model_kwargs=None,
    ):
        self.adata = adata.copy()
        self.preprocess_mode = preprocess_mode
        self._prepare_data()
        self._print_dataset_statistics(self.U)

        if model_kwargs is None:
            model_kwargs = {}
        model_kwargs["lambda_prior"] = model_kwargs.get("lambda_prior", 1e0)

        if isinstance(model_class, str):
            model_class = MODEL_REGISTRY[model_class]

        self.model = model_class(
            n_genes=self.n_genes,
            n_tfs=self.n_tfs,
            tf2gene_indicators=self.tf2gene_indicators,
            **model_kwargs,
        )
        self.state = None

    def _prepare_data(self):
        self.adata.X = self.adata.layers["counts"].copy()
        sc.pp.filter_genes(self.adata, min_cells=10)
        sc.pp.normalize_total(self.adata, target_sum=1)
        self.X = self.adata.X.toarray()
        if self.preprocess_mode == "normalized_concentration":
            self.X = self.X / self.X.max(0)
        elif self.preprocess_mode == "concentration":
            pass
        else:
            raise ValueError("Invalid preprocess mode")

        self.tf_list = self.adata.obs["consensus_target"].unique()
        self.tf_list = [t for t in self.tf_list if t in self.adata.var_names]
        self.n_tfs = len(self.tf_list)
        self.n_genes = self.X.shape[1]

        tf2gene_indicators = np.zeros((self.n_genes, self.n_tfs))
        for tf_idx, tf_knockdown in enumerate(self.tf_list):
            gene_idx = self.adata.var_names.get_loc(tf_knockdown)
            tf2gene_indicators[gene_idx, tf_idx] = 1
        self.tf2gene_indicators = jnp.array(tf2gene_indicators)

        U = []
        for cell_idx in range(self.X.shape[0]):
            tf_knockdown = self.adata.obs["consensus_target"].iloc[cell_idx]
            if tf_knockdown in self.tf_list:
                res_ = np.zeros((self.n_tfs,))
                tf_idx = self.tf_list.index(tf_knockdown)
                res_[tf_idx] = 1
                U.append(res_)
            else:
                U.append(np.zeros((self.n_tfs,)))
        self.U = jnp.array(U)

    def _print_dataset_statistics(self, u_):
        n_perturbed = (u_.sum(axis=1) > 0).sum()
        n_total = u_.shape[0]
        print(f"\nDataset info:")
        print(f"  Total cells: {n_total}")
        print(f"  Perturbed cells: {n_perturbed} ({100*n_perturbed/n_total:.1f}%)")
        print(
            f"  Control cells: {n_total - n_perturbed} ({100*(n_total - n_perturbed)/n_total:.1f}%)"
        )
        print(f"  Number of TFs: {self.n_tfs}")
        print(f"  Number of genes: {self.n_genes}\n")

    @jax.jit
    def _train_step(self, state, x, u):
        def loss_fn(params):
            variables = {"params": params}
            loss_dict = state.apply_fn(variables, x, u)
            return loss_dict["loss"], loss_dict

        (_, loss_dict), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss_dict, grads

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
        batch_size=None,
        early_stopping_patience=20,
        early_stopping_metric="loss",
        log_gradients_every=100,
    ):
        x_ = jnp.array(self.X)
        u_ = jnp.array(self.U)
        n_obs = x_.shape[0]

        dummy_x = jnp.zeros((4, self.n_genes))
        dummy_u = jnp.zeros((4, self.n_tfs))
        params = self.model.init(jax.random.PRNGKey(0), dummy_x, dummy_u)["params"]

        optimizer = optax.adam(learning_rate=learning_rate)
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=params, tx=optimizer
        )

        loss_history = []
        pbar = tqdm(range(n_epochs))
        hasnt_improved_counter = 0
        best_loss = 1e10

        for i in pbar:
            if batch_size is None:
                self.state, loss_dict, grads = self._train_step(self.state, x_, u_)
                loss_history.append(loss_dict)
            else:
                permutation = jax.random.permutation(jax.random.PRNGKey(i), n_obs)
                x_perm = x_[permutation]
                u_perm = u_[permutation]

                epoch_losses = []
                for j in range(0, n_obs, batch_size):
                    x_batch = x_perm[j : j + batch_size]
                    u_batch = u_perm[j : j + batch_size]
                    self.state, loss_dict, grads = self._train_step(self.state, x_batch, u_batch)
                    epoch_losses.append(loss_dict)

                # Average losses for the epoch for logging and early stopping
                avg_loss_dict = {
                    k: jnp.mean(jnp.array([d[k] for d in epoch_losses])) for k in epoch_losses[0]
                }
                loss_history.append(avg_loss_dict)
                loss_dict = avg_loss_dict

            # Log gradients intermittently
            if log_gradients_every > 0 and i % log_gradients_every == 0:
                step_name = f"Epoch {i}"
                self._log_gradient_norms(step_name, grads)
                self._log_model_diagnostics(step_name, loss_dict)

            pbar.set_postfix(loss=f'{loss_dict["loss"].item()}')
            if loss_dict[early_stopping_metric] < best_loss:
                best_loss = loss_dict[early_stopping_metric]
                hasnt_improved_counter = 0
            else:
                hasnt_improved_counter += 1
            if hasnt_improved_counter > early_stopping_patience:
                break

        return pd.DataFrame(loss_history).astype(float)

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
