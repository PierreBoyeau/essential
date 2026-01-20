import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm
from flax.training import train_state
import pandas as pd
import scanpy as sc


from .models import MODEL_REGISTRY


class Trainer:
    def __init__(
        self,
        adata: sc.AnnData,
        model_class,
        expression_type,
        model_kwargs=None,
        perturbation_col="consensus_target",
        control_key="nontargeting",
    ):
        self.adata = adata.copy()
        self.preprocess_mode = expression_type

        self.X = None
        self.X0 = None
        self.U = None
        self.n_genes = None
        self.perturbation_col = perturbation_col
        self.control_key = control_key
        self.nns = None
        self.nn_index = None

        # Indices for train/val split
        self.train_indices = None
        self.val_indices = None

        self._prepare_data()

        if model_kwargs is None:
            model_kwargs = {}
        model_kwargs["lambda_prior"] = model_kwargs.get("lambda_prior", 1e0)
        if isinstance(model_class, str):
            model_class = MODEL_REGISTRY[model_class]

        n_obs = self.X.shape[0]
        self.model = model_class(
            n_obs=n_obs,
            n_genes=self.n_genes,
            **model_kwargs,
        )
        self.random_key = jax.random.PRNGKey(0)
        self.state = None
        self.epoch_history_df = None
        self.step_history_df = None
        self.topk_history_df = None

    def _prepare_data(self):
        raise NotImplementedError("Subclasses must implement _prepare_data")

    @staticmethod
    @jax.jit
    def _eval_step(state, batch_dict):
        variables = {"params": state.params}
        # Unpack batch_dict as keyword arguments if necessary, or pass directly
        # Assuming model.apply takes kwargs corresponding to batch_dict keys
        loss_dict = state.apply_fn(variables, **batch_dict)
        return loss_dict

    @staticmethod
    @jax.jit
    def _train_step(state, batch_dict):
        def loss_fn(params):
            variables = {"params": params}
            loss_dict = state.apply_fn(variables, **batch_dict)
            return loss_dict["loss"], loss_dict

        (_, loss_dict), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss_dict, grads

    def split_train_val(self, train_size, random_seed=0):
        """
        Splits the dataset into training and validation sets.
        Default implementation uses index permutation on self.x_.
        """
        n_obs = self.x_.shape[0]
        split_key = jax.random.PRNGKey(random_seed)
        indices = jax.random.permutation(split_key, n_obs)
        train_idx_int = int(n_obs * train_size)

        self.train_indices = indices[:train_idx_int]
        self.val_indices = indices[train_idx_int:]

        print(f"Data split: {len(self.train_indices)} train, {len(self.val_indices)} val")

    def get_dataloader(self, batch_size, key, split):
        """
        Abstract method to be implemented by subclasses.
        Should yield batch dictionaries for the specified split.

        Parameters
        ----------
        batch_size : int
            Size of the batch.
        key : jax.random.PRNGKey
            Random key for shuffling/sampling.
        split : str
            'train' or 'val'.
        """
        raise NotImplementedError("Subclasses must implement get_dataloader")

    def _run_epoch(self, state, dataloader, is_training):
        epoch_losses = []
        step_history_for_epoch = []
        grads = None

        for batch_dict in dataloader:
            if is_training:
                state, loss_dict, grads = self._train_step(state, batch_dict)
                step_history_for_epoch.append(loss_dict)
            else:
                loss_dict = self._eval_step(state, batch_dict)
            epoch_losses.append(loss_dict)

        avg_loss_dict = {}
        if epoch_losses:
            avg_loss_dict = {
                k: jnp.mean(jnp.array([d[k] for d in epoch_losses])) for k in epoch_losses[0]
            }

        return state, avg_loss_dict, step_history_for_epoch, grads

    def _log_gradient_norms(self, name, grads):
        grad_norms = {k: float(jnp.linalg.norm(v)) for k, v in grads.items()}
        print(f"\n{name} Gradient norms:")
        for param_name, norm in grad_norms.items():
            print(f"  {param_name}: {norm:.6e}")

    def _log_parameter_values(self, name):
        params = self.state.params
        print(f"\n{name} Parameter ranges:")
        for param_name, param_value in params.items():
            min_val = float(jnp.min(param_value))
            max_val = float(jnp.max(param_value))
            print(f"  {param_name}: min={min_val:.6e}, max={max_val:.6e}")

    def _cleanup_after_fit(self):
        """
        Hook for cleaning up large data attributes after fitting.
        Subclasses can override this to delete specific attributes.
        """
        pass

    def fit(
        self,
        learning_rate=1e-3,
        n_epochs=5000,
        batch_size=4096,
        train_size=0.9,
        early_stopping_patience=500,
        early_stopping_metric="loss",
        log_every_n_steps=100,
        optimizer=None,
        batch_size_eval=10000,
        gradient_clip_norm=None,
    ):
        key = self.random_key

        # Initialize parameters
        params = self.model.init_params(jax.random.PRNGKey(0))

        if optimizer is None:
            if gradient_clip_norm is not None:
                optimizer = optax.chain(
                    optax.clip_by_global_norm(gradient_clip_norm),
                    optax.adam(learning_rate=learning_rate),
                )
            else:
                optimizer = optax.adam(learning_rate=learning_rate)
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=params, tx=optimizer
        )

        # Prepare data splits
        self.split_train_val(train_size)

        # Adjust eval batch size
        if self.val_indices is not None:
            if batch_size_eval is None or batch_size_eval > len(self.val_indices):
                batch_size_eval = len(self.val_indices)

        train_epoch_history = []
        train_step_history = []
        val_epoch_history = []
        pbar = tqdm(range(n_epochs))

        # Step-based early stopping
        total_steps = 0
        best_step = 0
        best_loss = 1e10

        for i in pbar:
            # Generate fresh key for this epoch's dataloaders
            key, train_key, val_key = jax.random.split(key, 3)

            # Training
            train_loader = self.get_dataloader(batch_size, train_key, split="train")
            self.state, avg_train_loss_dict, train_steps, grads = self._run_epoch(
                self.state, train_loader, is_training=True
            )
            train_step_history.extend(train_steps)
            train_epoch_history.append(avg_train_loss_dict)

            # Update step counter
            if self.train_indices is not None:
                steps_this_epoch = len(self.train_indices) // batch_size
                total_steps += steps_this_epoch

            # Log gradients intermittently
            if log_every_n_steps > 0 and i % log_every_n_steps == 0 and grads is not None:
                step_name = f"Epoch {i} (step {total_steps})"
                self._log_gradient_norms(step_name, grads)
                print("--------------------------------")
                self._log_parameter_values(step_name)
                print("--------------------------------")

            # Validation
            if self.val_indices is not None and len(self.val_indices) > 0:
                val_loader = self.get_dataloader(batch_size_eval, val_key, split="val")
                _, avg_val_loss_dict, _, _ = self._run_epoch(
                    self.state, val_loader, is_training=False
                )
            else:
                avg_val_loss_dict = {}

            pbar_postfix = {f"train_{k}": f"{v.item():.2E}" for k, v in avg_train_loss_dict.items()}
            if avg_val_loss_dict:
                pbar_postfix.update(
                    {f"val_{k}": f"{v.item():.2E}" for k, v in avg_val_loss_dict.items()}
                )
                avg_val_loss_dict["step"] = total_steps
                val_epoch_history.append(avg_val_loss_dict)
            pbar_postfix["epoch"] = total_steps
            pbar.set_postfix(pbar_postfix)

            if early_stopping_metric in avg_val_loss_dict:
                metric_to_check = avg_val_loss_dict[early_stopping_metric]
            elif early_stopping_metric in avg_train_loss_dict:
                metric_to_check = avg_train_loss_dict[early_stopping_metric]
            else:
                # Fallback or warning if metric is missing
                available = list(avg_val_loss_dict.keys()) + list(avg_train_loss_dict.keys())
                raise ValueError(
                    f"Early stopping metric '{early_stopping_metric}' not found in loss dicts. "
                    f"Available metrics: {set(available)}"
                )

            if metric_to_check < best_loss:
                best_loss = metric_to_check
                best_step = total_steps

            steps_without_improvement = total_steps - best_step
            if steps_without_improvement > early_stopping_patience:
                print(f"\nEarly stopping: no improvement for {steps_without_improvement} steps")
                break

        self.random_key = key  # Update the key

        train_df = pd.DataFrame(train_epoch_history).add_prefix("train_")
        val_df = pd.DataFrame(val_epoch_history).add_prefix("val_")
        self.epoch_history_df = pd.concat([train_df, val_df], axis=1).astype(float)
        self.step_history_df = pd.DataFrame(train_step_history).astype(float)

        # Call cleanup hook
        self._cleanup_after_fit()

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
        if processed_Amat.shape == (self.n_genes, self.n_genes):
            Amat_ = pd.DataFrame(
                processed_Amat, index=self.adata.var_names, columns=self.adata.var_names
            )
        else:
            raise ValueError(f"Invalid Amat shape: {processed_Amat.shape}")
        return self.process_interaction_matrix(Amat_, return_square=return_square, delta=delta)

    @staticmethod
    def process_interaction_matrix(amat_df, return_square=True, delta=None):
        amat_df = amat_df.copy()
        if return_square:
            return amat_df

        Amat_unstack = (
            amat_df.unstack()
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

    def get_results(self, delta, ref_db, transpose_amat=False):
        df = self.get_interaction_matrix(return_square=False, delta=delta)
        return Trainer.get_results_from_interactions(df, ref_db, transpose_amat)

    @staticmethod
    def get_results_from_interactions(df, ref_db, transpose_amat=False):
        assert "target_gene" in ref_db.columns and "regulator_gene" in ref_db.columns
        assert "is_evidence" in ref_db.columns

        if transpose_amat:
            print("Transposing Amat...")
            df = df.rename(
                columns={"target_gene": "regulator_gene", "regulator_gene": "target_gene"}
            )
        df = df.merge(ref_db, on=["target_gene", "regulator_gene"], how="left").assign(
            is_evidence=lambda x: x["is_evidence"].fillna(False),
            is_tp=lambda x: (x["is_evidence"] & x["decision"]).astype(int),
        )
        return df
