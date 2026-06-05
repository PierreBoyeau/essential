import json
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import scanpy as sc
from flax.training import train_state
from scipy.sparse import issparse
from tqdm import tqdm

from .cellbox_steady_state import CellBoxSteadyState
from .metrics import profile_metrics


class CellBoxEstimator:
    """Training and inference wrapper for CellBoxSteadyState.

    Parameters
    ----------
    adata:
        AnnData with cells as rows and genes as columns.
    Amask:
        Optional (n_genes, n_genes) binary mask applied element-wise to A.
    perturbation_col:
        obs column that contains the knocked-out gene name (or control_key).
    control_key:
        Value in perturbation_col indicating un-perturbed / non-targeting cells.
    standardize_inputs:
        If True, standardize the regulator state fed into the sigmoid argument
        using per-gene control-cell statistics (centers the unperturbed state
        near argument 0, keeping the sigmoid in its responsive region). The
        model output stays on the raw data scale.
    """

    def __init__(
        self,
        adata: sc.AnnData,
        Amask: jnp.ndarray | None = None,
        perturbation_col: str = "consensus_target",
        control_key: str = "nontargeting",
        standardize_inputs: bool = False,
    ):
        self.adata = adata.copy()
        self.perturbation_col = perturbation_col
        self.control_key = control_key
        self.standardize_inputs = standardize_inputs

        self._prepare_data()

        control_mean, control_std = self._compute_predictor_stats()
        epsilon_init = 2.0 * control_mean
        x_mean, x_std = (control_mean, control_std) if standardize_inputs else (None, None)

        self.model = CellBoxSteadyState(
            n_obs=self.n_obs,
            n_genes=self.n_genes,
            Amask=Amask,
            x_mean=x_mean,
            x_std=x_std,
            epsilon_init=epsilon_init,
        )
        self.state = None
        self.epoch_history_df = None
        self.train_indices = None
        self.val_indices = None
        self._random_key = jax.random.PRNGKey(0)

    def _adata_to_arrays(self, adata: sc.AnnData) -> tuple[jnp.ndarray, jnp.ndarray]:
        X = adata.X
        if issparse(X):
            X = X.toarray()
        n_obs = X.shape[0]

        U = np.zeros((n_obs, self.n_genes), dtype=np.float32)
        for i, pert in enumerate(adata.obs[self.perturbation_col]):
            if pert in self.adata.var_names:
                U[i, self.adata.var_names.get_loc(pert)] = 1.0

        return jnp.array(X, dtype=jnp.float32), jnp.array(U, dtype=jnp.float32)

    def _prepare_data(self):
        X = self.adata.X
        if issparse(X):
            X = X.toarray()
        self.n_obs, self.n_genes = X.shape
        self.x_, self.u_ = self._adata_to_arrays(self.adata)

    def _compute_predictor_stats(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Per-gene mean/std over control cells for input standardization.

        Centering on control (rather than the perturbed mix) puts the
        unperturbed state at sigmoid-argument 0. Falls back to all cells if
        no control cells are present; flat genes get unit scale.
        """
        X = self.adata.X
        if issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)

        is_control = np.asarray(self.adata.obs[self.perturbation_col] == self.control_key)
        Xc = X[is_control] if is_control.any() else X

        mean = Xc.mean(0)
        std = Xc.std(0)
        std = np.where(std > 1e-3, std, 1.0)
        return jnp.array(mean, dtype=jnp.float32), jnp.array(std, dtype=jnp.float32)

    def _split_train_val(
        self,
        train_size: float,
        seed: int = 0,
        by_perturbation: bool = False,
        max_val_perturbations: int | None = None,
    ):
        key = jax.random.PRNGKey(seed)
        if not by_perturbation:
            idx = jax.random.permutation(key, self.n_obs)
            n_train = int(self.n_obs * train_size)
            self.train_indices = idx[:n_train]
            self.val_indices = idx[n_train:]
        else:
            perts = self.adata.obs[self.perturbation_col].values
            unique_perts = np.unique(perts)
            perm = np.array(jax.random.permutation(key, len(unique_perts)))
            unique_perts = unique_perts[perm]
            n_train_perts = int(len(unique_perts) * train_size)
            val_perts = unique_perts[n_train_perts:]
            if max_val_perturbations is not None and len(val_perts) > max_val_perturbations:
                val_perts = val_perts[:max_val_perturbations]
            train_perts = set(unique_perts) - set(val_perts)
            all_idx = np.arange(self.n_obs)
            mask = np.isin(perts, list(train_perts))
            self.train_indices = jnp.array(all_idx[mask])
            self.val_indices = jnp.array(all_idx[~mask])
        print(f"Split: {len(self.train_indices)} train, {len(self.val_indices)} val")

    def get_dataloader(self, indices, batch_size: int, shuffle_key=None):
        """Yield batches of (xt, u) for the given index set."""
        n = len(indices)
        if shuffle_key is not None:
            indices = jax.random.permutation(shuffle_key, indices)

        for start in range(0, n - batch_size + 1, batch_size):
            idx = indices[start : start + batch_size]
            yield {"xt": self.x_[idx], "u": self.u_[idx]}

    def get_dataloader_rollout(self, indices, batch_size: int, shuffle_key=None):
        """Yield (x0_control, x_gt, u) batches for rollout training.

        x0 is a randomly sampled control cell used as the ODE starting point;
        xt holds the observed perturbed expression (the target). Control cells
        are excluded from the target pool — they generate trivial u=0 examples
        that dominate the gradient without teaching perturbation dynamics.
        """
        perts = np.asarray(self.adata.obs[self.perturbation_col])
        ctrl_idx = np.where(perts == self.control_key)[0]

        indices = np.asarray(indices)
        indices = indices[~np.isin(indices, ctrl_idx)]

        n = len(indices)
        if shuffle_key is not None:
            shuffle_key, ctrl_key = jax.random.split(shuffle_key)
            indices = np.asarray(jax.random.permutation(shuffle_key, indices))
            rng = np.random.default_rng(int(ctrl_key[0]) & 0x7FFFFFFF)
        else:
            rng = np.random.default_rng(0)

        for start in range(0, n - batch_size + 1, batch_size):
            idx = indices[start : start + batch_size]
            x0 = self.x_[rng.choice(ctrl_idx, size=len(idx), replace=True)]
            yield {"x0": x0, "xt": self.x_[idx], "u": self.u_[idx]}

    def _make_rollout_train_step(self, n_steps: int):
        """Return a JIT-compiled rollout train step with n_steps baked in."""
        model = self.model

        def _predict_batch(params, batch):
            return jax.vmap(
                lambda x0, u: model.apply(
                    {"params": params},
                    x0,
                    u,
                    method=model.predict_steady_state,
                    n_steps=n_steps,
                )
            )(batch["x0"], batch["u"])

        @jax.jit
        def step(state, batch):
            def loss_fn(params):
                x_pred = _predict_batch(params, batch)
                mask = 1.0 - batch["u"]
                loss = jnp.mean(jnp.sum(((x_pred - batch["xt"]) * mask) ** 2, axis=1))
                return loss, {"loss": loss, "reco_loss": loss}

            (_, out), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            return state.apply_gradients(grads=grads), out

        return step

    @staticmethod
    @jax.jit
    def _train_step(state, batch):
        def loss_fn(params):
            out = state.apply_fn({"params": params}, **batch)
            return out["loss"], out

        (_, out), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, out

    @staticmethod
    @jax.jit
    def _eval_step(state, batch):
        return state.apply_fn({"params": state.params}, **batch)

    def fit(
        self,
        learning_rate: float = 1e-3,
        n_epochs: int = 5000,
        batch_size: int = 4096,
        train_size: float = 0.9,
        early_stopping_patience: int = 500,
        early_stopping_metric: str = "reco_loss",
        early_stopping_mode: str = "min",
        log_every_n_epochs: int = 100,
        gradient_clip_norm: float | None = None,
        split_by_perturbation: bool = False,
        max_val_perturbations: int | None = None,
        validate_every_n_epochs: int = 0,
        n_val_control: int = 64,
        n_val_steps: int = 100,
        train_mode: str = "reconstruction",
        n_train_steps: int = 100,
    ):
        key = self._random_key
        self._split_train_val(
            train_size,
            by_perturbation=split_by_perturbation,
            max_val_perturbations=max_val_perturbations,
        )

        params = self.model.init_params(jax.random.PRNGKey(0))
        if gradient_clip_norm is not None:
            tx = optax.chain(
                optax.clip_by_global_norm(gradient_clip_norm),
                optax.adam(learning_rate),
            )
        else:
            tx = optax.adam(learning_rate)

        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=tx)

        if train_mode == "reconstruction":
            _step = self._train_step
            _loader = self.get_dataloader
        elif train_mode == "rollout":
            _step = self._make_rollout_train_step(n_train_steps)
            _loader = self.get_dataloader_rollout
        else:
            raise ValueError(
                f"train_mode must be 'reconstruction' or 'rollout', got {train_mode!r}"
            )

        train_history, val_history = [], []
        best_metric = -float("inf") if early_stopping_mode == "max" else float("inf")
        best_epoch = 0

        pbar = tqdm(range(n_epochs))
        for epoch in pbar:
            key, train_key, val_key = jax.random.split(key, 3)

            train_outs = []
            for batch in _loader(self.train_indices, batch_size, train_key):
                self.state, out = _step(self.state, batch)
                train_outs.append(out)

            avg_train = {
                k: float(jnp.mean(jnp.stack([o[k] for o in train_outs]))) for k in train_outs[0]
            }

            val_outs = []
            for batch in self.get_dataloader(
                self.val_indices, min(batch_size, len(self.val_indices)), val_key
            ):
                val_outs.append(self._eval_step(self.state, batch))

            if val_outs:
                avg_val = {
                    k: float(jnp.mean(jnp.stack([o[k] for o in val_outs]))) for k in val_outs[0]
                }
            else:
                avg_val = dict(avg_train)

            if validate_every_n_epochs > 0 and epoch % validate_every_n_epochs == 0:
                pert_metrics = self.validate_perturbations(
                    n_control=n_val_control, n_steps=n_val_steps
                )
                avg_val.update(pert_metrics)
                if early_stopping_metric in pert_metrics:
                    pbar.write(
                        f"[epoch {epoch}] {early_stopping_metric} = {pert_metrics[early_stopping_metric]:.4f}"
                    )

            train_history.append(avg_train)
            val_history.append(avg_val)

            monitor = avg_val.get(early_stopping_metric)
            if monitor is not None and not np.isnan(monitor):
                improved = (
                    monitor > best_metric if early_stopping_mode == "max" else monitor < best_metric
                )
                if improved:
                    best_metric, best_epoch = monitor, epoch

            if log_every_n_epochs > 0 and epoch % log_every_n_epochs == 0:
                pbar.set_postfix(
                    {f"train_{k}": f"{v:.2E}" for k, v in avg_train.items()}
                    | {f"val_{k}": f"{v:.2E}" for k, v in avg_val.items()}
                )

            if epoch - best_epoch > early_stopping_patience:
                print(
                    f"\nEarly stopping at epoch {epoch} (no improvement for {epoch - best_epoch} epochs)"
                )
                break

        self._random_key = key
        self.epoch_history_df = pd.DataFrame(
            [
                {f"train_{k}": v for k, v in t.items()} | {f"val_{k}": v for k, v in v_.items()}
                for t, v_ in zip(train_history, val_history)
            ]
        )
        del self.x_, self.u_

    def predict(
        self,
        adata: sc.AnnData,
        perturbation: str | None = None,
        perturbation_idx: int | None = None,
        batch_size: int = 128,
        n_steps: int = 100,
    ):
        """Predict steady-state expression for each cell in adata under a fixed perturbation.

        Parameters
        ----------
        adata:
            Cells to predict for (genes must match training adata). Only expression
            values are used; perturbation info from obs is ignored.
        perturbation:
            Gene name to knock out. Exactly one of perturbation / perturbation_idx must be given.
        perturbation_idx:
            Integer index into var_names. Exactly one of perturbation / perturbation_idx must be given.
        """
        if self.state is None:
            raise RuntimeError("Call .fit() before .predict()")
        if (perturbation is None) == (perturbation_idx is None):
            raise ValueError("Provide exactly one of perturbation or perturbation_idx")

        if perturbation is not None:
            perturbation_idx = self.adata.var_names.get_loc(perturbation)

        X = adata.X
        if issparse(X):
            X = X.toarray()
        x_ = jnp.array(X, dtype=jnp.float32)

        u = jnp.zeros(self.n_genes, dtype=jnp.float32).at[perturbation_idx].set(1.0)

        predict_fn = jax.vmap(
            lambda y, u: self.model.apply(
                {"params": self.state.params},
                y,
                u,
                method=self.model.predict_steady_state,
                n_steps=n_steps,
            ),
            in_axes=(0, None),
        )
        preds = []
        for i in range(0, x_.shape[0], batch_size):
            preds.append(np.asarray(predict_fn(x_[i : i + batch_size], u)))
        return np.concatenate(preds, axis=0)

    def get_val_perturbations(self) -> list[str]:
        """Return gene names of perturbations assigned to the validation split."""
        if self.val_indices is None:
            return []
        perts = self.adata.obs[self.perturbation_col].iloc[np.array(self.val_indices)]
        unique = perts[perts != self.control_key].unique()
        return [p for p in unique if p in self.adata.var_names]

    def validate_perturbations(
        self, n_control: int = 64, n_steps: int = 100, seed: int = 0
    ) -> dict:
        """Mean ``profile_metrics`` over held-out (val) perturbations."""
        if self.state is None:
            raise RuntimeError("Call .fit() before .validate_perturbations()")
        val_perts = self.get_val_perturbations()
        obs = self.adata.obs[self.perturbation_col]
        is_control = np.asarray(obs == self.control_key)
        if not val_perts or not is_control.any():
            return {}

        X = self.adata.X
        if issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)

        Xc = X[is_control]
        mu_control = Xc.mean(0)
        if n_control is not None and Xc.shape[0] > n_control:
            rng = np.random.default_rng(seed)
            Xc = Xc[rng.choice(Xc.shape[0], n_control, replace=False)]
        adata_control = sc.AnnData(Xc)

        per_pert = []
        for pert in val_perts:
            mu_gt = X[np.asarray(obs == pert)].mean(0)
            mu_pred = self.predict(adata_control, perturbation=pert, n_steps=n_steps).mean(0)
            per_pert.append(profile_metrics(mu_gt, mu_pred, mu_control))

        return {k: float(np.nanmean([m[k] for m in per_pert])) for k in per_pert[0]}

    def get_Amat(self) -> pd.DataFrame:
        """Return the learned interaction matrix as a DataFrame (genes × genes)."""
        if self.state is None:
            raise RuntimeError("Call .fit() before .get_Amat()")
        A = self.model.apply({"params": self.state.params}, method=self.model.get_Amat)
        return pd.DataFrame(A, index=self.adata.var_names, columns=self.adata.var_names)
