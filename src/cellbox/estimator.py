import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import scanpy as sc
from flax.training import train_state
from scipy.sparse import issparse
from scipy.special import expit
from tqdm import tqdm

from .cellbox_metabolite import CellBoxMetabolite
from .cellbox_steady_state import CellBoxSteadyState
from .metrics import profile_metrics


def summarize_saturation(args, threshold: float = 4.0) -> dict:
    """Summary stats of sigmoid arguments (e.g. from ``get_saturation_args``).

    ``frac_saturated`` is the share of ``|arg| > threshold`` (σ' ≲ 0.018 at 4);
    ``mean_sigmoid_grad`` is the mean σ'(arg), i.e. local responsiveness.
    """
    args = np.asarray(args)
    g = expit(args)
    return {
        "mean_abs_arg": float(np.abs(args).mean()),
        "median_abs_arg": float(np.median(np.abs(args))),
        "frac_saturated": float((np.abs(args) > threshold).mean()),
        "mean_sigmoid_grad": float((g * (1.0 - g)).mean()),
    }


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
    metabolite_key:
        Optional ``adata.obsm`` key holding a per-cell metabolite LFC vector. When
        set, a :class:`CellBoxMetabolite` model is used and the per-regulator gate
        ``g(M) = 1 + h(M)`` modulates the TF term. When ``None`` (default) the plain
        :class:`CellBoxSteadyState` is used and behavior is unchanged.
    """

    def __init__(
        self,
        adata: sc.AnnData,
        Amask: jnp.ndarray | None = None,
        perturbation_col: str = "consensus_target",
        control_key: str = "nontargeting",
        standardize_inputs: bool = False,
        metabolite_key: str | None = None,
    ):
        self.adata = adata.copy()
        self.perturbation_col = perturbation_col
        self.control_key = control_key
        self.standardize_inputs = standardize_inputs
        self.metabolite_key = metabolite_key

        self._prepare_data()

        control_mean, control_std = self._compute_predictor_stats()
        epsilon_init = 2.0 * control_mean
        x_mean, x_std = (control_mean, control_std) if standardize_inputs else (None, None)

        if metabolite_key is not None:
            self.model = CellBoxMetabolite(
                n_obs=self.n_obs,
                n_genes=self.n_genes,
                n_metabolites=self.n_metabolites,
                Amask=Amask,
                x_mean=x_mean,
                x_std=x_std,
                epsilon_init=epsilon_init,
            )
        else:
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
        if self.metabolite_key is not None:
            M = np.asarray(self.adata.obsm[self.metabolite_key], dtype=np.float32)
            self.n_metabolites = M.shape[1]
            self.m_ = jnp.array(M)
        else:
            self.n_metabolites = 0
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
            # Optionally cap the number of held-out validation perturbations;
            # the surplus is returned to the training pool.
            if max_val_perturbations is not None and len(val_perts) > max_val_perturbations:
                val_perts = val_perts[:max_val_perturbations]
            train_perts = set(unique_perts) - set(val_perts)
            all_idx = np.arange(self.n_obs)
            mask = np.isin(perts, list(train_perts))
            self.train_indices = jnp.array(all_idx[mask])
            self.val_indices = jnp.array(all_idx[~mask])
        print(f"Split: {len(self.train_indices)} train, {len(self.val_indices)} val")

    def get_dataloader(self, indices, batch_size: int, shuffle_key=None):
        """Yield batches of (xt, u[, m]) for the given index set."""
        n = len(indices)
        if shuffle_key is not None:
            indices = jax.random.permutation(shuffle_key, indices)

        for start in range(0, n - batch_size + 1, batch_size):
            idx = indices[start : start + batch_size]
            batch = {"xt": self.x_[idx], "u": self.u_[idx]}
            if self.metabolite_key is not None:
                batch["m"] = self.m_[idx]
            yield batch

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

        train_history, val_history = [], []
        best_metric = -float("inf") if early_stopping_mode == "max" else float("inf")
        best_epoch = 0

        pbar = tqdm(range(n_epochs))
        for epoch in pbar:
            key, train_key, val_key = jax.random.split(key, 3)

            # Training epoch
            train_outs = []
            for batch in self.get_dataloader(self.train_indices, batch_size, train_key):
                self.state, out = self._train_step(self.state, batch)
                train_outs.append(out)

            avg_train = {
                k: float(jnp.mean(jnp.stack([o[k] for o in train_outs]))) for k in train_outs[0]
            }

            # Validation epoch
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

            # Periodic multi-step validation: the true downstream metric
            # (fixed-point rollout from control on held-out perturbations).
            if validate_every_n_epochs > 0 and epoch % validate_every_n_epochs == 0:
                avg_val.update(
                    self.validate_perturbations(n_control=n_val_control, n_steps=n_val_steps)
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
        self._cleanup_after_fit()

    def _cleanup_after_fit(self):
        del self.x_, self.u_
        if self.metabolite_key is not None:
            del self.m_

    def _metabolite_for_perturbation(self, perturbation: str) -> jnp.ndarray:
        """Mean metabolite LFC vector for cells belonging to ``perturbation``."""
        obs = self.adata.obs[self.perturbation_col]
        M = np.asarray(self.adata.obsm[self.metabolite_key], dtype=np.float32)
        rows = M[np.asarray(obs == perturbation)]
        if rows.shape[0] == 0:
            raise ValueError(
                f"No cells found for perturbation '{perturbation}' in obsm['{self.metabolite_key}']"
            )
        return jnp.array(rows.mean(0))

    def predict(
        self,
        adata: sc.AnnData,
        perturbation: str | None = None,
        perturbation_idx: int | None = None,
        m: jnp.ndarray | None = None,
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
        m:
            Metabolite LFC vector for the condition, shape ``(n_metabolites,)``. Required
            when ``metabolite_key`` was set and ``perturbation`` is not found in training
            adata obs (e.g. external conditions). When ``perturbation`` is known, the
            mean metabolite vector for that condition is looked up automatically.
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

        if self.metabolite_key is not None:
            if m is None:
                m = self._metabolite_for_perturbation(perturbation)
            predict_fn = jax.vmap(
                lambda y, u, m: self.model.apply(
                    {"params": self.state.params},
                    y,
                    u,
                    m,
                    method=self.model.predict_steady_state,
                    n_steps=n_steps,
                ),
                in_axes=(0, None, None),
            )
            preds = []
            for i in range(0, x_.shape[0], batch_size):
                preds.append(np.asarray(predict_fn(x_[i : i + batch_size], u, m)))
        else:
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

    def get_saturation_args(
        self,
        adata: sc.AnnData,
        perturbation: str | None = None,
        perturbation_idx: int | None = None,
        m: jnp.ndarray | None = None,
        batch_size: int = 128,
    ):
        """Sigmoid arguments A·(g(M)⊙ỹ) − p·u + b for each cell in a query adata (cf. predict).

        Returns an (n_cells, n_genes) array; large |arg| means the gene sits in a
        flat tail of the sigmoid (saturated). This is a single forward step from
        the cells' own expression, not the iterated steady state. With no
        perturbation given, the un-knocked-out (u=0) arguments are returned.
        Pass the result to ``summarize_saturation`` for summary stats.
        """
        if self.state is None:
            raise RuntimeError("Call .fit() before .get_saturation_args()")
        if perturbation is not None and perturbation_idx is not None:
            raise ValueError("Provide at most one of perturbation or perturbation_idx")
        if perturbation is not None:
            perturbation_idx = self.adata.var_names.get_loc(perturbation)

        u = jnp.zeros(self.n_genes, dtype=jnp.float32)
        if perturbation_idx is not None:
            u = u.at[perturbation_idx].set(1.0)

        X = adata.X
        if issparse(X):
            X = X.toarray()
        x_ = jnp.array(X, dtype=jnp.float32)

        if self.metabolite_key is not None:
            if m is None and perturbation is not None:
                m = self._metabolite_for_perturbation(perturbation)
            arg_fn = jax.vmap(
                lambda y, u, m: self.model.apply(
                    {"params": self.state.params}, y, u, m, method=self.model.preactivation
                ),
                in_axes=(0, None, None),
            )
            args = []
            for i in range(0, x_.shape[0], batch_size):
                args.append(np.asarray(arg_fn(x_[i : i + batch_size], u, m)))
        else:
            arg_fn = jax.vmap(
                lambda y, u: self.model.apply(
                    {"params": self.state.params}, y, u, method=self.model.preactivation
                ),
                in_axes=(0, None),
            )
            args = []
            for i in range(0, x_.shape[0], batch_size):
                args.append(np.asarray(arg_fn(x_[i : i + batch_size], u)))
        return np.concatenate(args, axis=0)

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
        """Mean ``profile_metrics`` over held-out (val) perturbations.

        This is the true downstream task: for each val perturbation, the model's
        steady state is rolled out from a control-cell subsample, reduced to a
        predicted mean profile, and scored against the observed perturbed mean
        (control mean is the LFC reference). Returns the mean of each metric
        across val perturbations (NaNs ignored), or ``{}`` when there are no val
        perturbations / no control cells.

        Far costlier than a reco-loss epoch, so meant to run every few epochs on
        a control subsample (``n_control``).
        """
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
            m_pert = (
                self._metabolite_for_perturbation(pert) if self.metabolite_key is not None else None
            )
            mu_pred = self.predict(
                adata_control, perturbation=pert, m=m_pert, n_steps=n_steps
            ).mean(0)
            per_pert.append(profile_metrics(mu_gt, mu_pred, mu_control))

        return {k: float(np.nanmean([m[k] for m in per_pert])) for k in per_pert[0]}

    def get_Amat(self) -> pd.DataFrame:
        """Return the learned interaction matrix as a DataFrame (genes × genes)."""
        if self.state is None:
            raise RuntimeError("Call .fit() before .get_Amat()")
        A = self.model.apply({"params": self.state.params}, method=self.model.get_Amat)
        return pd.DataFrame(A, index=self.adata.var_names, columns=self.adata.var_names)

    def get_interaction_matrix(
        self, return_square: bool = True, delta: float | None = None
    ) -> pd.DataFrame:
        """Return the interaction matrix, optionally in long format with filtering.

        Parameters
        ----------
        return_square:
            If True, return a (G×G) DataFrame. Otherwise return long-format rows.
        delta:
            When return_square=False, only keep pairs with |score| > delta.
        """
        Amat_df = self.get_Amat()
        if return_square:
            return Amat_df

        long = (
            Amat_df.unstack()
            .rename("signed_score")
            .reset_index()
            .rename(columns={"level_0": "regulator_gene", "level_1": "target_gene"})
            .assign(
                target_gene=lambda d: d["target_gene"].str.lower(),
                regulator_gene=lambda d: d["regulator_gene"].str.lower(),
                score=lambda d: d["signed_score"].abs(),
            )
        )
        if delta is not None:
            long = long[long["score"] > delta].copy()
        return long

    def get_results(
        self, delta: float, ref_db: pd.DataFrame, transpose_amat: bool = False
    ) -> pd.DataFrame:
        """Merge interaction predictions with a reference database for evaluation.

        Parameters
        ----------
        delta:
            Score threshold; pairs above this are predicted positives.
        ref_db:
            DataFrame with columns target_gene, regulator_gene, is_evidence.
        transpose_amat:
            Swap regulator/target labels before merging (useful if orientation is ambiguous).
        """
        df = self.get_interaction_matrix(return_square=False, delta=None)
        df["decision"] = df["score"] > delta

        if transpose_amat:
            df = df.rename(
                columns={"target_gene": "regulator_gene", "regulator_gene": "target_gene"}
            )

        return df.merge(ref_db, on=["target_gene", "regulator_gene"], how="left").assign(
            is_evidence=lambda d: d["is_evidence"].fillna(False),
            is_tp=lambda d: (d["is_evidence"] & d["decision"]).astype(int),
        )
