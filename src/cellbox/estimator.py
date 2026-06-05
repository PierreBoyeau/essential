import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse

from .base_estimator import BaseEstimator
from .cellbox_steady_state import CellBoxSteadyState
from .metrics import profile_metrics


class REGISTRY_KEYS:
    X_KEY = "xt"
    U_KEY = "u"
    X0_KEY = "x0"


class CellBoxEstimator(BaseEstimator):

    def __init__(
        self,
        perturbation_col: str = "consensus_target",
        control_key: str = "nontargeting",
        standardize_inputs: bool = False,
        train_mode: str = "reconstruction",
        n_rollout_train: int = 100,
        n_val_control: int = 64,
        n_rollout_val: int = 100,
    ):
        self.perturbation_col = perturbation_col
        self.control_key = control_key
        self.standardize_inputs = standardize_inputs
        self.train_mode = train_mode
        self.n_rollout_train = n_rollout_train
        self.n_val_control = n_val_control
        self.n_rollout_val = n_rollout_val

        self.adata = None
        self.model = None
        self.n_genes = None
        self.n_obs = None
        self.state = None
        self.epoch_history_df = None
        self.train_indices = None
        self.val_indices = None
        self._random_key = jax.random.PRNGKey(0)

        self._capture_init_params(locals())

    # ── fit override (adds adata + Amask, then delegates) ────────────────────

    def fit(self, adata: sc.AnnData, Amask: jnp.ndarray | None = None, **kwargs):
        self.adata = adata.copy()
        self._prepare_data()

        control_mean, control_std = self._compute_predictor_stats()
        x_mean, x_std = (control_mean, control_std) if self.standardize_inputs else (None, None)
        self.model = CellBoxSteadyState(
            n_obs=self.n_obs,
            n_genes=self.n_genes,
            Amask=Amask,
            x_mean=x_mean,
            x_std=x_std,
            epsilon_init=2.0 * control_mean,
        )
        super().fit(**kwargs)

    # ── persistence hooks ─────────────────────────────────────────────────────

    def _extra_state(self) -> dict:
        return {"n_genes": self.n_genes, "var_names": list(self.adata.var_names)}

    def _restore_extra_state(self, extra: dict):
        self.n_genes = extra["n_genes"]
        self.adata = sc.AnnData(var=pd.DataFrame(index=extra["var_names"]))
        self.model = CellBoxSteadyState(n_obs=0, n_genes=self.n_genes)

    # ── BaseEstimator: abstract implementations ──────────────────────────────

    def _prepare_data(self):
        is_ctrl = np.asarray(self.adata.obs[self.perturbation_col]) == self.control_key

        adata_pert = self.adata[~is_ctrl] if self.train_mode == "rollout" else self.adata
        adata_ctrl = self.adata[is_ctrl]

        self.n_genes = self.adata.n_vars
        self._adata_train = adata_pert
        self.x_, self.u_ = self._adata_to_arrays(adata_pert)
        self.n_obs = self.x_.shape[0]

        X_ctrl = adata_ctrl.X
        if issparse(X_ctrl):
            X_ctrl = X_ctrl.toarray()
        self.x_ctrl_ = jnp.array(X_ctrl, dtype=jnp.float32)

    def _get_tensors(
        self, indices: np.ndarray, rng: np.random.Generator | None
    ) -> dict[str, jnp.ndarray]:
        rng = rng or np.random.default_rng(0)
        return {
            REGISTRY_KEYS.X_KEY: self.x_[indices],
            REGISTRY_KEYS.U_KEY: self.u_[indices],
            REGISTRY_KEYS.X0_KEY: self.x_ctrl_[
                rng.choice(len(self.x_ctrl_), size=len(indices), replace=True)
            ],
        }

    def _get_model_input(self, tensors: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
        return tensors

    def _free_data_cache(self):
        del self.x_, self.u_, self.x_ctrl_

    def _make_predict_step(self):
        model = self.model
        n_steps = self.n_rollout_val

        @jax.jit
        def predict_fn(params, x, u):
            return jax.vmap(
                lambda _x, _u: model.apply(
                    {"params": params},
                    _x,
                    _u,
                    method=model.predict_steady_state,
                    n_steps=n_steps,
                ),
                in_axes=(0, None),
            )(x, u)

        return predict_fn

    def _get_predict_dataloader(
        self,
        adata: sc.AnnData,
        batch_size: int,
        perturbation: str | None = None,
        perturbation_idx: int | None = None,
    ):
        if (perturbation is None) == (perturbation_idx is None):
            raise ValueError("Provide exactly one of perturbation or perturbation_idx")
        if perturbation is not None:
            perturbation_idx = self.adata.var_names.get_loc(perturbation)

        X = adata.X
        if issparse(X):
            X = X.toarray()
        x = jnp.array(X, dtype=jnp.float32)
        u = jnp.zeros(self.n_genes, dtype=jnp.float32).at[perturbation_idx].set(1.0)

        for start in range(0, x.shape[0], batch_size):
            yield {REGISTRY_KEYS.X_KEY: x[start : start + batch_size], REGISTRY_KEYS.U_KEY: u}

    def _predict_batch(self, tensors: dict[str, jnp.ndarray]) -> np.ndarray:
        return np.asarray(
            self._jit_predict(
                self.state.params, tensors[REGISTRY_KEYS.X_KEY], tensors[REGISTRY_KEYS.U_KEY]
            )
        )

    # ── BaseEstimator: overrides ─────────────────────────────────────────────

    def _make_train_step(self):
        if self.train_mode == "reconstruction":
            return super()._make_train_step()
        return self._make_rollout_train_step(self.n_rollout_train)

    def _periodic_validate(self, epoch: int) -> dict:
        return self.validate_perturbations(n_control=self.n_val_control)

    # ── rollout-specific step ────────────────────────────────────────────────

    def _make_rollout_train_step(self, n_steps: int):
        model = self.model

        def _extract(tensors):
            return (
                tensors[REGISTRY_KEYS.X0_KEY],
                tensors[REGISTRY_KEYS.X_KEY],
                tensors[REGISTRY_KEYS.U_KEY],
            )

        @jax.jit
        def step(state, tensors):
            x0, xt, u = _extract(tensors)

            def loss_fn(params):
                x_pred = jax.vmap(
                    lambda _x0, _u: model.apply(
                        {"params": params},
                        _x0,
                        _u,
                        method=model.predict_steady_state,
                        n_steps=n_steps,
                    )
                )(x0, u)
                mask = 1.0 - u
                loss = jnp.mean(jnp.sum(((x_pred - xt) * mask) ** 2, axis=1))
                return loss, {"loss": loss, "reco_loss": loss}

            (_, out), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            return state.apply_gradients(grads=grads), out

        return step

    # ── public API ───────────────────────────────────────────────────────────

    def validate_perturbations(self, n_control: int = 64, seed: int = 0) -> dict:
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
        if n_control is not None and Xc.shape[0] > n_control:
            rng = np.random.default_rng(seed)
            Xc = Xc[rng.choice(Xc.shape[0], n_control, replace=False)]
        adata_control = sc.AnnData(Xc)
        mu_control = X[is_control].mean(0)

        per_pert = []
        for pert in val_perts:
            mu_gt = X[np.asarray(obs == pert)].mean(0)
            mu_pred = self.predict(adata_control, perturbation=pert).mean(0)
            per_pert.append(profile_metrics(mu_gt, mu_pred, mu_control))

        return {k: float(np.nanmean([m[k] for m in per_pert])) for k in per_pert[0]}

    def get_Amat(self) -> pd.DataFrame:
        if self.state is None:
            raise RuntimeError("Call .fit() before .get_Amat()")
        A = self.model.apply({"params": self.state.params}, method=self.model.get_Amat)
        return pd.DataFrame(A, index=self.adata.var_names, columns=self.adata.var_names)

    # ── helpers ──────────────────────────────────────────────────────────────

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

    def _compute_predictor_stats(self) -> tuple[jnp.ndarray, jnp.ndarray]:
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

    @staticmethod
    def prepare_data(
        adata_path,
        min_library_size,
        perturbation_col,
        normalization,
    ):
        from essential.data import load_regulondb_full

        def _normalize_counts(adata, norm_cfg):
            """Apply the configured count-normalization scheme in place on ``adata.X``."""
            adata.X = adata.layers["reads"].copy()
            method = norm_cfg.method

            if method == "none":
                return

            sc.pp.normalize_total(adata, target_sum=norm_cfg.target_sum)
            if method == "library":
                return

            if method == "log1p":
                sc.pp.log1p(adata)
                return

            if method == "quantile_clip":
                X = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
                upper = np.percentile(X, norm_cfg.clip_percentile, axis=0)
                upper = np.where(upper > 0, upper, 1.0)
                adata.X = np.clip(X, 0, upper) / upper
                return

            raise ValueError(f"Unknown normalization method: {method!r}")

        # load adata
        adata = sc.read_h5ad(adata_path)
        adata.obs["library_size"] = adata.layers["reads"].sum(1).A1
        adata = adata[adata.obs["library_size"] > min_library_size].copy()
        adata = adata[adata.obs[perturbation_col].notna()].copy()
        _normalize_counts(adata, normalization)

        ref_db = load_regulondb_full()
        ref_db = ref_db.loc[lambda x: x["ri_type"].str.startswith("TF")]

        var_lower = adata.var_names.str.lower()
        name_to_idx = pd.Series(np.arange(len(var_lower)), index=var_lower)
        n = len(adata.var_names)
        Amask = np.zeros((n, n), dtype=np.float32)
        pairs = ref_db[["target_gene", "regulator_gene"]].copy()
        pairs["t_idx"] = pairs["target_gene"].map(name_to_idx)
        pairs["r_idx"] = pairs["regulator_gene"].map(name_to_idx)
        pairs = pairs.dropna(subset=["t_idx", "r_idx"])
        Amask[pairs["t_idx"].astype(int).values, pairs["r_idx"].astype(int).values] = 1.0

        return adata, Amask
