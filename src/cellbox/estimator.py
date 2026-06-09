import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scanpy as sc
from flax import traverse_util
from scipy.sparse import issparse

from .base_estimator import BaseEstimator
from .cellbox_steady_state import CellBoxSteadyState
from .cellbox_steady_state_ds import CellBoxSteadyStateDS, CellBoxSteadyStateNBDS
from .cellbox_steady_state_nb import CellBoxSteadyStateNB
from .metrics import profile_metrics


def _a_scale_from_mask(Amask, n_genes: int) -> float:
    """Glorot rescaling so Var(A·norm(x))_i ≈ 1 given the active entries per row."""
    if Amask is None:
        return 1.0
    active = np.asarray(Amask) * (1.0 - np.eye(n_genes))
    k_eff = max(float(active.sum()) / n_genes, 1.0)
    return float(np.sqrt(n_genes / k_eff))


class REGISTRY_KEYS:
    X_KEY = "xt"
    U_KEY = "u"
    X0_KEY = "x0"


class CellBoxEstimator(BaseEstimator):

    def __init__(
        self,
        perturbation_col: str = "consensus_target",
        control_key: str = "nontargeting",
        model_type: str = "gaussian",
        standardize_inputs: bool = False,
        train_mode: str = "reconstruction",
        n_rollout_train: int = 100,
        n_val_control: int = 64,
        n_rollout_val: int = 100,
        layer: str = "log1p",
        layer_eval: str = "",
        reg_embed_dim: int = 16,
        reg_hidden_dim: int = 16,
    ):
        self.perturbation_col = perturbation_col
        self.control_key = control_key
        self.model_type = model_type
        self.standardize_inputs = standardize_inputs
        self.train_mode = train_mode
        self.n_rollout_train = n_rollout_train
        self.n_val_control = n_val_control
        self.n_rollout_val = n_rollout_val
        self.layer = layer
        self.layer_eval = layer_eval or layer
        self.reg_embed_dim = reg_embed_dim
        self.reg_hidden_dim = reg_hidden_dim

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

    # ── fit override ─────────────────────────────────────────────────────────

    def fit(self, adata: sc.AnnData, Amask: jnp.ndarray | None = None, **kwargs):
        self.adata = adata.copy()
        self._amask = Amask
        self._prepare_data()

        a_scale = _a_scale_from_mask(Amask, self.n_genes)
        if self.model_type == "nb":
            x_mean, x_std = self._compute_logcp10k_stats()
            self.model = CellBoxSteadyStateNB(
                n_obs=self.n_obs,
                n_genes=self.n_genes,
                Amask=Amask,
                x_mean=x_mean,
                x_std=x_std,
                epsilon_init=2.0 * jnp.maximum(x_mean, 1e-3),
                a_scale=a_scale,
            )
        elif self.model_type == "nb_ds":
            x_mean, x_std = self._compute_logcp10k_stats()
            self.model = CellBoxSteadyStateNBDS(
                n_obs=self.n_obs,
                n_genes=self.n_genes,
                Amask=Amask,
                x_mean=x_mean,
                x_std=x_std,
                epsilon_init=2.0 * jnp.maximum(x_mean, 1e-3),
                reg_embed_dim=self.reg_embed_dim,
                reg_hidden_dim=self.reg_hidden_dim,
            )
        elif self.model_type == "gaussian_ds":
            control_mean, control_std = self._compute_predictor_stats()
            x_mean = control_mean if self.standardize_inputs else None
            x_std = control_std if self.standardize_inputs else None
            self.model = CellBoxSteadyStateDS(
                n_obs=self.n_obs,
                n_genes=self.n_genes,
                Amask=Amask,
                x_mean=x_mean,
                x_std=x_std,
                epsilon_init=2.0 * control_mean,
                reg_embed_dim=self.reg_embed_dim,
                reg_hidden_dim=self.reg_hidden_dim,
            )
        else:
            control_mean, control_std = self._compute_predictor_stats()
            x_mean = control_mean if self.standardize_inputs else None
            x_std = control_std if self.standardize_inputs else None
            self.model = CellBoxSteadyState(
                n_obs=self.n_obs,
                n_genes=self.n_genes,
                Amask=Amask,
                x_mean=x_mean,
                x_std=x_std,
                epsilon_init=2.0 * control_mean,
                a_scale=a_scale,
            )
        super().fit(**kwargs)

    # ── persistence hooks ─────────────────────────────────────────────────────

    def _extra_state(self) -> dict:
        extra = {"n_genes": self.n_genes, "var_names": list(self.adata.var_names)}
        if (
            self.model_type in ("gaussian_ds", "nb_ds")
            and getattr(self, "_amask", None) is not None
        ):
            amask_np = np.array(self._amask)
            rows, cols = np.where(amask_np > 0)
            extra["amask_edges"] = [rows.tolist(), cols.tolist()]
        return extra

    def _restore_extra_state(self, extra: dict):
        self.n_genes = extra["n_genes"]
        self.adata = sc.AnnData(var=pd.DataFrame(index=extra["var_names"]))

        Amask = None
        if "amask_edges" in extra:
            rows, cols = extra["amask_edges"]
            amask_np = np.zeros((self.n_genes, self.n_genes), dtype=np.float32)
            amask_np[rows, cols] = 1.0
            Amask = jnp.array(amask_np)

        if self.model_type == "nb":
            self.model = CellBoxSteadyStateNB(n_obs=0, n_genes=self.n_genes)
        elif self.model_type == "nb_ds":
            self.model = CellBoxSteadyStateNBDS(
                n_obs=0,
                n_genes=self.n_genes,
                Amask=Amask,
                reg_embed_dim=self.reg_embed_dim,
                reg_hidden_dim=self.reg_hidden_dim,
            )
        elif self.model_type == "gaussian_ds":
            self.model = CellBoxSteadyStateDS(
                n_obs=0,
                n_genes=self.n_genes,
                Amask=Amask,
                reg_embed_dim=self.reg_embed_dim,
                reg_hidden_dim=self.reg_hidden_dim,
            )
        else:
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

        self.x_ctrl_ = jnp.array(self._get_X(adata_ctrl), dtype=jnp.float32)

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

    def _postprocessing(self):
        del self.x_, self.u_, self.x_ctrl_
        super()._postprocessing()

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

        x = jnp.array(self._get_X(adata), dtype=jnp.float32)
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

        if self.model_type in ("nb", "nb_ds"):

            @jax.jit
            def step(state, tensors):
                x0 = tensors[REGISTRY_KEYS.X0_KEY]
                xt = tensors[REGISTRY_KEYS.X_KEY]
                u = tensors[REGISTRY_KEYS.U_KEY]

                def loss_fn(params):
                    out = model.apply({"params": params}, xt, u, x0=x0, n_steps=n_steps)
                    return out["loss"], out

                (_, out), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
                flat_grads = traverse_util.flatten_dict(grads, sep="/")
                out = {
                    **out,
                    **{f"grad_norm/{k}": jnp.linalg.norm(v) for k, v in flat_grads.items()},
                }
                return state.apply_gradients(grads=grads), out

        else:

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
                        ),
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

        X = self._to_eval_space(self._get_X(self.adata))

        Xc_raw = self._get_X(self.adata)[is_control]
        if n_control is not None and Xc_raw.shape[0] > n_control:
            rng = np.random.default_rng(seed)
            Xc_raw = Xc_raw[rng.choice(Xc_raw.shape[0], n_control, replace=False)]
        adata_control = sc.AnnData(Xc_raw)
        if self.layer:
            adata_control.layers[self.layer] = Xc_raw

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

    def _get_X(self, adata: sc.AnnData) -> np.ndarray:
        X = adata.layers[self.layer] if self.layer else adata.X
        if issparse(X):
            X = X.toarray()
        return np.asarray(X, dtype=np.float32)

    def _to_eval_space(self, X: np.ndarray) -> np.ndarray:
        """For NB models, convert raw counts to log-CP10K to match predict_steady_state output."""
        if self.model_type in ("nb", "nb_ds"):
            lib = X.sum(1, keepdims=True)
            return np.log1p(X / (lib + 1e-6) * 1e4)
        return X

    def _adata_to_arrays(self, adata: sc.AnnData) -> tuple[jnp.ndarray, jnp.ndarray]:
        X = self._get_X(adata)
        n_obs = X.shape[0]
        U = np.zeros((n_obs, self.n_genes), dtype=np.float32)
        for i, pert in enumerate(adata.obs[self.perturbation_col]):
            if pert in self.adata.var_names:
                U[i, self.adata.var_names.get_loc(pert)] = 1.0
        return jnp.array(X, dtype=jnp.float32), jnp.array(U, dtype=jnp.float32)

    def _compute_predictor_stats(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        X = self._get_X(self.adata)
        is_control = np.asarray(self.adata.obs[self.perturbation_col] == self.control_key)
        Xc = X[is_control] if is_control.any() else X
        mean = Xc.mean(0)
        std = Xc.std(0)
        std = np.where(std > 1e-3, std, 1.0)
        return jnp.array(mean, dtype=jnp.float32), jnp.array(std, dtype=jnp.float32)

    def _compute_logcp10k_stats(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Log-CP10K mean/std of control cells, used for NB model normalisation."""
        X_raw = self._get_X(self.adata)
        is_control = np.asarray(self.adata.obs[self.perturbation_col] == self.control_key)
        Xc = X_raw[is_control] if is_control.any() else X_raw
        lib = Xc.sum(1, keepdims=True)
        Xc_lcp10k = np.log1p(Xc / (lib + 1e-6) * 1e4)
        mean = Xc_lcp10k.mean(0)
        std = Xc_lcp10k.std(0)
        std = np.where(std > 1e-3, std, 1.0)
        return jnp.array(mean, dtype=jnp.float32), jnp.array(std, dtype=jnp.float32)
