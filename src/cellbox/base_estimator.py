import inspect
import json
from abc import ABC, abstractmethod
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pandas as pd
import scanpy as sc
from flax.training import train_state
from tqdm import tqdm


class BaseEstimator(ABC):

    # ── abstract interface ───────────────────────────────────────────────────

    @abstractmethod
    def _prepare_data(self): ...

    @abstractmethod
    def _get_tensors(
        self, indices: np.ndarray, rng: np.random.Generator | None
    ) -> dict[str, jnp.ndarray]: ...

    @abstractmethod
    def _get_model_input(self, tensors: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]: ...

    @abstractmethod
    def _make_predict_step(self): ...

    @abstractmethod
    def _get_predict_dataloader(self, adata: sc.AnnData, batch_size: int, **kwargs): ...

    @abstractmethod
    def _predict_batch(self, tensors: dict[str, jnp.ndarray]) -> np.ndarray: ...

    # ── overrideable hooks ───────────────────────────────────────────────────

    def _build_train_state(self, tx) -> train_state.TrainState:
        params = self.model.init_params(jax.random.PRNGKey(0))
        return train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=tx)

    def _free_data_cache(self):
        pass

    def _preprocess_indices(self, indices: np.ndarray) -> np.ndarray:
        return indices

    def _make_train_step(self):
        get_input = self._get_model_input

        @jax.jit
        def step(state, tensors):
            def loss_fn(params):
                out = state.apply_fn({"params": params}, **get_input(tensors))
                return out["loss"], out

            (_, out), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            return state.apply_gradients(grads=grads), out

        return step

    def _make_eval_step(self):
        get_input = self._get_model_input

        @jax.jit
        def step(state, tensors):
            return state.apply_fn({"params": state.params}, **get_input(tensors))

        return step

    def _periodic_validate(self, epoch: int) -> dict:
        return {}

    def _extra_state(self) -> dict:
        """Data-derived state (not __init__ args) needed to reconstruct the model on load."""
        return {}

    def _restore_extra_state(self, extra: dict):
        """Re-hydrate data-derived state and build self.model before params are loaded."""
        pass

    # ── init param capture ───────────────────────────────────────────────────

    def _capture_init_params(self, init_locals: dict):
        sig = inspect.signature(type(self).__init__)
        self._init_params = {k: init_locals[k] for k in sig.parameters if k != "self"}

    # ── helpers ──────────────────────────────────────────────────────────────

    def get_val_perturbations(self) -> list[str]:
        if self.val_indices is None:
            return []
        adata_train = getattr(self, "_adata_train", self.adata)
        perts = adata_train.obs[self.perturbation_col].iloc[np.array(self.val_indices)]
        unique = perts[perts != self.control_key].unique()
        return [p for p in unique if p in self.adata.var_names]

    # ── dataloader ───────────────────────────────────────────────────────────

    def get_dataloader(self, indices, batch_size: int, shuffle_key=None):
        indices = np.asarray(self._preprocess_indices(np.asarray(indices)))
        n = len(indices)
        rng = None
        if shuffle_key is not None:
            seed = int(np.asarray(shuffle_key)[0]) & 0x7FFFFFFF
            rng = np.random.default_rng(seed)
            indices = rng.permutation(indices)
        for start in range(0, n - batch_size + 1, batch_size):
            yield self._get_tensors(indices[start : start + batch_size], rng)

    # ── train / val split ────────────────────────────────────────────────────

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
            adata_train = getattr(self, "_adata_train", self.adata)
            perts = adata_train.obs[self.perturbation_col].values
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

    # ── training loop ────────────────────────────────────────────────────────

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
        checkpoint_dir: str | None = None,
        checkpoint_every_n_epochs: int = 0,
    ):
        key = self._random_key
        self._split_train_val(
            train_size,
            by_perturbation=split_by_perturbation,
            max_val_perturbations=max_val_perturbations,
        )

        if gradient_clip_norm is not None:
            tx = optax.chain(
                optax.clip_by_global_norm(gradient_clip_norm),
                optax.adam(learning_rate),
            )
        else:
            tx = optax.adam(learning_rate)

        self.state = self._build_train_state(tx)
        self._jit_predict = self._make_predict_step()
        _step = self._make_train_step()
        _eval_step = self._make_eval_step()

        train_history, val_history = [], []
        best_metric = -float("inf") if early_stopping_mode == "max" else float("inf")
        best_epoch = 0

        pbar = tqdm(range(n_epochs))
        for epoch in pbar:
            key, train_key, val_key = jax.random.split(key, 3)

            train_outs = []
            for tensors in self.get_dataloader(self.train_indices, batch_size, train_key):
                self.state, out = _step(self.state, tensors)
                train_outs.append(out)

            avg_train = {
                k: float(jnp.mean(jnp.stack([o[k] for o in train_outs]))) for k in train_outs[0]
            }

            val_outs = []
            for tensors in self.get_dataloader(
                self.val_indices, min(batch_size, len(self.val_indices)), val_key
            ):
                val_outs.append(_eval_step(self.state, tensors))

            avg_val = (
                {k: float(jnp.mean(jnp.stack([o[k] for o in val_outs]))) for k in val_outs[0]}
                if val_outs
                else dict(avg_train)
            )

            if validate_every_n_epochs > 0 and epoch % validate_every_n_epochs == 0:
                pert_metrics = self._periodic_validate(epoch)
                avg_val.update(pert_metrics)
                if early_stopping_metric in pert_metrics:
                    pbar.write(
                        f"[epoch {epoch}] {early_stopping_metric}"
                        f" = {pert_metrics[early_stopping_metric]:.4f}"
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

            if checkpoint_every_n_epochs > 0 and epoch % checkpoint_every_n_epochs == 0:
                self.save(Path(checkpoint_dir) / f"epoch_{epoch:05d}")

            if epoch - best_epoch > early_stopping_patience:
                print(
                    f"\nEarly stopping at epoch {epoch}"
                    f" (no improvement for {epoch - best_epoch} epochs)"
                )
                break

        self._random_key = key
        self.epoch_history_df = pd.DataFrame(
            [
                {f"train_{k}": v for k, v in t.items()} | {f"val_{k}": v for k, v in v_.items()}
                for t, v_ in zip(train_history, val_history)
            ]
        )
        self._free_data_cache()

        if checkpoint_dir is not None:
            self.save(Path(checkpoint_dir) / "final")

    def predict(self, adata: sc.AnnData, batch_size: int = 128, **kwargs) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Call .fit() before .predict()")
        return np.concatenate(
            [
                self._predict_batch(tensors)
                for tensors in self._get_predict_dataloader(adata, batch_size, **kwargs)
            ]
        )

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | Path):
        if self.state is None:
            raise RuntimeError("Call .fit() before .save()")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "config.json", "w") as f:
            json.dump({"init_params": self._init_params, "extra": self._extra_state()}, f, indent=2)
        ocp.PyTreeCheckpointer().save(str(path / "checkpoint"), {"params": self.state.params})

    @classmethod
    def load(cls, path: str | Path) -> "BaseEstimator":
        path = Path(path)
        with open(path / "config.json") as f:
            saved = json.load(f)
        inst = cls(**saved["init_params"])
        inst._restore_extra_state(saved["extra"])
        template = inst.model.init_params(jax.random.PRNGKey(0))
        restored = ocp.PyTreeCheckpointer().restore(
            str(path / "checkpoint"), item={"params": template}
        )
        inst.state = train_state.TrainState.create(
            apply_fn=inst.model.apply, params=restored["params"], tx=optax.adam(1e-3)
        )
        inst._jit_predict = inst._make_predict_step()
        return inst
