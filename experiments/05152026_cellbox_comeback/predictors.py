"""Perturbation predictor registry and baseline implementations.

Adding a baseline: subclass ``Predictor``, implement ``fit`` and ``predict``,
then decorate with ``@register("name")``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import scanpy as sc
from scipy import sparse
from tqdm import tqdm

from data import _as_dense
from legacy.cellbox import CellBoxEstimator

REGISTRY: dict[str, type] = {}


def register(name: str):
    def deco(cls):
        REGISTRY[name] = cls
        return cls

    return deco


def build(name: str, config, adata_train: sc.AnnData, Amask):
    if name not in REGISTRY:
        raise ValueError(f"Unknown predictor {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name](config, adata_train, Amask)


class Predictor(ABC):
    @abstractmethod
    def fit(self): ...

    @abstractmethod
    def predict(self, adata_control: sc.AnnData, target: str) -> np.ndarray: ...

    def collect_predictions(self, adata_test, adata_control, perturbation_col):
        """Return a result dict with ``X_gt0``, ``X_gt``, ``X_pred``, ``test_targets``."""
        targets = list(adata_test.obs[perturbation_col].astype(str).unique())
        X_gt, X_pred = [], []
        for target in tqdm(targets):
            X_gt.append(_as_dense(adata_test[adata_test.obs[perturbation_col] == target].X))
            X_pred.append(self.predict(adata_control, target))
        return dict(
            X_gt0=_as_dense(adata_control.X),
            X_gt=X_gt,
            X_pred=X_pred,
            test_targets=targets,
            var_names=list(adata_test.var_names),
        )


@register("cellbox")
class CellBoxPredictor(Predictor):
    def __init__(self, config, adata_train: sc.AnnData, Amask):
        mcfg = config.models.cellbox
        self.estimator = CellBoxEstimator(
            adata_train,
            Amask if mcfg.filter_regulators else None,
            perturbation_col=config.perturbation_col,
            control_key=config.control_key,
            standardize_inputs=mcfg.standardize_inputs,
            train_mode=mcfg.train_mode,
            n_rollout_train=mcfg.n_rollout_train,
            n_rollout_val=mcfg.n_rollout_val,
            n_val_control=mcfg.n_val_control,
        )
        self._fit_kwargs = mcfg.training.to_dict()

    def fit(self):
        self.estimator.fit(**self._fit_kwargs)

    def predict(self, adata_control: sc.AnnData, target: str) -> np.ndarray:
        return np.asarray(self.estimator.predict(adata_control, perturbation=target))


@register("mean")
class MeanPredictor(Predictor):
    """Return the control cells unchanged (ignoring the perturbation target)."""

    def __init__(self, config, adata_train: sc.AnnData, Amask):
        pass

    def fit(self):
        pass

    def predict(self, adata_control: sc.AnnData, target: str) -> np.ndarray:
        return _as_dense(adata_control.X)
