"""AnnData → jnp arrays for TF→gene prediction.

prepare_layers writes the two layers every other function in this package
assumes (``counts``: raw counts, ``lcp10k``: log-CP10K).  TFStandardizer is
fit once on a reference AnnData (with a control mask) and then maps any
AnnData to a TFArrays NamedTuple of (X_tf, Y_raw, lib).
"""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from scipy import sparse


class TFArrays(NamedTuple):
    """Inputs and targets for one forward pass."""

    X_tf: jnp.ndarray  # (n_cells, n_tfs)    normalised log-CP10K of TFs
    Y_raw: jnp.ndarray  # (n_cells, n_genes)  raw counts
    lib: jnp.ndarray  # (n_cells,)          library size (Σ raw counts)


def _dense(X):
    if sparse.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def _layer_to_dense(adata, layer):
    if layer not in adata.layers:
        raise KeyError(
            f"adata.layers[{layer!r}] missing — call tf_prediction.prepare_layers(adata) first"
        )
    return _dense(adata.layers[layer])


def prepare_layers(adata, *, raw_layer: str = "reads"):
    """Add ``counts`` and ``lcp10k`` layers from a raw-count source layer.

    Mutates ``adata`` in place and returns it for chaining.
    """
    raw = _dense(adata.layers[raw_layer])
    lib = raw.sum(1, keepdims=True)
    adata.layers["counts"] = raw
    adata.layers["lcp10k"] = np.log1p(raw / (lib + 1e-6) * 1e4).astype(np.float32)
    return adata


class TFStandardizer:
    """Fit log-CP10K TF mean/std and the per-gene control mean on a reference
    AnnData; then map any AnnData (with the same gene axis) to ``TFArrays``.

    Parameters
    ----------
    tf_cols, tf_mu, tf_sigma : np.ndarray
        Indices into ``adata.var_names`` for the TF genes, and their
        control-cell mean/std in log-CP10K space.
    ctrl_lcp_mean : np.ndarray
        (n_genes,) control-cell mean in log-CP10K space — used as the frozen
        ``x_mean`` of every model in this package.
    """

    def __init__(self, tf_cols, tf_mu, tf_sigma, ctrl_lcp_mean):
        self.tf_cols = np.asarray(tf_cols, dtype=np.int64)
        self.tf_mu = np.asarray(tf_mu, dtype=np.float32)
        self.tf_sigma = np.asarray(tf_sigma, dtype=np.float32)
        self.ctrl_lcp_mean = np.asarray(ctrl_lcp_mean, dtype=np.float32)

    @classmethod
    def fit(cls, adata, *, tf_cols, control_mask, eps: float = 1e-3):
        ctrl_lcp = _layer_to_dense(adata, "lcp10k")[np.asarray(control_mask)]
        tf_mu = ctrl_lcp[:, tf_cols].mean(0)
        tf_sigma = ctrl_lcp[:, tf_cols].std(0)
        tf_sigma = np.where(tf_sigma > eps, tf_sigma, 1.0)
        ctrl_lcp_mean = ctrl_lcp.mean(0)
        return cls(tf_cols, tf_mu, tf_sigma, ctrl_lcp_mean)

    def transform(self, adata) -> TFArrays:
        lcp = _layer_to_dense(adata, "lcp10k")
        raw = _layer_to_dense(adata, "counts")
        X_tf = (lcp[:, self.tf_cols] - self.tf_mu) / self.tf_sigma
        lib = raw.sum(1)
        return TFArrays(
            X_tf=jnp.asarray(X_tf, dtype=jnp.float32),
            Y_raw=jnp.asarray(raw, dtype=jnp.float32),
            lib=jnp.asarray(lib, dtype=jnp.float32),
        )
