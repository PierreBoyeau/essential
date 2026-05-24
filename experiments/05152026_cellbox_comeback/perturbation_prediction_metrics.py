"""Metrics for held-out perturbation prediction.

The core is :func:`profile_metrics`: given three ``(n_genes,)`` mean expression
profiles -- observed (``mu_gt``), predicted (``mu_pred``) and control
(``mu_control``) -- it returns scalar MSE and squared-Pearson metrics on both
the mean profile and the log fold-change vs control (``lfc = mu - mu_control``),
each computed over all genes and over the top-K DEGs (largest ``|lfc_gt|``).

It is pure and framework-agnostic, so the same function serves test-time
reporting and in-training validation; the jax↔host boundary sits at the
predicted mean profile.

:class:`PerturbationPredictionMetrics` is a thin wrapper that reduces
per-perturbation cell arrays to those mean profiles and tabulates the result.
"""

import numpy as np
import pandas as pd

# profile_metrics lives in the cellbox package (single source of truth, shared
# with CellBoxEstimator.validate_perturbations); re-exported here for the API.
from cellbox.metrics import profile_metrics


class PerturbationPredictionMetrics:
    """Tabulate :func:`profile_metrics` over a set of held-out perturbations.

    Parameters
    ----------
    n_top_degs:
        Number of top DEGs (by observed ``|mu_gt - mu_control|``) used for the
        ``*_top_degs`` metrics.
    """

    def __init__(self, n_top_degs: int = 20):
        self.n_top_degs = n_top_degs

    def compute_metrics(self, X_gt0, X_gt, X_pred, targets=None, output_path=None):
        """Per-perturbation scalar metrics, one row per target.

        Parameters
        ----------
        X_gt0:
            (n_cells0, n_genes) observed control cells.
        X_gt, X_pred:
            Lists of (n_cells, n_genes) arrays, aligned with each other and with
            ``targets``.
        targets:
            Optional perturbation names; defaults to integer positions.
        output_path:
            If given, the DataFrame is written there as CSV.
        """
        if len(X_gt) != len(X_pred):
            raise ValueError(f"X_gt ({len(X_gt)}) and X_pred ({len(X_pred)}) length mismatch")
        if targets is None:
            targets = list(range(len(X_gt)))
        mu_control = np.asarray(X_gt0).mean(0)

        rows = []
        for target, Xg, Xp in zip(targets, X_gt, X_pred):
            Xg, Xp = np.asarray(Xg), np.asarray(Xp)
            rows.append(
                {
                    "target": target,
                    "n_cells_gt": Xg.shape[0],
                    "n_cells_pred": Xp.shape[0],
                    **profile_metrics(Xg.mean(0), Xp.mean(0), mu_control, self.n_top_degs),
                }
            )
        metrics = pd.DataFrame(rows).set_index("target")

        if output_path is not None:
            metrics.to_csv(output_path)
        return metrics

    def from_result(self, result, output_path=None):
        """Compute metrics from a ``run()`` result dict."""
        df = self.compute_metrics(
            result["X_gt0"], result["X_gt"], result["X_pred"], targets=result["test_targets"]
        )
        df["model_name"] = result["model_name"]
        df = df.reset_index()
        if output_path is not None:
            df.to_csv(output_path, index=False)
        return df
