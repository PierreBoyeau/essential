"""Diagnostic utilities for the CellBox comeback experiment.

A quick sanity check on whether a (regulator, target) gene pair shows the
saturating dose-response CellBox assumes, i.e. whether the target expression
``y`` is well described by a sigmoid of the regulator expression ``x``::

    y ~= a * sigmoid(X @ b + c)

where ``X`` is ``(n, d)`` and ``b`` is a ``d``-vector. Passing a 1-D ``x``
is equivalent to ``d = 1``.

This module is utilities only -- no data loading or plotting side effects.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def _sse_and_grad(params, x, y, fit_intercept):
    """Sum-of-squared-errors loss and its gradient for the sigmoid fit.

    ``x`` is always ``(n, d)`` here.  Parameter layout:
    ``[a, b_0, ..., b_{d-1}]`` and, when ``fit_intercept``, ``c`` appended.
    """
    d = x.shape[1]
    a = params[0]
    b = params[1 : 1 + d]
    c = params[1 + d] if fit_intercept else 0.0

    sig = expit(x @ b + c)
    dsig = sig * (1.0 - sig)
    r = y - a * sig

    sse = np.sum(r**2)
    grad = np.empty(len(params))
    grad[0] = -2.0 * np.sum(r * sig)
    grad[1 : 1 + d] = -2.0 * (x.T @ (r * a * dsig))
    if fit_intercept:
        grad[1 + d] = -2.0 * np.sum(r * a * dsig)
    return sse, grad


def fit_sigmoid(x, y, fit_intercept=True, n_restarts=5, seed=42):
    """Least-squares fit of ``y ~= a * sigmoid(X @ b + c)``.

    Pass ``fit_intercept=False`` to force ``c = 0``. Uses multiple random
    restarts of L-BFGS-B to avoid getting stuck in poor local minima.

    Parameters
    ----------
    x, y:
        ``x`` may be 1-D ``(n,)`` or 2-D ``(n, d)``; 1-D is treated as
        ``(n, 1)``.  ``y`` is 1-D of length ``n``.
    fit_intercept:
        Whether to fit the intercept ``c`` (otherwise ``c = 0``).
    n_restarts:
        Number of random initialisations; the best fit is kept.
    seed:
        Seed for the restart initialisation.

    Returns
    -------
    dict with keys ``a`` (scalar), ``b`` (array of shape ``(d,)``), ``c``
    (scalar) and ``sse`` (residual sum of squares).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    y = np.asarray(y, dtype=float).ravel()

    n, d = x.shape
    rng = np.random.default_rng(seed)
    n_params = 1 + d + (1 if fit_intercept else 0)

    best = None
    for _ in range(n_restarts):
        p0 = rng.uniform(-3, 3, size=n_params)
        res = minimize(
            _sse_and_grad,
            p0,
            args=(x, y, fit_intercept),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if best is None or res.fun < best.fun:
            best = res

    a = float(best.x[0])
    b = best.x[1 : 1 + d].copy()
    c = float(best.x[1 + d]) if fit_intercept else 0.0
    return {"a": a, "b": b, "c": c, "sse": float(best.fun)}


def sigmoid_predict(x, params):
    """Evaluate ``a * sigmoid(X @ b + c)`` using a ``fit_sigmoid`` result."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    return params["a"] * expit(x @ params["b"] + params["c"])
