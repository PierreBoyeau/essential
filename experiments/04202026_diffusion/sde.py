"""
Stationary SDE causal model with linear drift.

Observational model:
    dx_t = (A x_t + b) dt + sigma dW_t

Under a shift intervention on variable u:
    dx_t = (A x_t + b + delta_u) dt + sigma dW_t

where delta_u is a d-dimensional vector that is zero everywhere except
at index u, where it takes a learned scalar specific to that intervention
target.  sigma is a learned diagonal matrix (constant, not state-dependent).
"""

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import glorot_normal, ones, zeros


class SDE(nn.Module):
    """
    Linear drift SDE causal model.

    Parameters
    ----------
    n_vars : int
        Dimension d of the state space.

    Learned parameters
    ------------------
    A     : (n_vars, n_vars)  weight matrix of the drift
    b     : (n_vars,)         bias of the drift
    delta : (n_vars,)         one shift scalar per possible intervention target
    log_sigma_diag : (n_vars,)  log of the diagonal entries of sigma
    """

    n_vars: int

    def setup(self):
        self.A = self.param("A", glorot_normal(), (self.n_vars, self.n_vars))
        self.b = self.param("b", ones, (self.n_vars,))
        self.delta = self.param("delta", zeros, (self.n_vars,))
        self.log_sigma_diag = self.param("log_sigma_diag", zeros, (self.n_vars,))

    def forward(self, x: jnp.ndarray, u: int | None = None):
        """
        Evaluate the drift and diffusion at x under intervention u.

        Parameters
        ----------
        x : jnp.ndarray, shape (n_vars,) or (batch, n_vars)
            State at which to evaluate the SDE.
        u : int or None
            Index of the intervened variable (0-indexed).
            None indicates the observational regime (no shift).

        Returns
        -------
        f     : jnp.ndarray, same shape as x
                Drift  f(x) = A x + b  (+delta_u under intervention).
        c     : jnp.ndarray, shape (n_vars,)
                Diagonal of the diffusion matrix sigma = diag(c).
        A     : jnp.ndarray, shape (n_vars, n_vars)
                Weight matrix (returned for inspection / KDS computation).
        """
        A = self.A

        # f(x) = Ax + b
        # x @ A.T handles both (n_vars,) and (batch, n_vars) inputs uniformly.
        f = x @ A.T + self.b

        # Shift intervention: add delta[u] at position u, zero elsewhere.
        if u is not None:
            delta_u = jnp.zeros(self.n_vars).at[u].set(self.delta[u])
            f = f + delta_u

        # Return the diagonal vector c = exp(log_sigma_diag) rather than the
        # full (d, d) matrix.  The caller only ever needs c_i^2 * H_ii, which
        # does not require materialising the off-diagonal entries.
        c = jnp.exp(self.log_sigma_diag)  # (n_vars,)

        return f, c, A

    @staticmethod
    def param_stats(params: dict):
        """Print min / max / mean / std and NaN count for each parameter."""
        entries = {
            "A": np.asarray(params["A"]).ravel(),
            "b": np.asarray(params["b"]).ravel(),
            "delta": np.asarray(params["delta"]).ravel(),
            "log_sigma": np.asarray(params["log_sigma_diag"]).ravel(),
            "sigma": np.exp(np.asarray(params["log_sigma_diag"])).ravel(),
        }
        print(f"{'param':<14} {'min':>10} {'max':>10} {'mean':>10} {'std':>10} {'nan':>5}")
        print("-" * 62)
        for name, v in entries.items():
            nan = np.isnan(v).sum()
            f = v[np.isfinite(v)]
            if len(f):
                print(
                    f"{name:<14} {f.min():>10.3e} {f.max():>10.3e} {f.mean():>10.3e} {f.std():>10.3e} {nan:>5}"
                )
            else:
                print(f"{name:<14} {'':>10} {'':>10} {'':>10} {'':>10} {nan:>5}  ← all NaN")
