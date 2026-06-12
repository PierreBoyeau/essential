import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import glorot_normal, zeros
from numpyro.distributions import NegativeBinomial2


def _const_init(value):
    return lambda key, shape, dtype=jnp.float32: jnp.asarray(value, dtype=dtype)


def _scaled_glorot(scale: float):
    def init(key, shape, dtype=jnp.float32):
        return scale * glorot_normal()(key, shape, dtype)

    return init


class CellBoxSteadyStateNB(nn.Module):
    n_genes: int
    n_obs: int
    lambda_prior: float = 1.0
    Amask: jnp.ndarray | None = None
    x_mean: jnp.ndarray | None = None  # log-CP10K mean of control cells
    x_std: jnp.ndarray | None = None  # log-CP10K std of control cells
    epsilon_init: jnp.ndarray | None = None  # log-CP10K scale
    a_scale: float = 1.0  # glorot rescaling for masked A
    mean_mode: str = "absolute"  # "absolute" | "residual" (control-anchored)

    def setup(self):
        G = self.n_genes

        _amask = self.Amask if self.Amask is not None else jnp.ones((G, G))
        self.A_ = self.param("A_", _scaled_glorot(self.a_scale), (G, G))
        self.b_ = self.param("b_", zeros, (G,))
        _eps = self.epsilon_init if self.epsilon_init is not None else jnp.zeros(G)
        # self.epsilon_ = self.param("epsilon_", _const_init(_eps), (G,))
        self.epsilon_ = self.param("epsilon_", zeros, (G,))
        _x_mean = self.x_mean if self.x_mean is not None else jnp.zeros(G)
        _x_std = self.x_std if self.x_std is not None else jnp.ones(G)
        self.Amask_ = self.param("Amask_", _const_init(_amask), (G, G))
        self.x_mean_ = self.param("x_mean_", _const_init(_x_mean), (G,))
        self.x_std_ = self.param("x_std_", _const_init(_x_std), (G,))

        self.overdispersion_ = self.param("overdispersion_", zeros, (G,))
        self.library_network = nn.Sequential(
            [
                nn.Dense(128, kernel_init=glorot_normal(), bias_init=zeros),
                nn.softplus,
                nn.Dense(1, kernel_init=glorot_normal(), bias_init=zeros),
                nn.softplus,
            ],
        )

    def init_params(self, key):
        dummy_xt = jnp.zeros((4, self.n_genes))
        dummy_u = jnp.zeros((4, self.n_genes))
        return self.init(key, dummy_xt, dummy_u)["params"]

    def get_Amat(self):
        A = self.A_ * (1.0 - jnp.eye(self.n_genes))
        return A * jax.lax.stop_gradient(self.Amask_)

    def _raw_to_logcp10k(self, y):
        """Single-obs raw counts → log-CP10K."""
        lib = jnp.sum(y)
        return jnp.log1p(y / (lib + 1e-6) * 1e4)

    def normalize_predictor(self, y):
        """y: log-CP10K (single obs) → zero-mean / unit-std (frozen stats)."""
        return (y - jax.lax.stop_gradient(self.x_mean_)) / jax.lax.stop_gradient(self.x_std_)

    def preactivation(self, y, u):
        """y: log-CP10K (single obs)."""
        A = self.get_Amat()
        p = 10 * jnp.ones(self.n_genes)
        return jnp.dot(A, self.normalize_predictor(y)) - p * u + self.b_

    def predict(self, y, u):
        """y: log-CP10K → log-CP10K (single obs)."""
        return jnp.exp(self.epsilon_) * nn.sigmoid(self.preactivation(y, u))

    def predict_steady_state(self, y, u, n_steps=100):
        """y: raw counts → log-CP10K mean response (single obs, for evaluation).

        Deterministic mean prediction; mirrors the mean construction in __call__
        (including the control-anchored residual). Library cancels in log-CP10K space.
        """
        y0 = self._raw_to_logcp10k(y)
        x_pred = jax.lax.fori_loop(0, n_steps, lambda _, v: self.predict(v, u), y0)
        if self.mean_mode == "residual":
            x0_pred = jax.lax.fori_loop(
                0, n_steps, lambda _, v: self.predict(v, jnp.zeros_like(u)), y0
            )
            x_pred = self.x_mean_ + x_pred - x0_pred
        return x_pred

    def _rollout_lcp10k(self, x0, u, n_steps):
        """Roll out from raw-count controls x0 under perturbation u → log-CP10K steady state."""
        return jax.vmap(
            lambda _x0, _u: jax.lax.fori_loop(
                0, n_steps, lambda _, v: self.predict(v, _u), self._raw_to_logcp10k(_x0)
            )
        )(x0, u)

    def __call__(
        self, xt: jnp.ndarray, u: jnp.ndarray, x0: jnp.ndarray | None = None, n_steps: int = 0
    ) -> dict:
        """
        xt: (batch, genes) raw counts — always the NB likelihood target.
        x0: (batch, genes) raw counts — if given and n_steps > 0, roll out from x0;
            otherwise reconstruct from xt in one step.
        """
        lib = xt.sum(-1, keepdims=True)  # (batch, 1)
        xt_lcp10k = jnp.log1p(xt / (lib + 1e-6) * 1e4)

        if x0 is not None and n_steps > 0:
            # rollout: iterate from control x0, evaluate against xt
            x_pred_lcp10k = self._rollout_lcp10k(x0, u, n_steps)
            args = jax.vmap(self.preactivation)(x_pred_lcp10k, u)
            if self.mean_mode == "residual":
                # control-anchored residual: population control mean + predicted (perturbed −
                # control) shift, with the frozen control rollout as a baseline-error correction.
                x0_pred_lcp10k = jax.lax.stop_gradient(
                    self._rollout_lcp10k(x0, jnp.zeros_like(u), n_steps)
                )
                x_pred_lcp10k = jax.lax.stop_gradient(self.x_mean_) + x_pred_lcp10k - x0_pred_lcp10k
        else:
            # reconstruction: one step from xt
            args = jax.vmap(self.preactivation)(xt_lcp10k, u)
            x_pred_lcp10k = jnp.exp(self.epsilon_) * nn.sigmoid(args)

        # shared NB likelihood head
        xconcat = jnp.log1p(xt)
        # lib_scale = self.library_network(xconcat)  # (batch, 1), ≈ total_counts / 1e4
        lib_scale = xt.sum(-1, keepdims=True) / 1e4
        mean = jnp.maximum(
            jnp.expm1(x_pred_lcp10k) * lib_scale, 1e-8
        )  # (batch, genes), raw-count scale
        overdispersion = jnp.exp(self.overdispersion_)
        lkl = NegativeBinomial2(mean=mean, concentration=overdispersion).log_prob(xt)
        mask = 1.0 - u

        reco_loss = -jnp.mean(lkl * mask)
        frac_saturated = jnp.sum((jnp.abs(args) > 4.0) * mask) / jnp.sum(mask)
        return {"loss": reco_loss, "reco_loss": reco_loss, "frac_saturated": frac_saturated}
