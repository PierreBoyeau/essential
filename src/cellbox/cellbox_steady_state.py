import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import glorot_normal, zeros


class CellBoxSteadyState(nn.Module):
    """CellBox steady-state model.

    Forward map (per gene j): x_j = ε_j · σ(aⱼᵀ x̃ − pⱼ uⱼ + bⱼ)
    where x̃ is the (optionally) standardized regulator state. Trained by
    matching the fixed-point condition x* = f(x*, u).

    The output stays on the raw data scale (ε ⊙ σ(·) ∈ [0, ε]); only the
    sigmoid argument reads a standardized view of the state, so the
    fixed point is unchanged (x̃ is a frozen affine function of x).
    """

    n_genes: int
    n_obs: int
    lambda_prior: float = 1.0
    Amask: jnp.ndarray | None = None
    x_mean: jnp.ndarray | None = None
    x_std: jnp.ndarray | None = None
    epsilon_init: jnp.ndarray | None = None

    def setup(self):
        G = self.n_genes
        self.A_ = self.param("A_", glorot_normal(), (G, G))
        # b stays at 0 so the centered control state sits at σ(0) = 0.5.
        self.b_ = self.param("b_", zeros, (G,))
        # ε sets the output range; init at ~2x the control mean so that, with
        # σ = 0.5 at baseline, the output already matches the control level and
        # the mean-chasing gradient has no reason to push b into saturation.
        eps_init = lambda key, shape, dtype=jnp.float32: jnp.asarray(self.epsilon_init, dtype=dtype)
        self.epsilon_ = self.param("epsilon_", eps_init, (G,))

    def init_params(self, key):
        dummy_xt = jnp.zeros((4, self.n_genes))
        dummy_u = jnp.zeros((4, self.n_genes))
        return self.init(key, dummy_xt, dummy_u)["params"]

    def get_Amat(self):
        A = self.A_ * (1.0 - jnp.eye(self.n_genes))
        if self.Amask is not None:
            A = A * self.Amask
        return A

    def get_epsilon(self):
        return self.epsilon_

    def get_p(self):
        return 10 * jnp.ones(self.n_genes)

    def normalize_predictor(self, y):
        """Standardize the regulator state fed into the sigmoid argument.

        Subtracts ``x_mean`` and divides by ``x_std`` when those frozen
        per-gene statistics are available, so the pre-activation ``A @ ỹ``
        sits in the responsive region of the sigmoid instead of saturating.
        Returns ``y`` unchanged when neither is set.
        """
        if self.x_mean is not None:
            y = y - self.x_mean
        if self.x_std is not None:
            y = y / self.x_std
        return y

    def preactivation(self, y, u):
        """Sigmoid argument A·ỹ − p·u + b — the quantity whose magnitude drives saturation."""
        A = self.get_Amat()
        return jnp.dot(A, self.normalize_predictor(y)) - self.get_p() * u + self.b_

    def predict(self, y, u):
        """Single-sample forward pass: x = ε ⊙ σ(A ỹ − p ⊙ u + b)."""
        return self.get_epsilon() * nn.sigmoid(self.preactivation(y, u))

    def predict_steady_state(self, y, u, n_steps=100):
        """Iterate the forward map to approximate the steady state."""
        return jax.lax.fori_loop(0, n_steps, lambda _, val: self.predict(val, u), y)

    def __call__(self, xt: jnp.ndarray, u: jnp.ndarray) -> dict:
        args = jax.vmap(self.preactivation)(xt, u)
        x_pred = self.get_epsilon() * nn.sigmoid(args)

        mask = 1.0 - u
        deltas = (x_pred - xt) * mask
        reco_loss = jnp.mean(jnp.sum(deltas**2, axis=1))

        frac_saturated = jnp.sum((jnp.abs(args) > 4.0) * mask) / jnp.sum(mask)
        return {"loss": reco_loss, "reco_loss": reco_loss, "frac_saturated": frac_saturated}
