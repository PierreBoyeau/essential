import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import glorot_normal, zeros
from numpyro.distributions import NegativeBinomial2


def _const_init(value):
    return lambda key, shape, dtype=jnp.float32: jnp.asarray(value, dtype=dtype)


class CellBoxSteadyStateNB(nn.Module):
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
        self.b_ = self.param("b_", zeros, (G,))
        _eps = self.epsilon_init if self.epsilon_init is not None else jnp.zeros(G)
        self.epsilon_ = self.param("epsilon_", _const_init(_eps), (G,))

        _amask = self.Amask if self.Amask is not None else jnp.ones((G, G))
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

    def normalize_predictor(self, y):
        x = jnp.log1p(y)
        x = x - jnp.log1p(jax.lax.stop_gradient(self.x_mean_))
        return x / jax.lax.stop_gradient(self.x_std_)

    def preactivation(self, y, u):
        A = self.get_Amat()
        p = 10 * jnp.ones(self.n_genes)
        return jnp.dot(A, self.normalize_predictor(y)) - p * u + self.b_

    def predict(self, y, u):
        out = self.epsilon_ * nn.sigmoid(self.preactivation(y, u))
        return out

    def predict_steady_state(self, y, u, n_steps=100):
        out = jax.lax.fori_loop(0, n_steps, lambda _, val: self.predict(val, u), y)
        return jnp.log(1 + 1e4 * out)

    def __call__(self, xt: jnp.ndarray, u: jnp.ndarray, x0: jnp.ndarray | None = None) -> dict:
        args = jax.vmap(self.preactivation)(xt, u)
        x_pred = self.epsilon_ * nn.sigmoid(args)

        obs_lib = xt.sum(axis=-1, keepdims=True)
        xconcat = jnp.concatenate([obs_lib, self.normalize_predictor(x_pred)], axis=-1)
        lib_size = self.library_network(xconcat)

        mean = x_pred * lib_size
        overdispersion = jnp.exp(self.overdispersion_)
        lkl = NegativeBinomial2(mean=mean, concentration=overdispersion).log_prob(xt)
        mask = 1.0 - u

        reco_loss = -jnp.mean(lkl * mask)

        frac_saturated = jnp.sum((jnp.abs(args) > 4.0) * mask) / jnp.sum(mask)
        return {"loss": reco_loss, "reco_loss": reco_loss, "frac_saturated": frac_saturated}
