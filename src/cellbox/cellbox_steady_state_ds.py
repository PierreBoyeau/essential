import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import glorot_normal, zeros
from numpyro.distributions import NegativeBinomial2

from .regulator_net import RegulatorNet


def _const_init(value):
    return lambda key, shape, dtype=jnp.float32: jnp.asarray(value, dtype=dtype)


class CellBoxSteadyStateDS(nn.Module):
    """Gaussian CellBox with DeepSets regulator aggregation.

    Identical to CellBoxSteadyState except A @ normalize(y) is replaced by
    RegulatorNet(normalize(y)). Requires Amask.
    """

    n_genes: int
    n_obs: int
    lambda_prior: float = 1.0
    Amask: jnp.ndarray | None = None
    x_mean: jnp.ndarray | None = None
    x_std: jnp.ndarray | None = None
    epsilon_init: jnp.ndarray | None = None
    reg_embed_dim: int = 16
    reg_hidden_dim: int = 16

    def setup(self):
        G = self.n_genes
        _eps = self.epsilon_init if self.epsilon_init is not None else jnp.zeros(G)
        self.epsilon_ = self.param("epsilon_", _const_init(_eps), (G,))
        self.b_ = self.param("b_", zeros, (G,))
        _x_mean = self.x_mean if self.x_mean is not None else jnp.zeros(G)
        _x_std = self.x_std if self.x_std is not None else jnp.ones(G)
        self.x_mean_ = self.param("x_mean_", _const_init(_x_mean), (G,))
        self.x_std_ = self.param("x_std_", _const_init(_x_std), (G,))
        self.reg_net = RegulatorNet(
            Amask=self.Amask,
            embed_dim=self.reg_embed_dim,
            hidden_dim=self.reg_hidden_dim,
        )

    def init_params(self, key):
        dummy_xt = jnp.zeros((4, self.n_genes))
        dummy_u = jnp.zeros((4, self.n_genes))
        return self.init(key, dummy_xt, dummy_u)["params"]

    def normalize_predictor(self, y):
        return (y - jax.lax.stop_gradient(self.x_mean_)) / jax.lax.stop_gradient(self.x_std_)

    def preactivation(self, y, u):
        p = 10 * jnp.ones(self.n_genes)
        return self.reg_net(self.normalize_predictor(y)) - p * u + self.b_

    def predict(self, y, u):
        return self.epsilon_ * nn.sigmoid(self.preactivation(y, u))

    def predict_steady_state(self, y, u, n_steps=100):
        return jax.lax.fori_loop(0, n_steps, lambda _, val: self.predict(val, u), y)

    def __call__(self, xt: jnp.ndarray, u: jnp.ndarray, x0: jnp.ndarray | None = None) -> dict:
        args = jax.vmap(self.preactivation)(xt, u)
        x_pred = self.epsilon_ * nn.sigmoid(args)

        mask = 1.0 - u
        deltas = (x_pred - xt) * mask
        reco_loss = jnp.mean(jnp.sum(deltas**2, axis=1))

        frac_saturated = jnp.sum((jnp.abs(args) > 4.0) * mask) / jnp.sum(mask)
        return {"loss": reco_loss, "reco_loss": reco_loss, "frac_saturated": frac_saturated}


class CellBoxSteadyStateNBDS(nn.Module):
    """Negative-binomial CellBox with DeepSets regulator aggregation.

    Identical to CellBoxSteadyStateNB except A @ normalize(y) is replaced by
    RegulatorNet(normalize(y)). Requires Amask.
    """

    n_genes: int
    n_obs: int
    lambda_prior: float = 1.0
    Amask: jnp.ndarray | None = None
    x_mean: jnp.ndarray | None = None
    x_std: jnp.ndarray | None = None
    epsilon_init: jnp.ndarray | None = None
    reg_embed_dim: int = 16
    reg_hidden_dim: int = 16

    def setup(self):
        G = self.n_genes
        self.epsilon_ = self.param("epsilon_", zeros, (G,))
        self.b_ = self.param("b_", zeros, (G,))
        _x_mean = self.x_mean if self.x_mean is not None else jnp.zeros(G)
        _x_std = self.x_std if self.x_std is not None else jnp.ones(G)
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
        self.reg_net = RegulatorNet(
            Amask=self.Amask,
            embed_dim=self.reg_embed_dim,
            hidden_dim=self.reg_hidden_dim,
        )

    def init_params(self, key):
        dummy_xt = jnp.zeros((4, self.n_genes))
        dummy_u = jnp.zeros((4, self.n_genes))
        return self.init(key, dummy_xt, dummy_u)["params"]

    def _raw_to_logcp10k(self, y):
        lib = jnp.sum(y)
        return jnp.log1p(y / (lib + 1e-6) * 1e4)

    def normalize_predictor(self, y):
        return (y - jax.lax.stop_gradient(self.x_mean_)) / jax.lax.stop_gradient(self.x_std_)

    def preactivation(self, y, u):
        p = 10 * jnp.ones(self.n_genes)
        return self.reg_net(self.normalize_predictor(y)) - p * u + self.b_

    def predict(self, y, u):
        return jnp.exp(self.epsilon_) * nn.sigmoid(self.preactivation(y, u))

    def predict_steady_state(self, y, u, n_steps=100):
        y0 = self._raw_to_logcp10k(y)
        ypred_logcp10k = jax.lax.fori_loop(0, n_steps, lambda _, val: self.predict(val, u), y0)
        ypred_scale = y.sum()
        mean = jnp.expm1(ypred_logcp10k) * ypred_scale / 1e4
        overd = jnp.exp(self.overdispersion_)
        dist = NegativeBinomial2(mean=mean, concentration=overd)
        samples = dist.sample(jax.random.PRNGKey(0), (100,))
        # samples.mean(0)
        samples = jnp.log1p(samples / ypred_scale * 1e4)
        return samples.mean(0)

    def __call__(
        self, xt: jnp.ndarray, u: jnp.ndarray, x0: jnp.ndarray | None = None, n_steps: int = 0
    ) -> dict:
        lib = xt.sum(-1, keepdims=True)
        xt_lcp10k = jnp.log1p(xt / (lib + 1e-6) * 1e4)

        if x0 is not None and n_steps > 0:
            x_pred_lcp10k = jax.vmap(
                lambda _x0, _u: jax.lax.fori_loop(
                    0, n_steps, lambda _, v: self.predict(v, _u), self._raw_to_logcp10k(_x0)
                )
            )(x0, u)
            args = jax.vmap(self.preactivation)(x_pred_lcp10k, u)
        else:
            args = jax.vmap(self.preactivation)(xt_lcp10k, u)
            x_pred_lcp10k = jnp.exp(self.epsilon_) * nn.sigmoid(args)

        xconcat = jnp.log1p(xt)
        lib_scale = self.library_network(xconcat)
        mean = jnp.maximum(jnp.expm1(x_pred_lcp10k) * lib_scale, 1e-8)
        overdispersion = jnp.exp(self.overdispersion_)
        lkl = NegativeBinomial2(mean=mean, concentration=overdispersion).log_prob(xt)
        mask = 1.0 - u

        reco_loss = -jnp.mean(lkl * mask)
        frac_saturated = jnp.sum((jnp.abs(args) > 4.0) * mask) / jnp.sum(mask)
        return {"loss": reco_loss, "reco_loss": reco_loss, "frac_saturated": frac_saturated}
