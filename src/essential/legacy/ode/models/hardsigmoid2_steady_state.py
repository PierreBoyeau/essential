import diffrax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import constant, glorot_normal, normal, ones, zeros

from .base_model import BaseModel


class HardSigmoid2SteadyStateModel(BaseModel):
    def setup(self):
        self.Amat_ = self.param("Amat_", glorot_normal(), (self.n_genes, self.n_genes))
        self.bias_term_sigmoid_ = self.param("bias_term_sigmoid_", zeros, (self.n_genes))
        self.scale_factor_ = self.param("scale_factor_", ones, (self.n_genes))

    def init_params(self, key):
        dummy_x0 = jnp.zeros((4, self.n_genes))
        dummy_xt = jnp.zeros((4, self.n_genes))
        params = self.init(key, dummy_x0, dummy_xt)["params"]
        return params

    def get_Amat(self):
        Amat = self.Amat_ * (1.0 - jnp.eye(self.n_genes))
        if self.Amask is not None:
            Amat = Amat * self.Amask
        return Amat

    def get_scale_factor(self):
        return nn.softplus(self.scale_factor_)

    def get_perturbation_effect(self):
        return self.perturbation_effect_

    def predict(self, y, u):
        A_mat = self.get_Amat()
        y_ko = y * (1.0 - u)
        conc_contribution = jnp.einsum("gj,j->g", A_mat, y_ko) + self.bias_term_sigmoid_

        alpha = nn.sigmoid(conc_contribution)
        alpha = self.get_scale_factor() * alpha
        return alpha

    def __call__(self, xt: jnp.ndarray, u: jnp.ndarray, **kwargs) -> dict:
        A_mat = self.get_Amat()
        rhs = jax.vmap(self.predict, in_axes=(0, 0))(xt, u)

        deltas = rhs - xt
        deltas = deltas * (1.0 - u)  # do not learn KO genes
        reco_loss = self.get_reconstruction_loss(deltas)
        l1_prior = self.get_laplace_prior(A_mat, self.lambda_prior) / self.n_obs
        loss = reco_loss + l1_prior
        return {"loss": loss, "reco_loss": reco_loss, "l1_prior": l1_prior}

    def predict_steady_state(self, y, u, n_steps=100):
        def body_fun(i, val):
            return self.predict(val, u)

        return jax.lax.fori_loop(0, n_steps, body_fun, y)
