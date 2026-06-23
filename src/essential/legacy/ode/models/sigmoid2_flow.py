import diffrax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import constant, glorot_normal, normal, ones, zeros

from .base_model import BaseModel


class Sigmoid2FlowModel(BaseModel):
    def setup(self):
        self.Amat_ = self.param("Amat_", glorot_normal(), (self.n_genes, self.n_genes))
        self.decay_ = self.param("decay_", ones, (self.n_genes))
        self.bias_term_sigmoid_ = self.param("bias_term_sigmoid_", zeros, (self.n_genes))
        self.perturbation_effect_ = self.param("perturbation_effect_", ones, (self.n_genes))
        self.scale_factor_ = self.param("scale_factor_", ones, (self.n_genes))

    def get_Amat(self):
        Amat = self.Amat_ * (1.0 - jnp.eye(self.n_genes))
        if self.Amask is not None:
            Amat = Amat * self.Amask
        return Amat

    def get_decay(self):
        return nn.softplus(self.decay_)

    def get_scale_factor(self):
        return nn.softplus(self.scale_factor_)

    def get_perturbation_effect(self):
        return self.perturbation_effect_

    def ode_fn(self, y, u):
        A_mat = self.get_Amat()
        u_gene = self.get_perturbation_effect() * u
        conc_contribution = jnp.einsum("gj,j->g", A_mat, y) + self.bias_term_sigmoid_ - u_gene

        alpha = nn.sigmoid(conc_contribution)
        alpha = self.get_scale_factor() * alpha
        beta = self.get_decay() * y
        return alpha - beta

    def __call__(self, x0: jnp.ndarray, x1: jnp.ndarray, t: jnp.ndarray, u: jnp.ndarray) -> dict:
        xt = t[:, None] * x0 + (1 - t[:, None]) * x1
        vtheta = jax.vmap(self.ode_fn, in_axes=(0, 0))(xt, u)
        ut = x1 - x0
        loss = jnp.mean((vtheta - ut) ** 2)
        return {"loss": loss, "reco_loss": loss}
