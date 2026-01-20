import jax
import jax.numpy as jnp
from flax.linen.initializers import glorot_normal, normal, zeros, constant, ones
import flax.linen as nn
import diffrax

from .base_model import BaseModel


class Sigmoid2SteadyStateModel(BaseModel):
    def setup(self):
        self.Amat_ = self.param("Amat_", glorot_normal(), (self.n_genes, self.n_genes))
        self.bias_term_sigmoid_ = self.param("bias_term_sigmoid_", zeros, (self.n_genes))
        self.scale_factor_ = self.param("scale_factor_", ones, (self.n_genes))

    def get_Amat(self):
        Amat = self.Amat_ * (1.0 - jnp.eye(self.n_genes))
        if self.Amask is not None:
            Amat = Amat * self.Amask
        return Amat

    def get_scale_factor(self):
        return nn.softplus(self.scale_factor_)

    def predict(self, y, u):
        A_mat = self.get_Amat()
        conc_contribution = jnp.einsum("gj,j->g", A_mat, y) + self.bias_term_sigmoid_ - 10 * u

        alpha = nn.sigmoid(conc_contribution)
        alpha = self.get_scale_factor() * alpha
        return alpha

    def __call__(self, xt: jnp.ndarray, u: jnp.ndarray, **kwargs) -> dict:
        A_mat = self.get_Amat()
        x_pred = jax.vmap(self.predict, in_axes=(0, 0))(xt, u)
        
        deltas = x_pred - xt
        deltas = deltas * (1.0 - u)

        reco_loss = self.get_reconstruction_loss(deltas)
        l1_prior = self.get_laplace_prior(A_mat, self.lambda_prior) / self.n_obs
        loss = reco_loss + l1_prior
        return {"loss": loss, "reco_loss": reco_loss, "l1_prior": l1_prior}
