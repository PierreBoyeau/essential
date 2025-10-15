import jax.numpy as jnp
from flax.linen.initializers import normal
import flax.linen as nn

from .base_model import BaseModel


class MultiplicativeKnockdownModel(BaseModel):
    """Steady-state model with multiplicative knockdown of regulatory inputs.

    Model: dx/dt = (1 - u) ⊙ (Ax) - γx = 0

    Knockdown reduces regulatory input multiplicatively. Diagonal of A is zero
    to separate cross-regulation from decay.
    """

    def setup(self):
        self.Amat_ = self.param("Amat_", normal(), (self.n_genes, self.n_genes))
        self.decay_ = self.param("decay_", normal(), (self.n_genes))

    def get_Amat(self):
        Amat = self.Amat_ * (1.0 - jnp.eye(self.n_genes))
        return Amat

    def get_decay(self):
        return nn.softplus(self.decay_)

    def __call__(self, x0: jnp.ndarray, xt: jnp.ndarray, t: jnp.ndarray, u: jnp.ndarray) -> dict:
        Amat = self.get_Amat()
        decay = self.get_decay()

        cross_terms = jnp.einsum("gj,nj->ng", Amat, xt)
        perturb_term = jnp.einsum("gf,nf->ng", self.tf2gene_indicators, u)
        product_contribution = (1.0 - perturb_term) * cross_terms

        decay_contribution = decay * xt

        derivative = product_contribution - decay_contribution
        reco_loss = jnp.mean(derivative**2)
        N = xt.shape[0]
        lap_density = jnp.abs(Amat) * self.lambda_prior
        l1_prior = jnp.sum(lap_density) / N
        loss = reco_loss + l1_prior

        diagnostics = {
            "cross_terms_mean": jnp.mean(jnp.abs(cross_terms)),
            "perturb_term_mean": jnp.mean(jnp.abs(perturb_term)),
            "u_active_fraction": jnp.mean(u.sum(axis=1) > 0),
        }

        return {
            "loss": loss,
            "reco_loss": reco_loss,
            "l1_prior": l1_prior,
            **diagnostics,
        }
