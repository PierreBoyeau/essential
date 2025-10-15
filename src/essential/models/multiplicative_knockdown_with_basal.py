import jax.numpy as jnp
from flax.linen.initializers import normal
import flax.linen as nn

from .base_model import BaseModel


class MultiplicativeKnockdownWithBasal(BaseModel):
    """Steady-state model with basal transcription and multiplicative knockdown.

    Model: dx/dt = β + (1 - u) ⊙ (Ax) - γx = 0

    Knockdown reduces regulatory input multiplicatively. Basal transcription β
    represents constitutive expression independent of regulatory control.
    Diagonal of A is zero to separate cross-regulation from decay.
    """

    def setup(self):
        self.Amat_ = self.param("Amat_", normal(), (self.n_genes, self.n_genes))
        self.decay_ = self.param("decay_", normal(), (self.n_genes))
        self.basal_transcription_ = self.param("basal_transcription_", normal(), (self.n_genes))

    def get_Amat(self):
        Amat = self.Amat_ * (1.0 - jnp.eye(self.n_genes))
        return Amat

    def get_decay(self):
        return nn.softplus(self.decay_)

    def get_basal_transcription(self):
        return nn.softplus(self.basal_transcription_)

    def __call__(self, x: jnp.ndarray, u: jnp.ndarray) -> dict:
        Amat = self.get_Amat()
        decay = self.get_decay()
        basal_transcription = self.get_basal_transcription()

        cross_terms = jnp.einsum("gj,nj->ng", Amat, x)
        perturb_term = jnp.einsum("gf,nf->ng", self.tf2gene_indicators, u)
        product_contribution = basal_transcription + (1.0 - perturb_term) * cross_terms

        decay_contribution = decay * x

        derivative = product_contribution - decay_contribution
        reco_loss = jnp.mean(derivative**2)
        N = x.shape[0]
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
