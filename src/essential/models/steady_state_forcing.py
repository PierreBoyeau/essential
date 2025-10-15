import jax.numpy as jnp
from flax.linen.initializers import normal
import flax.linen as nn

from .base_model import BaseModel


class SteadyStateForcingModel(BaseModel):
    """Steady-state model with external forcing perturbation.

    Model: Σ_j A_{ij} x_j = u_i
    Issue: Treats knockdown as external forcing (unphysical at steady-state)
    """

    def setup(self):
        self.Amat_ = self.param("Amat_", normal(), (self.n_genes, self.n_genes))
        self.bvec_ = self.param("bvec_", normal(), (self.n_tfs))

    def get_Amat(self):
        # Amat = self.Amat_ * (1 - jnp.eye(self.n_genes))
        # this was a mistake, we should allow for diagonal terms
        # to capture decay of regulatory elements
        return self.Amat_

    def get_bvec(self):
        return -nn.softplus(self.bvec_)

    def __call__(self, x: jnp.ndarray, u: jnp.ndarray) -> dict:
        indic_times_param = u * self.get_bvec()
        perturb_contribution = jnp.einsum("gf,nf->ng", self.tf2gene_indicators, indic_times_param)
        A_mat = self.get_Amat()
        conc_contribution = jnp.einsum("gj,nj->ng", A_mat, x)

        # reco_loss = jnp.mean((conc_contribution + perturb_contribution)**2)
        # l1_prior = jnp.mean(jnp.abs(A_mat))
        # loss = reco_loss + self.lambda_prior * l1_prior

        N = x.shape[0]
        reco_loss = jnp.mean((conc_contribution + perturb_contribution) ** 2)
        lap_density = jnp.abs(A_mat) * self.lambda_prior
        l1_prior = jnp.sum(lap_density) / N
        loss = reco_loss + l1_prior
        return {"loss": loss, "reco_loss": reco_loss, "l1_prior": l1_prior}
