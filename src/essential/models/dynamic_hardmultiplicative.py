import jax
import jax.numpy as jnp
from flax.linen.initializers import normal
import flax.linen as nn
import diffrax

from .base_model import BaseModel


class DynamicHardMultiplicativeModel(BaseModel):
    def setup(self):
        self.Amat_ = self.param("Amat_", normal(), (self.n_genes, self.n_genes))
        self.decay_ = self.param("decay_", normal(), (self.n_genes))

        self.solver = diffrax.Heun()
        self.saveat = diffrax.SaveAt(t1=True)
        self.adjoint = diffrax.DirectAdjoint()

    def get_Amat(self):
        Amat = self.Amat_ * (1.0 - jnp.eye(self.n_genes))
        return Amat

    def get_decay(self):
        return nn.softplus(self.decay_)

    def __call__(self, x0: jnp.ndarray, xt: jnp.ndarray, t: jnp.ndarray, u: jnp.ndarray) -> dict:
        A_mat = self.get_Amat()
        decay = self.get_decay()

        def solve_single(x_i, u_i):
            perturb_term = jnp.einsum("gf,f->g", self.tf2gene_indicators, u_i)

            def ode_fn(t, y, args):
                cross_terms = jnp.einsum("gj,j->g", A_mat, y)
                product_contribution = (1.0 - perturb_term) * cross_terms
                decay_contribution = decay * y
                return product_contribution - decay_contribution

            ode_term = diffrax.ODETerm(ode_fn)
            sol = diffrax.diffeqsolve(
                ode_term,
                self.solver,
                t0=0.0,
                t1=1.0,
                dt0=0.1,
                y0=x_i,
                saveat=self.saveat,
                adjoint=self.adjoint,
            )
            return sol.ys

        xpred = jax.vmap(solve_single, in_axes=(0, 0))(x0, u)
        reco_loss = jnp.mean((xpred - xt) ** 2)
        l1_prior = jnp.mean(jnp.abs(A_mat))
        loss = reco_loss + self.lambda_prior * l1_prior
        return {"loss": loss, "reco_loss": reco_loss, "l1_prior": l1_prior}
