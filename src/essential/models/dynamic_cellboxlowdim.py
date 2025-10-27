import jax
import jax.numpy as jnp
from flax.linen.initializers import normal
import flax.linen as nn
import diffrax

from .base_model import BaseModel


class DynamicCellboxLowDimModel(BaseModel):
    def setup(self):
        self.factors_ = self.param("factors_", normal(), (self.n_genes, 256))
        self.loadings_ = self.param("loadings_", normal(), (256, self.n_genes))
        self.bvec_ = self.param("bvec_", normal(), (self.n_tfs))

        self.solver = diffrax.Heun()
        self.saveat = diffrax.SaveAt(t1=True)
        self.adjoint = diffrax.DirectAdjoint()

    def get_Amat(self):
        return self.factors_ @ self.loadings_

    def get_bvec(self):
        return -nn.softplus(self.bvec_)

    def simulate(self, x0: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray):
        bvec = self.get_bvec()

        def solve_single(x_i, u_i, t_i):
            indic_times_param_i = u_i * bvec
            perturb_i = jnp.einsum("gf,f->g", self.tf2gene_indicators, indic_times_param_i)

            def ode_fn(t, y, args):
                conc_contribution = jnp.einsum("fj,j->f", self.loadings_, y)
                conc_contribution = jnp.einsum("gf,f->g", self.factors_, conc_contribution)
                return conc_contribution + perturb_i

            ode_term = diffrax.ODETerm(ode_fn)
            sol = diffrax.diffeqsolve(
                ode_term,
                self.solver,
                t0=0.0,
                t1=jnp.squeeze(t_i),
                dt0=0.1,
                y0=x_i,
                saveat=self.saveat,
                adjoint=self.adjoint,
            )
            return sol.ys[-1]

        return jax.vmap(solve_single, in_axes=(0, 0, 0))(x0, u, t)

    def __call__(self, x0: jnp.ndarray, xt: jnp.ndarray, t: jnp.ndarray, u: jnp.ndarray) -> dict:
        A_mat = self.get_Amat()
        xpred = self.simulate(x0, u, t)
        reco_loss = jnp.mean((xpred - xt) ** 2)
        # l1_prior = jnp.mean(jnp.abs(A_mat))
        # loss = reco_loss + self.lambda_prior * l1_prior
        loss = reco_loss
        return {"loss": loss, "reco_loss": reco_loss, "l1_prior": 0.0}
