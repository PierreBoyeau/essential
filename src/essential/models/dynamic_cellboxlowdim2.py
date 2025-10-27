import jax
import jax.numpy as jnp
from flax.linen.initializers import normal
import flax.linen as nn
import diffrax

from .base_model import BaseModel


class DynamicCellboxLowDimModel2(BaseModel):
    n_latent: int

    def setup(self):
        self.factors_ = self.param("factors_", normal(), (self.n_genes, self.n_latent))
        self.loadings_ = self.param("loadings_", normal(), (self.n_latent, self.n_genes))
        self.bvec_ = self.param("bvec_", normal(), (self.n_tfs))

        # Heun solver - empirically fastest for this problem
        self.solver = diffrax.Heun()
        self.saveat = diffrax.SaveAt(t1=True)
        self.adjoint = diffrax.DirectAdjoint()

    def get_Amat(self):
        return self.factors_ @ self.loadings_

    def get_bvec(self):
        return -nn.softplus(self.bvec_)

    def simulate(self, x0: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray):
        bvec = self.get_bvec()

        z0 = jnp.einsum("fg,ng->nf", self.loadings_, x0)

        def solve_single(z0_i, u_i, t_i):
            indic_times_param_i = u_i * bvec
            perturb_i = jnp.einsum("gf,f->g", self.tf2gene_indicators, indic_times_param_i)

            Amat_z = self.loadings_ @ self.factors_
            p_vec = self.loadings_ @ perturb_i

            def ode_fn(t, z, args):
                return Amat_z @ z + p_vec

            ode_term = diffrax.ODETerm(ode_fn)
            sol = diffrax.diffeqsolve(
                ode_term,
                self.solver,
                t0=0.0,
                t1=jnp.squeeze(t_i),
                dt0=0.1,
                y0=z0_i,
                saveat=self.saveat,
                adjoint=self.adjoint,
            )
            return sol.ys[-1]

        zt = jax.vmap(solve_single, in_axes=(0, 0, 0))(z0, u, t)
        x = jnp.einsum("gf,nf->ng", self.factors_, zt)
        return x

    def __call__(self, x0: jnp.ndarray, xt: jnp.ndarray, t: jnp.ndarray, u: jnp.ndarray) -> dict:
        A_mat = self.get_Amat()
        xpred = self.simulate(x0, u, t)
        reco_loss = jnp.mean((xpred - xt) ** 2)
        # l1_prior = jnp.mean(jnp.abs(A_mat))
        # loss = reco_loss + self.lambda_prior * l1_prior
        loss = reco_loss
        return {"loss": loss, "reco_loss": reco_loss, "l1_prior": 0.0}
