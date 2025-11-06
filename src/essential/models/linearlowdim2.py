import jax
import jax.numpy as jnp
from flax.linen.initializers import normal
import flax.linen as nn
import diffrax

from .base_model import BaseModel


class LinearLowDim2Model(BaseModel):
    def setup(self):
        assert self.n_latent is not None, "n_latent must be specified for LinearLowDim2Model"
        self.factors_ = self.param("factors_", normal(), (self.n_genes, self.n_latent))
        self.loadings_ = self.param("loadings_", normal(), (self.n_latent, self.n_genes))
        self.bvec_ = self.param("bvec_", normal(), (self.n_tfs))

        # Heun solver - empirically fastest for this problem
        self.solver = diffrax.Heun()
        self.saveat = diffrax.SaveAt(t1=True)
        self.adjoint = diffrax.DirectAdjoint()

    def get_Amat(self):
        Amat = self.factors_ @ self.loadings_
        if self.Amask is not None:
            Amat = Amat * self.Amask
        return Amat

    def get_bvec(self):
        return -nn.softplus(self.bvec_)

    def ode_fn(self, y, u):
        bvec = self.get_bvec()
        z = jnp.einsum("fg,g->f", self.loadings_, y)
        indic_times_param_i = u * bvec
        perturb_i = jnp.einsum("gf,f->g", self.tf2gene_indicators, indic_times_param_i)
        Amat_z = self.loadings_ @ self.factors_
        p_vec = self.loadings_ @ perturb_i
        dzdt = Amat_z @ z + p_vec
        dxdt = jnp.einsum("gf,f->g", self.factors_, dzdt)
        return dxdt

    def simulate(self, x0: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray):
        z0 = jnp.einsum("fg,ng->nf", self.loadings_, x0)

        def solve_single(z0_i, u_i, t_i):
            def ode_fn_diffrax(t, z, args):
                bvec = self.get_bvec()
                indic_times_param_i = u_i * bvec
                perturb_i = jnp.einsum("gf,f->g", self.tf2gene_indicators, indic_times_param_i)
                Amat_z = self.loadings_ @ self.factors_
                p_vec = self.loadings_ @ perturb_i
                return Amat_z @ z + p_vec

            ode_term = diffrax.ODETerm(ode_fn_diffrax)
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
        if self.mode == "dynamic":
            xpred = self.simulate(x0, u, t)
            deltas = xpred - xt
        else:
            dxdt = jax.vmap(self.ode_fn, in_axes=(0, 0))(xt, u)
            deltas = dxdt

        reco_loss = self.get_reconstruction_loss(deltas)
        l1_prior_factors = self.get_laplace_prior(self.factors_, self.lambda_prior) / self.n_obs
        l1_prior_loadings = self.get_laplace_prior(self.loadings_, self.lambda_prior) / self.n_obs
        l1_prior = l1_prior_factors + l1_prior_loadings
        loss = reco_loss + l1_prior
        return {"loss": loss, "reco_loss": reco_loss, "l1_prior": l1_prior}
