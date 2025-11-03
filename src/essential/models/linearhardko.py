import jax
import jax.numpy as jnp
from flax.linen.initializers import normal
import flax.linen as nn
import diffrax

from .base_model import BaseModel


class LinearHardKoModel(BaseModel):
    def setup(self):
        self.Amat_ = self.param("Amat_", normal(), (self.n_genes, self.n_genes))

        # Heun solver - empirically fastest for this problem
        self.solver = diffrax.Heun()
        self.saveat = diffrax.SaveAt(t1=True)
        self.adjoint = diffrax.DirectAdjoint()

    def get_Amat(self):
        return self.Amat_

    def ode_fn(self, y, u):
        A_mat = self.get_Amat()
        # u is not used in this model
        conc_contribution = jnp.einsum("gj,j->g", A_mat, y)
        return conc_contribution

    def simulate(self, x0: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray):
        def solve_single(x_i, u_i, t_i):
            perturb_i = jnp.einsum("gf,f->g", self.tf2gene_indicators, u_i)
            # hard setting the KO gene to 0
            x_i_ = x_i * (1.0 - perturb_i)

            def ode_fn_diffrax(t, y, args):
                return self.ode_fn(y, u_i)

            ode_term = diffrax.ODETerm(ode_fn_diffrax)
            sol = diffrax.diffeqsolve(
                ode_term,
                self.solver,
                t0=0.0,
                t1=jnp.squeeze(t_i),
                dt0=0.1,
                y0=x_i_,
                saveat=self.saveat,
                adjoint=self.adjoint,
            )
            return sol.ys[-1]

        return jax.vmap(solve_single, in_axes=(0, 0, 0))(x0, u, t)

    def __call__(self, x0: jnp.ndarray, xt: jnp.ndarray, t: jnp.ndarray, u: jnp.ndarray) -> dict:
        A_mat = self.get_Amat()
        if self.mode == "dynamic":
            xpred = self.simulate(x0, u, t)
            reco_loss = jnp.mean((xpred - xt) ** 2)
        else:
            perturb_i = jnp.einsum("gf,nf->ng", self.tf2gene_indicators, u)
            xt_ = xt * (1.0 - perturb_i)
            dxdt = jax.vmap(self.ode_fn, in_axes=(0, 0))(xt_, u)
            reco_loss = jnp.mean(dxdt**2)
        l1_prior = jnp.mean(jnp.abs(A_mat))
        loss = reco_loss + self.lambda_prior * l1_prior
        return {"loss": loss, "reco_loss": reco_loss, "l1_prior": l1_prior}
