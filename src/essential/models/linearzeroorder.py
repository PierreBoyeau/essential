import jax
import jax.numpy as jnp
from flax.linen.initializers import normal
import flax.linen as nn
import diffrax

from .base_model import BaseModel


class LinearZeroOrderModel(BaseModel):
    def setup(self):
        self.Amat_ = self.param("Amat_", normal(), (self.n_genes, self.n_genes))
        self.bvec_ = self.param("bvec_", normal(), (self.n_tfs))
        self.bias_term_ = self.param("bias_term_", normal(), (self.n_genes))

        # Heun solver - empirically fastest for this problem
        self.solver = diffrax.Heun()
        self.saveat = diffrax.SaveAt(t1=True)
        self.adjoint = diffrax.DirectAdjoint()

    def get_Amat(self):
        Amat = self.Amat_
        if self.Amask is not None:
            Amat = Amat * self.Amask
        return Amat

    def get_bvec(self):
        return -nn.softplus(self.bvec_)

    def ode_fn(self, y, u):
        A_mat = self.get_Amat()
        bvec = self.get_bvec()

        indic_times_param_i = u * bvec
        constant_term = jnp.einsum("gf,f->g", self.tf2gene_indicators, indic_times_param_i)
        constant_term = constant_term + self.bias_term_

        conc_contribution = jnp.einsum("gj,j->g", A_mat, y)
        return conc_contribution + constant_term

    def simulate(self, x0: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray):
        def solve_single(x_i, u_i, t_i):
            def ode_fn_diffrax(t, y, args):
                return self.ode_fn(y, u_i)

            ode_term = diffrax.ODETerm(ode_fn_diffrax)
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
        if self.mode == "dynamic":
            xpred = self.simulate(x0, u, t)
            deltas = xpred - xt
        else:
            dxdt = jax.vmap(self.ode_fn, in_axes=(0, 0))(xt, u)
            deltas = dxdt

        reco_loss = self.get_reconstruction_loss(deltas)
        l1_prior = self.get_laplace_prior(A_mat, self.lambda_prior) / self.n_obs
        loss = reco_loss + l1_prior
        return {"loss": loss, "reco_loss": reco_loss, "l1_prior": l1_prior}
