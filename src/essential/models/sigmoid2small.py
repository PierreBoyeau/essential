import jax
import jax.numpy as jnp
from flax.linen.initializers import glorot_normal, normal, zeros, constant, ones
import flax.linen as nn
import diffrax

from .base_model import BaseModel


class Sigmoid2SmallModel(BaseModel):
    def setup(self):
        self.Amat_ = self.param("Amat_", glorot_normal(), (self.n_genes, self.n_tfs))
        self.decay_ = self.param("decay_", zeros, (self.n_genes))
        self.bias_term_sigmoid_ = self.param("bias_term_sigmoid_", zeros, (self.n_genes))
        self.perturbation_effect_ = self.param("perturbation_effect_", zeros, (self.n_tfs))
        self.scale_factor_ = self.param("scale_factor_", ones, (self.n_genes))
        self.solver = diffrax.Heun()
        self.saveat = diffrax.SaveAt(t1=True)
        self.adjoint = diffrax.DirectAdjoint()

    def get_Amat(self):
        return self.Amat_

    def get_decay(self):
        return nn.softplus(self.decay_)

    def get_perturbation_effect(self):
        return self.perturbation_effect_

    def ode_fn(self, y, u):
        A_mat = self.get_Amat()
        u_gene = jnp.einsum("gf,f->g", self.tf2gene_indicators, u * self.get_perturbation_effect())
        y_tf = jnp.einsum("gf,g->f", self.tf2gene_indicators, y)
        conc_contribution = jnp.einsum("gf,f->g", A_mat, y_tf) + self.bias_term_sigmoid_ - u_gene
        alpha = nn.sigmoid(conc_contribution)
        beta = self.get_decay() * y
        return alpha - beta

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
