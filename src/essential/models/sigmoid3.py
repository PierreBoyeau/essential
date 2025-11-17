import jax
import jax.numpy as jnp
from flax.linen.initializers import glorot_normal, normal, zeros, constant, ones
import flax.linen as nn
import diffrax

from .base_model import BaseModel


class Sigmoid3Model(BaseModel):
    def setup(self):
        self.Amat_ = self.param("Amat_", glorot_normal(), (self.n_genes, self.n_genes))
        self.decay_ = self.param("decay_", ones, (self.n_genes))
        self.perturbation_effect_ = self.param("perturbation_effect_", zeros, (self.n_tfs))
        self.scale_factor_ = self.param("scale_factor_", ones, (self.n_genes))
        self.solver = diffrax.Heun()
        self.saveat = diffrax.SaveAt(t1=True)
        self.adjoint = diffrax.DirectAdjoint()

    def get_Amat(self):
        Amat = self.Amat_ * (1.0 - jnp.eye(self.n_genes))
        if self.Amask is not None:
            Amat = Amat * self.Amask
        return Amat

    def get_decay(self):
        return nn.softplus(self.decay_)

    def get_perturbation_effect(self):
        return self.perturbation_effect_

    def get_scale_factor(self):
        return nn.softplus(self.scale_factor_)

    def ode_fn(self, y, u):
        A_mat = self.get_Amat()
        where_gene_perturbed = self.tf2gene_indicators.sum(axis=1).flatten()
        A_mat_ = A_mat * (1.0 - where_gene_perturbed[:, None])
        u_gene = jnp.einsum("gf,f->g", self.tf2gene_indicators, u * self.get_perturbation_effect())
        conc_contribution = jnp.einsum("gj,j->g", A_mat_, y) - u_gene
        alpha = self.get_scale_factor() * nn.sigmoid(conc_contribution)
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
