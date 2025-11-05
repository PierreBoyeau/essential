import jax
import jax.numpy as jnp
from flax.linen.initializers import glorot_normal, normal, zeros, constant, ones
import flax.linen as nn
import diffrax

from .base_model import BaseModel


class Sigmoid2Model(BaseModel):
    def setup(self):
        self.Amat_ = self.param("Amat_", glorot_normal(), (self.n_genes, self.n_genes))
        self.decay_ = self.param("decay_", zeros, (self.n_genes))
        self.bias_term_sigmoid_ = self.param("bias_term_sigmoid_", zeros, (self.n_genes))
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
        # return nn.softplus(self.perturbation_effect_)
        return self.perturbation_effect_

    def ode_fn(self, y, u):
        A_mat = self.get_Amat()
        # u_effect = jnp.einsum("gf,f->g", self.tf2gene_indicators, u)
        # y_effective = y * (1.0 - u_effect)
        # y_effective = y
        # conc_contribution = jnp.einsum("gj,j->g", A_mat, y_effective) + self.bias_term_sigmoid_

        u_gene = jnp.einsum("gf,f->g", self.tf2gene_indicators, u * self.get_perturbation_effect())
        conc_contribution = jnp.einsum("gj,j->g", A_mat, y) + self.bias_term_sigmoid_ - u_gene
        # conc_contribution = jnp.einsum("gj,j->g", A_mat, y) - u_gene
        # alpha = self.scale_factor_ * nn.sigmoid(conc_contribution)
        alpha = nn.sigmoid(conc_contribution)
        beta = self.get_decay() * y
        # jax.debug.print(
        #     "alpha max={x}, min={y}, avg={avg}", x=alpha.max(), y=alpha.min(), avg=alpha.mean()
        # )
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
            reco_loss = jnp.mean((xpred - xt) ** 2)
        else:
            dxdt = jax.vmap(self.ode_fn, in_axes=(0, 0))(xt, u)
            reco_loss = jnp.mean(dxdt**2)

        l1_prior = jnp.mean(jnp.abs(A_mat))
        loss = reco_loss + self.lambda_prior * l1_prior
        return {"loss": loss, "reco_loss": reco_loss, "l1_prior": l1_prior}
