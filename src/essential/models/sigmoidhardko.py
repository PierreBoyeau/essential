import jax
import jax.numpy as jnp
from flax.linen.initializers import glorot_normal, normal, zeros, constant
import flax.linen as nn
import diffrax

from .base_model import BaseModel


class SigmoidHardKoModel(BaseModel):
    def setup(self):
        self.Amat_ = self.param("Amat_", glorot_normal(), (self.n_genes, self.n_genes))

        # Initialize decay_ such that softplus(decay_) is centered around 1.0.
        # The inverse of softplus(x) is log(exp(x) - 1). For x=1, this is ~0.54.
        # We initialize from a narrow normal distribution around this value.
        decay_init_mean = jnp.log(jnp.exp(1.0) - 1.0)
        self.decay_ = self.param("decay_", normal(), (self.n_genes))

        # Initialize sigmoid bias to a negative value to start in an "off" state.
        self.bias_term_sigmoid_ = self.param("bias_term_sigmoid_", constant(-5.0), (self.n_genes))
        self.solver = diffrax.Heun()
        self.saveat = diffrax.SaveAt(t1=True)
        self.adjoint = diffrax.DirectAdjoint()

    def get_Amat(self):
        return self.Amat_ * (1.0 - jnp.eye(self.n_genes))

    def get_decay(self):
        return nn.softplus(self.decay_)
        # return self.decay_

    def ode_fn(self, y, u):
        A_mat = self.get_Amat()
        conc_contribution = jnp.einsum("gj,j->g", A_mat, y) + self.bias_term_sigmoid_
        alpha = nn.sigmoid(conc_contribution)
        beta = self.get_decay() * y
        return alpha - beta

    def simulate(self, x0: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray):
        def solve_single(x_i, u_i, t_i):
            perturb_i = jnp.einsum("gf,f->g", self.tf2gene_indicators, u_i)
            mask_expressed = 1.0 - perturb_i
            x_i_ = x_i * mask_expressed

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
            mask_expressed = 1.0 - perturb_i
            xt_ = xt * mask_expressed
            dxdt = jax.vmap(self.ode_fn, in_axes=(0, 0))(xt_, u)
            reco_loss = jnp.mean(dxdt**2)

        l1_prior = jnp.mean(jnp.abs(A_mat))
        loss = reco_loss + self.lambda_prior * l1_prior
        return {"loss": loss, "reco_loss": reco_loss, "l1_prior": l1_prior}
