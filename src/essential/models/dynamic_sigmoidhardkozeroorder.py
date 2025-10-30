import jax
import jax.numpy as jnp
from flax.linen.initializers import normal
import flax.linen as nn
import diffrax

from .base_model import BaseModel


class DynamicSigmoidHardKoZeroOrderModel(BaseModel):
    def setup(self):
        self.Amat_ = self.param("Amat_", normal(), (self.n_genes, self.n_genes))
        self.decay_ = self.param("decay_", normal(), (self.n_genes))
        self.bias_term_sigmoid_ = self.param("bias_term_sigmoid_", normal(), (self.n_genes))
        self.basal_term_ = self.param("basal_term_", normal(), (self.n_genes))
        self.solver = diffrax.Heun()
        self.saveat = diffrax.SaveAt(t1=True)
        self.adjoint = diffrax.DirectAdjoint()

    def get_Amat(self):
        return self.Amat_ * (1.0 - jnp.eye(self.n_genes))

    def simulate(self, x0: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray):
        A_mat = self.get_Amat()

        def solve_single(x_i, u_i, t_i):
            perturb_i = jnp.einsum("gf,f->g", self.tf2gene_indicators, u_i)
            mask_expressed = 1.0 - perturb_i
            x_i_ = x_i * mask_expressed

            def ode_fn(t, y, args):
                conc_contribution = jnp.einsum("gj,j->g", A_mat, y) + self.bias_term_sigmoid_
                alpha = nn.sigmoid(conc_contribution) + self.basal_term_
                beta = self.decay_ * y
                return alpha - beta

            ode_term = diffrax.ODETerm(ode_fn)
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
        xpred = self.simulate(x0, u, t)
        reco_loss = jnp.mean((xpred - xt) ** 2)
        l1_prior = jnp.mean(jnp.abs(A_mat))
        loss = reco_loss + self.lambda_prior * l1_prior
        return {"loss": loss, "reco_loss": reco_loss, "l1_prior": l1_prior}
