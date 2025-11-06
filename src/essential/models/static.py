import jax
import jax.numpy as jnp
from flax.linen.initializers import normal
import flax.linen as nn
import diffrax

from .base_model import BaseModel


class StaticModel(BaseModel):
    def setup(self):
        self.means_ = self.param("means_", normal(), (self.n_genes))

    def get_means(self):
        return self.means_

    def __call__(self, x0: jnp.ndarray, xt: jnp.ndarray, t: jnp.ndarray, u: jnp.ndarray) -> dict:
        means = self.get_means()
        deltas = x0 - means
        reco_loss = self.get_reconstruction_loss(deltas)
        return {"loss": reco_loss, "reco_loss": reco_loss, "l1_prior": 0.0}
