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
        reco_loss = jnp.mean((x0 - means) ** 2)
        return {"loss": reco_loss, "reco_loss": reco_loss}
