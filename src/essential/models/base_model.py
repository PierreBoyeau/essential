import flax.linen as nn
import jax.numpy as jnp


class BaseModel(nn.Module):
    """Base model for ODEs."""

    n_genes: int
    n_tfs: int
    tf2gene_indicators: jnp.ndarray
    lambda_prior: float

    def init_params(self, key):
        """
        Initialize model parameters using dummy data.
        """
        dummy_x0 = jnp.zeros((4, self.n_genes))
        dummy_x = jnp.zeros((4, self.n_genes))
        dummy_t = jnp.ones((4,))
        dummy_u = jnp.zeros((4, self.n_tfs))
        params = self.init(key, dummy_x0, dummy_x, dummy_t, dummy_u)["params"]
        return params
