import flax.linen as nn
import jax.numpy as jnp


class BaseModel(nn.Module):
    """Base model for ODEs."""

    n_obs: int
    n_genes: int
    lambda_prior: float = 1.0
    mode: str = "dynamic"
    Amask: jnp.ndarray | None = None
    Aweight: jnp.ndarray | None = None
    n_latent: int | None = None

    def setup(self):
        pass

    def init_params(self, key):
        """
        Initialize model parameters using dummy data.
        """
        dummy_x0 = jnp.zeros((4, self.n_genes))
        dummy_x = jnp.zeros((4, self.n_genes))
        dummy_t = jnp.ones((4,))
        dummy_u = jnp.zeros((4, self.n_genes))
        params = self.init(key, dummy_x0, dummy_x, dummy_t, dummy_u)["params"]
        return params

    def get_laplace_prior(self, mat, lambda_prior):
        if self.Aweight is None:
            return lambda_prior * jnp.sum(jnp.abs(mat))
        else:
            return lambda_prior * jnp.sum(self.Aweight * jnp.abs(mat))

    @staticmethod
    def get_reconstruction_loss(deltas):
        return jnp.mean(jnp.sum(deltas**2, axis=1))
