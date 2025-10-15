import flax.linen as nn
import jax.numpy as jnp


class BaseModel(nn.Module):
    """Base model for ODEs."""

    n_genes: int
    n_tfs: int
    tf2gene_indicators: jnp.ndarray
    lambda_prior: float
