import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import glorot_normal, zeros


class RegulatorNet(nn.Module):
    """DeepSets aggregation over a gene's regulators, edge-based.

    Works on the n_edges active edges from Amask rather than a padded
    (G, max_reg) tensor.  Intermediate tensors scale as O(n_edges) instead of
    O(G * max_reg), which matters when the mask is sparse (here: 6 070 edges
    vs 80 325 padded slots — 13× smaller, preventing OOM in the backward pass).

    For each active edge (src -> dst):
        feat_k  = [y_norm[src_k], embed[src_k]]       (1 + embed_dim,)
        h_k     = tanh(phi(feat_k))                    (hidden_dim,)
    Then per target gene i:
        pooled_i = sum_{k : dst_k == i} h_k            (hidden_dim,)
        out_i    = rho(pooled_i)                        scalar
    Genes with no regulators get pooled = 0, so out_i = rho(0) (a learned bias).
    """

    Amask: jnp.ndarray  # (G, G)
    embed_dim: int = 16
    hidden_dim: int = 16

    def setup(self):
        if self.Amask is None:
            raise ValueError("RegulatorNet requires Amask")

        amask_np = np.array(self.Amask)
        G = amask_np.shape[0]
        # Amask[i, j] = 1  =>  gene j regulates gene i
        rows, cols = np.where(amask_np > 0)  # rows = targets, cols = sources
        self._src = np.array(cols, dtype=np.int32)  # (n_edges,) — numpy = compile-time constant
        self._dst = np.array(rows, dtype=np.int32)  # (n_edges,)
        self._G = G

        self.embed = self.param("embed", glorot_normal(), (G, self.embed_dim))
        self.phi = nn.Dense(self.hidden_dim, kernel_init=glorot_normal(), bias_init=zeros)
        self.rho = nn.Dense(1, kernel_init=glorot_normal(), bias_init=zeros)

    def __call__(self, y_norm: jnp.ndarray) -> jnp.ndarray:
        """y_norm: (G,) normalized expression -> (G,) preactivation contribution."""
        src_expr = y_norm[self._src]  # (n_edges,)
        src_emb = self.embed[self._src]  # (n_edges, embed_dim)
        feat = jnp.concatenate([src_expr[:, None], src_emb], axis=-1)  # (n_edges, 1+embed_dim)
        h = jnp.tanh(self.phi(feat))  # (n_edges, hidden_dim)
        pooled = jax.ops.segment_sum(h, self._dst, self._G)  # (G, hidden_dim)
        return self.rho(pooled).squeeze(-1)  # (G,)
