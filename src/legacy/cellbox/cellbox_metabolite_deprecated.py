import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import zeros

from .cellbox_steady_state import CellBoxSteadyState


class CellBoxMetabolite(CellBoxSteadyState):
    """CellBox steady state with metabolite-gated regulation.

    Forward map (per gene i): x_i = ε_i · σ(Σ_j A_ij · g_j(M) · x̃_j − p_i u_i + b_i)
    where ``M`` is the condition's metabolite LFC vector and ``g(M) = 1 + h(M)``
    is a per-regulator gate produced by a linear map ``h`` of ``M``.

    ``h`` is zero-initialized, so at init ``g ≡ 1`` and the model reduces exactly
    to :class:`CellBoxSteadyState`. The gate depends only on the regulator ``j``
    (not the edge), acting elementwise on the regulator state before ``A``.
    """

    n_metabolites: int = 0

    def setup(self):
        super().setup()
        # h(M): linear map metabolites -> per-regulator gate offset, zero-init
        # so g = 1 + h(M) starts at the identity gate (= plain CellBox).
        # self.gate_net = nn.Dense(self.n_genes, kernel_init=zeros, bias_init=zeros, use_bias=False)

        gate_rank = 10
        self.gate_factor = nn.Dense(gate_rank, use_bias=False)
        self.gate_net = nn.Dense(self.n_genes, kernel_init=zeros, bias_init=zeros, use_bias=False)

    def init_params(self, key):
        dummy_xt = jnp.zeros((4, self.n_genes))
        dummy_u = jnp.zeros((4, self.n_genes))
        dummy_m = jnp.zeros((4, self.n_metabolites))
        return self.init(key, dummy_xt, dummy_u, dummy_m)["params"]

    def gate(self, m):
        """Pe r-regulator multiplicative gate g(M) = 1 + h(M), shape (n_genes,)."""
        return 1.0 + self.gate_net(self.gate_factor(m))

    def preactivation(self, y, u, m):
        A = self.get_Amat()
        return jnp.dot(A, self.gate(m) * self.normalize_predictor(y)) - self.get_p() * u + self.b_

    def predict(self, y, u, m):
        return self.get_epsilon() * nn.sigmoid(self.preactivation(y, u, m))

    def predict_steady_state(self, y, u, m, n_steps=100):
        return jax.lax.fori_loop(0, n_steps, lambda _, val: self.predict(val, u, m), y)

    def __call__(self, xt: jnp.ndarray, u: jnp.ndarray, m: jnp.ndarray) -> dict:
        args = jax.vmap(lambda y, u_i, m_i: self.preactivation(y, u_i, m_i))(xt, u, m)
        x_pred = self.get_epsilon() * nn.sigmoid(args)

        mask = 1.0 - u
        deltas = (x_pred - xt) * mask
        reco_loss = jnp.mean(jnp.sum(deltas**2, axis=1))

        frac_saturated = jnp.sum((jnp.abs(args) > 4.0) * mask) / jnp.sum(mask)
        # gate_kernel = self.variables["params"]["gate_net"]["kernel"]
        # gate_l1 = jnp.mean(jnp.abs(gate_kernel))
        # Effective (n_metabolites, n_genes) kernel is mat1 @ mat2.
        mat1 = self.variables["params"]["gate_factor"]["kernel"]
        mat2 = self.variables["params"]["gate_net"]["kernel"]
        gate_kernel = mat1 @ mat2
        gate_l1 = jnp.mean(jnp.abs(gate_kernel))

        # loss = reco_loss + 1e-1 * gate_l1
        loss = reco_loss
        return {
            "loss": loss,
            "reco_loss": reco_loss,
            "frac_saturated": frac_saturated,
            "gate_l1": gate_l1,
        }
