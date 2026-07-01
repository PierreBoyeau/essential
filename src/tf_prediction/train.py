"""Minibatch Adam training on mean NB-NLL.

``fit`` works with any nn.Module whose ``__call__(x_tf, lib)`` returns
``(mean, concentration)`` — i.e. all models in this package.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from tqdm import tqdm

from .models import nb_nll


def _make_step(model):
    @jax.jit
    def step(state, x_tf, y_raw, lib):
        def loss_fn(params):
            mean, conc = model.apply({"params": params}, x_tf, lib)
            return nb_nll(mean, conc, y_raw).mean()

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss

    return step


def fit(
    model,
    X_tf,
    Y_raw,
    lib,
    *,
    n_epochs: int,
    batch_size: int,
    lr: float,
    key=None,
    tqdm_: bool = True,
):
    """Train ``model`` by minibatch Adam on mean NB NLL.

    Both parameter init and per-epoch shuffling are derived from a single
    ``key`` (default ``PRNGKey(0)``); pass a different key to vary either.

    Returns ``(final_state, history)`` where
    ``history = {'train_nll': list[float]}`` (one entry per epoch).
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    init_key, shuffle_key = jax.random.split(key, 2)

    init_x = jnp.asarray(X_tf[:4])
    init_lib = jnp.asarray(lib[:4])
    params = model.init(init_key, init_x, init_lib)["params"]
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(lr),
    )
    step = _make_step(model)

    shuffle_seed = int(shuffle_key[0]) & 0x7FFFFFFF
    rng = np.random.default_rng(shuffle_seed)
    history: list[float] = []
    n = X_tf.shape[0]
    iterator = tqdm(range(n_epochs)) if tqdm_ else range(n_epochs)
    for _ in iterator:
        idx = rng.permutation(n)
        losses = []
        for s in range(0, n - batch_size + 1, batch_size):
            b = idx[s : s + batch_size]
            state, loss = step(
                state,
                jnp.asarray(X_tf[b]),
                jnp.asarray(Y_raw[b]),
                jnp.asarray(lib[b]),
            )
            losses.append(float(loss))
        history.append(float(np.mean(losses)))

    return state, {"train_nll": history}
