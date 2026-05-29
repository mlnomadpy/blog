"""The generic training loop: one jitted SGD step, shared by every loss.

The only thing that changes between losses is the scalar ``loss_fn`` and
whether we re-project onto the sphere after the update. Everything else — the
optimizer, the gradient, the JIT — is identical. That is the payoff of writing
each loss as a single differentiable scalar.
"""

from __future__ import annotations

import functools

import jax
import numpy as np
import optax

from data import accuracy
from losses import Cfg, l2normalize, make_masks


def make_step(loss_fn, lr: float, on_sphere: bool, masks):
    """Build a jitted ``(z, key, param) -> (z, loss)`` step for one loss."""
    opt = optax.sgd(lr)

    @jax.jit
    def step(z, opt_state, key, param):
        loss, grads = jax.value_and_grad(loss_fn)(z, None, key, param, m=masks)
        updates, opt_state = opt.update(grads, opt_state)
        z = optax.apply_updates(z, updates)
        if on_sphere:
            z = l2normalize(z)
        return z, opt_state, loss

    return step, opt


def run(cfg: Cfg, z0: np.ndarray, labels: np.ndarray, seed: int, stride: int):
    """Run the optimization, capturing a frame every ``stride`` steps.

    Returns a list of ``(z_snapshot, loss, accuracy)`` tuples — the raw
    material the renderer turns into GIF frames.
    """
    import jax.numpy as jnp

    masks = make_masks(labels)
    step, opt = make_step(cfg.fn, cfg.lr, cfg.on_sphere, masks)

    z = jnp.asarray(z0)
    if cfg.on_sphere:
        z = l2normalize(z)
    opt_state = opt.init(z)
    key = jax.random.key(seed)

    frames = []

    def snap(t, z, loss):
        zc = np.asarray(z)
        frames.append((zc, float(loss), accuracy(zc, labels)))

    snap(0, z, np.nan)
    for t in range(1, cfg.steps + 1):
        key, sub = jax.random.split(key)
        z, opt_state, loss = step(z, opt_state, sub, cfg.param)
        if t % stride == 0 or t == cfg.steps:
            snap(t, z, loss)

    return frames
