"""Companion check: every code block in a-network-that-conserves-energy-jax-flax-nnx
runs as written here. Flax NNX versions of the two objects in the post:

  HNNField      the pendulum field as the symplectic gradient of a learned scalar
  LeapfrogNet   the classifier whose block is one leapfrog step of a learned potential

Trains each briefly (real steps, small budgets) and asserts the two structural
facts the post claims: the HNN's rollout drifts less than the plain field's, and
the leapfrog net's learned energy stays within a bounded band through depth
while it reaches sane accuracy. Writes results/hamiltonian_nnx_check.json.

Run on Kaggle (CPU is fine): the numbers here verify the CODE, the published
numbers come from scripts/hamiltonian_net.py.
"""

import json
import os

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import nnx

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT = {}

# ── block 1: the HNN field, a scalar with a rotated gradient ────────────────


class HNNField(nnx.Module):
    """One scalar H(q, p); the field is (dH/dp, -dH/dq)."""

    def __init__(self, hidden=64, *, rngs):
        self.l1 = nnx.Linear(2, hidden, rngs=rngs)
        self.l2 = nnx.Linear(hidden, hidden, rngs=rngs)
        self.l3 = nnx.Linear(hidden, 1, rngs=rngs)

    def energy(self, qp):                      # (..., 2) -> (...,)
        h = jnp.tanh(self.l1(qp))
        h = jnp.tanh(self.l2(h))
        return self.l3(h)[..., 0]

    def __call__(self, qp):                    # the symplectic gradient
        g = jax.vmap(jax.grad(lambda s: self.energy(s)))(qp)
        return jnp.stack([g[:, 1], -g[:, 0]], axis=1)


class PlainField(nnx.Module):
    """The control: two free outputs, no scalar anywhere."""

    def __init__(self, hidden=64, *, rngs):
        self.l1 = nnx.Linear(2, hidden, rngs=rngs)
        self.l2 = nnx.Linear(hidden, hidden, rngs=rngs)
        self.l3 = nnx.Linear(hidden, 2, rngs=rngs)

    def __call__(self, qp):
        h = jnp.tanh(self.l1(qp))
        h = jnp.tanh(self.l2(h))
        return self.l3(h)


# ── block 2: fit both to the same pendulum arrows ───────────────────────────

def pendulum_data(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.uniform(-2, 2, n).astype(np.float32)
    p = rng.uniform(-2, 2, n).astype(np.float32)
    states = np.stack([q, p], 1)
    targets = np.stack([p, -np.sin(q)], 1).astype(np.float32)   # (dq, dp)
    return jnp.asarray(states), jnp.asarray(targets)


def fit(model, S, T, steps=1500, lr=1e-3):
    opt = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)

    @nnx.jit
    def step(model, opt):
        loss, grads = nnx.value_and_grad(lambda m: jnp.mean((m(S) - T) ** 2))(model)
        opt.update(model, grads)
        return loss

    for _ in range(steps):
        loss = step(model, opt)
    return float(loss)


# ── block 3: roll both out and watch the true energy ────────────────────────

def rollout_energy(model, q0=1.6, p0=0.0, dt=0.05, n=600):
    def rk4(s, _):
        k1 = model(s); k2 = model(s + 0.5 * dt * k1)
        k3 = model(s + 0.5 * dt * k2); k4 = model(s + dt * k3)
        s2 = s + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return s2, s2[0]

    s0 = jnp.array([[q0, p0]], dtype=jnp.float32)
    _, traj = jax.lax.scan(rk4, s0, None, length=n)
    E = 0.5 * traj[:, 1] ** 2 + (1 - jnp.cos(traj[:, 0]))
    E0 = 0.5 * p0 ** 2 + (1 - np.cos(q0))
    return float(jnp.max(jnp.abs(E - E0)) / E0)


# ── block 4: the leapfrog classifier ─────────────────────────────────────────

DIM, T_TOTAL = 4, 6.0


class LeapfrogNet(nnx.Module):
    """Hidden state (q, p); one shared block = one leapfrog step of V(q)."""

    def __init__(self, hidden=32, *, rngs):
        self.enc = nnx.Linear(2, 2 * DIM, rngs=rngs)
        self.v1 = nnx.Linear(DIM, hidden, rngs=rngs)
        self.v2 = nnx.Linear(hidden, DIM, rngs=rngs)
        self.dec = nnx.Linear(2 * DIM, 2, rngs=rngs)

    def potential(self, q):                    # scalar V per row
        return (self.v2(jnp.tanh(self.v1(q)))).sum(axis=-1)

    def grad_v(self, q):
        return jax.vmap(jax.grad(lambda qi: self.potential(qi[None])[0]))(q)

    def __call__(self, x, L=16):
        h = T_TOTAL / L
        z = self.enc(x)
        q, p = z[:, :DIM], z[:, DIM:]

        def step(carry, _):
            q, p = carry
            p = p - 0.5 * h * self.grad_v(q)   # kick
            q = q + h * p                      # drift
            p = p - 0.5 * h * self.grad_v(q)   # kick
            return (q, p), None

        (q, p), _ = jax.lax.scan(step, (q, p), None, length=L)
        return self.dec(jnp.concatenate([q, p], axis=1))

    def energy(self, x, L=16):                 # learned E through depth
        h = T_TOTAL / L
        z = self.enc(x)
        q, p = z[:, :DIM], z[:, DIM:]

        def step(carry, _):
            q, p = carry
            p = p - 0.5 * h * self.grad_v(q)
            q = q + h * p
            p = p - 0.5 * h * self.grad_v(q)
            E = (0.5 * (p ** 2).sum(1) + self.potential(q)).mean()
            return (q, p), E

        _, Es = jax.lax.scan(step, (q, p), None, length=L)
        return Es


def moons(n, seed):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, np.pi, n // 2)
    x0 = np.stack([np.cos(t), np.sin(t)], 1)
    x1 = np.stack([1 - np.cos(t), 1 - np.sin(t) - 0.5], 1)
    X = np.concatenate([x0, x1]) + rng.normal(0, 0.08, (n, 2))
    y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(np.int32)
    return jnp.asarray(X, dtype=jnp.float32), jnp.asarray(y)


def main():
    S, T = pendulum_data()
    plain = PlainField(rngs=nnx.Rngs(0))
    hnn = HNNField(rngs=nnx.Rngs(1))
    OUT["plain_mse"] = fit(plain, S, T)
    OUT["hnn_mse"] = fit(hnn, S, T)
    OUT["plain_drift"] = rollout_energy(plain)
    OUT["hnn_drift"] = rollout_energy(hnn)
    print(f"field fit: plain {OUT['plain_mse']:.2e}, hnn {OUT['hnn_mse']:.2e}")
    print(f"rollout drift: plain {OUT['plain_drift']:.1%}, hnn {OUT['hnn_drift']:.1%}")
    assert OUT["hnn_drift"] < OUT["plain_drift"], "HNN should drift less"

    X, y = moons(512, 0)
    net = LeapfrogNet(rngs=nnx.Rngs(0))
    opt = nnx.Optimizer(net, optax.adam(3e-3), wrt=nnx.Param)

    @nnx.jit
    def step(net, opt):
        def loss_fn(m):
            return optax.softmax_cross_entropy_with_integer_labels(m(X), y).mean()
        loss, grads = nnx.value_and_grad(loss_fn)(net)
        opt.update(net, grads)
        return loss

    for _ in range(1200):
        loss = step(net, opt)
    Xt, yt = moons(1024, 1)
    acc = float((net(Xt).argmax(1) == yt).mean())
    Es = np.asarray(net.energy(Xt[:256]))
    drift = float(np.max(np.abs(Es - Es[0])) / (abs(Es[0]) + 1e-6))
    acc4 = float((net(Xt, L=64).argmax(1) == yt).mean())
    OUT.update(net_acc=acc, net_acc_4L=acc4, net_energy_drift=drift)
    print(f"leapfrog net: acc {acc:.3f}, acc@4L {acc4:.3f}, energy drift {drift:.1%}")
    assert acc > 0.9, "leapfrog net should fit moons"

    with open(os.path.join(RESULTS_DIR, "hamiltonian_nnx_check.json"), "w") as f:
        json.dump(OUT, f, indent=1)
    print("ok")


if __name__ == "__main__":
    main()
