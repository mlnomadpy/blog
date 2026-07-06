"""momentum_nnx_check.py -- runs the exact code blocks of the JAX companion post
(momentum-resnet-jax-flax-nnx.mdx) so every printed number in the post is from a
real run. Flax NNX version of scripts/momentum_resnet.py's Part B, plus the
Kepler check and the exact rewind in float32 and float64.

Run on Kaggle: python3 kgl.py --entry momentum_nnx_check.py --slug blog-momentum-nnx \
    --expdir <repo>/scripts --pip "flax optax"
"""
import json
import math
import os

import jax
jax.config.update("jax_enable_x64", True)   # only the rewind uses it; training casts to f32
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

OUT = os.path.join(os.getcwd(), "results")
os.makedirs(OUT, exist_ok=True)
report = {}

# ── the Kepler check (post section 1) ────────────────────────────────────────
r0, v0 = np.array([1.0, 0.0]), np.array([0.0, 0.9])
E0 = 0.5 * 0.9**2 - 1.0                       # -0.595, an ellipse
period = 2 * np.pi * (-1 / (2 * E0)) ** 1.5

def accel(r):
    return -r / np.linalg.norm(r) ** 3

def euler(r, v, dt):
    a = accel(r)
    return r + dt * v, v + dt * a             # both updates from the stale state

def leapfrog(r, v, dt):
    v = v + 0.5 * dt * accel(r)                # half kick
    r = r + dt * v                             # drift with the FRESH velocity
    return r, v + 0.5 * dt * accel(r)          # half kick

dt, n = 0.02, int(round(20 * period / 0.02))
re_, ve_, rl_, vl_ = r0, v0, r0, v0
apo = 0.0; dev = 0.0
for _ in range(n):
    re_, ve_ = euler(re_, ve_, dt)
    rl_, vl_ = leapfrog(rl_, vl_, dt)
    apo = max(apo, np.linalg.norm(re_))
    dev = max(dev, abs(0.5 * vl_ @ vl_ - 1 / np.linalg.norm(rl_) - E0))
drift = (0.5 * ve_ @ ve_ - 1 / np.linalg.norm(re_) - E0) / abs(E0)
print(f"Euler:    energy drift {drift:+.1%} over 20 orbits, farthest reach {apo:.2f}")
print(f"leapfrog: max energy deviation {dev/abs(E0):.3%}")
report["kepler"] = dict(euler_drift=float(drift), euler_apo=float(apo),
                        leapfrog_dev=float(dev / abs(E0)))

# ── the data (post section 3) ────────────────────────────────────────────────
def make_rings(n, rng, noise=0.04):
    """Class 0 is a disk; class 1 is an annulus that walls it in at every angle."""
    n2 = n // 2
    th, r_in = rng.uniform(0, 2 * np.pi, n2), 0.6 * np.sqrt(rng.uniform(0, 1, n2))
    th2, r_out = rng.uniform(0, 2 * np.pi, n2), rng.uniform(1.3, 1.8, n2)
    X = np.concatenate([np.stack([r_in * np.cos(th), r_in * np.sin(th)], 1),
                        np.stack([r_out * np.cos(th2), r_out * np.sin(th2)], 1)])
    X += rng.normal(0, noise, X.shape)
    y = np.repeat(np.arange(2), n2).astype(np.int32)
    return X.astype(np.float32), y

# ── the architecture (post section 2) ────────────────────────────────────────
class MomentumResNet(nnx.Module):
    """v' = mu v + (1 - mu) F_l(x);  x' = x + h v'.  At mu = 0 this IS the
    plain residual network x' = x + h F_l(x): one dial from Euler to Newton."""

    def __init__(self, L=32, width=2, hidden=16, mu=0.9, T=8.0, *, rngs):
        gain1 = nnx.initializers.normal(1.0 / math.sqrt(width))
        gain2 = nnx.initializers.normal(1.0 / math.sqrt(hidden))
        self.blocks = [(nnx.Linear(width, hidden, kernel_init=gain1, rngs=rngs),
                        nnx.Linear(hidden, width, kernel_init=gain2, rngs=rngs))
                       for _ in range(L)]
        self.readout = nnx.Linear(width, 2, kernel_init=gain1, rngs=rngs)
        self.mu, self.h = mu, T / L

    def __call__(self, x, return_state=False):
        v = jnp.zeros_like(x)
        for lin1, lin2 in self.blocks:
            f = lin2(jnp.tanh(lin1(x)))            # the vector field F_l(x)
            v = self.mu * v + (1 - self.mu) * f    # the block writes to the LEDGER
            x = x + self.h * v                     # the ledger moves the state
        return (self.readout(x), x, v) if return_state else self.readout(x)

# ── training (post section 3) ────────────────────────────────────────────────
def train(mu, seed=0, steps=4000):
    rng = np.random.default_rng(1000 + seed)
    Xtr, ytr = map(jnp.asarray, make_rings(1024, rng))
    Xte, yte = map(jnp.asarray, make_rings(2048, rng))
    model = MomentumResNet(mu=mu, rngs=nnx.Rngs(seed))
    opt = nnx.Optimizer(model, optax.adam(3e-3), wrt=nnx.Param)

    @nnx.jit
    def step(model, opt):
        def loss_fn(m):
            return optax.softmax_cross_entropy_with_integer_labels(
                m(Xtr), ytr).mean()
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        opt.update(model, grads)
        return loss

    for _ in range(steps):
        loss = step(model, opt)
    acc = (model(Xte).argmax(1) == yte).mean()
    return model, Xte, float(acc)

accs = {}
models = {}
for mu in (0.0, 0.9):
    models[mu], Xte, accs[mu] = train(mu)
    print(f"rings, L=32, mu={mu}: test accuracy {accs[mu]*100:.2f}%")
report["accs"] = {str(k): v for k, v in accs.items()}

# ── the rewind (post section on reversibility) ───────────────────────────────
def rewind(model, xL, vL):
    """From the final (x, v) alone, solve every block for its own past."""
    x, v = xL, vL
    for lin1, lin2 in reversed(model.blocks):
        x = x - model.h * v                        # undo the drift
        f = lin2(jnp.tanh(lin1(x)))
        v = (v - (1 - model.mu) * f) / model.mu    # undo the deposit: divide by mu
    return x, v

rew = {}
for mu in (0.9, 0.6, 0.3):
    model, _, acc = (models[0.9], None, accs[0.9]) if mu == 0.9 else train(mu)
    for dtype, name in ((jnp.float32, "float32"), (jnp.float64, "float64")):
        X0 = Xte[:64].astype(dtype)
        _, xL, vL = model(X0, return_state=True)
        x0, _ = rewind(model, xL, vL)
        err = float(jnp.abs(x0 - X0).max())
        rew[f"{mu}/{name}"] = err
        print(f"rewind mu={mu} {name}: max |x0_rec - x0| = {err:.1e}")
report["rewind"] = rew

with open(os.path.join(OUT, "momentum_nnx_check.json"), "w") as f:
    json.dump(report, f, indent=1)
print("MOMENTUM_NNX_CHECK_DONE")
