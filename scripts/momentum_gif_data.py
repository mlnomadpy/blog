"""momentum_gif_data.py -- training telemetry for the momentum-ResNet companion GIFs.

The published bundle public/momentum-resnet/*.json already carries everything the
companion needs EXCEPT per-training-step telemetry: the exported weights cover the
velocity-ledger and rewind renders (inference only), trajectories.json covers the
depth time-lapse, cliff.json/results.json cover the exactness ceiling, and the
Kepler GIF is pure ODE math. This script reruns ONE training, the momentum net the
explainer crowns (rings, L=32, mu=0.9, seed 0, the 100.0% net), with the exact
same data / init / optimizer as scripts/momentum_resnet.py, and dumps dense
per-snapshot telemetry so the "training crystallization" GIF can animate the real
process:

  - decision-probability grid (class-1 softmax prob) on a fixed 96x96 plane,
    every SNAP_EVERY steps,
  - test accuracy (2048 points) per snapshot,
  - full per-step training-loss curve,
  - hidden trajectories of 16 spread probe test points (all 33 states) per snapshot.

Writes results/momentum_gif_data.npz (one bundle). Run on Kaggle via kgl.py:
  python3 kgl.py --entry momentum_gif_data.py --slug blog-momentum-gifs \
      --expdir <repo>/scripts --pip "flax optax"
"""
import math
import os
import time

import numpy as np

import jax
import jax.numpy as jnp
import optax

SEED = int(os.environ.get("SEED", "0"))
STEPS = int(os.environ.get("STEPS", "4000"))
SNAP_EVERY = int(os.environ.get("SNAP_EVERY", "50"))
LR = 3e-3
WIDTH, HIDDEN = 2, 16
L = 32
MU = 0.9
T_TOTAL = 8.0
H = T_TOTAL / L
N_TRAIN, N_TEST = 1024, 2048
GRID_N, GRID_LIM = 96, 2.1

OUT = os.path.join(os.getcwd(), "results")
os.makedirs(OUT, exist_ok=True)


def make_rings(n, rng, noise=0.04):
    """Identical to scripts/momentum_resnet.py: disk (class 0) inside annulus."""
    n2 = n // 2
    th0 = rng.uniform(0, 2 * math.pi, n2)
    r_in = 0.6 * np.sqrt(rng.uniform(0, 1, n2))
    th1 = rng.uniform(0, 2 * math.pi, n2)
    r_out = rng.uniform(1.3, 1.8, n2)
    a = np.stack([r_in * np.cos(th0), r_in * np.sin(th0)], 1)
    b = np.stack([r_out * np.cos(th1), r_out * np.sin(th1)], 1)
    X = np.concatenate([a, b]) + rng.normal(0, noise, (n2 * 2, 2))
    y = np.concatenate([np.zeros(n2), np.ones(n2)]).astype(np.int32)
    return X.astype(np.float32), y


def init_params(key):
    k1, k2, k3 = jax.random.split(key, 3)
    s1 = 1.0 / math.sqrt(WIDTH)
    s2 = 1.0 / math.sqrt(HIDDEN)
    return {
        "W1": jax.random.normal(k1, (L, WIDTH, HIDDEN)) * s1,
        "b1": jnp.zeros((L, HIDDEN)),
        "W2": jax.random.normal(k2, (L, HIDDEN, WIDTH)) * s2,
        "b2": jnp.zeros((L, WIDTH)),
        "Wr": jax.random.normal(k3, (WIDTH, 2)) * s1,
        "br": jnp.zeros((2,)),
    }


def forward(params, X, collect=False):
    v0 = jnp.zeros_like(X)

    def block(carry, p):
        x, v = carry
        f = jnp.tanh(x @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]
        v = MU * v + (1.0 - MU) * f
        x = x + H * v
        return (x, v), ((x, v) if collect else None)

    blocks = {k: params[k] for k in ("W1", "b1", "W2", "b2")}
    (xL, vL), traj = jax.lax.scan(block, (X, v0), blocks)
    logits = xL @ params["Wr"] + params["br"]
    if collect:
        xs = jnp.concatenate([X[None], traj[0]], 0)   # [L+1, n, 2]
        vs = jnp.concatenate([v0[None], traj[1]], 0)  # [L+1, n, 2]
        return logits, xs, vs
    return logits


def main():
    t0 = time.time()
    rng = np.random.default_rng(1000 + SEED)
    Xtr, ytr = make_rings(N_TRAIN, rng)
    Xte, yte = make_rings(N_TEST, rng)
    params = init_params(jax.random.PRNGKey(SEED))

    g = np.linspace(-GRID_LIM, GRID_LIM, GRID_N).astype(np.float32)
    GX, GY = np.meshgrid(g, g)
    Xgrid = jnp.array(np.stack([GX.ravel(), GY.ravel()], 1))

    # 16 spread probe points, balanced by class, ordered by angle (as the export)
    idx = []
    for c in (0, 1):
        cand = np.where(yte == c)[0]
        order = np.argsort(np.arctan2(Xte[cand, 1], Xte[cand, 0]))
        idx.extend(cand[order[:: max(1, len(cand) // 8)]][:8])
    idx = np.array(idx)
    Xprobe = jnp.array(Xte[idx])

    opt = optax.adam(LR)
    opt_state = opt.init(params)
    Xtr_j, ytr_j, Xte_j, yte_j = map(jnp.array, (Xtr, ytr, Xte, yte))

    @jax.jit
    def step(params, opt_state):
        def lf(p):
            return optax.softmax_cross_entropy_with_integer_labels(
                forward(p, Xtr_j), ytr_j).mean()
        l, grads = jax.value_and_grad(lf)(params)
        updates, opt_state = opt.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, l

    @jax.jit
    def snapshot(params):
        probs = jax.nn.softmax(forward(params, Xgrid), -1)[:, 1]
        acc = (jnp.argmax(forward(params, Xte_j), 1) == yte_j).mean()
        _, xs, vs = forward(params, Xprobe, collect=True)
        return probs, acc, xs, vs

    grids, accs, snap_steps, trajs, vels = [], [], [], [], []
    losses = np.zeros(STEPS, np.float32)

    def take(params, i):
        probs, acc, xs, vs = snapshot(params)
        grids.append(np.asarray(probs, np.float16).reshape(GRID_N, GRID_N))
        accs.append(float(acc))
        trajs.append(np.asarray(xs, np.float32))   # [L+1, 16, 2]
        vels.append(np.asarray(vs, np.float32))
        snap_steps.append(i)

    take(params, 0)
    for i in range(STEPS):
        params, opt_state, l = step(params, opt_state)
        losses[i] = float(l)
        if (i + 1) % SNAP_EVERY == 0 or i == STEPS - 1:
            take(params, i + 1)

    print(f"trained rings L={L} mu={MU} seed={SEED}: "
          f"final test acc {accs[-1]*100:.2f}%, final loss {losses[-1]:.2e}, "
          f"{len(accs)} snapshots, {time.time()-t0:.0f}s")

    np.savez_compressed(
        os.path.join(OUT, "momentum_gif_data.npz"),
        grids=np.stack(grids), accs=np.array(accs, np.float32),
        snap_steps=np.array(snap_steps, np.int32), losses=losses,
        trajs=np.stack(trajs), vels=np.stack(vels),
        probe_y=yte[idx], probe_X=Xte[idx],
        Xtr=Xtr, ytr=ytr, grid_lim=np.float32(GRID_LIM),
        meta=np.array([L, MU, H, SEED, STEPS], np.float32))
    print("wrote", os.path.join(OUT, "momentum_gif_data.npz"),
          os.path.getsize(os.path.join(OUT, "momentum_gif_data.npz")) // 1024, "KB")
    print("MOMENTUM_GIF_DATA_DONE")


if __name__ == "__main__":
    main()
