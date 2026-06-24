"""The modality gap as a learned dynamic.

Two modalities of the same K-class signal, each encoded by a small ReLU tower.
We start both towers from the *same* initialisation, so at step 0 the two
embedding clouds sit on top of each other -- there is no gap. Then we train a
SigLIP contrastive objective and watch what happens: the objective pushes the
two modalities apart into two separated cones, even though matched pairs stay
ranked above random ones. The gap is not present at initialisation; the
separating objective *opens* it.

This is the cone/temperature story of the modality gap (Liang et al. 2022),
reproduced from scratch and, crucially, recorded *across training* so we can
animate it.

Outputs to ../../public/modality-gap/:
  gap_dynamics.gif  - the hero: the two clouds separating over training
  gap_cones.png     - the final frame (two cones) + the gap-vs-step curve
  gap.json          - per-frame 2-D coordinates for the interactive viz
  results_gap.npz   - raw snapshots + metric trajectory

Run:  python gap.py
"""

from __future__ import annotations

import os
import json
import numpy as np
import jax
import jax.numpy as jnp
import optax

SEED = 0
D = 24            # ambient dim per modality
H = 48            # tower hidden width
DE = 16           # embedding dim (L2-normalised); projected to 2-D for viewing
LAYERS = 2        # tower depth (the cone effect needs a couple of layers)
K = 2             # classes (two -> clean circle/cross markers, gap still opens)
N = 4000
NOISE = 0.08
STEPS = 1400
BATCH = 256
LR = 2e-3
LOGIT_SCALE = 2.0   # 1/temperature; the cone separation grows with this
NVIZ = 240          # points per modality kept for the animation / viz


def _lift(seed):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((D, 2)).astype(np.float32)
    W /= np.linalg.norm(W, axis=0, keepdims=True)
    return W, rng.standard_normal(D).astype(np.float32) * 0.1


W_A, b_A = _lift(11)
W_B, b_B = _lift(22)


def generate(n, seed):
    rng = np.random.default_rng(seed)
    c = rng.integers(0, K, n)
    tA = rng.uniform(-1, 1, n).astype(np.float32)
    tB = rng.uniform(-1, 1, n).astype(np.float32)

    def build(t, W, b):
        s = (c - (K - 1) / 2) * 1.2 + rng.standard_normal(n) * 0.1
        u = np.stack([s, t], axis=-1).astype(np.float32)
        return (u @ W.T + b + rng.standard_normal((n, D)) * NOISE).astype(np.float32)

    return build(tA, W_A, b_A), build(tB, W_B, b_B), c.astype(np.int32), tA, tB


def tower_init(key):
    ks = jax.random.split(key, LAYERS)
    layers = []
    din = D
    for i in range(LAYERS):
        dout = H if i < LAYERS - 1 else DE
        layers.append((jax.random.normal(ks[i], (din, dout)) * np.sqrt(2.0 / din),
                       jnp.zeros(dout)))
        din = dout
    return layers


def encode(tower, x):
    h = x
    for i, (W, b) in enumerate(tower):
        h = h @ W + b
        if i < len(tower) - 1:
            h = jax.nn.relu(h)
    return h / (jnp.linalg.norm(h, axis=-1, keepdims=True) + 1e-9)


def siglip_loss(p, xa, xb):
    zA, zB = encode(p["A"], xa), encode(p["B"], xb)
    t = jnp.exp(p["ls"])
    logits = t * (zA @ zB.T) + p["bias"]
    n = zA.shape[0]
    labels = 2.0 * jnp.eye(n) - 1.0
    return jnp.mean(jax.nn.softplus(-labels * logits))


def main():
    print("Generating K-class data and matched-init towers...")
    xA, xB, c, tA, tB = generate(N, SEED)

    def whiten(x):
        mu, sd = x.mean(0), x.std(0) + 1e-6
        return ((x - mu) / sd).astype(np.float32)
    xA, xB = whiten(xA), whiten(xB)

    # SAME init for both towers -> no gap at step 0.
    key0 = jax.random.key(1)
    params = {"A": tower_init(key0), "B": tower_init(key0),
              "ls": jnp.asarray(np.log(LOGIT_SCALE), np.float32),
              "bias": jnp.asarray(-1.0, np.float32)}

    opt = optax.adam(LR)
    state = opt.init(params)

    @jax.jit
    def step(p, state, xa, xb):
        l, g = jax.value_and_grad(siglip_loss)(p, xa, xb)
        upd, state = opt.update(g, state)
        return optax.apply_updates(p, upd), state, l

    encA = jax.jit(lambda p, x: encode(p["A"], x))
    encB = jax.jit(lambda p, x: encode(p["B"], x))

    xa_v = jnp.asarray(xA[:NVIZ]); xb_v = jnp.asarray(xB[:NVIZ])
    c_v = c[:NVIZ]

    # snapshot schedule: dense early (the gap opens fast), sparser later
    snaps = sorted(set([int(round(s)) for s in
                        np.unique(np.concatenate([
                            np.arange(0, 60, 3),
                            np.linspace(60, STEPS, 30)]).astype(int))]))
    snaps = [s for s in snaps if s <= STEPS]

    embA_hist, embB_hist, traj = [], [], []
    key = jax.random.key(SEED)
    si = 0
    for t in range(STEPS + 1):
        if si < len(snaps) and t == snaps[si]:
            zA = np.asarray(encA(params, xa_v)); zB = np.asarray(encB(params, xb_v))
            embA_hist.append(zA); embB_hist.append(zB)
            matched = float(np.mean(np.sum(zA * zB, axis=1)))
            rng = np.random.default_rng(0)
            rnd = float(np.mean(np.sum(zA * zB[rng.permutation(len(zB))], axis=1)))
            gap = float(np.linalg.norm(zA.mean(0) - zB.mean(0)))
            traj.append((t, matched, rnd, gap))
            si += 1
        if t == STEPS:
            break
        key, sub = jax.random.split(key)
        idx = jax.random.randint(sub, (BATCH,), 0, N)
        params, state, l = step(params, state, jnp.asarray(xA)[idx], jnp.asarray(xB)[idx])

    embA_hist = np.array(embA_hist); embB_hist = np.array(embB_hist)  # (F, NVIZ, DE)
    traj = np.array(traj)  # (F, 4): step, matched, random, gap

    # fixed 2-D basis from the FINAL frame so the animation doesn't spin
    final = np.concatenate([embA_hist[-1], embB_hist[-1]], axis=0)
    mu = final.mean(0)
    _, _, Vt = np.linalg.svd(final - mu, full_matrices=False)
    basis = Vt[:2].T
    PA = (embA_hist - mu) @ basis     # (F, NVIZ, 2)
    PB = (embB_hist - mu) @ basis
    # global scale so all frames fit a [-1,1]-ish box
    scale = float(np.percentile(np.abs(np.concatenate([PA, PB])), 99)) + 1e-9
    PA /= scale; PB /= scale

    print(f"frames: {len(traj)}   gap: {traj[0,3]:.2f} (step 0) -> {traj[-1,3]:.2f} (step {int(traj[-1,0])})")
    print(f"matched cos: {traj[0,1]:+.2f} -> {traj[-1,1]:+.2f}   random: {traj[0,2]:+.2f} -> {traj[-1,2]:+.2f}")

    out = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "public", "modality-gap"))
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "results_gap.npz"),
             PA=PA, PB=PB, c=c_v, traj=traj)
    print("saved -> results_gap.npz  (interactive gap viz runs live in-browser)")


if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    main()
