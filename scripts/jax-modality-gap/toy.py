"""Separate or represent: the choice, in its smallest form.

The abstract toy behind "Separate or Represent: The Modality Gap Is a Choice."
No radio, no transformers. Four distributions, two modalities, two classes, and
a single linear projector per modality.

  Two classes  c in {0, 1}.
  A within-class factor  t in [0, 1]  (position along an arc) that is drawn
  *independently* for each modality. So a matched pair (A_i, B_i) shares its
  class and nothing else: t is modality-unique.

  Modality A:  x_A = lift_A(arc(c, t_A)) + noise        (R^D)
  Modality B:  x_B = lift_B(arc(c, t_B)) + noise        (R^D)

Both modalities are the two-moons shape pushed into R^D by a different random
linear map. Class picks the moon; t walks along it.

We give a single linear projector per modality (same 2-D bottleneck for every
model -- so capacity is held fixed and only the *objective* differs) and train
two ways:

  separate  : align A_i with B_i using the SigLIP pairwise-sigmoid loss.
  represent : reconstruct each modality's own input (a linear autoencoder).

Then we ask the same space to do two jobs:

  classify  : recover the class  c           (the separable slice)
  represent : recover the factor t           (the within-class whole)

SigLIP keeps c (the only thing the two modalities agree on) and discards t,
because t carries no cross-modal agreement -- aligning on it is impossible, so
the loss collapses it. The autoencoder keeps t, because it has to put it back.
Same data, same linear capacity; the objective decides what survives.

Outputs: results_toy.npz (+ toy.json for the interactive viz) in
../../public/modality-gap/.  Render with toy_render.py.

Run:  python toy.py
"""

from __future__ import annotations

import os
import json
import numpy as np
import jax
import jax.numpy as jnp
import optax

SEED = 0
D = 24            # ambient dimension of each modality (intrinsic structure is 2-D)
DEMB = 2          # shared bottleneck / embedding dim (same for SigLIP and the AE)
CLASS_GAP = 1.0   # class-axis offset (+/-) before jitter
CLASS_JIT = 0.12  # within-class jitter on the class axis
T_SPAN = 1.0      # within-class factor t spans [-T_SPAN, T_SPAN]
NOISE = 0.08      # gaussian noise added in ambient space
N_TRAIN = 6000
N_TEST = 2000
STEPS = 3000
BATCH = 256
LR = 1e-2


def _lift(key, n_in=2):
    """A fixed random linear map (2 -> D) with roughly unit-energy columns,
    so the 2-D structure (class axis, t axis) is hidden in R^D but fully
    recoverable by a *linear* probe."""
    rng = np.random.default_rng(key)
    W = rng.standard_normal((D, n_in)).astype(np.float32)
    W /= np.linalg.norm(W, axis=0, keepdims=True)
    b = rng.standard_normal((D,)).astype(np.float32) * 0.1
    return W, b


W_A, b_A = _lift(11)
W_B, b_B = _lift(22)


def generate(n, seed):
    """Return (x_A, x_B, c, t_A, t_B). The intrinsic coordinate is 2-D and
    *linear*: one axis is the class (two separated clusters), the orthogonal
    axis is the within-class factor t. A matched pair (x_A[i], x_B[i]) shares
    its class c[i]; the within-class factors t_A[i], t_B[i] are independent
    draws -- so class is all the two modalities agree on."""
    rng = np.random.default_rng(seed)
    c = rng.integers(0, 2, n)
    tA = rng.uniform(-T_SPAN, T_SPAN, n).astype(np.float32)
    tB = rng.uniform(-T_SPAN, T_SPAN, n).astype(np.float32)

    def build(t, W, b):
        s = np.where(c == 1, CLASS_GAP, -CLASS_GAP) + rng.standard_normal(n) * CLASS_JIT
        u = np.stack([s, t], axis=-1).astype(np.float32)        # (n, 2): [class, t]
        return (u @ W.T + b + rng.standard_normal((n, D)) * NOISE).astype(np.float32)

    xA = build(tA, W_A, b_A)
    xB = build(tB, W_B, b_B)
    return xA, xB, c.astype(np.int32), tA, tB


def l2n(x):
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


# ── separate: two linear projectors aligned with SigLIP ─────────────────────
def make_siglip(key):
    ks = jax.random.split(key, 2)
    return dict(
        PA=jax.random.normal(ks[0], (D, DEMB)) * 0.1,
        PB=jax.random.normal(ks[1], (D, DEMB)) * 0.1,
        logit_scale=jnp.log(2.0),
        bias=jnp.asarray(-1.0),
    )


def siglip_embed_A(p, xA):
    return l2n(xA @ p["PA"])


def siglip_embed_B(p, xB):
    return l2n(xB @ p["PB"])


def siglip_loss(p, xA, xB):
    zA, zB = siglip_embed_A(p, xA), siglip_embed_B(p, xB)
    t = jnp.exp(p["logit_scale"])
    logits = t * (zA @ zB.T) + p["bias"]
    n = zA.shape[0]
    labels = 2.0 * jnp.eye(n) - 1.0            # +1 on the matched diagonal, -1 off
    return jnp.mean(jax.nn.softplus(-labels * logits))


# ── represent: a linear autoencoder per modality (same bottleneck) ──────────
def make_ae(key):
    ks = jax.random.split(key, 4)
    return dict(
        EA=jax.random.normal(ks[0], (D, DEMB)) * 0.1,
        DA=jax.random.normal(ks[1], (DEMB, D)) * 0.1,
        EB=jax.random.normal(ks[2], (D, DEMB)) * 0.1,
        DB=jax.random.normal(ks[3], (DEMB, D)) * 0.1,
    )


def ae_embed_A(p, xA):
    return xA @ p["EA"]


def ae_embed_B(p, xB):
    return xB @ p["EB"]


def ae_loss(p, xA, xB):
    recA = ae_embed_A(p, xA) @ p["DA"]
    recB = ae_embed_B(p, xB) @ p["DB"]
    return jnp.mean((recA - xA) ** 2) + jnp.mean((recB - xB) ** 2)


def train(loss_fn, params, data, steps, lr):
    opt = optax.adam(lr)
    state = opt.init(params)
    key = jax.random.key(SEED)

    @jax.jit
    def step(params, state, batch):
        l, g = jax.value_and_grad(loss_fn)(params, *batch)
        upd, state = opt.update(g, state)
        return optax.apply_updates(params, upd), state, l

    n = data[0].shape[0]
    for t in range(steps):
        key, sub = jax.random.split(key)
        idx = jax.random.randint(sub, (BATCH,), 0, n)
        batch = tuple(jnp.asarray(d)[idx] for d in data)
        params, state, l = step(params, state, batch)
    return params


# ── probes ───────────────────────────────────────────────────────────────
def ridge_fit(X, Y, lam=1.0):
    X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ Y)


def ridge_pred(W, X):
    X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    return X @ W


def class_acc(tr, ytr, te, yte):
    """Linear probe for the class -- the separable slice."""
    Y = np.eye(2)[ytr]
    W = ridge_fit(tr, Y)
    pred = ridge_pred(W, te).argmax(1)
    return float((pred == yte).mean())


def recover_t(tr, ttr, te, tte):
    """Linear probe for the within-class factor -- the representable whole.
    Returns R^2 (1 = fully recovered, 0 = no better than the mean)."""
    W = ridge_fit(tr, ttr[:, None])
    pred = ridge_pred(W, te).ravel()
    ss_res = float(np.sum((pred - tte) ** 2))
    ss_tot = float(np.sum((tte - tte.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def within_class_retrieval(emb, c, t, k=10, seed=0):
    """How well does nearest-neighbour retrieval *within a class* respect the
    within-class factor t? For each query we take its k embedding-nearest
    same-class neighbours and measure the mean |dt| to them, normalised by the
    mean |dt| to random same-class items. <1 means the space preserves t (so
    you can retrieve the right neighbour); ~1 means t was discarded."""
    rng = np.random.default_rng(seed)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    ratios = []
    for cls in (0, 1):
        idx = np.where(c == cls)[0]
        E = emb[idx]; T = t[idx]
        sim = E @ E.T
        np.fill_diagonal(sim, -np.inf)
        nn = np.argsort(-sim, axis=1)[:, :k]
        dt_nn = np.abs(T[:, None] - T[nn]).mean(1)
        # baseline: mean |dt| to k random same-class items
        rand = rng.integers(0, len(idx), (len(idx), k))
        dt_rand = np.abs(T[:, None] - T[rand]).mean(1)
        ratios.append(dt_nn / (dt_rand + 1e-9))
    return float(np.concatenate(ratios).mean())


def main():
    print("Generating four distributions (2 classes x 2 modalities)...")
    xA, xB, c, tA, tB = generate(N_TRAIN, SEED)
    xA_te, xB_te, c_te, tA_te, tB_te = generate(N_TEST, SEED + 1)

    # whiten each modality (per-dim zero-mean, unit-var) using train statistics
    def whitener(x):
        mu, sd = x.mean(0), x.std(0) + 1e-6
        return lambda z: ((z - mu) / sd).astype(np.float32)
    wA, wB = whitener(xA), whitener(xB)
    xA, xB = wA(xA), wB(xB)
    xA_te, xB_te = wA(xA_te), wB(xB_te)

    print("Training SEPARATE  (SigLIP-aligned linear projectors)...")
    sp = train(siglip_loss, make_siglip(jax.random.key(SEED)), (xA, xB), STEPS, LR)
    s_embA = jax.jit(lambda x: siglip_embed_A(sp, x))
    s_embB = jax.jit(lambda x: siglip_embed_B(sp, x))
    sA = np.asarray(s_embA(jnp.asarray(xA))); sA_te = np.asarray(s_embA(jnp.asarray(xA_te)))
    sB_te = np.asarray(s_embB(jnp.asarray(xB_te)))

    print("Training REPRESENT (linear autoencoder, same bottleneck)...")
    ap = train(ae_loss, make_ae(jax.random.key(SEED + 3)), (xA, xB), STEPS, LR)
    a_embA = jax.jit(lambda x: ae_embed_A(ap, x))
    a_embB = jax.jit(lambda x: ae_embed_B(ap, x))
    aA = np.asarray(a_embA(jnp.asarray(xA))); aA_te = np.asarray(a_embA(jnp.asarray(xA_te)))
    aB_te = np.asarray(a_embB(jnp.asarray(xB_te)))

    # The two jobs, probed on modality A's embedding.
    s_class = class_acc(sA, c, sA_te, c_te)
    a_class = class_acc(aA, c, aA_te, c_te)
    s_t = recover_t(sA, tA, sA_te, tA_te)
    a_t = recover_t(aA, tA, aA_te, tA_te)
    s_ret = within_class_retrieval(sA_te, c_te, tA_te)
    a_ret = within_class_retrieval(aA_te, c_te, tA_te)

    # Cross-modal alignment: cosine of matched (A_i, B_i) pairs vs random pairs.
    matched = np.sum(sA_te * sB_te, axis=1)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(sB_te))
    random = np.sum(sA_te * sB_te[perm], axis=1)

    print("\n==================== RESULTS (probed on modality A) ====================")
    print(f"                          SEPARATE (SigLIP)   REPRESENT (autoencoder)")
    print(f"classify  class acc  (^)      {s_class*100:5.1f}%               {a_class*100:5.1f}%")
    print(f"represent recover-t  R^2 (^)  {s_t:6.2f}                {a_t:6.2f}")
    print(f"retrieve  within-cls dt  (v)  {s_ret:6.2f}                {a_ret:6.2f}")
    print(f"(retrieval ratio: 1.0 = t discarded, <1 = t preserved)")
    print(f"\nSigLIP matched-pair cosine: {matched.mean():+.2f}   random: {random.mean():+.2f}")
    print("========================================================================\n")

    out = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "public", "modality-gap"))
    os.makedirs(out, exist_ok=True)

    # raw arrays for matplotlib figures
    np.savez(os.path.join(out, "results_toy.npz"),
             xA=xA_te[:1500], xB=xB_te[:1500], c=c_te[:1500],
             tA=tA_te[:1500], tB=tB_te[:1500],
             sA=sA_te[:1500], sB=sB_te[:1500],
             aA=aA_te[:1500], aB=aB_te[:1500],
             s_class=s_class, a_class=a_class, s_t=s_t, a_t=a_t,
             s_ret=s_ret, a_ret=a_ret,
             matched=matched, random=random)
    print("saved ->", os.path.join(out, "results_toy.npz"),
          "  (interactive computes live; no JSON needed)")


if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    main()
