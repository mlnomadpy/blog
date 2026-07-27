"""Lazy training: frozen random Yat prototypes + a trained linear readout.

The question: freeze a bank of m Yat units at random initialization (the
prototypes W, and the kernel scalars b/eps, never train), fit ONLY the linear
readout on top, and sweep m. How many frozen random features does it take to
match what training the features buys? Four arms, identical data and head:

  yat_frozen   W ~ lecun-normal init, FROZEN; head trained
               (random features: the Rahimi-Recht move, with the Yat kernel)
  yat_data     W = m random TRAINING IMAGES, frozen; head trained
               (data-sampled centers: the Nystrom flavor; every neuron is a
               real picture, no training anywhere but the readout)
  yat_trained  same init as yat_frozen (same seed), EVERYTHING trains
               (the feature-learning upper line; also logs the relative
               prototype movement ||W_end - W_0||_F / ||W_0||_F, the
               laziness statistic)
  yat_data_trained  data-seeded init, everything trains (what training buys
               over the frozen data-seeded bank, and how far centers move)
  rbf_data     Gaussian units exp(-||x-w||^2 / 2 sigma^2), centers = random
               training images, sigma at the median heuristic, frozen
               (the classical RBF network of Broomhead-Lowe)
  rbf_data_trained  same, everything trains. The kernel book's deep-learning
               chapter claims Gaussian centers are effectively immobile
               under gradient descent (the gradient carries the dying
               exponential); the Yat unit's tail is polynomial, so its
               centers should move. Both movements measured.
  relu_frozen  frozen random Linear + ReLU, head trained (the random-feature
               Monte Carlo of the arc-cosine kernel, Cho-Saul)

Fashion-MNIST raw pixels (the series' home turf: raw-pixel constructed head
79%, hand-built features 83.3%, trained backbone 85.7%). Width sweep
m = 16 .. 8192, 3 seeds. Writes results/yat_lazy.json + prototype samples
for the explainer viz. Kaggle GPU.

Env knobs: SMOKE=1 (tiny), SWEEP_LRS="1e-3,3e-3" (LR sweep mode at m=256),
LR_HEAD / LR_FULL (overrides), WIDTHS="16,64" (subset), SEEDS_N=3.
"""

import json
import os
import time

import numpy as np

import jax
import jax.numpy as jnp
import optax
from flax import nnx

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
SMOKE = os.environ.get("SMOKE", "0") == "1"

WIDTHS = [int(w) for w in os.environ.get(
    "WIDTHS", "16,32,64,128,256,512,1024,2048,4096,8192").split(",")]
SEEDS = tuple(range(int(os.environ.get("SEEDS_N", "1" if SMOKE else "3"))))
ARMS = tuple(os.environ.get(
    "ARMS", "yat_frozen,yat_data,yat_trained,yat_data_trained,"
            "rbf_data,rbf_data_trained,relu_frozen").split(","))
TRAINED_ARMS = {"yat_trained", "yat_data_trained", "rbf_data_trained"}
SWEEP_LRS = [float(x) for x in os.environ.get("SWEEP_LRS", "").split(",") if x]
EPOCHS = int(os.environ.get("EPOCHS", "2" if SMOKE else "12"))
BATCH = 256
LR_HEAD = float(os.environ.get("LR_HEAD", "3e-3"))
LR_FULL = float(os.environ.get("LR_FULL", "3e-3"))
if SMOKE:
    WIDTHS = [16, 64]


def load_fmnist():
    import torchvision  # only for the download; everything else is numpy/JAX
    tr = torchvision.datasets.FashionMNIST("/tmp/fmnist", train=True, download=True)
    te = torchvision.datasets.FashionMNIST("/tmp/fmnist", train=False, download=True)
    Xtr = (tr.data.numpy().astype(np.float32) / 255.0).reshape(-1, 784)
    Xte = (te.data.numpy().astype(np.float32) / 255.0).reshape(-1, 784)
    ytr = tr.targets.numpy().astype(np.int32)
    yte = te.targets.numpy().astype(np.int32)
    if SMOKE:
        Xtr, ytr = Xtr[:8000], ytr[:8000]
    return Xtr, ytr, Xte, yte


Xtr, ytr, Xte, yte = load_fmnist()
print(f"[data] train {Xtr.shape} test {Xte.shape}")


class YatLayer(nnx.Module):
    """The series' Yat layer: kernel units, no activation function.
    b and eps live behind a softplus (b0 = eps0 = 0.5)."""

    def __init__(s, d_in, m, *, rngs, init="lecun"):
        if init == "lecun":
            W0 = nnx.initializers.lecun_normal()(rngs.params(), (m, d_in))
        else:  # data-seeded: m random training images as prototypes
            idx = jax.random.choice(rngs.params(), len(Xtr), (m,), replace=False)
            W0 = jnp.asarray(Xtr)[idx]
        s.W = nnx.Param(W0)
        s.log_b = nnx.Param(jnp.full((), jnp.log(jnp.expm1(0.5))))
        s.log_eps = nnx.Param(jnp.full((), jnp.log(jnp.expm1(0.5))))

    def __call__(s, x):
        b = jax.nn.softplus(s.log_b.value)
        eps = jax.nn.softplus(s.log_eps.value)
        dot = x @ s.W.value.T
        d2 = (jnp.sum(x * x, -1, keepdims=True)
              + jnp.sum(s.W.value ** 2, -1) - 2 * dot)
        return (dot + b) ** 2 / (jnp.maximum(d2, 0.0) + eps)


class RbfLayer(nnx.Module):
    """Gaussian units exp(-||x - w||^2 / 2 sigma^2), the classical RBF
    network. sigma initialized at the median distance heuristic (median
    ||x - w|| over a data sample at init), learned through softplus when
    the arm trains."""

    def __init__(s, d_in, m, *, rngs):
        idx = jax.random.choice(rngs.params(), len(Xtr), (m,), replace=False)
        W0 = jnp.asarray(Xtr)[idx]
        s.W = nnx.Param(W0)
        smp = jnp.asarray(Xtr[:512])
        d2 = (jnp.sum(smp * smp, -1, keepdims=True)
              + jnp.sum(W0 ** 2, -1) - 2 * smp @ W0.T)
        sigma0 = jnp.sqrt(jnp.median(jnp.maximum(d2, 0.0)))
        s.log_sigma = nnx.Param(jnp.log(jnp.expm1(sigma0)))

    def __call__(s, x):
        sigma = jax.nn.softplus(s.log_sigma.value)
        d2 = (jnp.sum(x * x, -1, keepdims=True)
              + jnp.sum(s.W.value ** 2, -1) - 2 * x @ s.W.value.T)
        return jnp.exp(-jnp.maximum(d2, 0.0) / (2 * sigma ** 2))


class Net(nnx.Module):
    def __init__(s, arm, m, *, rngs):
        if arm.startswith("relu"):
            s.feat = ReluLayer(784, m, rngs=rngs)
        elif arm.startswith("rbf"):
            s.feat = RbfLayer(784, m, rngs=rngs)
        else:
            init = "data" if arm.startswith("yat_data") else "lecun"
            s.feat = YatLayer(784, m, rngs=rngs, init=init)
        s.head = nnx.Linear(m, 10, rngs=rngs)

    def __call__(s, x):
        return s.head(s.feat(x))


class ReluLayer(nnx.Module):
    def __init__(s, d_in, m, *, rngs):
        s.lin = nnx.Linear(d_in, m, rngs=rngs)

    def __call__(s, x):
        return jax.nn.relu(s.lin(x))


def run(arm, m, seed, lr=None):
    frozen = arm not in TRAINED_ARMS
    lr = lr or (LR_HEAD if frozen else LR_FULL)
    model = Net(arm, m, rngs=nnx.Rngs(seed))
    W0 = (np.asarray(model.feat.W.value).copy()
          if not arm.startswith("relu") else None)
    # frozen arms: only the head's params receive gradients
    trainable = (nnx.All(nnx.Param, nnx.PathContains("head"))
                 if frozen else nnx.Param)
    opt = nnx.Optimizer(model, optax.adamw(lr, weight_decay=1e-4), wrt=trainable)

    @nnx.jit
    def train_step(model, opt, x, y):
        def loss_fn(m_):
            lg = m_(x)
            return optax.softmax_cross_entropy_with_integer_labels(lg, y).mean()
        loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, trainable))(model)
        opt.update(model, grads)
        return loss

    @nnx.jit
    def acc_batch(model, x, y):
        return (model(x).argmax(-1) == y).mean()

    def test_acc():
        accs = [float(acc_batch(model, jnp.asarray(Xte[i:i + 2000]),
                                jnp.asarray(yte[i:i + 2000])))
                for i in range(0, len(Xte), 2000)]
        return 100 * float(np.mean(accs))

    rng = np.random.default_rng(seed)
    best, curve = 0.0, []
    t0 = time.time()
    # TRAJ=1: snapshot the first 16 prototypes + per-unit movement each epoch
    # (the companion's noise-to-garment and travel-race figures)
    TRAJ = os.environ.get("TRAJ", "0") == "1" and not arm.startswith("relu")
    snaps, moves = [], []
    if TRAJ:
        snaps.append(np.asarray(model.feat.W.value[:16], dtype=np.float16))
    for ep in range(EPOCHS):
        perm = rng.permutation(len(Xtr))
        for i in range(0, len(Xtr) - BATCH + 1, BATCH):
            idx = perm[i:i + BATCH]
            train_step(model, opt, jnp.asarray(Xtr[idx]), jnp.asarray(ytr[idx]))
        a = test_acc()
        best = max(best, a)
        curve.append(round(a, 2))
        if TRAJ:
            Wn = np.asarray(model.feat.W.value)
            snaps.append(Wn[:16].astype(np.float16))
            if W0 is not None:
                mv = np.linalg.norm(Wn - W0, axis=1) / (np.linalg.norm(W0, axis=1) + 1e-12)
                moves.append(mv.astype(np.float16))
    if TRAJ:
        np.savez_compressed(
            os.path.join(RESULTS_DIR, f"traj_{arm}_m{m}_s{seed}.npz"),
            snaps=np.stack(snaps), moves=np.stack(moves) if moves else np.zeros(0),
            acc=np.array(curve, dtype=np.float32))
    out = dict(arm=arm, m=m, seed=seed, lr=lr, best_acc=best, curve=curve,
               secs=round(time.time() - t0, 1))
    if arm.startswith("yat"):
        out["b"] = float(jax.nn.softplus(model.feat.log_b.value))
        out["eps"] = float(jax.nn.softplus(model.feat.log_eps.value))
    if arm.startswith("rbf"):
        out["sigma"] = float(jax.nn.softplus(model.feat.log_sigma.value))
    if arm in TRAINED_ARMS and W0 is not None:
        Wf = np.asarray(model.feat.W.value)
        out["w_move_rel"] = float(np.linalg.norm(Wf - W0) / (np.linalg.norm(W0) + 1e-12))
        # per-unit movement too: the distribution matters (a few units moving
        # far reads differently than all units creeping)
        mv = np.linalg.norm(Wf - W0, axis=1) / (np.linalg.norm(W0, axis=1) + 1e-12)
        out["w_move_median"] = float(np.median(mv))
        out["w_move_max"] = float(mv.max())
    print(f"[{arm} m={m} s{seed}] best {best:.2f}% lr={lr} "
          + (f"move {out.get('w_move_rel', 0):.3f} " if arm in TRAINED_ARMS else "")
          + f"({out['secs']}s)", flush=True)
    return out


def export_samples():
    """Prototype galleries for the explainer viz: what a frozen random
    neuron looks like vs a data-seeded one (m = 256, seed 0)."""
    smp = {}
    for arm in ("yat_frozen", "yat_data"):
        model = Net(arm, 256, rngs=nnx.Rngs(0))
        W = np.asarray(model.feat.W.value)[:24]
        smp[arm] = W.astype(np.float16)
    np.savez_compressed(os.path.join(RESULTS_DIR, "yat_lazy_prototypes.npz"), **smp)


def main():
    t0 = time.time()
    rows = []
    if SWEEP_LRS:
        for arm in ("yat_frozen", "yat_trained"):
            for lr in SWEEP_LRS:
                r = run(arm, 256, 0, lr=lr)
                r["sweep"] = True
                rows.append(r)
        with open(os.path.join(RESULTS_DIR, "yat_lazy_sweep.json"), "w") as f:
            json.dump(rows, f)
        return
    # per-arm LR fairness: bracket each arm's rate at m=256 (one seed),
    # then run the full width grid at that arm's own best rate.
    # trained arms were pre-bracketed around 1e-2 (kgl_blog-yatlazy-lrsweep*);
    # head-only arms keep climbing past 3e-1, so their grid extends to 1.0
    GRIDS = {True: [3e-3, 1e-2, 3e-2], False: [3e-2, 1e-1, 3e-1, 1.0]}
    chosen = {}
    for arm in ARMS:
        env = "LR_FULL" if arm in TRAINED_ARMS else "LR_HEAD"
        if os.environ.get(env):
            chosen[arm] = float(os.environ[env])
            continue
        if SMOKE:
            chosen[arm] = 3e-3
            continue
        cells = []
        for lr in GRIDS[arm in TRAINED_ARMS]:
            r = run(arm, 256, 0, lr=lr)
            r["sweep"] = True
            cells.append(r)
            rows.append(r)
        best = max(cells, key=lambda r: r["best_acc"])
        chosen[arm] = best["lr"]
        print(f"[lr] {arm}: {chosen[arm]}")
    for arm in ARMS:
        for m in WIDTHS:
            for seed in SEEDS:
                rows.append(run(arm, m, seed, lr=chosen[arm]))
    export_samples()
    with open(os.path.join(RESULTS_DIR, "yat_lazy.json"), "w") as f:
        json.dump(dict(rows=rows, widths=WIDTHS, epochs=EPOCHS,
                       n_train=len(Xtr)), f)
    print(f"done in {time.time() - t0:.0f}s -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
