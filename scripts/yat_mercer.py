"""Mercer's microscope: train a Yat network and export everything needed to
eigendecompose its kernel through training.

The post's object is the trained network's own kernel,
K_net(x, y) = sum_u phi_{w_u}(x) phi_{w_u}(y), the finite-rank Mercer kernel
the bank defines at each moment of training. This script trains the standard
YatMLP (m = 256 prototypes + linear readout, the lazy-training post's
yat_trained arm, same protocol) and snapshots the FULL model every epoch:
prototypes, kernel scalars, head. The eigen-analysis (spectra, kernel-target
alignment, mode truncation, mode galleries) is local, in
export_mercer_viz.py: a seconds-scale replay of these weights.

Also trains the frozen twin (same init, head-only) so every spectral
statement about "what training did" has its lazy control.

Arms x 3 seeds for the headline numbers; per-epoch snapshots for seed 0.
Kaggle GPU. Env: SMOKE=1, EPOCHS, M (width, default 256).
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
M = int(os.environ.get("M", "256"))
EPOCHS = int(os.environ.get("EPOCHS", "2" if SMOKE else "12"))
BATCH = 256
SEEDS = (0,) if SMOKE else (0, 1, 2)
LRS = {"trained": 1e-2, "frozen": 1.0}   # the lazy post's bracketed rates


def load_fmnist():
    import torchvision
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
print(f"[data] train {Xtr.shape} test {Xte.shape} m={M}")


class YatLayer(nnx.Module):
    def __init__(s, d_in, m, *, rngs):
        s.W = nnx.Param(nnx.initializers.lecun_normal()(rngs.params(), (m, d_in)))
        s.log_b = nnx.Param(jnp.full((), jnp.log(jnp.expm1(0.5))))
        s.log_eps = nnx.Param(jnp.full((), jnp.log(jnp.expm1(0.5))))

    def __call__(s, x):
        b = jax.nn.softplus(s.log_b.value)
        eps = jax.nn.softplus(s.log_eps.value)
        dot = x @ s.W.value.T
        d2 = (jnp.sum(x * x, -1, keepdims=True)
              + jnp.sum(s.W.value ** 2, -1) - 2 * dot)
        return (dot + b) ** 2 / (jnp.maximum(d2, 0.0) + eps)


class Net(nnx.Module):
    def __init__(s, m, *, rngs):
        s.feat = YatLayer(784, m, rngs=rngs)
        s.head = nnx.Linear(m, 10, rngs=rngs)

    def __call__(s, x):
        return s.head(s.feat(x))


def snapshot(model):
    return dict(
        W=np.asarray(model.feat.W.value, dtype=np.float16),
        b=float(jax.nn.softplus(model.feat.log_b.value)),
        eps=float(jax.nn.softplus(model.feat.log_eps.value)),
        A=np.asarray(model.head.kernel.value, dtype=np.float16),
        bias=np.asarray(model.head.bias.value, dtype=np.float16),
    )


def run(arm, seed):
    frozen = arm == "frozen"
    lr = LRS[arm]
    model = Net(M, rngs=nnx.Rngs(seed))
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
        return 100 * float(np.mean([
            float(acc_batch(model, jnp.asarray(Xte[i:i + 2000]),
                            jnp.asarray(yte[i:i + 2000])))
            for i in range(0, len(Xte), 2000)]))

    rng = np.random.default_rng(seed)
    snaps = [snapshot(model)]          # every seed keeps its trajectory endpoints
    best, curve = 0.0, []
    t0 = time.time()
    for ep in range(EPOCHS):
        perm = rng.permutation(len(Xtr))
        for i in range(0, len(Xtr) - BATCH + 1, BATCH):
            idx = perm[i:i + BATCH]
            train_step(model, opt, jnp.asarray(Xtr[idx]), jnp.asarray(ytr[idx]))
        a = test_acc()
        best = max(best, a)
        curve.append(round(a, 2))
        if seed == 0 or ep == EPOCHS - 1:
            snaps.append(snapshot(model))
    if True:
        np.savez_compressed(
            os.path.join(RESULTS_DIR, f"mercer_{arm}_s{seed}.npz"),
            **{f"W{e}": s["W"] for e, s in enumerate(snaps)},
            **{f"A{e}": s["A"] for e, s in enumerate(snaps)},
            **{f"bias{e}": s["bias"] for e, s in enumerate(snaps)},
            b=np.array([s["b"] for s in snaps], dtype=np.float32),
            eps=np.array([s["eps"] for s in snaps], dtype=np.float32))
    print(f"[{arm} s{seed}] best {best:.2f}% ({time.time()-t0:.0f}s)", flush=True)
    return dict(arm=arm, seed=seed, best_acc=best, curve=curve, lr=lr, m=M)


def main():
    rows = [run(arm, seed) for arm in ("trained", "frozen") for seed in SEEDS]
    with open(os.path.join(RESULTS_DIR, "yat_mercer.json"), "w") as f:
        json.dump(dict(rows=rows, m=M, epochs=EPOCHS), f)
    for arm in ("trained", "frozen"):
        accs = [r["best_acc"] for r in rows if r["arm"] == arm]
        print(f"{arm}: {np.mean(accs):.2f} +- {np.std(accs):.2f}")


if __name__ == "__main__":
    main()
