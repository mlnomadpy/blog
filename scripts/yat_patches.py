"""Parts instead of pictures: one shared Yat MLP over image patches, mean-pooled.

    image -> P x P patches -> ONE shared Yat bank per patch -> mean over
    patches -> linear readout

Mean pooling is linear, so the logits are exactly the AVERAGE OF PER-PATCH
VOTES:  A^T mean_p phi(x_p) = mean_p (A^T phi(x_p)).  Every patch casts its own
ballot and the model is the tally, which gives exact spatial attribution with
no gradients and no probes.

The point is what this does to the audit in scripts/yat_protocol.py. Whole
images sit ~300 squared units away from every prototype, so the softening eps
never engages and the trained net leans on the alignment channel. Patches live
in a small dense space, so three predictions follow:

  1  eps stops being vestigial (distances collapse toward it)
  2  the proximity channel starts carrying concepts
  3  concept support narrows, which is what would make the exact row edit work

Patch size is the granularity dial: 28 (the whole image, the control), 14, 7,
4. Writes results/yat_patches.json + per-size weights for the local audit.
Kaggle GPU. Env: SMOKE=1, SIZES=28,14,7,4, EPOCHS, M, SEEDS_N.
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
SIZES = [int(s) for s in os.environ.get("SIZES", "28,14,7,4").split(",")]
EPOCHS = int(os.environ.get("EPOCHS", "2" if SMOKE else "12"))
M = int(os.environ.get("M", "64" if SMOKE else "256"))
SEEDS = tuple(range(int(os.environ.get("SEEDS_N", "1" if SMOKE else "3"))))
BATCH = 256
LR = float(os.environ.get("LR", "1e-2"))


def load():
    import torchvision
    tr = torchvision.datasets.FashionMNIST("/tmp/fmnist", train=True, download=True)
    te = torchvision.datasets.FashionMNIST("/tmp/fmnist", train=False, download=True)
    f = lambda ds: (ds.data.numpy().astype(np.float32) / 255).reshape(-1, 28, 28)
    Xtr, Xte = f(tr), f(te)
    if SMOKE:
        Xtr = Xtr[:8000]
        return Xtr, tr.targets.numpy()[:8000].astype(np.int32), Xte, te.targets.numpy().astype(np.int32)
    return Xtr, tr.targets.numpy().astype(np.int32), Xte, te.targets.numpy().astype(np.int32)


Xtr, ytr, Xte, yte = load()
print(f"[data] {Xtr.shape} -> {Xte.shape}")


def to_patches(X, P):
    """(n,28,28) -> (n, npatch, P*P), non-overlapping."""
    n, g = X.shape[0], 28 // P
    return (X.reshape(n, g, P, g, P).transpose(0, 1, 3, 2, 4)
             .reshape(n, g * g, P * P))


class YatBank(nnx.Module):
    """The series' Yat layer, applied to whatever vector it is handed."""

    def __init__(s, d_in, m, *, rngs):
        s.W = nnx.Param(nnx.initializers.lecun_normal()(rngs.params(), (m, d_in)))
        s.log_b = nnx.Param(jnp.full((), jnp.log(jnp.expm1(0.5))))
        s.log_eps = nnx.Param(jnp.full((), jnp.log(jnp.expm1(0.5))))

    def __call__(s, x):                      # x: (..., d_in)
        b = jax.nn.softplus(s.log_b.value)
        eps = jax.nn.softplus(s.log_eps.value)
        dot = x @ s.W.value.T
        d2 = (jnp.sum(x * x, -1, keepdims=True)
              + jnp.sum(s.W.value ** 2, -1) - 2 * dot)
        return (dot + b) ** 2 / (jnp.maximum(d2, 0.0) + eps)


class PatchNet(nnx.Module):
    """Shared bank over patches, mean pooled, linear readout."""

    def __init__(s, P, m, *, rngs):
        s.P = P
        s.bank = YatBank(P * P, m, rngs=rngs)
        s.head = nnx.Linear(m, 10, rngs=rngs)

    def __call__(s, xp):                     # xp: (B, npatch, P*P)
        f = s.bank(xp)                       # (B, npatch, m): a vote per patch
        return s.head(f.mean(1))

    def patch_votes(s, xp):                  # the exact per-patch ballots
        return s.head(s.bank(xp))


def run(P, seed):
    Ptr, Pte = to_patches(Xtr, P), to_patches(Xte, P)
    npatch = Ptr.shape[1]
    model = PatchNet(P, M, rngs=nnx.Rngs(seed))
    opt = nnx.Optimizer(model, optax.adamw(LR, weight_decay=1e-4), wrt=nnx.Param)

    @nnx.jit
    def step(model, opt, x, y):
        def loss_fn(m_):
            return optax.softmax_cross_entropy_with_integer_labels(m_(x), y).mean()
        loss, g = nnx.value_and_grad(loss_fn)(model)
        opt.update(model, g)
        return loss

    @nnx.jit
    def acc_b(model, x, y):
        return (model(x).argmax(-1) == y).mean()

    def test_acc():
        return 100 * float(np.mean([
            float(acc_b(model, jnp.asarray(Pte[i:i + 1000]), jnp.asarray(yte[i:i + 1000])))
            for i in range(0, len(Pte), 1000)]))

    rng = np.random.default_rng(seed)
    best, curve = 0.0, []
    t0 = time.time()
    for ep in range(EPOCHS):
        perm = rng.permutation(len(Ptr))
        for i in range(0, len(Ptr) - BATCH + 1, BATCH):
            idx = perm[i:i + BATCH]
            step(model, opt, jnp.asarray(Ptr[idx]), jnp.asarray(ytr[idx]))
        a = test_acc()
        best = max(best, a)
        curve.append(round(a, 2))
    out = dict(P=P, npatch=int(npatch), seed=seed, m=M, lr=LR,
               best_acc=best, curve=curve, secs=round(time.time() - t0, 1),
               b=float(jax.nn.softplus(model.bank.log_b.value)),
               eps=float(jax.nn.softplus(model.bank.log_eps.value)))
    # the distance scale is the whole point of the eps audit, so measure it here
    Wv = np.asarray(model.bank.W.value)
    smp = Pte[:2000].reshape(-1, P * P)
    d2 = ((smp ** 2).sum(1, keepdims=True) + (Wv ** 2).sum(1)
          - 2 * smp @ Wv.T)
    out["median_d2"] = float(np.median(np.maximum(d2, 0)))
    out["min_d2"] = float(np.maximum(d2, 0).min())
    out["eps_ratio"] = out["eps"] / out["median_d2"]
    print(f"[P={P} s{seed}] {best:.2f}%  patches={npatch}  b={out['b']:.3f} "
          f"eps={out['eps']:.4f}  eps/median d^2={out['eps_ratio']:.2e} "
          f"({out['secs']}s)", flush=True)
    if seed == 0:
        np.savez_compressed(
            os.path.join(RESULTS_DIR, f"patches_P{P}.npz"),
            W=Wv.astype(np.float32), A=np.asarray(model.head.kernel.value, np.float32),
            bias=np.asarray(model.head.bias.value, np.float32),
            b=out["b"], eps=out["eps"],
            votes=np.asarray(model.patch_votes(jnp.asarray(Pte[:400])), np.float16),
            vote_labels=yte[:400])
    return out


def main():
    rows = [run(P, s) for P in SIZES for s in SEEDS]
    with open(os.path.join(RESULTS_DIR, "yat_patches.json"), "w") as f:
        json.dump(dict(rows=rows, sizes=SIZES, m=M, epochs=EPOCHS), f)
    print()
    for P in SIZES:
        a = [r["best_acc"] for r in rows if r["P"] == P]
        r0 = next(r for r in rows if r["P"] == P)
        print(f"P={P:2d} ({r0['npatch']:2d} patches): {np.mean(a):.2f} +- {np.std(a):.2f}"
              f"   eps/median d^2 = {r0['eps_ratio']:.2e}")


if __name__ == "__main__":
    main()
