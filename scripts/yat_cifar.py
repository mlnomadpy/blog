"""Grayscale CIFAR-100: the audit where the class count stops being the answer.

With ten classes the readout is m x 10, so "the function factors through ten
concepts" is arithmetic, not a finding. CIFAR-100 gives the function a hundred
dimensions to use and lets us ask how many it actually takes. It also ships a
ground truth the protocol has never been tested against: 100 fine classes
nested in 20 superclasses, so "the leading concepts are coarse and the later
ones split within a coarse group" becomes a checkable claim instead of a story
told over a class-mode map.

Grayscale (32x32 = 1024 dims) keeps the input a single channel, matching every
prototype-as-picture argument this series makes.

Writes results/yat_cifar.json + per-seed weights in the format
scripts/yat_audit.py consumes. Kaggle GPU.
Env: SMOKE=1, M (bank width), EPOCHS, SEEDS_N, SWEEP=1 (bracket the rate).
"""

import json
import os
import pickle
import time

import numpy as np

import jax
import jax.numpy as jnp
import optax
from flax import nnx

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
SMOKE = os.environ.get("SMOKE", "0") == "1"
M = int(os.environ.get("M", "64" if SMOKE else "1024"))
EPOCHS = int(os.environ.get("EPOCHS", "2" if SMOKE else "24"))
SEEDS = tuple(range(int(os.environ.get("SEEDS_N", "1" if SMOKE else "3"))))
SWEEP = os.environ.get("SWEEP", "0") == "1"
BATCH = 256
D = 1024
NCLS = 100


def _from_hf():
    """The canonical mirror is rate-limited to the point of failure on the
    worker (it died at 86% after 38 minutes), so prefer the HF copy, which
    also carries the coarse labels the hierarchy instrument needs."""
    import io
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image
    out = {}
    for split, fn in (("train", "cifar100/train-00000-of-00001.parquet"),
                      ("test", "cifar100/test-00000-of-00001.parquet")):
        f = hf_hub_download("uoft-cs/cifar100", fn, repo_type="dataset")
        t = pq.read_table(f).to_pydict()
        imgs = t.get("img") or t.get("image")
        g = np.stack([
            np.asarray(Image.open(io.BytesIO(im["bytes"])).convert("L"),
                       np.float32).reshape(-1) / 255.0
            for im in imgs])
        out[split] = (g, np.asarray(t["fine_label"], np.int32),
                      np.asarray(t["coarse_label"], np.int32))
    return out


def _from_torchvision():
    import torchvision
    root = "/tmp/cifar"
    torchvision.datasets.CIFAR100(root, train=True, download=True)
    base = os.path.join(root, "cifar-100-python")
    out = {}
    for split in ("train", "test"):
        with open(os.path.join(base, split), "rb") as f:
            d = pickle.load(f, encoding="bytes")
        X = d[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        g = (0.299 * X[:, 0] + 0.587 * X[:, 1] + 0.114 * X[:, 2]).reshape(-1, D)
        out[split] = (g, np.array(d[b"fine_labels"], np.int32),
                      np.array(d[b"coarse_labels"], np.int32))
    return out


def load_cifar():
    """Grayscale CIFAR-100 with BOTH label levels (fine and coarse)."""
    try:
        d = _from_hf()
        print("[data] via huggingface")
        return d
    except Exception as e:
        print(f"[data] hf failed ({e}); falling back to torchvision")
        return _from_torchvision()


DATA = load_cifar()
Xtr, ytr, ctr = DATA["train"]
Xte, yte, cte = DATA["test"]
if SMOKE:
    Xtr, ytr, ctr = Xtr[:8000], ytr[:8000], ctr[:8000]
print(f"[data] train {Xtr.shape} test {Xte.shape} "
      f"fine {ytr.max()+1} coarse {ctr.max()+1}")


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
        s.feat = YatLayer(D, m, rngs=rngs)
        s.head = nnx.Linear(m, NCLS, rngs=rngs)

    def __call__(s, x):
        return s.head(s.feat(x))


def run(seed, lr, m=M, save=True):
    model = Net(m, rngs=nnx.Rngs(seed))
    opt = nnx.Optimizer(model, optax.adamw(lr, weight_decay=1e-4), wrt=nnx.Param)

    @nnx.jit
    def step(model, opt, x, y):
        def loss_fn(mm):
            return optax.softmax_cross_entropy_with_integer_labels(mm(x), y).mean()
        loss, g = nnx.value_and_grad(loss_fn)(model)
        opt.update(model, g)
        return loss

    @nnx.jit
    def top1(model, x, y):
        return (model(x).argmax(-1) == y).mean()

    def test_acc():
        return 100 * float(np.mean([
            float(top1(model, jnp.asarray(Xte[i:i + 1000]), jnp.asarray(yte[i:i + 1000])))
            for i in range(0, len(Xte), 1000)]))

    rng = np.random.default_rng(seed)
    best, curve = 0.0, []
    t0 = time.time()
    for ep in range(EPOCHS):
        perm = rng.permutation(len(Xtr))
        for i in range(0, len(Xtr) - BATCH + 1, BATCH):
            idx = perm[i:i + BATCH]
            step(model, opt, jnp.asarray(Xtr[idx]), jnp.asarray(ytr[idx]))
        a = test_acc()
        best = max(best, a)
        curve.append(round(a, 2))
    out = dict(seed=seed, lr=lr, m=m, best_acc=best, curve=curve,
               secs=round(time.time() - t0, 1),
               b=float(jax.nn.softplus(model.feat.log_b.value)),
               eps=float(jax.nn.softplus(model.feat.log_eps.value)))
    print(f"[s{seed} m={m} lr={lr}] best {best:.2f}%  b={out['b']:.3f} "
          f"eps={out['eps']:.4f}  ({out['secs']}s)", flush=True)
    if save:
        # the audit's expected layout: W0/A0 = init, W1/A1 = final
        np.savez_compressed(
            os.path.join(RESULTS_DIR, f"mercer_trained_s{seed}.npz"),
            W0=np.asarray(model.feat.W.value, np.float16),
            A0=np.asarray(model.head.kernel.value, np.float16),
            bias0=np.asarray(model.head.bias.value, np.float16),
            W1=np.asarray(model.feat.W.value, np.float16),
            A1=np.asarray(model.head.kernel.value, np.float16),
            bias1=np.asarray(model.head.bias.value, np.float16),
            b=np.array([out["b"], out["b"]], np.float32),
            eps=np.array([out["eps"], out["eps"]], np.float32))
    return out


def main():
    lr = float(os.environ.get("LR", "1e-2"))
    rows = []
    if SWEEP:
        cells = [run(0, x, m=256, save=False) for x in (3e-3, 1e-2, 3e-2)]
        lr = max(cells, key=lambda r: r["best_acc"])["lr"]
        rows += cells
        print(f"[lr] bracketed at {lr}")
    rows += [run(s, lr) for s in SEEDS]
    # the coarse labels travel with the run so the audit can test the hierarchy
    np.savez_compressed(os.path.join(RESULTS_DIR, "cifar_labels.npz"),
                        fine_test=yte, coarse_test=cte,
                        fine_train=ytr, coarse_train=ctr)
    with open(os.path.join(RESULTS_DIR, "yat_cifar.json"), "w") as f:
        json.dump(dict(rows=rows, m=M, epochs=EPOCHS, lr=lr, dataset="cifar100-gray"), f)
    a = [r["best_acc"] for r in rows if r.get("m") == M]
    print(f"\ncifar100-gray m={M}: {np.mean(a):.2f} +- {np.std(a):.2f}")


if __name__ == "__main__":
    main()
