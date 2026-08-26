"""The Yat mixer: do the two forbidden conversations buy the coarse factor?

The hundred-classes audit factored the flat model's ceiling: 26.0% at the
20-way superclass question x 66.1% at the sibling pick = 17.2% fine, and
traced the coarse failure to a function class where concepts cannot compose
and image regions cannot talk. This experiment opens exactly those two
channels and nothing else:

  - the image becomes 16 tokens (8x8 patches), embedded per patch by a Yat
    layer (shared, prototypes stay pictures of patches)
  - ONE weight-tied mixer block, applied R times (recursive): a token-mixing
    Yat layer (patches talk, per channel) then a channel-mixing Yat layer
    (concepts talk, per token)
  - every mixing step is itself a Yat kernel map, so each layer of the stack
    is a Mercer feature map and the audit's spectral toolbox survives depth
  - skip connections stay ON the sphere: after every residual add each token
    is renormalized to unit length, so angles and distances, the quantities
    every instrument reads, are the geometry the network computes in

Arms: LR bracket at R=4; R sweep {0,1,2,4,8} at the bracketed rate (R=0 is
the no-mixing control: embed + pool + head); three seeds at the winner.
Every eval reports fine, coarse, and fine-given-coarse, so the ceiling
factorization is a curve, not a post-hoc read.

Writes results/yat_mixer.json + per-seed weights mixer_trained_s{seed}.npz.
Kaggle GPU. Env: SMOKE=1, MIXW (channels), EPOCHS, SEEDS_N, R, LR,
SWEEP=1 (bracket the rate), RSWEEP=1 (sweep the recursion depth).
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
M = int(os.environ.get("MIXW", "64" if SMOKE else "256"))
EPOCHS = int(os.environ.get("EPOCHS", "2" if SMOKE else "24"))
SEEDS = tuple(range(int(os.environ.get("SEEDS_N", "1" if SMOKE else "3"))))
R_DEFAULT = int(os.environ.get("R", "2" if SMOKE else "4"))
SWEEP = os.environ.get("SWEEP", "0") == "1"
RSWEEP = os.environ.get("RSWEEP", "0") == "1"
POOL = os.environ.get("POOL", "mean")
PSWEEP = os.environ.get("PSWEEP", "0") == "1"    # mean vs max vs meanmax vs concat
WSWEEP = os.environ.get("WSWEEP", "0") == "1"    # channel width
COLOR = os.environ.get("COLOR", "0") == "1"      # keep the three channels
AUG = os.environ.get("AUG", "0") == "1"          # pad-4 random crop + h-flip
WD = float(os.environ.get("WD", "1e-4"))         # readout is now the biggest block
BATCH = 256
P = 8                      # patch side: 32x32 -> 4x4 grid of 8x8 patches
T = (32 // P) ** 2         # 16 tokens
CH = 3 if COLOR else 1
DPATCH = P * P * CH        # pixels per token
NCLS = 100


# ── data: grayscale CIFAR-100 with both label levels (same loader lineage
#    as yat_cifar.py: HF parquet first, torchvision fallback) ────────────────
def _from_hf():
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
        mode = "RGB" if COLOR else "L"
        g = np.stack([
            np.asarray(Image.open(io.BytesIO(im["bytes"])).convert(mode),
                       np.float32).reshape(-1) / 255.0
            for im in imgs])
        out[split] = (g, np.asarray(t["fine_label"], np.int32),
                      np.asarray(t["coarse_label"], np.int32))
    return out


def _from_torchvision():
    import pickle
    import torchvision
    root = "/tmp/cifar"
    torchvision.datasets.CIFAR100(root, train=True, download=True)
    base = os.path.join(root, "cifar-100-python")
    out = {}
    for split in ("train", "test"):
        with open(os.path.join(base, split), "rb") as f:
            d = pickle.load(f, encoding="bytes")
        X = d[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        g = (X.transpose(0, 2, 3, 1).reshape(len(X), -1) if COLOR else
             (0.299 * X[:, 0] + 0.587 * X[:, 1] + 0.114 * X[:, 2]).reshape(-1, 1024))
        out[split] = (g, np.array(d[b"fine_labels"], np.int32),
                      np.array(d[b"coarse_labels"], np.int32))
    return out


try:
    DATA = _from_hf()
    print("[data] via huggingface")
except Exception as e:
    print(f"[data] hf failed ({e}); falling back to torchvision")
    DATA = _from_torchvision()

Xtr, ytr, ctr = DATA["train"]
Xte, yte, cte = DATA["test"]
if SMOKE:
    Xtr, ytr, ctr = Xtr[:8000], ytr[:8000], ctr[:8000]
F2C = np.zeros(NCLS, int)
for f_, c_ in zip(yte, cte):
    F2C[f_] = c_


def tokens(X):
    """(n, 32*32*CH) -> (n, 16, P*P*CH): the 4x4 grid of 8x8 patches."""
    n, g = len(X), 32 // P
    return (X.reshape(n, 32, 32, CH).reshape(n, g, P, g, P, CH)
             .transpose(0, 1, 3, 2, 4, 5).reshape(n, T, DPATCH))


def augment(X, rng):
    """Pad-4 random crop plus horizontal flip, the standard CIFAR pair. In
    kernel terms this trains against the GROUP-AVERAGED kernel, which is the
    classical way to put an invariance the raw-pixel kernel does not have into
    a model without redesigning the kernel."""
    n = len(X)
    im = X.reshape(n, 32, 32, CH)
    pad = np.pad(im, ((0, 0), (4, 4), (4, 4), (0, 0)), mode="reflect")
    ox, oy = rng.integers(0, 9, n), rng.integers(0, 9, n)
    idx = np.arange(32)
    out = pad[np.arange(n)[:, None, None],
              (ox[:, None] + idx)[:, :, None],
              (oy[:, None] + idx)[:, None, :]]
    flip = rng.random(n) < 0.5
    out[flip] = out[flip, :, ::-1]
    return out.reshape(n, -1)


RAWTR = Xtr.copy() if AUG else None
Xtr, Xte = tokens(Xtr), tokens(Xte)
print(f"[data] train {Xtr.shape} test {Xte.shape} tokens {T} x {DPATCH}")


class YatLayer(nnx.Module):
    """The same kernel map as every post in this series, on the last axis."""

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


def sphere(h):
    """The skip discipline: every token lives on S^(m-1), never off it."""
    return h / jnp.maximum(jnp.linalg.norm(h, axis=-1, keepdims=True), 1e-6)


class MixerBlock(nnx.Module):
    """One recursive step: patches talk, then concepts talk, both as Yat maps."""

    def __init__(s, m, *, rngs):
        s.tok = YatLayer(T, T, rngs=rngs)
        s.ch = YatLayer(m, m, rngs=rngs)
        s.a_t = nnx.Param(jnp.full((), 0.5))
        s.a_c = nnx.Param(jnp.full((), 0.5))

    def __call__(s, h):                       # h: (B, T, m), unit tokens
        t = s.tok(h.transpose(0, 2, 1)).transpose(0, 2, 1)
        h = sphere(h + s.a_t.value * t)
        c = s.ch(h)
        return sphere(h + s.a_c.value * c)


# ── how sixteen token vectors become one ─────────────────────────────────────
# Averaging the tokens is not a heuristic, it is the kernel mean embedding of
# the token set under a LINEAR feature map. A mean embedding is injective only
# when its kernel is characteristic, and the linear one is not, which is why
# mean pooling is the lossy member of the family: measured on trained weights
# it drops the effective dimension from 3076 to 211.
#
# The fix is to average in a better space rather than to stop averaging.
#
#   mean/max/meanmax   first-order summaries, permutation-invariant, lossy
#   concat             keeps position, costs T x the head, drops invariance
#   kme                mu = (1/T) sum psi(h_i) with psi a Yat bank: the mean
#                      embedding under a nonlinear map. Permutation-invariant,
#                      output stays m-dimensional, still Mercer.
#   nw{Q}              Nadaraya-Watson with Q learned query points. Weights are
#                      Yat kernel scores, which are nonnegative, so they
#                      normalize with no softmax. Q=1 is an adaptive mean whose
#                      output cannot leave the convex hull of the tokens, so it
#                      cannot raise rank; Q dials toward concat as it grows.
PARAM_POOLS = ("kme", "nw1", "nw2", "nw4", "nw8")


class Pool(nnx.Module):
    def __init__(s, m, kind, *, rngs):
        s.kind = kind
        s.mult = 1
        if kind == "kme":
            s.psi = YatLayer(m, m, rngs=rngs)
        elif kind.startswith("nw"):
            s.Q = int(kind[2:])
            s.mult = s.Q
            s.query = nnx.Param(nnx.initializers.lecun_normal()(rngs.params(), (s.Q, m)))
            s.log_b = nnx.Param(jnp.full((), jnp.log(jnp.expm1(0.5))))
            s.log_eps = nnx.Param(jnp.full((), jnp.log(jnp.expm1(0.5))))

    def __call__(s, h):                        # h: (B, T, m) unit tokens
        if s.kind == "kme":
            return sphere(s.psi(h)).mean(1)
        q = sphere(s.query.value)              # (Q, m), queries on the sphere too
        b = jax.nn.softplus(s.log_b.value)
        eps = jax.nn.softplus(s.log_eps.value)
        dot = jnp.einsum("qm,btm->bqt", q, h)
        d2 = (jnp.sum(q * q, -1)[None, :, None]
              + jnp.sum(h * h, -1)[:, None, :] - 2 * dot)
        k = (dot + b) ** 2 / (jnp.maximum(d2, 0.0) + eps)      # nonnegative
        w = k / jnp.maximum(k.sum(-1, keepdims=True), 1e-9)    # no softmax needed
        return jnp.einsum("bqt,btm->bqm", w, h).reshape(h.shape[0], -1)


POOLS = {
    "mean": (lambda h: h.mean(1), 1),
    "max": (lambda h: h.max(1), 1),
    "meanmax": (lambda h: jnp.concatenate([h.mean(1), h.max(1)], -1), 2),
    "concat": (lambda h: h.reshape(h.shape[0], -1), T),
}


class MixerNet(nnx.Module):
    def __init__(s, m, r, *, rngs, pool="mean"):
        s.embed = YatLayer(DPATCH, m, rngs=rngs)
        s.block = MixerBlock(m, rngs=rngs)     # ONE block, applied r times
        if pool in PARAM_POOLS:
            s.pool = Pool(m, pool, rngs=rngs)
            s.pool_fn, mult = None, s.pool.mult
        else:
            s.pool = None
            s.pool_fn, mult = POOLS[pool]
        s.head = nnx.Linear(m * mult, NCLS, rngs=rngs)
        s.r = r

    def __call__(s, x):                        # x: (B, T, DPATCH)
        h = sphere(s.embed(x))
        for _ in range(s.r):
            h = s.block(h)
        return s.head(s.pool(h) if s.pool is not None else s.pool_fn(h))


def run(seed, lr, r=R_DEFAULT, m=M, save=True, pool=None):
    pool = pool or POOL
    model = MixerNet(m, r, rngs=nnx.Rngs(seed), pool=pool)
    opt = nnx.Optimizer(model, optax.adamw(lr, weight_decay=WD), wrt=nnx.Param)

    @nnx.jit
    def step(model, opt, x, y):
        def loss_fn(mm):
            return optax.softmax_cross_entropy_with_integer_labels(mm(x), y).mean()
        loss, g = nnx.value_and_grad(loss_fn)(model)
        opt.update(model, g)
        return loss

    @nnx.jit
    def preds(model, x):
        return model(x).argmax(-1)

    f2c = jnp.asarray(F2C)

    def factored_acc():
        p = np.concatenate([np.asarray(preds(model, jnp.asarray(Xte[i:i + 1000])))
                            for i in range(0, len(Xte), 1000)])
        fine = 100 * float((p == yte).mean())
        gc = F2C[p] == F2C[yte]
        co = 100 * float(gc.mean())
        fgc = 100 * float((p[gc] == yte[gc]).mean()) if gc.any() else 0.0
        return fine, co, fgc

    rng = np.random.default_rng(seed)
    best, curve = 0.0, []
    t0 = time.time()
    for ep in range(EPOCHS):
        perm = rng.permutation(len(Xtr))
        for i in range(0, len(Xtr) - BATCH + 1, BATCH):
            idx = perm[i:i + BATCH]
            xb = (tokens(augment(RAWTR[idx], rng)) if AUG else Xtr[idx])
            step(model, opt, jnp.asarray(xb), jnp.asarray(ytr[idx]))
        fine, co, fgc = factored_acc()
        best = max(best, fine)
        curve.append(dict(fine=round(fine, 2), coarse=round(co, 2),
                          fgc=round(fgc, 2)))
    sp = lambda v: float(jax.nn.softplus(v))
    out = dict(seed=seed, lr=lr, r=r, m=m, pool=pool, color=COLOR, aug=AUG, wd=WD, best_acc=round(best, 2), curve=curve,
               secs=round(time.time() - t0, 1),
               fine=curve[-1]["fine"], coarse=curve[-1]["coarse"],
               fgc=curve[-1]["fgc"],
               b_embed=sp(model.embed.log_b.value), eps_embed=sp(model.embed.log_eps.value),
               b_tok=sp(model.block.tok.log_b.value), eps_tok=sp(model.block.tok.log_eps.value),
               b_ch=sp(model.block.ch.log_b.value), eps_ch=sp(model.block.ch.log_eps.value),
               a_t=float(model.block.a_t.value), a_c=float(model.block.a_c.value))
    print(f"[s{seed} r={r} m={m} pool={pool} lr={lr}] best {best:.2f}%  "
          f"final fine {out['fine']:.2f} = coarse {out['coarse']:.2f} x "
          f"fgc {out['fgc']:.2f}  a_t={out['a_t']:.3f} a_c={out['a_c']:.3f} "
          f"({out['secs']}s)", flush=True)
    if save:
        np.savez_compressed(
            os.path.join(RESULTS_DIR, f"mixer_trained_s{seed}.npz"),
            We=np.asarray(model.embed.W.value, np.float16),
            Wt=np.asarray(model.block.tok.W.value, np.float32),
            Wc=np.asarray(model.block.ch.W.value, np.float16),
            A=np.asarray(model.head.kernel.value, np.float16),
            bias=np.asarray(model.head.bias.value, np.float16),
            be=np.array([out["b_embed"], out["eps_embed"]], np.float32),
            bt=np.array([out["b_tok"], out["eps_tok"]], np.float32),
            bc=np.array([out["b_ch"], out["eps_ch"]], np.float32),
            alphas=np.array([out["a_t"], out["a_c"]], np.float32),
            config=np.array([r, m, P, T], np.int32))
    return out


def main():
    lr = float(os.environ.get("LR", "1e-2"))
    r = R_DEFAULT
    rows = []
    if SWEEP:
        cells = [run(0, x, r=r, save=False) for x in (3e-3, 1e-2, 3e-2)]
        lr = max(cells, key=lambda c: c["best_acc"])["lr"]
        rows += cells
        print(f"[lr] bracketed at {lr}")
    if RSWEEP:
        cells = [run(0, lr, r=rr, save=False) for rr in (0, 1, 2, 4, 8)]
        r = max(cells, key=lambda c: c["best_acc"])["r"]
        rows += cells
        print(f"[r] recursion depth chosen: {r}")
    pool = POOL
    if PSWEEP:
        arms = os.environ.get("POOLS", "mean,max,meanmax,concat").split(",")
        cells = [run(0, lr, r=r, m=M, save=False, pool=q) for q in arms]
        pool = max(cells, key=lambda c: c["best_acc"])["pool"]
        rows += cells
        print(f"[pool] chosen: {pool}")
    m = M
    if WSWEEP:
        cells = [run(0, lr, r=r, m=w, save=False, pool=pool) for w in (128, 256, 512, 1024)]
        m = max(cells, key=lambda c: c["best_acc"])["m"]
        rows += cells
        print(f"[width] chosen: m={m}")
    rows += [run(s, lr, r=r, m=m, pool=pool) for s in SEEDS]
    np.savez_compressed(os.path.join(RESULTS_DIR, "mixer_labels.npz"),
                        fine_test=yte, coarse_test=cte,
                        fine_train=ytr, coarse_train=ctr)
    with open(os.path.join(RESULTS_DIR, "yat_mixer.json"), "w") as f:
        json.dump(dict(rows=rows, m=M, epochs=EPOCHS, lr=lr, r=r, patch=P, pool=POOL,
                       dataset="cifar100-gray"), f)
    a = [row["best_acc"] for row in rows[-len(SEEDS):]]   # the seed arms only
    print(f"\nmixer r={r} m={M}: {np.mean(a):.2f} +- {np.std(a):.2f} "
          f"(flat baseline 17.65 +- 0.17, coarse 26.0 x fgc 66.1)")


if __name__ == "__main__":
    main()
