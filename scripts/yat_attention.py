"""BxC: Yat attention (the compatibility kernel in the score slot).

why-attention-needs-qk-projections closes on the construction this script
tests: keep the Q/K role asymmetry, but replace the bilinear-then-exp score
with a genuine Mercer kernel BETWEEN the roles:

    s_ij = kappa(f_Q(x_i), f_K(x_j)),
    kappa(a, b) = (a . b)^2 / (||a - b||^2 + eps)      (the Yat kernel)

kappa is nonnegative by construction, so attention needs NO softmax: weights
are kappa / sum(kappa) over the causal window, a true Nadaraya-Watson smoother
(the reading attention-is-a-kernel established). The exp() was only ever there
to make arbitrary bilinear scores positive; a positive-definite kernel arrives
positive.

Existence-proof framing (the Yat program's standing rule): the claim is that a
transformer whose attention scores come from a bounded Mercer kernel trains by
plain gradient descent and lands in the pack on quality, while inheriting what
exp(q.k) cannot give:

  E1  quality: parameter-matched char-level GPTs on Shakespeare, softmax vs
      yat attention, 3 seeds, best-val BPC. Tie or near-tie expected.
  E2  bounded scores: max attention score per layer through training. Softmax
      logits roam (their scale is a free gauge); yat scores are bounded by the
      kernel's geometry.
  E3  the mass signal: sum_j kappa (each query's total kernel mass) is a
      native confidence channel with no analogue under softmax (whose row sums
      are 1 by fiat). AUROC of total mass predicting next-token correctness.

Writes results/yat_attention.json (+ per-variant curves). Kaggle GPU.
"""

import json
import math
import os
import time
import urllib.request

import numpy as np

import jax
import jax.numpy as jnp
import optax
from flax import nnx

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
SMOKE = os.environ.get("SMOKE", "0") == "1"

FULL = dict(D=192, LAYERS=6, HEADS=4, T=128, FF=768, BATCH=64, STEPS=12000,
            LR=3e-4, DROP=0.1, EVAL_EVERY=500, VAL_BATCHES=16)
SMK = dict(D=64, LAYERS=2, HEADS=2, T=64, FF=128, BATCH=16, STEPS=60,
           LR=3e-4, DROP=0.0, EVAL_EVERY=30, VAL_BATCHES=2)
CFG = SMK if SMOKE else FULL
SEEDS = (0,) if (SMOKE or os.environ.get("TELEMETRY", "0") == "1") else (0, 1, 2)
VARIANTS = ("softmax", "yat")
EPS = 1.0

TINY_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_corpus():
    raw = urllib.request.urlopen(TINY_URL, timeout=60).read().decode("utf-8", "ignore")
    return raw


TEXT = load_corpus()
CHARS = sorted(set(TEXT))
V = len(CHARS)
STOI = {c: i for i, c in enumerate(CHARS)}
DATA = np.array([STOI[c] for c in TEXT], dtype=np.int32)
NSPLIT = int(0.9 * len(DATA))
TRAIN, VAL = DATA[:NSPLIT], DATA[NSPLIT:]
print(f"[data] chars={len(DATA):,} vocab={V}")


def get_batch(split, bs, seed):
    d = TRAIN if split == "train" else VAL
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(d) - CFG["T"] - 1, bs)
    x = np.stack([d[i:i + CFG["T"]] for i in ix])
    y = np.stack([d[i + 1:i + CFG["T"] + 1] for i in ix])
    return jnp.asarray(x), jnp.asarray(y)


class Attention(nnx.Module):
    """Causal MHA; `variant` picks the score: exp(bilinear) or the Yat kernel."""

    def __init__(s, variant, dm, heads, drop, *, rngs):
        s.variant, s.h, s.dh = variant, heads, dm // heads
        s.wq = nnx.Linear(dm, dm, rngs=rngs)
        s.wk = nnx.Linear(dm, dm, rngs=rngs)
        s.wv = nnx.Linear(dm, dm, rngs=rngs)
        s.wo = nnx.Linear(dm, dm, rngs=rngs)
        s.drop = nnx.Dropout(drop, rngs=rngs)

    def __call__(s, x, train=False, telemetry=None):
        B, T, D = x.shape
        split = lambda z: z.reshape(B, T, s.h, s.dh).transpose(0, 2, 1, 3)
        q, k, v = split(s.wq(x)), split(s.wk(x)), split(s.wv(x))
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        if s.variant == "softmax":
            logits = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(s.dh)
            logits = jnp.where(mask, logits, -jnp.inf)
            w = jax.nn.softmax(logits, axis=-1)
            if telemetry is not None:
                telemetry["score_max"].append(jnp.max(jnp.where(mask, logits, -jnp.inf)))
        else:
            dots = q @ k.transpose(0, 1, 3, 2)                       # (B,h,T,T)
            d2 = (jnp.sum(q * q, -1, keepdims=True)
                  + jnp.sum(k * k, -1, keepdims=True).swapaxes(-1, -2)
                  - 2 * dots)
            kappa = (dots * dots) / (jnp.maximum(d2, 0.0) + EPS)     # >= 0, Mercer
            kappa = jnp.where(mask, kappa, 0.0)
            mass = kappa.sum(-1, keepdims=True)                       # (B,h,T,1)
            w = kappa / (mass + 1e-9)
            if telemetry is not None:
                telemetry["score_max"].append(jnp.max(kappa))
                telemetry["mass"].append(mass[..., 0])                # (B,h,T)
        if telemetry is not None and "attn" in telemetry:
            telemetry["attn"].append(np.asarray(w[0], dtype=np.float16))  # (h,T,T)
        w = s.drop(w, deterministic=not train)
        out = (w @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
        return s.wo(out)


class Block(nnx.Module):
    def __init__(s, variant, dm, ff, heads, drop, *, rngs):
        s.ln1 = nnx.LayerNorm(dm, rngs=rngs)
        s.attn = Attention(variant, dm, heads, drop, rngs=rngs)
        s.ln2 = nnx.LayerNorm(dm, rngs=rngs)
        s.fc1 = nnx.Linear(dm, ff, rngs=rngs)
        s.fc2 = nnx.Linear(ff, dm, rngs=rngs)
        s.drop = nnx.Dropout(drop, rngs=rngs)

    def __call__(s, x, train=False, telemetry=None):
        x = x + s.attn(s.ln1(x), train=train, telemetry=telemetry)
        h = jax.nn.gelu(s.fc1(s.ln2(x)))
        x = x + s.drop(s.fc2(h), deterministic=not train)
        return x


class GPT(nnx.Module):
    def __init__(s, variant, *, rngs):
        c = CFG
        s.emb = nnx.Embed(V, c["D"], rngs=rngs)
        s.pos = nnx.Embed(c["T"], c["D"], rngs=rngs)
        s.blocks = nnx.data([Block(variant, c["D"], c["FF"], c["HEADS"], c["DROP"], rngs=rngs)
                             for _ in range(c["LAYERS"])])
        s.lnf = nnx.LayerNorm(c["D"], rngs=rngs)
        s.head = nnx.Linear(c["D"], V, rngs=rngs)

    def __call__(s, idx, train=False, telemetry=None):
        B, T = idx.shape
        x = s.emb(idx) + s.pos(jnp.arange(T))
        for b in s.blocks:
            x = b(x, train=train, telemetry=telemetry)
        return s.head(s.lnf(x))


def n_params(model):
    return sum(int(np.prod(v.shape)) for v in jax.tree.leaves(nnx.state(model, nnx.Param)))


def run_variant(variant, seed):
    model = GPT(variant, rngs=nnx.Rngs(seed))
    opt = nnx.Optimizer(model, optax.adamw(CFG["LR"], weight_decay=0.01), wrt=nnx.Param)
    print(f"[{variant} s{seed}] params={n_params(model):,}")

    @nnx.jit
    def train_step(model, opt, x, y):
        def loss_fn(m):
            lg = m(x, train=True)
            return optax.softmax_cross_entropy_with_integer_labels(lg, y).mean()
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        opt.update(model, grads)
        return loss

    @nnx.jit
    def val_loss(model, x, y):
        lg = model(x)
        return optax.softmax_cross_entropy_with_integer_labels(lg, y).mean()

    best_val, curve = float("inf"), []
    t0 = time.time()
    # telemetry mode: snapshot attention on one fixed val window at checkpoints
    TELEM = os.environ.get("TELEMETRY", "0") == "1"
    ckpt_steps = {0, CFG["STEPS"] // 10, CFG["STEPS"] // 2, CFG["STEPS"] - 1}
    snaps = {}
    x_fix, _ = get_batch("val", 1, 424242)

    def snapshot(tag):
        t = {"attn": [], "score_max": [], "mass": []}
        model(x_fix, telemetry=t)
        snaps[tag] = np.stack(t["attn"])          # (layers, heads, T, T) f16

    if TELEM:
        snapshot(0)
    for step in range(CFG["STEPS"]):
        x, y = get_batch("train", CFG["BATCH"], seed * 10_000_000 + step)
        loss = train_step(model, opt, x, y)
        if TELEM and step in ckpt_steps and step > 0:
            snapshot(step)
        if (step + 1) % CFG["EVAL_EVERY"] == 0 or step == CFG["STEPS"] - 1:
            vls = [float(val_loss(model, *get_batch("val", CFG["BATCH"], 77000 + i)))
                   for i in range(CFG["VAL_BATCHES"])]
            vl = float(np.mean(vls))
            best_val = min(best_val, vl)
            curve.append(dict(step=step + 1, val=vl))
            print(f"  [{variant} s{seed}] step {step+1} val {vl:.4f} "
                  f"best {best_val:.4f} ({time.time()-t0:.0f}s)", flush=True)

    # telemetry pass on held-out batches: scores + (yat) mass vs correctness
    tel = {"score_max": [], "mass": []}
    masses, correct = [], []
    for i in range(CFG["VAL_BATCHES"]):
        x, y = get_batch("val", CFG["BATCH"], 88000 + i)
        t = {"score_max": [], "mass": []}
        lg = model(x, telemetry=t)
        pred_ok = np.asarray(lg.argmax(-1) == y)                     # (B,T)
        tel["score_max"].extend([float(v) for v in t["score_max"]])
        if variant == "yat" and t["mass"]:
            # average kernel mass across layers and heads per token
            m_all = np.stack([np.asarray(mm) for mm in t["mass"]])   # (L, B, h, T)
            m_tok = m_all.mean(axis=(0, 2))                          # (B, T)
            masses.append(m_tok.ravel())
            correct.append(pred_ok.ravel())
    out = dict(variant=variant, seed=seed, best_val=best_val,
               bpc=best_val / math.log(2), curve=curve,
               score_max=tel["score_max"])
    if variant == "yat" and masses:
        m = np.concatenate(masses); c = np.concatenate(correct).astype(int)
        # AUROC of mass predicting correctness (rank-based)
        order = np.argsort(m)
        ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(len(m))
        pos = ranks[c == 1]
        auroc = (pos.sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * (c == 0).sum() + 1e-9)
        out["mass_auroc"] = float(auroc)
        out["mass_mean_correct"] = float(m[c == 1].mean())
        out["mass_mean_wrong"] = float(m[c == 0].mean())
        print(f"  [{variant} s{seed}] mass AUROC {auroc:.3f} "
              f"(mean mass correct {m[c==1].mean():.2f} vs wrong {m[c==0].mean():.2f})")
    if snaps:
        np.savez_compressed(os.path.join(RESULTS_DIR, f"attn_{variant}_s{seed}.npz"),
                            tokens=np.asarray(x_fix[0]),
                            **{f"step{k}": v for k, v in snaps.items()})
        # decoded prompt so figures can label the axes with real characters
        out["telemetry_prompt"] = "".join(CHARS[i] for i in np.asarray(x_fix[0]))
    return out


def main():
    t0 = time.time()
    rows = []
    for variant in VARIANTS:
        for seed in SEEDS:
            rows.append(run_variant(variant, seed))
    summary = {}
    for variant in VARIANTS:
        vals = [r["best_val"] for r in rows if r["variant"] == variant]
        summary[variant] = dict(best_val_mean=float(np.mean(vals)),
                                best_val_std=float(np.std(vals)),
                                bpc_mean=float(np.mean(vals) / math.log(2)))
        print(f"{variant}: best-val {np.mean(vals):.4f} +- {np.std(vals):.4f}")
    with open(os.path.join(RESULTS_DIR, "yat_attention.json"), "w") as f:
        json.dump(dict(summary=summary, runs=rows, cfg=CFG), f)
    print(f"done in {time.time()-t0:.0f}s -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
