r"""Is the mixer rank-limited or data-limited? Three questions, frozen weights.

`mixer_diagnose.py` found that the patch embedding followed by a mean over the
sixteen tokens collapses to an effective dimension of about 20 out of 256, and
that each round of mixing pushes that back up (20 -> 50 -> 211) with accuracy
following. That suggests the pooling, not the depth, is the bottleneck. Three
tests, none of which needs retraining:

  1. POOLING     ridge on the mean-pooled 256 against ridge on the sixteen
                 tokens CONCATENATED (16 x 256). If concatenation is far better
                 at r=0, the mean pool is throwing away usable signal and the
                 mixing rounds are buying back what it destroyed. Max-pool and
                 mean+max are included because they cost nothing to try.
  2. WIDTH       ridge accuracy against the number of retained channels. If it
                 is still climbing at 256 the layer wants to be wider; if flat,
                 the channels are redundant and width is not the lever.
                 (A proxy: subsampling trained channels is not the same as
                 training wider, which is why the real sweep is a separate run.)
  3. DATA        ridge accuracy against n, and the fitted exponent. Answers the
                 data question on the architecture that matters rather than on
                 the flat model.

Ridge is solved by one eigendecomposition per feature set, reused across the
whole lambda grid, so the 4096-dimensional concatenation is affordable.

Run: python scripts/mixer_capacity.py
Writes results/mixer_capacity.json
"""

import json
import os

import numpy as np

from export_mixer_viz import Mixer, tokens
from yat_audit import dataset
from yat_diagnose import cka, decay_exponent, eff_dim, onehot

HERE = os.path.dirname(os.path.abspath(__file__))
MIX = os.path.join(HERE, "results", "kgl_blog-mixer-v1")
C = 100
LAMS = np.logspace(-7, 3, 21)
NS = (1000, 2000, 5000, 10000, 25000, 50000)


def gram_chunked(make, X, y, C, chunk=2000):
    """Accumulate mu, X^T X and X^T Y without ever holding X."""
    d = make(X[:2]).shape[1]
    n = len(X)
    ssum = np.zeros(d)
    for i in range(0, n, chunk):
        ssum += make(X[i:i + chunk]).sum(0)
    mu = ssum / n
    G = np.zeros((d, d))
    B = np.zeros((d, C))
    Y = onehot(y, C)
    for i in range(0, n, chunk):
        A = make(X[i:i + chunk]) - mu
        G += A.T @ A
        B += A.T @ Y[i:i + chunk]
    return mu, G, B


def ridge_sweep(make, Xtr, ytr, Xte, yte, C, lams=LAMS, chunk=2000):
    """One eigendecomposition, then every lambda for free."""
    mu, G, B = gram_chunked(make, Xtr, ytr, C, chunk)
    w, V = np.linalg.eigh(G)
    w = np.maximum(w[::-1], 0.0)
    V = V[:, ::-1]
    VB = V.T @ B
    n = len(Xtr)
    best = (-1.0, None)
    for lam in lams:
        W = V @ (VB / (w + lam * n)[:, None])
        hit = 0
        for i in range(0, len(Xte), chunk):
            Z = make(Xte[i:i + chunk]) - mu
            hit += int((np.argmax(Z @ W, 1) == yte[i:i + chunk]).sum())
        acc = 100.0 * hit / len(Xte)
        if acc > best[0]:
            best = (acc, float(lam))
    return best[0], best[1], w


def main():
    Xtr, ytr, Xte, yte, names, coarse = dataset("cifar100g")
    Ttr, Tte = tokens(Xtr), tokens(Xte)
    mx = Mixer(os.path.join(MIX, "mixer_trained_s0.npz"))
    out = {}

    # ── 1. is the mean pool the bottleneck? ──
    print("── 1. pooling: what the mean over sixteen tokens costs ──")
    print(f"  {'readout on':<34} {'r=0':>18}   {'r=4':>18}")
    pools = {
        "mean over tokens (as built)": lambda h: h.mean(1),
        "max over tokens": lambda h: h.max(1),
        "mean and max": lambda h: np.concatenate([h.mean(1), h.max(1)], 1),
        "all 16 tokens concatenated": lambda h: h.reshape(len(h), -1),
    }
    rows = []
    for lab, pool in pools.items():
        cells = []
        for r in (0, 4):
            def make(Xt, pool=pool, r=r):
                h = mx.embed(Xt)
                for _ in range(r):
                    h = mx.block(h)
                return pool(h)
            acc, lam, w = ridge_sweep(make, Ttr, ytr, Tte, yte, C)
            sl, p = decay_exponent(w)
            cells.append(dict(r=r, acc=acc, p=p, dim=int(make(Ttr[:2]).shape[1]),
                              eff=eff_dim(w, 1e-4 * len(Xtr))))
        rows.append(dict(pool=lab, cells=cells))
        f = lambda c: f"{c['acc']:5.2f}% N={c['eff']:6.1f}"
        print(f"  {lab:<34} {f(cells[0]):>18}   {f(cells[1]):>18}")
    out["pooling"] = rows

    # ── 2. width: are the 256 channels redundant? ──
    print("\n── 2. width: ridge against retained channels (r=4, mean pool) ──")
    base = lambda Xt: mx.pooled(Xt)
    Ptr = np.concatenate([base(Ttr[i:i + 2000]) for i in range(0, len(Ttr), 2000)])
    Pte = np.concatenate([base(Tte[i:i + 2000]) for i in range(0, len(Tte), 2000)])
    rng = np.random.default_rng(0)
    wid = []
    for k in (16, 32, 64, 128, 192, 256):
        accs = []
        for rep in range(3 if k < 256 else 1):
            idx = rng.choice(256, k, replace=False) if k < 256 else np.arange(256)
            a, _, _ = ridge_sweep(lambda X, i=idx: X[:, i], Ptr, ytr, Pte, yte, C)
            accs.append(a)
        wid.append(dict(k=k, acc=float(np.mean(accs))))
        print(f"  {k:>4} channels   {np.mean(accs):6.2f}%")
    out["width"] = wid
    gain = wid[-1]["acc"] - wid[-2]["acc"]
    print(f"  last step (192 -> 256) bought {gain:+.2f} points"
          f"  -> {'WIDTH IS STILL PAYING' if gain > 0.4 else 'channels are redundant'}")

    # ── 3. data ──
    print("\n── 3. data: ridge against n (r=4, mean pool) ──")
    curve = []
    for n in NS:
        a, _, _ = ridge_sweep(lambda X: X, Ptr[:n], ytr[:n], Pte, yte, C)
        curve.append(dict(n=n, acc=a))
        print(f"  n={n:>6}   {a:6.2f}%")
    ns = np.array([c["n"] for c in curve], float)
    er = 100.0 - np.array([c["acc"] for c in curve])
    sl = float(np.polyfit(np.log(ns), np.log(er), 1)[0])
    ex = float(100 - np.exp(np.polyval(np.polyfit(np.log(ns), np.log(er), 1), np.log(5e5))))
    last = curve[-1]["acc"] - curve[-2]["acc"]
    out["data"] = dict(curve=curve, slope=sl, extrap_10x=ex, last_gain=last)
    print(f"  error ~ n^{sl:.3f}; last doubling {last:+.2f} pts; 10x data -> {ex:.1f}%")
    print(f"  -> {'DATA-LIMITED' if last > 0.5 else 'saturating in n'}")

    p = os.path.join(HERE, "results", "mixer_capacity.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    print(f"\nwrote {os.path.relpath(p, HERE)}")


if __name__ == "__main__":
    main()
