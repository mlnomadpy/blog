r"""Why is it stuck at 17 percent? A kernel-theory diagnosis, not an audit.

Four things can hold a kernel machine down, and classical theory gives each one
a different signature and a different fix:

  optimization   the readout SGD found is worse than the one the SAME features
                 admit. Signature: ridge on the frozen features beats the
                 trained model. Fix: train the head properly, change nothing.
  estimation     not enough data for this capacity. Signature: the learning
                 curve in n is still climbing at the full training set. Theory
                 (ch07): excess risk ~ n^{-2r/(2r+p)}; fit the exponent.
  approximation  the target is not well expressed by this kernel's leading
                 eigenfunctions. Signature: low centered kernel-target
                 alignment, fast spectral decay, small effective dimension.
  representation the kernel is on the wrong space. Signature: the Yat layer
                 does not beat a plain ridge on its own input, and a better
                 input beats both. Fix: compose, k(phi(x), phi(x')) is still
                 Mercer for any phi.

Everything below is a linear solve on frozen features, so the whole diagnosis
runs in a couple of minutes and needs no retraining. Ridge gives the BEST head
the features admit, which is the ceiling any readout could reach; comparing it
against the trained model separates optimization from everything else.

Run: python scripts/yat_diagnose.py
Writes results/yat_diagnose.json
"""

import json
import os

import numpy as np

from yat_audit import Model, dataset, load_models
from yat_forecast import features_chunked

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.join(HERE, "results", "kgl_blog-cifar-v1")
NS = (1000, 2000, 5000, 10000, 25000, 50000)
LAMS = np.logspace(-6, 3, 19)


def onehot(y, C):
    Y = np.zeros((len(y), C))
    Y[np.arange(len(y)), y] = 1.0
    return Y - 1.0 / C                      # centered targets


def ridge_acc(Xtr, ytr, Xte, yte, C, lams=LAMS):
    """Best linear readout the features admit, with lambda swept. Returns the
    top test accuracy and the lambda that got it. Solved in feature space, so
    the cost is m^3 and not n^3."""
    mu = Xtr.mean(0)
    A = Xtr - mu
    G = A.T @ A
    B = A.T @ onehot(ytr, C)
    Zte = Xte - mu
    best = (-1.0, None)
    ev = np.linalg.eigvalsh(G)
    for lam in lams:
        W = np.linalg.solve(G + lam * len(A) * np.eye(len(G)), B)
        acc = 100.0 * float((np.argmax(Zte @ W, 1) == yte).mean())
        if acc > best[0]:
            best = (acc, float(lam))
    return best[0], best[1], ev[::-1]


def decay_exponent(ev, lo=1, hi=None):
    """Fit lambda_k ~ k^{-1/p} on the log-log slope. p is the capacity exponent
    of ch07: bigger p means a slower decay, a richer kernel, and a slower rate."""
    ev = np.maximum(ev, 1e-30)
    hi = hi or min(len(ev), 200)
    k = np.arange(lo, hi) + 1.0
    s = np.polyfit(np.log(k), np.log(ev[lo:hi]), 1)[0]
    return float(-s), float(-1.0 / s) if s < 0 else float("inf")


def eff_dim(ev, lam):
    return float((ev / (ev + lam)).sum())


def cka(X, y, C):
    """Centered kernel-target alignment (ch12). How much of the label kernel
    this feature map's Gram can express. HKH centering matters: uncentered
    alignment ranks a perfect kernel below an imperfect one."""
    A = X - X.mean(0)
    Y = onehot(y, C)
    Y = Y - Y.mean(0)
    num = np.linalg.norm(A.T @ Y, "fro") ** 2
    den = np.linalg.norm(A.T @ A, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    return float(num / max(den, 1e-30))


def main():
    Xtr, ytr, Xte, yte, names, coarse = dataset("cifar100g")
    C = 100
    rep = {}
    print(f"CIFAR-100 grayscale, {len(Xtr)} train / {len(Xte)} test, {C} classes\n")

    models = load_models(BUNDLE, "trained", 12)
    seed, M = models[0]
    print(f"the model under diagnosis: one Yat bank, m={M.m}, on raw {M.d}-d pixels")

    F = features_chunked(M, Xtr)
    Fte = features_chunked(M, Xte)
    trained = 100.0 * float((np.argmax(Fte @ M.A + M.bias, 1) == yte).mean())

    # ── 1. optimization: is the trained head the best this bank admits? ──
    acc, lam, ev = ridge_acc(F, ytr, Fte, yte, C)
    p_slope, p_cap = decay_exponent(ev)
    rep["optimization"] = dict(trained=trained, ridge_ceiling=acc, lam=lam)
    print("\n── 1. optimization ──")
    print(f"  trained readout (SGD)          {trained:.2f}%")
    print(f"  best ridge head, same features {acc:.2f}%   (lambda {lam:.2g})")
    print(f"  -> {'THE HEAD IS UNDERTRAINED' if acc > trained + 1 else 'the head is at the ceiling of these features'}")

    # ── 2. approximation: what do the features carry? ──
    al_yat = cka(F, ytr, C)
    al_pix = cka(Xtr, ytr, C)
    rep["approximation"] = dict(cka_yat=al_yat, cka_pixels=al_pix,
                                decay_exponent=p_slope, capacity_p=p_cap,
                                eff_dim={str(l): eff_dim(ev, l * len(F))
                                         for l in (1e-6, 1e-4, 1e-2)},
                                top_eigen_share=float(ev[:15].sum() / ev.sum()))
    print("\n── 2. approximation (what the kernel can express) ──")
    print(f"  centered kernel-target alignment, Yat features {al_yat:.4f}")
    print(f"  same for raw pixels                            {al_pix:.4f}")
    print(f"  spectral decay lambda_k ~ k^-{p_slope:.2f}  (capacity exponent p = {p_cap:.2f})")
    print(f"  effective dimension N(lam): " + "  ".join(
        f"lam={l:g}: {eff_dim(ev, l*len(F)):.0f}" for l in (1e-6, 1e-4, 1e-2)))
    print(f"  top-15 eigenvalues hold {100*ev[:15].sum()/ev.sum():.1f}% of the trace")

    # ── 3. estimation: is it data? ──
    print("\n── 3. estimation (is it data?) ──")
    curve = []
    for n in NS:
        if n > len(F):
            continue
        a, _, _ = ridge_acc(F[:n], ytr[:n], Fte, yte, C)
        curve.append(dict(n=n, acc=a))
        print(f"  n={n:>6}  ridge {a:.2f}%")
    ns = np.array([c["n"] for c in curve], float)
    accs = np.array([c["acc"] for c in curve])
    err = 100.0 - accs
    sl = np.polyfit(np.log(ns), np.log(err), 1)[0]
    last = accs[-1] - accs[-2]
    rep["estimation"] = dict(curve=curve, err_slope=float(sl), last_gain=float(last),
                             extrap_10x=float(100 - np.exp(np.polyval(
                                 np.polyfit(np.log(ns), np.log(err), 1), np.log(ns[-1] * 10)))))
    print(f"  error ~ n^{sl:.3f}; last doubling bought {last:+.2f} points")
    print(f"  extrapolating 10x more data: {rep['estimation']['extrap_10x']:.1f}%")
    print(f"  -> {'DATA-LIMITED' if last > 0.5 else 'SATURATED: more data of this kind will not help'}")

    # ── 4. representation: is the kernel on the wrong space? ──
    print("\n── 4. representation (is the input the problem?) ──")
    rows = {}
    a_pix, _, ev_pix = ridge_acc(Xtr, ytr, Xte, yte, C)
    rows["raw grayscale pixels"] = a_pix
    rows[f"Yat bank m={M.m} on those pixels"] = acc

    # colour, to price what grayscale threw away
    try:
        import torchvision
        import pickle
        root = "/tmp/cifar"
        torchvision.datasets.CIFAR100(root, train=True, download=True)
        base = os.path.join(root, "cifar-100-python")
        got = {}
        for split in ("train", "test"):
            with open(os.path.join(base, split), "rb") as fh:
                d = pickle.load(fh, encoding="bytes")
            got[split] = (d[b"data"].astype(np.float32) / 255.0,
                          np.array(d[b"fine_labels"]))
        a_col, _, _ = ridge_acc(got["train"][0], got["train"][1],
                                got["test"][0], got["test"][1], C)
        rows["raw COLOUR pixels"] = a_col
    except Exception as e:
        print(f"  (colour check skipped: {e})")

    # a fixed feature map in front of the kernel: still Mercer, different space
    rng = np.random.default_rng(0)
    W = rng.normal(size=(M.d, 2048)) / np.sqrt(M.d)
    relu = lambda X: np.maximum(features_chunked_mm(X, W), 0)
    a_rf, _, _ = ridge_acc(relu(Xtr), ytr, relu(Xte), yte, C)
    rows["random ReLU features (2048)"] = a_rf

    for k, v in rows.items():
        print(f"  {k:<34} {v:6.2f}%")
    rep["representation"] = rows

    p = os.path.join(HERE, "results", "yat_diagnose.json")
    json.dump(rep, open(p, "w"), indent=1, default=float)
    print(f"\nwrote {os.path.relpath(p, HERE)}")


def features_chunked_mm(X, W, chunk=5000):
    return np.concatenate([X[i:i + chunk] @ W for i in range(0, len(X), chunk)])


if __name__ == "__main__":
    main()
