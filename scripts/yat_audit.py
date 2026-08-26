r"""The Yat audit, v2: an instrument, not a report.

v1 (yat_protocol.py) printed numbers from one seed with nothing to compare them
against. A reading is only evidence if you know three things about it: how much
it moves across seeds, what it would have been by chance, and whether the data
can resolve it at all. v2 enforces that for every instrument.

  every reading carries      value +- spread over seeds, a NULL from a matched
                             random control, and a VERDICT against a threshold
  concepts are gated         a concept is named only if its eigengap survives
                             bootstrap resampling; otherwise it is reported as
                             an unresolved rotation and NOT interpreted
  claims need two methods    the channel ledger runs two independent
                             decompositions and reports their agreement; a
                             single-method split is marked PROVISIONAL
  the measure is swept       concepts are recomputed under train / test /
                             class-balanced measures and compared by principal
                             angle, because spectral coordinates are
                             measure-dependent and conclusions should not be
  the write path is tested   deleting a concept's support rows is followed by a
                             REFIT, so "removed" is distinguished from "hidden"

Model-agnostic: anything that can hand back (W, b, eps, A, bias) and a
featurizer runs, so whole-image and patch networks go through identical code.

Run: python scripts/yat_audit.py [--bundle DIR] [--arm trained] [--patches P]
Writes results/yat_audit_<tag>.json
"""

import argparse
import glob
import json
import os

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
CLS = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
       "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]
# Instruments take their own generator. A module-global one was shared across
# every seed and every instrument, so two models drew correlated bootstraps and
# the seed spread understated itself; main() now hands each model its own.
RNG = np.random.default_rng(0)


# ══ the model under audit ════════════════════════════════════════════════════
class Model:
    """A Yat bank + linear readout. `P` patches the input, so the same audit
    runs on whole-image and patch networks without a second implementation."""

    def __init__(s, W, b, eps, A, bias, P=None):
        s.W, s.b, s.eps, s.A, s.bias = W, b, eps, A, bias
        s.m, s.d = W.shape
        s.C = A.shape[1]
        s.P = P

    def _patch(s, X):
        if s.P is None:
            return X[:, None, :]
        n, g = len(X), 28 // s.P
        return (X.reshape(n, 28, 28).reshape(n, g, s.P, g, s.P)
                 .transpose(0, 1, 3, 2, 4).reshape(n, g * g, s.P * s.P))

    def channels(s, X, eps=None, b=None):
        """alignment and proximity, per patch, kept apart."""
        eps = s.eps if eps is None else eps
        b = s.b if b is None else b
        xp = s._patch(X)
        n, np_, d = xp.shape
        flat = xp.reshape(-1, d)
        dot = flat @ s.W.T
        d2 = np.maximum((flat ** 2).sum(1, keepdims=True) + (s.W ** 2).sum(1) - 2 * dot, 0)
        return ((dot + b) ** 2).reshape(n, np_, -1), (1.0 / (d2 + eps)).reshape(n, np_, -1), d2

    def features(s, X, eps=None, b=None):
        N, Pi, _ = s.channels(X, eps, b)
        return (N * Pi).mean(1)                      # mean pooling (identity if P is None)


def load_models(bundle, arm, epoch, P=None):
    """Every seed present in the bundle."""
    out = []
    hit = lambda pat: sorted(glob.glob(os.path.join(bundle, pat)) +
                             glob.glob(os.path.join(bundle, "**", pat), recursive=True))
    if P is not None:
        f = hit(f"patches_P{P}.npz")[0]
        z = np.load(f)
        out.append((0, Model(z["W"].astype(np.float64), float(z["b"]), float(z["eps"]),
                             z["A"].astype(np.float64), z["bias"].astype(np.float64), P)))
        return out
    seen = set()
    for f in hit(f"mercer_{arm}_s*.npz"):
        if os.path.basename(f) in seen:
            continue
        seen.add(os.path.basename(f))
        seed = int(f.rsplit("_s", 1)[1].split(".")[0])
        z = np.load(f)
        e = epoch if f"W{epoch}" in z else max(
            int(k[1:]) for k in z.files if k.startswith("W") and k[1:].isdigit())
        out.append((seed, Model(z[f"W{e}"].astype(np.float64), float(z["b"][min(e, len(z["b"]) - 1)]),
                                float(z["eps"][min(e, len(z["eps"]) - 1)]),
                                z[f"A{e}"].astype(np.float64), z[f"bias{e}"].astype(np.float64))))
    return out


def dataset(name):
    """Returns Xtr, ytr, Xte, yte, names, coarse (or None). Any dataset that
    can answer this runs through the whole audit unchanged."""
    import torchvision
    if name == "fmnist":
        tr = torchvision.datasets.FashionMNIST("/tmp/fmnist", train=True, download=True)
        te = torchvision.datasets.FashionMNIST("/tmp/fmnist", train=False, download=True)
        f = lambda ds: (ds.data.numpy().astype(np.float32) / 255).reshape(-1, 784)
        return f(tr), tr.targets.numpy(), f(te), te.targets.numpy(), CLS, None
    if name == "cifar100g":
        import pickle
        try:                      # the fast mirror; the canonical one stalls
            import io
            import pyarrow.parquet as pq
            from huggingface_hub import hf_hub_download
            from PIL import Image
            got = {}
            for split in ("train", "test"):
                f = hf_hub_download("uoft-cs/cifar100",
                                    f"cifar100/{split}-00000-of-00001.parquet",
                                    repo_type="dataset")
                t = pq.read_table(f).to_pydict()
                imgs = t.get("img") or t.get("image")
                g = np.stack([np.asarray(Image.open(io.BytesIO(im["bytes"])).convert("L"),
                                         np.float32).reshape(-1) / 255.0 for im in imgs])
                got[split] = (g, np.asarray(t["fine_label"]), np.asarray(t["coarse_label"]))
            names = [f"c{i}" for i in range(100)]
            return (got["train"][0], got["train"][1], got["test"][0], got["test"][1],
                    names, (got["train"][2], got["test"][2]))
        except Exception as e:
            print(f"[data] hf failed ({e}); falling back")
        root = "/tmp/cifar"
        torchvision.datasets.CIFAR100(root, train=True, download=True)
        base = os.path.join(root, "cifar-100-python")
        got = {}
        for split in ("train", "test"):
            with open(os.path.join(base, split), "rb") as fh:
                d = pickle.load(fh, encoding="bytes")
            X = d[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
            g = (0.299 * X[:, 0] + 0.587 * X[:, 1] + 0.114 * X[:, 2]).reshape(-1, 1024)
            got[split] = (g, np.array(d[b"fine_labels"]), np.array(d[b"coarse_labels"]))
        with open(os.path.join(base, "meta"), "rb") as fh:
            meta = pickle.load(fh, encoding="bytes")
        names = [n.decode() for n in meta[b"fine_label_names"]]
        return (got["train"][0], got["train"][1], got["test"][0], got["test"][1],
                names, (got["train"][2], got["test"][2]))
    raise ValueError(name)


# ══ shared machinery ═════════════════════════════════════════════════════════
def concept_basis(M, Z):
    """Eigendecompose the LOGIT covariance: the coordinates the verdict uses."""
    C = Z.T @ Z / len(Z)
    S = M.A.T @ C @ M.A
    sig, E = np.linalg.eigh(S)
    return np.maximum(sig[::-1], 0), E[:, ::-1], C


def principal_angles(U, V):
    """Cosines of the principal angles between two subspaces, descending.

    U and V are expected to have orthonormal columns, which every basis coming
    out of `eigh` does. cos = 1 means the subspaces share that direction
    exactly; cos = 0 means it is orthogonal to everything on the other side.
    """
    return np.clip(np.linalg.svd(U.T @ V, compute_uv=False), 0.0, 1.0)


def subspace_agreement(U, V):
    """Mean principal-angle cosine: 1 if the two spans coincide, however each
    one happens to be rotated or ordered inside itself."""
    return float(principal_angles(U, V).mean())


def matched_agreement(U, V):
    """Per-axis agreement AFTER solving the assignment, so that an eigenvalue
    rank swap is not scored as a lost concept. Compare against the raw
    column-to-column number to see how much of a stability reading was really
    about ordering."""
    Cm = np.abs(U.T @ V)
    r, c = linear_sum_assignment(-Cm)
    return float(Cm[r, c].mean())


def summarize(vals):
    v = np.asarray(vals, float)
    return dict(mean=float(v.mean()), sd=float(v.std()), n=int(v.size))


def verdict(value, null, sd, threshold, above, labels):
    """A reading counts only if it clears its null by more than the seed spread."""
    margin = abs(value - null) / max(sd, 1e-9)
    hit = (value > threshold) if above else (value < threshold)
    if margin < 2:
        return f"INCONCLUSIVE (within {margin:.1f} sd of the null)"
    return labels[0] if hit else labels[1]


# ══ instruments ══════════════════════════════════════════════════════════════
def i_concepts(M, Ztr, Zte, yte, base, nboot=32, rng=None, ks=(5, 10, 20, 40)):
    """Y1 + the identifiability gate, per axis AND per subspace.

    The per-axis gate compares bootstrap column j to column j. That is a test of
    the axis and of its rank at once: if two neighbouring eigenvalues trade
    places under resampling, both columns score near zero even though the plane
    they span never moved. So this also reports the assignment-solved version of
    the same number, and the principal angles of the leading subspaces, which no
    amount of rotation or reordering inside them can disturb. Any gap between
    `stability` and `stability_matched` is ordering, not instability.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    sig, E, C = concept_basis(M, Ztr)
    acc = lambda P: 100.0 * float((((Zte @ M.A) @ P + base).argmax(1) == yte).mean())
    curve = [acc(E[:, :k] @ E[:, :k].T) for k in range(1, M.C + 1)]
    # bootstrap the measure: can the data resolve each axis, or is it a rotation?
    cos = np.zeros(M.C)
    matched = np.zeros(M.C)
    sub = {k: [] for k in ks if k <= M.C}
    dS = []
    n = len(Ztr)
    S0 = M.A.T @ (Ztr.T @ Ztr / n) @ M.A
    for _ in range(nboot):
        idx = rng.integers(0, n, n)
        Zb = Ztr[idx]
        _, Eb, _ = concept_basis(M, Zb)
        cos += np.abs(np.sum(E * Eb, 0))
        Cm = np.abs(E.T @ Eb)
        r, c = linear_sum_assignment(-Cm)
        matched[r] += Cm[r, c]
        for k in sub:
            sub[k].append(subspace_agreement(E[:, :k], Eb[:, :k]))
        Sb = M.A.T @ (Zb.T @ Zb / n) @ M.A
        dS.append(float(np.linalg.norm(Sb - S0, 2)))
    cos /= nboot
    matched /= nboot
    gap = np.diff(np.concatenate([sig, [0.0]])) / max(sig.sum(), 1e-30)
    return dict(sig=sig, E=E, C=C, curve=curve, stability=cos,
                stability_matched=matched,
                subspace={k: summarize(v) for k, v in sub.items()},
                perturbation=summarize(dS),
                gap=np.abs(gap), named=[bool(c > 0.9) for c in cos],
                named_matched=[bool(c > 0.9) for c in matched])


def i_channels(M, Xtr, Xte, E, nnull=16, rng=None):
    """Y2 by TWO independent methods, plus a random-concept null.

    method A: freeze one factor at its per-unit mean and re-score
    method B: shuffle one factor across samples (breaks its covariance with the
              other while preserving its marginal), and measure the loss
    """
    rng = rng if rng is not None else RNG
    Ntr, Ptr, _ = M.channels(Xtr)
    Nte, Pte, _ = M.channels(Xte)
    pool = lambda A_: A_.mean(1)
    both_te, both_tr = pool(Nte * Pte), pool(Ntr * Ptr)
    mu = both_tr.mean(0)
    perm = rng.permutation(len(Xte))
    banks = {
        "align_A": pool(Nte * Ptr.mean(0, keepdims=True).mean(0)[None, None, :]
                        if False else Nte * Ptr.mean((0, 1))[None, None, :]),
        "place_A": pool(Ntr.mean((0, 1))[None, None, :] * Pte),
        "align_B": pool(Nte * Pte[perm]),      # proximity decorrelated by shuffling
        "place_B": pool(Nte[perm] * Pte),      # alignment decorrelated
    }
    def r2(a, b):
        a = a - a.mean(); b = b - b.mean()
        return float((a @ b) ** 2 / max((a @ a) * (b @ b), 1e-30))
    out = []
    for j in range(M.C):
        v = M.A @ E[:, j]
        s = (both_te - mu) @ v
        rA = {k: r2(s, (banks[f"{k}_A"] - banks[f"{k}_A"].mean(0)) @ v) for k in ("align", "place")}
        rB = {k: r2(s, (banks[f"{k}_B"] - banks[f"{k}_B"].mean(0)) @ v) for k in ("align", "place")}
        fA = rA["align"] / max(rA["align"] + rA["place"], 1e-30)
        fB = rB["align"] / max(rB["align"] + rB["place"], 1e-30)
        out.append(dict(j=j, align_A=fA, align_B=fB, agree=abs(fA - fB),
                        place_A=1 - fA, place_B=1 - fB))
    # null: a random direction in logit space
    nulls = []
    for _ in range(nnull):
        e = rng.normal(size=M.C); e /= np.linalg.norm(e)
        v = M.A @ e
        s = (both_te - mu) @ v
        rA = {k: r2(s, (banks[f"{k}_A"] - banks[f"{k}_A"].mean(0)) @ v) for k in ("align", "place")}
        nulls.append(rA["align"] / max(rA["align"] + rA["place"], 1e-30))
    return out, summarize(nulls)


def i_eps(M, Xtr, Xte, yte):
    """Y3: does the softening engage, and where would it have to sit to."""
    _, _, d2 = M.channels(Xte)
    med, mn = float(np.median(d2)), float(d2.min())
    rows = []
    for mult in (1, 1e2, 1e4, 3e4, 1e5, 3e5, 1e6):
        eps = M.eps * mult
        Ftr, Fte = M.features(Xtr, eps=eps), M.features(Xte, eps=eps)
        mu = Ftr.mean(0)
        rows.append(dict(eps=eps, ratio=eps / med,
                         acc=100.0 * float((((Fte - mu) @ M.A + (mu @ M.A + M.bias)).argmax(1) == yte).mean())))
    base = rows[0]["acc"]
    breaks = next((r["ratio"] for r in rows if r["acc"] < base - 1.0), None)
    return dict(eps=M.eps, median_d2=med, min_d2=mn, ratio=M.eps / med,
                closest_in_eps=mn / M.eps, sweep=rows, breaks_at_ratio=breaks)


def i_support(M, Zte, yte, base, mu, E, Ztr, ytr, nnull=16, rng=None):
    """Y4: support, its null, and the silence-vs-erase test."""
    rng = rng if rng is not None else RNG
    def fit(Etr, ytr_, Ete, yte_, steps=700, bs=512, lr=3e-2):
        d, K = Etr.shape[1], M.C            # the class count comes from the model
        Aw = np.zeros((d, K)); bw = np.zeros(K)
        mA = np.zeros_like(Aw); vA = np.zeros_like(Aw); mb = np.zeros(K); vb = np.zeros(K)
        rr = np.random.default_rng(0)
        for t in range(1, steps + 1):
            i = rr.integers(0, len(Etr), bs)
            lg = Etr[i] @ Aw + bw
            lg -= lg.max(1, keepdims=True)
            p = np.exp(lg); p /= p.sum(1, keepdims=True)
            p[np.arange(bs), ytr_[i]] -= 1
            gA, gb = Etr[i].T @ p / bs, p.mean(0)
            for P_, G, M_, V_ in ((Aw, gA, mA, vA), (bw, gb, mb, vb)):
                M_ *= 0.9; M_ += 0.1 * G
                V_ *= 0.999; V_ += 0.001 * G * G
                P_ -= lr * (M_ / (1 - 0.9 ** t)) / (np.sqrt(V_ / (1 - 0.999 ** t)) + 1e-8)
        return 100.0 * float(((Ete @ Aw + bw).argmax(1) == yte_).mean())

    supp = lambda v: float(np.abs(v).sum() ** 2 / max((v ** 2).sum(), 1e-30))
    real = [supp(M.A @ E[:, j]) for j in range(M.C)]
    nulls = []
    for _ in range(nnull):
        e = rng.normal(size=M.C); e /= np.linalg.norm(e)
        nulls.append(supp(M.A @ e))
    # can the top-support rows be deleted, and does a refit bring the concept back?
    j = 0
    v = M.A @ E[:, 0]
    order = np.argsort(-np.abs(v))
    erase = []
    for k in (16, 64):
        keep = np.setdiff1d(np.arange(M.m), order[:k])
        erase.append(dict(k=int(k),
                          refit=fit(Ztr[:, keep], ytr, Zte[:, keep], yte)))
    return dict(support=real, null=summarize(nulls), erase=erase,
                intact_refit=fit(Ztr, ytr, Zte, yte))


def i_measure(M, Ztr, Zte, ytr, E, k=5):
    """Are the concepts a property of the network or of the population?

    Two readings per comparison, because they answer different questions and
    they disagree by a lot. The AXIS reading is the legacy one: |cos| between
    column j here and column j there. It scores near zero whenever two adjacent
    eigenvalues swap rank, even when the plane they span has not moved. The
    SUBSPACE reading is the mean cosine of the principal angles between the two
    top-k subspaces, which is invariant to any rotation or reordering inside
    them. Where the two disagree, the axes moved and the subspace did not.
    """
    outs = {}
    cols = lambda Ea, Eb: float(np.mean(np.abs(np.sum(Ea[:, :k] * Eb[:, :k], 0))))

    _, Ete, _ = concept_basis(M, Zte)
    outs["test"] = cols(E, Ete)
    outs["test_subspace"] = subspace_agreement(E[:, :k], Ete[:, :k])
    outs["test_matched"] = matched_agreement(E[:, :k], Ete[:, :k])

    # A genuinely class-balanced measure: every class present, equal counts.
    # (This loop used to run over range(10) and take 1000 rows per class, which
    # on a 100-class problem with ~80 rows per class quietly reduced to 800
    # points drawn from ten classes.)
    counts = [np.where(ytr == c)[0] for c in range(M.C)]
    per = min((len(i) for i in counts if len(i)), default=0)
    if per:
        idx = np.concatenate([i[:per] for i in counts if len(i)])
        _, Ebal, _ = concept_basis(M, Ztr[idx])
        outs["class_balanced"] = cols(E, Ebal)
        outs["balanced_subspace"] = subspace_agreement(E[:, :k], Ebal[:, :k])
        outs["balanced_matched"] = matched_agreement(E[:, :k], Ebal[:, :k])
        outs["balanced_per_class"] = int(per)
        outs["balanced_classes"] = int(sum(1 for i in counts if len(i)))
    return outs


def i_hierarchy(M, E, sig, ytr, coarse_tr, nnull=32, rng=None):
    """Y6: with a real taxonomy, is a concept a COARSE contrast or a fine one?

    Each concept is a vector over classes. Score how much of its class-loading
    variance is explained by superclass membership (between-group over total).
    A coarse concept separates superclasses; a fine one splits within them.
    The null is the same statistic on random concept directions.
    """
    rng = rng if rng is not None else RNG
    fine2coarse = np.zeros(M.C, int)
    for f_, c_ in zip(ytr, coarse_tr):
        fine2coarse[f_] = c_
    def eta2(e):
        gm = e.mean()
        num = 0.0
        for g in np.unique(fine2coarse):
            k = fine2coarse == g
            num += k.sum() * (e[k].mean() - gm) ** 2
        den = ((e - gm) ** 2).sum()
        return float(num / max(den, 1e-30))
    real = [eta2(E[:, j]) for j in range(M.C)]
    nulls = []
    for _ in range(nnull):
        e = rng.normal(size=M.C)
        nulls.append(eta2(e))
    w = sig / max(sig.sum(), 1e-30)
    top = float(np.average(real[:10], weights=w[:10] + 1e-12))
    bot = float(np.mean(real[max(0, M.C - 20):]))
    return dict(per_concept=real, null=summarize(nulls), top10=top, bottom20=bot)


# ══ the audit ════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=os.path.join(HERE, "results", "kgl_blog-mercer-v1"))
    ap.add_argument("--arm", default="trained")
    ap.add_argument("--epoch", type=int, default=12)
    ap.add_argument("--patches", type=int, default=None)
    ap.add_argument("--data", default="fmnist")
    a = ap.parse_args()

    models = load_models(a.bundle, a.arm, a.epoch, a.patches)
    assert models, f"no models in {a.bundle}"
    Xtr, ytr, Xte, yte, NAMES, COARSE = dataset(a.data)
    Xs = Xtr[:8000]; ys = ytr[:8000]
    per_seed = []
    print(f"=== Yat audit v2 :: {a.arm}"
          f"{' P=' + str(a.patches) if a.patches else ''} :: {len(models)} seed(s) ===")
    for seed, M in models:
        rng = np.random.default_rng(9_000 + seed)     # independent per model
        Ftr, Fte = M.features(Xs), M.features(Xte)
        mu = Ftr.mean(0); Ztr, Zte = Ftr - mu, Fte - mu
        base = mu @ M.A + M.bias
        acc = 100.0 * float(((Zte @ M.A + base).argmax(1) == yte).mean())
        con = i_concepts(M, Ztr, Zte, yte, base, rng=rng)
        ch, ch_null = i_channels(M, Xs, Xte, con["E"], rng=rng)
        ep = i_eps(M, Xs, Xte, yte)
        su = i_support(M, Zte, yte, base, mu, con["E"], Ztr, ys, rng=rng)
        me = i_measure(M, Ztr, Zte, ys, con["E"])
        hi = (i_hierarchy(M, con["E"], con["sig"], ys, COARSE[0][:len(ys)], rng=rng)
              if COARSE is not None else None)
        per_seed.append(dict(seed=seed, acc=acc, con=con, ch=ch, ch_null=ch_null,
                             eps=ep, sup=su, meas=me, hier=hi))
        print(f"  seed {seed}: {acc:.2f}%  named concepts "
              f"{sum(con['named'])}/{M.C} per axis, "
              f"{sum(con['named_matched'])}/{M.C} once ordering is solved"
              f"   eps ratio {ep['ratio']:.2e}")

    M0 = models[0][1]
    agg = lambda f: summarize([f(r) for r in per_seed])
    rep = dict(arm=a.arm, patches=a.patches, m=M0.m, seeds=[r["seed"] for r in per_seed],
               accuracy=agg(lambda r: r["acc"]))

    print("\n── readings (mean ± sd over seeds, against their nulls) ──")
    # concepts
    k9 = agg(lambda r: r["con"]["curve"][min(8, len(r['con']['curve']) - 1)])
    named = agg(lambda r: float(sum(r["con"]["named"])))
    namedm = agg(lambda r: float(sum(r["con"]["named_matched"])))
    subs = {k: agg(lambda r, k=k: r["con"]["subspace"][k]["mean"])
            for k in per_seed[0]["con"]["subspace"]}
    rep["concepts"] = dict(acc_at_9=k9, n_named=named, n_named_matched=namedm,
                           subspace=subs,
                           perturbation=agg(lambda r: r["con"]["perturbation"]["mean"]),
                           stability=agg(lambda r: float(np.mean(r["con"]["stability"][:5]))))
    print(f"  identifiable concepts     {named['mean']:.1f} ± {named['sd']:.1f} of {M0.C}"
          f"   (gate: bootstrap |cos| > 0.9)")
    print(f"    same gate, ordering solved  {namedm['mean']:.1f} ± {namedm['sd']:.1f}"
          f"   (the difference is rank swaps, not instability)")
    print("    leading subspaces, mean principal-angle cosine:  "
          + "  ".join(f"k={k}: {v['mean']:.3f}" for k, v in sorted(subs.items())))
    # channels, two methods
    nul = agg(lambda r: 1.0 - r["ch_null"]["mean"])          # null PLACE share
    dis = agg(lambda r: float(np.mean([c["agree"] for c in r["ch"][:5]])))
    rows = []
    print(f"  channel ledger, as EXCESS PLACE over a random direction"
          f" (null place share {100*nul['mean']:.0f}%):")
    for j in range(5):
        pA = agg(lambda r, j=j: r["ch"][j]["place_A"])
        exc = pA["mean"] - nul["mean"]
        sd = max(pA["sd"], nul["sd"], 1e-9)
        tag = ("PLACE-DRIVEN" if exc > 2 * sd else
               "DIRECTION-DRIVEN" if exc < -2 * sd else "indistinguishable from chance")
        rows.append(dict(j=j, place=pA, excess=exc, verdict=tag))
        print(f"    concept {j+1}: place {100*pA['mean']:4.0f}% ± {100*pA['sd']:.0f}"
              f"   excess {100*exc:+5.0f} pts   -> {tag}")
    rep["channels"] = dict(null_place=nul, per_concept=rows, mean_disagreement=dis,
                           method_agreement=("CORROBORATED" if dis["mean"] < 0.10
                                             else "PROVISIONAL: the two methods disagree"))
    print(f"    methods: {rep['channels']['method_agreement']} (mean |A-B| = {dis['mean']:.2f})")
    # eps
    er = agg(lambda r: r["eps"]["ratio"]); cl = agg(lambda r: r["eps"]["closest_in_eps"])
    rep["softening"] = dict(ratio=er, closest_in_eps=cl,
                            breaks_at=agg(lambda r: r["eps"]["breaks_at_ratio"] or 0.0),
                            verdict=("ENGAGES" if er["mean"] > 0.01 else "IDLE: the floor is never reached"))
    print(f"  eps / median distance²     {er['mean']:.2e} ± {er['sd']:.1e}"
          f"   closest point sits at {cl['mean']:.0f}× eps")
    print(f"    -> {rep['softening']['verdict']}")
    # support vs null
    sp = agg(lambda r: float(np.mean(r["sup"]["support"][:3])))
    sn = agg(lambda r: r["sup"]["null"]["mean"])
    ref = agg(lambda r: r["sup"]["intact_refit"])
    e16 = agg(lambda r: r["sup"]["erase"][0]["refit"]); e64 = agg(lambda r: r["sup"]["erase"][1]["refit"])
    localized = sp["mean"] < 0.5 * sn["mean"]
    rep["support"] = dict(real=sp, null=sn, refit_intact=ref, refit_del16=e16, refit_del64=e64,
                          verdict=("LOCALIZED" if localized else
                                   "NOT LOCALIZED: support is no narrower than a random direction"))
    print(f"  concept support            {sp['mean']:.0f} ± {sp['sd']:.0f} of {M0.m}"
          f"   null (random direction): {sn['mean']:.0f}")
    print(f"    -> {rep['support']['verdict']}")
    print(f"  refit after deleting rows  intact {ref['mean']:.1f}%  "
          f"-16 rows {e16['mean']:.1f}%  -64 rows {e64['mean']:.1f}%"
          f"  -> {'ERASED' if e64['mean'] < ref['mean'] - 5 else 'HIDDEN, NOT ERASED'}")
    # measure dependence
    mt = agg(lambda r: r["meas"]["test"]); mb = agg(lambda r: r["meas"]["class_balanced"])
    mts = agg(lambda r: r["meas"]["test_subspace"])
    mbs = agg(lambda r: r["meas"]["balanced_subspace"])
    rep["measure"] = dict(train_vs_test=mt, train_vs_balanced=mb,
                          test_subspace=mts, balanced_subspace=mbs,
                          balanced_per_class=per_seed[0]["meas"].get("balanced_per_class"),
                          balanced_classes=per_seed[0]["meas"].get("balanced_classes"),
                          verdict=("STABLE" if min(mt["mean"], mb["mean"]) > 0.9
                                   else "MEASURE-SENSITIVE"),
                          subspace_verdict=("STABLE" if min(mts["mean"], mbs["mean"]) > 0.9
                                            else "MEASURE-SENSITIVE"))
    print(f"  measure dependence         per axis: test {mt['mean']:.3f}, "
          f"class-balanced {mb['mean']:.3f}  -> {rep['measure']['verdict']}")
    print(f"    as subspaces               test {mts['mean']:.3f}, "
          f"class-balanced {mbs['mean']:.3f}  -> {rep['measure']['subspace_verdict']}")
    print(f"    (balanced measure: {per_seed[0]['meas'].get('balanced_classes')} classes"
          f" x {per_seed[0]['meas'].get('balanced_per_class')} rows)")

    if per_seed[0]["hier"] is not None:
        t10 = agg(lambda r: r["hier"]["top10"])
        b20 = agg(lambda r: r["hier"]["bottom20"])
        nl = agg(lambda r: r["hier"]["null"]["mean"])
        coarse_first = t10["mean"] > b20["mean"] + 2 * max(t10["sd"], b20["sd"], 1e-9)
        rep["hierarchy"] = dict(top10=t10, bottom20=b20, null=nl,
                                verdict=("COARSE FIRST: leading concepts separate superclasses"
                                         if coarse_first else
                                         "NO HIERARCHY: concept rank does not track the taxonomy"))
        print(f"  taxonomy alignment         top-10 concepts {t10['mean']:.3f} ± {t10['sd']:.3f}"
              f"   tail {b20['mean']:.3f}   null {nl['mean']:.3f}")
        print(f"    -> {rep['hierarchy']['verdict']}")

    tag = a.arm + (f"_P{a.patches}" if a.patches else "") + ("" if a.data == "fmnist" else "_" + a.data)
    p = os.path.join(HERE, "results", f"yat_audit_{tag}.json")
    with open(p, "w") as f:
        json.dump(rep, f, indent=1, default=float)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
