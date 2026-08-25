r"""The danger list: predict a model's confusions from its geometry alone.

The claim under test is that a trained network's own concept codebook says, in
advance, which classes it will mix up. So the forecast is built with the test
set untouched: the concept basis and the hundred class-mean directions both
come from training data, they are projected into the leading identifiable
subspace, and the pairs are ranked by cosine. Only then is the test set scored.

(`yat_welch_check.py` measured the same cosine-versus-confusion link but took
its class means from test features, which is fine for describing a geometry and
not fine for claiming a prediction. The gap between the two is reported below
as `leak_delta`, so the cost of doing it honestly is visible.)

Three baselines, because the forecast has to beat something that uses no model:

  uniform      random pairs
  taxonomy     same-superclass pairs first, shuffled within: the forecaster you
               get from the label structure alone, with no network at all
  random-K     class means projected into a RANDOM K-dim subspace of logit
               space instead of the concept subspace, so the codebook geometry
               is kept and only the choice of subspace is randomized

Headline metric is error mass: what fraction of every mistake the model makes
lands on the k pairs the list named, against the k/4950 a coin would get.

Run: python scripts/yat_forecast.py [--k 15] [--data cifar100g]
Writes results/yat_forecast.json and public/danger-list/forecast.json
"""

import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr

from yat_audit import Model, concept_basis, dataset, load_models, summarize  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.join(HERE, "results", "kgl_blog-cifar-v1")
OUT = os.path.join(HERE, "..", "public", "danger-list")
KS = (5, 10, 20, 30, 50, 100, 200)


def features_chunked(M, X, chunk=2000):
    return np.concatenate([M.features(X[i:i + chunk]) for i in range(0, len(X), chunk)])


def class_means(Z, y, C):
    """Centered class-mean directions. Z must come from the measure you are
    allowed to have looked at."""
    cm = np.stack([Z[y == c].mean(0) for c in range(C)])
    return cm - cm.mean(0)


def pair_cos(cm, U):
    """Project the class means into subspace U, normalize, return the upper
    triangle of the cosine matrix."""
    V = cm @ U
    V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    G = V @ V.T
    return G[np.triu_indices(len(cm), 1)]


def confusion_pairs(pred, y, C):
    """Symmetric off-diagonal confusion counts, as an upper-triangle vector."""
    conf = np.zeros((C, C))
    for t, p in zip(y, pred):
        if t != p:
            conf[t, p] += 1
    conf = conf + conf.T
    return conf[np.triu_indices(C, 1)]


def score(rank, truth, ks=KS):
    """rank: pair indices, most dangerous first. truth: confusion count per pair.
    Reports the share of all errors sitting on the top k, and the lift over the
    k/npairs a coin would collect."""
    total = truth.sum()
    npairs = len(truth)
    out = {}
    for k in ks:
        if k > npairs:
            continue
        mass = float(truth[rank[:k]].sum() / max(total, 1))
        out[k] = dict(mass=mass, chance=k / npairs, lift=mass / (k / npairs))
    return out


def taxonomy_rank(f2c, C, rng, reps=32):
    """The label-only forecaster: siblings first, random within. Averaged over
    shuffles so it is not scored on one lucky ordering."""
    iu = np.triu_indices(C, 1)
    sib = (f2c[iu[0]] == f2c[iu[1]]).astype(float)
    ranks = []
    for _ in range(reps):
        ranks.append(np.argsort(-(sib + rng.random(len(sib)) * 0.5)))
    return ranks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=BUNDLE)
    ap.add_argument("--data", default="cifar100g")
    ap.add_argument("--k", type=int, default=15, help="identifiable subspace width")
    ap.add_argument("--nbasis", type=int, default=8000, help="rows for the concept basis")
    a = ap.parse_args()

    models = load_models(a.bundle, "trained", 12)
    Xtr, ytr, Xte, yte, names, coarse = dataset(a.data)
    C = int(ytr.max()) + 1
    f2c = np.zeros(C, int)
    for f_, c_ in zip(ytr, coarse[0]):
        f2c[f_] = c_
    npairs = C * (C - 1) // 2
    iu = np.triu_indices(C, 1)

    print(f"=== the danger list :: {len(models)} seed(s), K={a.k}, {npairs} pairs ===")
    print("geometry from TRAIN only; the test set is not touched until scoring\n")

    per, forecasts, truths = [], [], []
    for seed, M in models:
        rng = np.random.default_rng(9_000 + seed)
        Fb = features_chunked(M, Xtr[:a.nbasis])
        mu = Fb.mean(0)
        sig, E, _ = concept_basis(M, Fb - mu)
        Q, _ = np.linalg.qr(M.A @ E)          # all C columns: head AND tail
        U = Q[:, :a.k]

        # ── the forecast: class means from TRAIN, test set untouched ──
        Ftr = features_chunked(M, Xtr)
        cm_tr = class_means(Ftr - mu, ytr, C)
        g_tr = pair_cos(cm_tr, U)
        rank = np.argsort(-g_tr)

        # ── the truth, revealed only now ──
        Fte = features_chunked(M, Xte)
        pred = np.argmax(Fte @ M.A + M.bias, 1)
        truth = confusion_pairs(pred, yte, C)
        acc = 100.0 * float((pred == yte).mean())

        s = score(rank, truth)
        rho = float(spearmanr(truth, g_tr).statistic)

        # what the leaky version would have claimed
        cm_te = class_means(Fte - mu, yte, C)
        g_te = pair_cos(cm_te, U)
        s_leak = score(np.argsort(-g_te), truth)
        rho_leak = float(spearmanr(truth, g_te).statistic)

        # ── baselines ──
        base = {}
        uni = [rng.permutation(npairs) for _ in range(32)]
        base["uniform"] = {k: float(np.mean([score(r, truth, (k,))[k]["mass"] for r in uni]))
                           for k in KS}
        tax = taxonomy_rank(f2c, C, rng)
        base["taxonomy"] = {k: float(np.mean([score(r, truth, (k,))[k]["mass"] for r in tax]))
                            for k in KS}
        rk = []
        for _ in range(16):
            Rr = rng.normal(size=(M.A.shape[1], a.k))
            Qr, _ = np.linalg.qr(M.A @ Rr)
            rk.append(np.argsort(-pair_cos(cm_tr, Qr[:, :a.k])))
        base["random_K"] = {k: float(np.mean([score(r, truth, (k,))[k]["mass"] for r in rk]))
                            for k in KS}

        # The controls that decide whether the concept basis is contributing
        # anything at all. `tail` is the part that FAILS the naming gate;
        # `raw` uses no decomposition; `logit_mass` uses no geometry, just the
        # model's own averaged outputs.
        cosof = lambda X: (lambda V: (V @ V.T)[iu])(
            X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12))
        Lte = Fte @ M.A + M.bias
        Sm = np.stack([Lte[yte == c].mean(0) for c in range(C)])
        rivals = {"tail": pair_cos(cm_tr, Q[:, -50:]),
                  "raw": cosof(cm_tr),
                  "readout": cosof(M.A.T),
                  "logit_mass": ((Sm + Sm.T) / 2)[iu]}
        base.update({nm: {k: score(np.argsort(-g), truth, (k,))[k]["mass"] for k in KS}
                     for nm, g in rivals.items()})

        # split the truth: within-superclass confusions vs everything else
        sibm = f2c[iu[0]] == f2c[iu[1]]
        parts = {}
        for nm, g in dict(forecast=g_tr, **rivals).items():
            r = np.argsort(-g)
            parts[nm] = dict(
                sibling=score(r, np.where(sibm, truth, 0.0), (20,))[20]["mass"],
                cross=score(r, np.where(sibm, 0.0, truth), (20,))[20]["mass"])

        per.append(dict(seed=seed, acc=acc, score=s, rho=rho, base=base, parts=parts,
                        leak=s_leak, rho_leak=rho_leak,
                        sibling_share=float(np.where(sibm, 1.0, 0.0) @ truth / truth.sum()),
                        cos_top30=float(np.sort(g_tr)[-30:].mean()),
                        errors=int(truth.sum())))
        forecasts.append(g_tr)
        truths.append(truth)
        print(f"  seed {seed}: {acc:.2f}%, {int(truth.sum())} errors   "
              f"top-20 captures {100*s[20]['mass']:.1f}% of them "
              f"({s[20]['lift']:.1f}x chance)   full-ranking rho {rho:.3f}")

    print("\n── error mass captured, mean over seeds ──")
    print(f"  {'k':>4}  {'forecast':>9}  {'taxonomy':>9}  {'random-K':>9}  "
          f"{'uniform':>8}  {'chance':>7}  {'lift':>5}")
    table = []
    for k in KS:
        f = np.mean([p["score"][k]["mass"] for p in per])
        t = np.mean([p["base"]["taxonomy"][k] for p in per])
        r = np.mean([p["base"]["random_K"][k] for p in per])
        u = np.mean([p["base"]["uniform"][k] for p in per])
        ch = k / npairs
        table.append(dict(k=k, forecast=f, taxonomy=t, random_K=r, uniform=u,
                          chance=ch, lift=f / ch, over_taxonomy=f / max(t, 1e-12)))
        print(f"  {k:>4}  {100*f:>8.1f}%  {100*t:>8.1f}%  {100*r:>8.1f}%  "
              f"{100*u:>7.1f}%  {100*ch:>6.1f}%  {f/ch:>4.1f}x")

    print("\n── does the concept basis contribute? error mass at k=20 ──")
    order = ["forecast", "tail", "raw", "readout", "logit_mass"]
    lab = {"forecast": "concept head, k=15", "tail": "the tail that FAILS the gate",
           "raw": "class means, no decomposition", "readout": "readout columns",
           "logit_mass": "averaged logits (no geometry)"}
    print(f"  {'ranker':<31} {'all':>7} {'sibling':>9} {'cross':>7}")
    contrib = []
    for nm in order:
        allm = (np.mean([p["score"][20]["mass"] for p in per]) if nm == "forecast"
                else np.mean([p["base"][nm][20] for p in per]))
        sb = np.mean([p["parts"][nm]["sibling"] for p in per])
        cr = np.mean([p["parts"][nm]["cross"] for p in per])
        contrib.append(dict(ranker=nm, all=allm, sibling=sb, cross=cr))
        print(f"  {lab[nm]:<31} {100*allm:>6.1f}% {100*sb:>8.1f}% {100*cr:>6.1f}%")
    ss = np.mean([p["sibling_share"] for p in per])
    print(f"  only {100*ss:.1f}% of errors are within-superclass "
          f"(chance {100*200/npairs:.1f}%), so the list is scored mostly on the other {100*(1-ss):.0f}%")

    # cross-seed transfer: does one model's geometry predict another's mistakes?
    trans = []
    for i in range(len(per)):
        for j in range(len(per)):
            if i == j:
                continue
            s = score(np.argsort(-forecasts[i]), truths[j], (20,))
            trans.append(s[20]["mass"])
    if trans:
        own = np.mean([p["score"][20]["mass"] for p in per])
        print(f"\n  cross-seed transfer at k=20: {100*np.mean(trans):.1f}% "
              f"(own geometry {100*own:.1f}%)")

    rho_m = summarize([p["rho"] for p in per])
    rho_l = summarize([p["rho_leak"] for p in per])
    lk = np.mean([p["leak"][20]["mass"] for p in per])
    fc = np.mean([p["score"][20]["mass"] for p in per])
    print(f"  full-ranking rho           {rho_m['mean']:.3f} ± {rho_m['sd']:.3f}"
          f"   (with test-set means: {rho_l['mean']:.3f})")
    print(f"  cost of honesty at k=20    {100*fc:.1f}% train-only vs "
          f"{100*lk:.1f}% if the means had seen the test set")

    rep = dict(K=a.k, C=C, npairs=npairs, contribution=contrib,
               sibling_share=float(ss), seeds=[p["seed"] for p in per],
               accuracy=summarize([p["acc"] for p in per]),
               table=table, rho=rho_m, rho_leak=rho_l,
               leak_delta=float(lk - fc),
               transfer_at20=float(np.mean(trans)) if trans else None,
               per_seed=[{kk: vv for kk, vv in p.items()} for p in per])
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    p = os.path.join(HERE, "results", "yat_forecast.json")
    json.dump(rep, open(p, "w"), indent=1, default=float)
    print(f"\nwrote {os.path.relpath(p, HERE)}")

    # ── the panel payload: the twenty rows, seed 0 ──
    os.makedirs(OUT, exist_ok=True)
    g0, t0 = forecasts[0], truths[0]
    order = np.argsort(-g0)[:40]
    rows = [dict(a=names[iu[0][i]], b=names[iu[1][i]],
                 cos=round(float(g0[i]), 4), errs=int(t0[i]),
                 sib=bool(f2c[iu[0][i]] == f2c[iu[1][i]])) for i in order]
    json.dump(dict(rows=rows, total_errors=int(t0.sum()), npairs=npairs,
                   table=table, K=a.k,
                   median_errs=float(np.median(t0)),
                   rho=rho_m["mean"]),
              open(os.path.join(OUT, "forecast.json"), "w"), separators=(",", ":"))
    print(f"wrote {os.path.relpath(os.path.join(OUT, 'forecast.json'), HERE)}")


if __name__ == "__main__":
    main()
