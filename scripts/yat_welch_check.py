"""Is the hundred-class codebook in the Welch regime?

The fifteen-ideas audit found the trained function factors through ~15
identifiable concept axes while carrying 100 classes. The Welch post
(welch-bound-good-latent-space) says: once C unit codes share d < C-1
dimensions, the centered simplex is impossible and the Welch bound sets a
crosstalk floor, rms cross-cosine >= sqrt((C-d)/(d(C-1))), attained only by
an equiangular tight frame that shares the crosstalk evenly.

This script measures where the trained network actually sits, from the
existing kgl_blog-cifar-v1 bundle (no new training):

  per seed:
    - concept basis from the audit's own machinery; top-15 subspace
      U = orth{A e_j : j < 15} in feature space
    - centered class-mean feature directions, projected into U, normalized
    - their pairwise cosine matrix vs the Welch floor for (C=100, d=15),
      vs a random codebook, vs a frame-potential-descended codebook
    - evenness: is the crosstalk shared (equiangular-ish) or dumped on a
      few pairs?
    - the confusion link: are the tightest-packed pairs the ones the
      network actually confuses on the test set?
    - sibling structure: same-superclass vs cross-superclass mean cosine
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yat_audit import load_models, dataset, concept_basis  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.join(HERE, "results", "kgl_blog-cifar-v1")
K = 15
C = 100
RNG = np.random.default_rng(0)


def welch_rms(c, d):
    return np.sqrt(max(c - d, 0) / (d * (c - 1)))


def offdiag(G):
    return G[~np.eye(len(G), dtype=bool)]


def pack_stats(V):
    """V: unit rows. Returns rms |cos|, max |cos|, mean cos, evenness."""
    G = V @ V.T
    o = offdiag(G)
    return dict(rms=float(np.sqrt((o ** 2).mean())), mx=float(np.abs(o).max()),
                mean=float(o.mean()), sd_abs=float(np.abs(o).std()))


def descend_frame_potential(c, d, steps=4000, lr=0.05, seed=1):
    """The best packing gradient descent on free unit vectors can buy."""
    r = np.random.default_rng(seed)
    V = r.normal(size=(c, d))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    for _ in range(steps):
        G = V @ V.T
        np.fill_diagonal(G, 0)
        g = 4 * G @ V
        g -= (g * V).sum(1, keepdims=True) * V   # tangent to the sphere
        V -= lr * g / c
        V /= np.linalg.norm(V, axis=1, keepdims=True)
    return V


def main():
    models = load_models(BUNDLE, "trained", 1)
    Xtr, ytr, Xte, yte, _, coarse = dataset("cifar100g")
    ctr, cte = coarse
    fine2coarse = np.zeros(C, int)
    for f_, c_ in zip(yte, cte):
        fine2coarse[f_] = c_
    sib = fine2coarse[:, None] == fine2coarse[None, :]
    np.fill_diagonal(sib, False)

    floor = welch_rms(C, K)
    print(f"Welch floor for C={C}, d={K}: rms cos = {floor:.4f} "
          f"(simplex impossible: needs d >= {C - 1})")

    best = descend_frame_potential(C, K)
    bs = pack_stats(best)
    x = RNG.normal(size=(C, K))
    rnd = pack_stats(x / np.linalg.norm(x, axis=1, keepdims=True))
    print(f"frame-potential descent (best free packing): rms {bs['rms']:.4f} "
          f"max {bs['mx']:.4f}  spread of |cos| {bs['sd_abs']:.4f}")
    print(f"random codebook:                             rms {rnd['rms']:.4f} "
          f"max {rnd['mx']:.4f}  spread of |cos| {rnd['sd_abs']:.4f}\n")

    agg = dict(rms=[], mx=[], mean=[], sd_abs=[], sib=[], cross=[],
               conf_cos=[], other_cos=[], rho=[], erank=[])
    coords, conf0 = [], None
    for seed, M in models:
        Ftr = M.features(Xtr[:8000])
        mu = Ftr.mean(0)
        Fte = M.features(Xte)
        Zte = Fte - mu
        Ztr = Ftr - mu

        sig, E, _ = concept_basis(M, Ztr)
        Q30, _ = np.linalg.qr(M.A @ E[:, :30])       # progressive basis: first k
        U = Q30[:, :K]                                # columns span the top-k axes

        cm = np.stack([Zte[yte == c].mean(0) for c in range(C)])
        cm -= cm.mean(0)                              # centered class means
        c30 = cm @ Q30
        coords.append(c30 / np.linalg.norm(c30, axis=1, keepdims=True))
        ev = np.linalg.eigvalsh(cm @ cm.T)
        erank = float(ev.sum() ** 2 / (ev ** 2).sum())

        V = cm @ U
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        st = pack_stats(V)
        G = V @ V.T

        # confusion link: symmetric off-diagonal confusion counts
        pred = np.argmax(Fte @ M.A + M.bias, 1)
        conf = np.zeros((C, C))
        for t, p in zip(yte, pred):
            if t != p:
                conf[t, p] += 1
        conf = conf + conf.T
        if conf0 is None:
            conf0 = conf
        iu = np.triu_indices(C, 1)
        cc, gg = conf[iu], G[iu]
        top = np.argsort(cc)[-30:]                    # 30 most-confused pairs
        rest = np.argsort(cc)[:-30]
        from scipy.stats import spearmanr
        rho = float(spearmanr(cc, gg).statistic)

        agg["rms"].append(st["rms"]); agg["mx"].append(st["mx"])
        agg["mean"].append(st["mean"]); agg["sd_abs"].append(st["sd_abs"])
        agg["sib"].append(float(G[sib].mean()))
        agg["cross"].append(float(G[~sib & ~np.eye(C, dtype=bool)].mean()))
        agg["conf_cos"].append(float(gg[top].mean()))
        agg["other_cos"].append(float(gg[rest].mean()))
        agg["rho"].append(rho)
        agg["erank"].append(erank)
        print(f"seed {seed}: rms {st['rms']:.4f} max {st['mx']:.4f} "
              f"mean {st['mean']:+.4f} |cos|sd {st['sd_abs']:.4f} "
              f"erank(full) {erank:.1f}")
        print(f"        siblings {G[sib].mean():+.4f} vs cross "
              f"{G[~sib & ~np.eye(C, dtype=bool)].mean():+.4f}   "
              f"top-30 confused pairs cos {gg[top].mean():+.4f} vs rest "
              f"{gg[rest].mean():+.4f}   spearman(conf, cos) {rho:.3f}")

    print("\n== across seeds ==")
    for k, v in agg.items():
        print(f"{k:10s} {np.mean(v):+.4f} +- {np.std(v):.4f}")
    out = dict(welch_floor=round(floor, 4), K=K, C=C,
               best_packing={k: round(v, 4) for k, v in bs.items()},
               random={k: round(v, 4) for k, v in rnd.items()},
               seeds={k: [round(float(x), 4) for x in v] for k, v in agg.items()})
    with open(os.path.join(HERE, "results", "yat_welch_check.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote results/yat_welch_check.json")

    # ── browser payload for the fifteen-ideas panel ──
    tax = json.load(open(os.path.join(HERE, "..", "public", "fifteen-ideas",
                                      "taxonomy.json")))
    pairs = [[int(i), int(j), int(conf0[i, j])]
             for i, j in zip(*np.triu_indices(C, 1)) if conf0[i, j] > 0]
    json.dump(dict(
        names=tax["names"], coarse_names=tax["coarse_names"],
        fine2coarse=tax["fine2coarse"],
        coords=[np.round(v, 4).tolist() for v in coords],
        conf=pairs,
        best=dict(rms=round(bs["rms"], 4), mx=round(bs["mx"], 4)),
        audit=dict(rms=round(float(np.mean(agg["rms"])), 4),
                   mx=round(float(np.mean(agg["mx"])), 4),
                   sib=round(float(np.mean(agg["sib"])), 4),
                   cross=round(float(np.mean(agg["cross"])), 4))),
        open(os.path.join(HERE, "..", "public", "fifteen-ideas", "welch.json"),
             "w"), separators=(",", ":"))
    print("wrote public/fifteen-ideas/welch.json")


if __name__ == "__main__":
    main()
