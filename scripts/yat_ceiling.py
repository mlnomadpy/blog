"""What is the 17.65% ceiling made of?

The packing measurement (yat_welch_check.py) said the hundred class codes
collide along taxonomy lines. This script converts that geometry into an
error budget, per seed, from the same kgl_blog-cifar-v1 bundle:

  - fine and coarse top-1 (coarse = map the fine prediction through the
    ground-truth superclass table)
  - fine accuracy GIVEN the superclass was right: the sibling pick
  - where the errors go: share landing inside the true superclass, against
    the 4/99 chance rate; share landing on tightly-packed pairs (cos > 0.7
    in the 15-dim identifiable subspace)
  - the truncation split: rebuild from the top-15 concept axes only and
    re-read coarse vs fine-given-coarse, so "the nameable head is a
    superclass machine, the sibling pick lives in the anonymous haze"
    becomes two curves instead of a sentence

Also patches public/fifteen-ideas/rebuild.json with the per-seed coarse
rebuild curves and the fine-to-superclass map, for the HundredRebuild panel.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yat_audit import load_models, dataset, concept_basis  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.join(HERE, "results", "kgl_blog-cifar-v1")
C, K = 100, 15


def main():
    models = load_models(BUNDLE, "trained", 1)
    Xtr, ytr, Xte, yte, _, coarse = dataset("cifar100g")
    ctr, cte = coarse
    f2c = np.zeros(C, int)
    for f_, c_ in zip(yte, cte):
        f2c[f_] = c_

    agg = {k: [] for k in ("fine", "co", "fine_gc", "err_in_super", "err_tight",
                           "co15", "fgc15", "fine15")}
    coarse_curves = []
    for seed, M in models:
        Ftr = M.features(Xtr[:8000])
        mu = Ftr.mean(0)
        Zte = M.features(Xte) - mu
        base = mu @ M.A + M.bias
        sig, E, _ = concept_basis(M, Ftr - mu)

        # exact logits, and the class codebook in the identifiable subspace
        L = Zte @ M.A + base
        pred = L.argmax(1)
        U, _ = np.linalg.qr(M.A @ E[:, :K])
        cm = np.stack([Zte[yte == c].mean(0) for c in range(C)])
        cm -= cm.mean(0)
        V = cm @ U
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        G = V @ V.T

        fine = 100 * (pred == yte).mean()
        co = 100 * (f2c[pred] == f2c[yte]).mean()
        gcm = f2c[pred] == f2c[yte]
        fine_gc = 100 * (pred[gcm] == yte[gcm]).mean()
        wrong = pred != yte
        err_in_super = 100 * (f2c[pred[wrong]] == f2c[yte[wrong]]).mean()
        err_tight = 100 * (G[yte[wrong], pred[wrong]] > 0.7).mean()
        n_tight = int((G[np.triu_indices(C, 1)] > 0.7).sum())

        # rebuild from k concept axes: logits = base + (Zte A E_k) E_k^T
        D = (Zte @ M.A) @ E
        curve_c = []
        for k in range(1, C + 1):
            Lk = base + D[:, :k] @ E[:, :k].T
            pk = Lk.argmax(1)
            curve_c.append(round(100 * float((f2c[pk] == f2c[yte]).mean()), 2))
            if k == K:
                fine15 = 100 * (pk == yte).mean()
                co15 = curve_c[-1]
                g15 = f2c[pk] == f2c[yte]
                fgc15 = 100 * (pk[g15] == yte[g15]).mean()
        coarse_curves.append(curve_c)

        for k, v in zip(agg, (fine, co, fine_gc, err_in_super, err_tight,
                              co15, fgc15, fine15)):
            agg[k].append(float(v))
        print(f"seed {seed}: fine {fine:.2f} coarse {co:.2f} "
              f"fine|coarse {fine_gc:.2f}  errors-in-superclass {err_in_super:.1f}% "
              f"errors-on-tight-pairs {err_tight:.1f}% ({n_tight} pairs of 4950)")
        print(f"        k=15: coarse {co15:.2f} fine {fine15:.2f} "
              f"fine|coarse {fgc15:.2f}")

    print("\n== across seeds ==")
    for k, v in agg.items():
        print(f"{k:14s} {np.mean(v):6.2f} +- {np.std(v):.2f}")
    chance_in_super = 100 * 4 / 99
    print(f"chance rate for an error landing in the true superclass: "
          f"{chance_in_super:.2f}%")

    out = dict({k: dict(mean=round(float(np.mean(v)), 2),
                        sd=round(float(np.std(v)), 2)) for k, v in agg.items()},
               chance_in_super=round(chance_in_super, 2), n_tight=n_tight)
    with open(os.path.join(HERE, "results", "yat_ceiling.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote results/yat_ceiling.json")

    rb_path = os.path.join(HERE, "..", "public", "fifteen-ideas", "rebuild.json")
    rb = json.load(open(rb_path))
    rb["coarse_curves"] = coarse_curves
    rb["fine2coarse"] = f2c.tolist()
    json.dump(rb, open(rb_path, "w"), separators=(",", ":"))
    print("patched public/fifteen-ideas/rebuild.json (+coarse_curves, +fine2coarse)")


if __name__ == "__main__":
    main()
