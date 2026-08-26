"""Browser payloads for the hundred-classes post.

Reuses the audit's own machinery (scripts/yat_audit.py) on the
kgl_blog-cifar-v1 bundle and writes public/fifteen-ideas/:

  rebuild.json   800 test images in logit coordinates + the concept basis +
                 the per-rank bootstrap stability, so the page can rebuild
                 the network from k of its 100 possible concepts live
  taxonomy.json  the top concepts' loadings over the 100 fine classes, the
                 fine-to-superclass map and names, and each concept's
                 superclass-alignment score against the random null
  support.json   concept support against the random-direction null, and the
                 delete-then-refit readings
  eps.json       the distance histogram, the trained softening per seed, and
                 the sweep's eps-follows-the-learning-rate readings
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yat_audit import (Model, load_models, dataset, concept_basis,  # noqa: E402
                       i_concepts, i_hierarchy, RNG)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "public", "fifteen-ideas")
os.makedirs(OUT, exist_ok=True)
BUNDLE = os.path.join(HERE, "results", "kgl_blog-cifar-v1")

COARSE_NAMES = [
    "aquatic mammals", "fish", "flowers", "food containers",
    "fruit and vegetables", "household electrical devices",
    "household furniture", "insects", "large carnivores",
    "large man-made outdoor things", "large natural outdoor scenes",
    "large omnivores and herbivores", "medium-sized mammals",
    "non-insect invertebrates", "people", "reptiles", "small mammals",
    "trees", "vehicles 1", "vehicles 2"]


def fine_names():
    """Class names from the parquet's own huggingface schema metadata."""
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        f = hf_hub_download("uoft-cs/cifar100",
                            "cifar100/test-00000-of-00001.parquet",
                            repo_type="dataset")
        meta = pq.read_schema(f).metadata
        info = json.loads(meta[b"huggingface"].decode())
        return info["info"]["features"]["fine_label"]["names"]
    except Exception as e:
        print(f"[names] schema metadata unavailable ({e}); using indices")
        return [f"class {i}" for i in range(100)]


def main():
    models = load_models(BUNDLE, "trained", 1)
    Xtr, ytr, Xte, yte, _, coarse = dataset("cifar100g")
    ctr, cte = coarse
    names = fine_names()
    fine2coarse = np.zeros(100, int)
    for f_, c_ in zip(ytr, ctr):
        fine2coarse[f_] = c_

    # per-seed: concepts + stability + curve + hierarchy
    per = []
    for seed, M in models:
        Xs = Xtr[:8000]
        Ftr, Fte = M.features(Xs), M.features(Xte)
        mu = Ftr.mean(0)
        Ztr, Zte = Ftr - mu, Fte - mu
        base = mu @ M.A + M.bias
        rng = np.random.default_rng(9_000 + seed)     # matches yat_audit.py
        con = i_concepts(M, Ztr, Zte, yte, base, rng=rng)
        hi = i_hierarchy(M, con["E"], con["sig"], ytr[:8000], ctr[:8000], rng=rng)
        per.append(dict(seed=seed, M=M, mu=mu, Zte=Zte, base=base, con=con, hi=hi))
        print(f"seed {seed}: named {sum(con['named'])} per axis, "
              f"{sum(con['named_matched'])} once ordering is solved; "
              f"acc@15 {con['curve'][14]:.2f} full {con['curve'][-1]:.2f}")

    s0 = per[0]
    M, con = s0["M"], s0["con"]

    # ── rebuild.json ──
    sub = np.sort(RNG.choice(len(yte), 800, replace=False))
    X16 = (Xte.reshape(-1, 32, 32).reshape(-1, 16, 2, 16, 2).mean((2, 4)) * 255)
    json.dump(dict(
        y=yte[sub].tolist(),
        names=names,
        thumbs=[np.round(X16[i]).astype(int).reshape(-1).tolist() for i in sub[:24]],
        L=np.round(s0["Zte"][sub] @ M.A, 3).tolist(),
        base=np.round(s0["base"], 4).tolist(),
        E=np.round(con["E"], 5).tolist(),
        curves=[[round(v, 2) for v in p["con"]["curve"]] for p in per],
        stability=np.round(np.mean([p["con"]["stability"] for p in per], 0), 3).tolist(),
        stability_matched=np.round(
            np.mean([p["con"]["stability_matched"] for p in per], 0), 3).tolist(),
        named_mean=float(np.mean([sum(p["con"]["named"]) for p in per])),
        named_matched_mean=float(np.mean([sum(p["con"]["named_matched"]) for p in per])),
        subspace={str(k): round(float(np.mean([p["con"]["subspace"][k]["mean"] for p in per])), 4)
                  for k in per[0]["con"]["subspace"]},
        full=round(float(np.mean([p["con"]["curve"][-1] for p in per])), 2)),
        open(os.path.join(OUT, "rebuild.json"), "w"), separators=(",", ":"))

    # ── taxonomy.json ──
    NC = 24
    json.dump(dict(
        names=names, coarse_names=COARSE_NAMES, fine2coarse=fine2coarse.tolist(),
        loadings=np.round(con["E"][:, :NC].T, 4).tolist(),
        eta=[np.round(p["hi"]["per_concept"][:60], 4).tolist() for p in per],
        null=dict(mean=round(per[0]["hi"]["null"]["mean"], 4),
                  sd=round(per[0]["hi"]["null"]["sd"], 4)),
        top10=[round(p["hi"]["top10"], 4) for p in per],
        bottom20=[round(p["hi"]["bottom20"], 4) for p in per],
        stability=np.round(np.mean([p["con"]["stability"][:NC] for p in per], 0), 3).tolist()),
        open(os.path.join(OUT, "taxonomy.json"), "w"), separators=(",", ":"))

    # ── support.json + eps.json, straight from the audit report + run json ──
    rep = json.load(open(os.path.join(HERE, "results", "yat_audit_trained_cifar100g.json")))
    run = json.load(open([os.path.join(r, "yat_cifar.json") for r in
                          [BUNDLE, os.path.join(BUNDLE, "exp", "results")]
                          if os.path.exists(os.path.join(r, "yat_cifar.json"))][0]))
    supp_per_concept = []
    for p in per:
        supp_per_concept.append([
            float(np.abs(v := p["M"].A @ p["con"]["E"][:, j]).sum() ** 2 / max((v ** 2).sum(), 1e-30))
            for j in range(16)])
    nulls = []
    for _ in range(64):
        e = RNG.normal(size=M.C); e /= np.linalg.norm(e)
        v = M.A @ e
        nulls.append(float(np.abs(v).sum() ** 2 / max((v ** 2).sum(), 1e-30)))
    json.dump(dict(m=int(M.m), per_concept=np.round(supp_per_concept, 1).tolist(),
                   nulls=np.round(nulls, 1).tolist(),
                   audit=rep["support"]),
              open(os.path.join(OUT, "support.json"), "w"), separators=(",", ":"))

    _, _, d2 = M.channels(Xte)
    lg = np.log10(np.maximum(d2.ravel()[::7], 1e-6))
    hist, edges = np.histogram(lg, bins=48)
    json.dump(dict(hist=hist.tolist(), edges=np.round(edges, 3).tolist(),
                   eps=[round(float(mm.eps), 4) for _, mm in models],
                   median_d2=float(np.median(d2)), min_d2=float(d2.min()),
                   sweep=[dict(lr=r["lr"], eps=round(r["eps"], 3), b=round(r["b"], 3),
                               acc=round(r["best_acc"], 2))
                          for r in run["rows"] if r.get("m") == 256]),
              open(os.path.join(OUT, "eps.json"), "w"), separators=(",", ":"))

    for f in sorted(os.listdir(OUT)):
        print(f"{f:14s} {os.path.getsize(os.path.join(OUT, f))/1e6:.2f} MB")


if __name__ == "__main__":
    main()
