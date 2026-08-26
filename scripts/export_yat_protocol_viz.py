"""Browser payloads for the interrogation-protocol post.

Reuses scripts/yat_protocol.py (the audit itself) and writes
public/yat-protocol/:

  eps.json      the distance distribution the softening is supposed to soften,
                the trained eps, and the measured accuracy along the eps dial
  basis.json    800 test images in logit coordinates + the concept basis, so
                the page can rebuild the network from k concepts live, against
                the variance basis measured on the full test set
  ledger.json   per concept: the score under both channels and under each one
                frozen, so the page can measure the alignment/proximity split
                itself; plus each concept rendered as a signed picture
  support.json  the prototype-by-concept weight matrix and every prototype
                picture, so a reader can see how broad a concept's support is
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yat_protocol import YatModel, fmnist, y1_concepts, y3_eps, CLS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "public", "yat-protocol")
os.makedirs(OUT, exist_ok=True)
BUNDLE = os.path.join(HERE, "results", "kgl_blog-mercer-v1")


def down14(v):
    """784-vector -> 14x14 list (2x2 mean pooling)."""
    return v.reshape(28, 28).reshape(14, 2, 14, 2).mean((1, 3)).reshape(-1)


def main():
    M = YatModel.load(BUNDLE, "trained", 12)
    Xtr, ytr, Xte, yte = fmnist()
    Ftr, Fte = M.features(Xtr), M.features(Xte)
    mu = Ftr.mean(0)
    Ztr, Zte = Ftr - mu, Fte - mu
    base = mu @ M.A + M.bias
    res = y1_concepts(M, Ztr, Zte, yte, base)

    rng = np.random.default_rng(0)
    sub = np.sort(rng.choice(len(yte), 800, replace=False))
    X14 = (Xte.reshape(-1, 28, 28).reshape(-1, 14, 2, 14, 2).mean((2, 4)) * 255)
    thumbs = np.round(X14[sub]).astype(int).reshape(len(sub), -1).tolist()

    # ── eps: the scale mismatch, plus the dial ──
    _, _, d2 = M.channels(Xte)
    lg = np.log10(np.maximum(d2.ravel(), 1e-6))
    hist, edges = np.histogram(lg, bins=48, range=(float(lg.min()), float(lg.max())))
    eps_res = y3_eps(M, Xtr, Xte, yte,
                     mults=(1, 10, 100, 1e3, 1e4, 3e4, 6e4, 1e5, 2e5, 3e5, 6e5))
    json.dump(dict(trained_eps=M.eps, median_d2=eps_res["median_d2"],
                   min_d2=eps_res["min_d2"], ratio=eps_res["ratio"],
                   hist=hist.tolist(), edges=np.round(edges, 3).tolist(),
                   sweep=[dict(eps=r["eps"], ratio=r["ratio"], acc=round(r["acc"], 2))
                          for r in eps_res["sweep"]]),
              open(os.path.join(OUT, "eps.json"), "w"), separators=(",", ":"))

    # ── basis: rebuild the network from k concepts, live ──
    L = (Zte[sub] @ M.A)                      # logit coordinates (800 x 10)
    json.dump(dict(classes=CLS, y=yte[sub].tolist(), thumbs=thumbs,
                   L=np.round(L, 3).tolist(), base=np.round(base, 4).tolist(),
                   E=np.round(res["E"], 5).tolist(),
                   full=round(res["full"], 2),
                   fun_curve=[[k, round(v, 2)] for k, v in res["fun_curve"]],
                   var_curve=[[k, round(v, 2)] for k, v in res["var_curve"]]),
              open(os.path.join(OUT, "basis.json"), "w"), separators=(",", ":"))

    # ── ledger: the two channels, per concept, on the same 800 images ──
    Ntr, Ptr, _ = M.channels(Xtr)
    Nte, Pte, _ = M.channels(Xte)
    banks = {"both": (Nte * Pte, (Ntr * Ptr).mean(0)),
             "align": (Nte * Ptr.mean(0), (Ntr * Ptr.mean(0)).mean(0)),
             "place": (Ntr.mean(0) * Pte, (Ntr.mean(0) * Ptr).mean(0))}
    concepts = []
    for c in res["concepts"]:
        v = c["v"]
        sc = {k: ((F - m0)[sub] @ v) for k, (F, m0) in banks.items()}
        # each score standardized so the panel compares shapes, not units
        z = {k: (s - s.mean()) / (s.std() + 1e-12) for k, s in sc.items()}
        concepts.append(dict(
            j=c["j"], gain=round(c["gain"], 1), contrast=c["contrast"],
            smooth=round(c["smooth_frac"], 4), support=round(c["support"], 1),
            e=np.round(c["e"], 4).tolist(),
            pic=np.round(down14(c["pic"]), 5).tolist(),
            both=np.round(z["both"], 3).tolist(),
            align=np.round(z["align"], 3).tolist(),
            place=np.round(z["place"], 3).tolist(),
            top_units=c["top_units"]))
    json.dump(dict(classes=CLS, y=yte[sub].tolist(), concepts=concepts),
              open(os.path.join(OUT, "ledger.json"), "w"), separators=(",", ":"))

    # ── support: who carries what, with every prototype as a picture ──
    V = np.stack([c["v"] for c in res["concepts"]], 1)      # (m, 10)
    protos = np.stack([down14(w) for w in M.W])
    lo, hi = float(protos.min()), float(protos.max())
    json.dump(dict(classes=CLS,
                   V=np.round(V / (np.abs(V).max(0) + 1e-12), 4).tolist(),
                   support=[round(c["support"], 1) for c in res["concepts"]],
                   contrast=[c["contrast"] for c in res["concepts"]],
                   protos=np.round((protos - lo) / (hi - lo + 1e-12), 4).tolist()),
              open(os.path.join(OUT, "support.json"), "w"), separators=(",", ":"))

    for f in sorted(os.listdir(OUT)):
        print(f"{f:14s} {os.path.getsize(os.path.join(OUT, f))/1e6:.2f} MB")


if __name__ == "__main__":
    main()
