"""Analysis + browser payloads for the patches-in-conversation post.

Reads the kgl_blog-mixer-v1 bundle (3 seeds of the recursive Yat mixer,
trained at R=4) and does four local measurements the Kaggle run did not:

  1. verifies the numpy forward reproduces the run's reported accuracy
  2. DEPTH TRANSFER: the block is weight-tied, so the trained operator can be
     applied any number of times; run r = 0..12 on the full test set and read
     the factorization (fine, coarse, fine-given-coarse) at every depth
  3. the concept audit at the pooled layer: same instrument as the flat model
     (logit-covariance basis + bootstrap identifiability gate); does the
     nameable count move past fifteen when concepts can compose?
  4. eps against the distances each Yat layer actually sees (embed on patch
     pixels, token mix on channel profiles, channel mix on sphere tokens):
     which softenings idle, which engage

then writes public/patches-in-conversation/:
  weights.json  seed-0 weights + dials, so panels run the real network live
  story.json    training curves, the R ladder, depth transfer, audit, eps
  thumbs.json   24 test images (full pixels) + labels + names
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yat_audit import dataset, i_concepts  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.join(HERE, "results", "kgl_blog-mixer-v1")
OUT = os.path.join(HERE, "..", "public", "patches-in-conversation")
os.makedirs(OUT, exist_ok=True)
P, T, DPATCH, C = 8, 16, 64, 100
RNG = np.random.default_rng(0)


def tokens(X):
    n, g = len(X), 32 // P
    return (X.reshape(n, 32, 32).reshape(n, g, P, g, P)
             .transpose(0, 1, 3, 2, 4).reshape(n, T, DPATCH))


def yat(x, W, b, eps):
    """(..., d) -> (..., m): the kernel map on the last axis."""
    dot = x @ W.T
    d2 = np.maximum((x ** 2).sum(-1, keepdims=True) + (W ** 2).sum(-1) - 2 * dot, 0)
    return (dot + b) ** 2 / (d2 + eps)


def sphere(h):
    return h / np.maximum(np.linalg.norm(h, axis=-1, keepdims=True), 1e-6)


class Mixer:
    def __init__(s, npz):
        z = np.load(npz)
        s.We = z["We"].astype(np.float64)
        s.Wt = z["Wt"].astype(np.float64)
        s.Wc = z["Wc"].astype(np.float64)
        s.A = z["A"].astype(np.float64)
        s.bias = z["bias"].astype(np.float64)
        s.be, s.bt, s.bc = z["be"], z["bt"], z["bc"]
        s.a_t, s.a_c = [float(v) for v in z["alphas"]]
        s.r = int(z["config"][0])
        s.m = int(z["config"][1])

    def embed(s, Xtok):
        return sphere(yat(Xtok, s.We, s.be[0], s.be[1]))

    def block(s, h):
        t = yat(h.transpose(0, 2, 1), s.Wt, s.bt[0], s.bt[1]).transpose(0, 2, 1)
        h = sphere(h + s.a_t * t)
        c = yat(h, s.Wc, s.bc[0], s.bc[1])
        return sphere(h + s.a_c * c)

    def pooled(s, Xtok, r=None):
        h = s.embed(Xtok)
        for _ in range(s.r if r is None else r):
            h = s.block(h)
        return h.mean(1)

    def logits(s, Xtok, r=None):
        return s.pooled(Xtok, r) @ s.A + s.bias


def factored(pred, yte, f2c):
    fine = 100 * float((pred == yte).mean())
    gc = f2c[pred] == f2c[yte]
    co = 100 * float(gc.mean())
    fgc = 100 * float((pred[gc] == yte[gc]).mean()) if gc.any() else 0.0
    return round(fine, 2), round(co, 2), round(fgc, 2)


def batched_logits(M, Xtok, r=None, bs=1000):
    return np.concatenate([M.logits(Xtok[i:i + bs], r)
                           for i in range(0, len(Xtok), bs)])


def main():
    Xtr, ytr, Xte, yte, _, coarse = dataset("cifar100g")
    ctr, cte = coarse
    f2c = np.zeros(C, int)
    for f_, c_ in zip(yte, cte):
        f2c[f_] = c_
    Ttr, Tte = tokens(Xtr[:8000]), tokens(Xte)
    tax = json.load(open(os.path.join(HERE, "..", "public", "fifteen-ideas",
                                      "taxonomy.json")))
    run = json.load(open(os.path.join(BUNDLE, "yat_mixer.json")))

    models = [(s, Mixer(os.path.join(BUNDLE, f"mixer_trained_s{s}.npz")))
              for s in range(3)]

    # 1. reproduce the run
    for s, M in models:
        pred = batched_logits(M, Tte).argmax(1)
        fine, co, fgc = factored(pred, yte, f2c)
        row = [r for r in run["rows"] if r.get("seed") == s][-1]
        print(f"seed {s}: numpy {fine:.2f} = {co:.2f} x {fgc:.2f}  "
              f"(kaggle final {row['fine']:.2f} = {row['coarse']:.2f} x {row['fgc']:.2f})")

    # 2. depth transfer on the full test set
    transfer = []
    for s, M in models:
        rowT = []
        for r in range(13):
            pred = batched_logits(M, Tte, r=r).argmax(1)
            rowT.append(factored(pred, yte, f2c))
        transfer.append(rowT)
        print(f"seed {s} transfer fine: {[v[0] for v in rowT]}")

    # 3. the concept audit at the pooled layer (same instrument as the flat model)
    audit = []
    for s, M in models:
        Ftr = np.concatenate([M.pooled(Ttr[i:i + 1000]) for i in range(0, len(Ttr), 1000)])
        Fte = np.concatenate([M.pooled(Tte[i:i + 1000]) for i in range(0, len(Tte), 1000)])
        mu = Ftr.mean(0)
        base = mu @ M.A + M.bias

        class Shim:
            A = M.A
            C = 100
        con = i_concepts(Shim, Ftr - mu, Fte - mu, yte, base,
                         rng=np.random.default_rng(9_000 + s))
        named = int(sum(con["named"]))
        namedm = int(sum(con["named_matched"]))
        audit.append(dict(named=named, named_matched=namedm,
                          stability=np.round(con["stability"], 3).tolist(),
                          stability_matched=np.round(con["stability_matched"], 3).tolist(),
                          subspace={str(k): round(v["mean"], 4)
                                    for k, v in con["subspace"].items()},
                          curve=[round(v, 2) for v in con["curve"]]))
        print(f"seed {s}: identifiable concepts {named} of 100 per axis, "
              f"{namedm} once ordering is solved "
              f"(acc@15 {con['curve'][14]:.2f}, full {con['curve'][-1]:.2f}); "
              f"subspace k=40 {con['subspace'][40]['mean']:.3f}")

    # 4. eps vs the distances each layer sees (seed 0, one test batch)
    s0 = models[0][1]
    Xb = Tte[:2000]
    eps_rows = []
    d2e = ((Xb[..., None, :] - s0.We[None, None]) ** 2).sum(-1)
    eps_rows.append(dict(layer="embed", eps=float(s0.be[1]),
                         med=float(np.median(d2e)), min=float(d2e.min())))
    h = s0.embed(Xb)
    u = h.transpose(0, 2, 1).reshape(-1, T)[::7]
    d2t = np.maximum((u ** 2).sum(-1, keepdims=True) + (s0.Wt ** 2).sum(-1) - 2 * u @ s0.Wt.T, 0)
    eps_rows.append(dict(layer="token", eps=float(s0.bt[1]),
                         med=float(np.median(d2t)), min=float(d2t.min())))
    v = h.reshape(-1, s0.m)[::7]
    d2c = np.maximum((v ** 2).sum(-1, keepdims=True) + (s0.Wc ** 2).sum(-1) - 2 * v @ s0.Wc.T, 0)
    eps_rows.append(dict(layer="channel", eps=float(s0.bc[1]),
                         med=float(np.median(d2c)), min=float(d2c.min())))
    for e in eps_rows:
        print(f"eps[{e['layer']}] = {e['eps']:.4f}  median d2 {e['med']:.3f}  "
              f"min {e['min']:.5f}  ratio {e['eps'] / max(e['med'], 1e-12):.4f}")

    # ── payloads ──
    s0z = models[0][1]
    json.dump(dict(
        We=np.round(s0z.We, 3).tolist(), Wt=np.round(s0z.Wt, 4).tolist(),
        Wc=np.round(s0z.Wc, 3).tolist(), A=np.round(s0z.A, 3).tolist(),
        bias=np.round(s0z.bias, 4).tolist(),
        be=[round(float(x), 5) for x in s0z.be],
        bt=[round(float(x), 5) for x in s0z.bt],
        bc=[round(float(x), 5) for x in s0z.bc],
        a_t=round(s0z.a_t, 4), a_c=round(s0z.a_c, 4), r=s0z.r, m=s0z.m),
        open(os.path.join(OUT, "weights.json"), "w"), separators=(",", ":"))

    seed_rows = run["rows"][-3:]
    ladder = [r for r in run["rows"] if r["lr"] == run["lr"]][:5]
    json.dump(dict(
        curves=[r["curve"] for r in seed_rows],
        alphas=[[round(r["a_t"], 3), round(r["a_c"], 3)] for r in run["rows"]],
        lrs=[r["lr"] for r in run["rows"]],
        best=[r["best_acc"] for r in run["rows"]],
        ladder=[dict(r=r["r"], fine=r["fine"], coarse=r["coarse"], fgc=r["fgc"],
                     best=r["best_acc"], a_t=round(r["a_t"], 3)) for r in ladder],
        transfer=transfer,
        audit=audit,
        eps=eps_rows,
        flat_stability=json.load(open(os.path.join(
            HERE, "..", "public", "fifteen-ideas", "rebuild.json")))["stability"],
        flat=dict(fine=17.19, coarse=26.02, fgc=66.09, best=17.65, named=15.3)),
        open(os.path.join(OUT, "story.json"), "w"), separators=(",", ":"))

    sub = np.sort(RNG.choice(len(yte), 24, replace=False))
    json.dump(dict(
        idx=sub.tolist(),
        imgs=[np.round(Xte[i] * 255).astype(int).tolist() for i in sub],
        y=yte[sub].tolist(), coarse=cte[sub].tolist(),
        names=tax["names"], coarse_names=tax["coarse_names"],
        fine2coarse=tax["fine2coarse"]),
        open(os.path.join(OUT, "thumbs.json"), "w"), separators=(",", ":"))

    for f in sorted(os.listdir(OUT)):
        print(f"{f:14s} {os.path.getsize(os.path.join(OUT, f)) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
