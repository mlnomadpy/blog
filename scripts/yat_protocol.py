r"""The Yat interrogation protocol: a standard audit for a trained Yat network.

Everything here exploits something specific to the Yat unit

    phi_w(x) = (w . x + b)^2 / (||x - w||^2 + eps)
               \___________/   \_________________/
                 alignment          proximity

and would be meaningless or impossible for a generic kernel: the unit is a
PRODUCT of two named channels, its centres live in input space (so anything in
feature space renders back as a picture), and b/eps are dials with kernel
meaning that can be turned after training.

  Y1  concepts      the network's function factors through a <= C-dimensional
                    projection (the readout is m x C), so eigendecomposing
                    S = A^T C A gives the concepts the model actually uses,
                    ranked by output gain, exact at rank C. Each renders as a
                    signed combination of prototype PICTURES.
  Y2  channel       freeze one channel at its per-unit data mean and re-measure
      ledger        each concept: is this concept carried by alignment (a
                    direction) or by proximity (a place)?
  Y3  eps dial      sweep the softening length with the weights frozen. Reports
                    eps / median distance^2: the admissibility audit, i.e.
                    whether the trained model is anywhere near the regime the
                    kernel's theory describes.
  Y4  support       V = A E gives each prototype's contribution to each concept.
      + write path  Concepts are READ here; the write path is the series' exact
                    row edit, so this also tests deleting a concept's support
                    rows and measures targeted vs collateral damage.
  Y5  b dial        b buys the universality theorem through the linear term of
                    (w.x + b)^2; sweep it to see which concepts depend on it.
  H   hygiene       bootstrap the measure and report each concept's stability,
                    so no one names a direction the data cannot resolve.

Run: python scripts/yat_protocol.py [--bundle DIR] [--arm trained] [--epoch 12]
Writes results/yat_protocol_<arm>.json (+ concept pictures npz).
"""

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLS = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
       "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]


# ── the model under audit ────────────────────────────────────────────────────
class YatModel:
    """A trained Yat layer + linear readout, with its two channels kept apart."""

    def __init__(s, W, b, eps, A, bias):
        s.W, s.b, s.eps, s.A, s.bias = W, b, eps, A, bias
        s.m, s.d = W.shape
        s.C = A.shape[1]

    def channels(s, X, eps=None, b=None):
        """The two halves of the kernel, separately: alignment and proximity."""
        eps = s.eps if eps is None else eps
        b = s.b if b is None else b
        dot = X @ s.W.T
        d2 = np.maximum((X ** 2).sum(1, keepdims=True) + (s.W ** 2).sum(1) - 2 * dot, 0)
        return (dot + b) ** 2, 1.0 / (d2 + eps), d2

    def features(s, X, eps=None, b=None):
        N, Pinv, _ = s.channels(X, eps, b)
        return N * Pinv

    @staticmethod
    def load(bundle, arm="trained", epoch=12):
        z = np.load(os.path.join(bundle, f"mercer_{arm}_s0.npz"))
        return YatModel(z[f"W{epoch}"].astype(np.float64), float(z["b"][epoch]),
                        float(z["eps"][epoch]), z[f"A{epoch}"].astype(np.float64),
                        z[f"bias{epoch}"].astype(np.float64))


def fmnist():
    import torchvision
    tr = torchvision.datasets.FashionMNIST("/tmp/fmnist", train=True, download=True)
    te = torchvision.datasets.FashionMNIST("/tmp/fmnist", train=False, download=True)
    f = lambda ds: (ds.data.numpy().astype(np.float32) / 255).reshape(-1, 784)
    return f(tr), tr.targets.numpy(), f(te), te.targets.numpy()


# ── Y1: the concepts the network actually uses ───────────────────────────────
def y1_concepts(M, Ztr, Zte, yte, base):
    """Eigendecompose the OUTPUT covariance, not the feature covariance."""
    C = Ztr.T @ Ztr / len(Ztr)
    S = M.A.T @ C @ M.A                        # (C x C): covariance of the logits
    sig, E = np.linalg.eigh(S)
    sig, E = np.maximum(sig[::-1], 0), E[:, ::-1]
    lam, U = np.linalg.eigh(C)                 # the Mercer spectrum, for pricing
    lam, U = np.maximum(lam[::-1], 0), U[:, ::-1]

    acc = lambda P: 100.0 * float((((Zte @ M.A) @ P + base).argmax(1) == yte).mean())
    full = acc(np.eye(M.C))
    fun_curve = [(k, acc(E[:, :k] @ E[:, :k].T)) for k in range(1, M.C + 1)]
    var_curve = []
    for k in [1, 2, 4, 8, 16, 32, 64, 128, M.m]:
        P = U[:, :k] @ U[:, :k].T
        var_curve.append((k, 100.0 * float(((Zte @ P @ M.A + base).argmax(1) == yte).mean())))

    concepts = []
    for j in range(M.C):
        e = E[:, j]
        v = M.A @ e                            # weights over prototypes
        c = U.T @ v
        price = float((c ** 2 / np.maximum(lam, 1e-12)).sum())
        smooth = float((c ** 2)[:16].sum() / max((c ** 2).sum(), 1e-30))
        support = float(np.abs(v).sum() ** 2 / max((v ** 2).sum(), 1e-30))
        pic = (v[:, None] * M.W).sum(0) / max(np.abs(v).sum(), 1e-30)
        concepts.append(dict(
            j=j, gain=float(sig[j]), price=price, smooth_frac=smooth,
            support=support, contrast=[CLS[int(np.argmax(e))], CLS[int(np.argmin(e))]],
            e=e.tolist(), v=v, pic=pic,
            top_units=[int(i) for i in np.argsort(-np.abs(v))[:12]]))
    return dict(concepts=concepts, sig=sig, E=E, lam=lam, U=U, full=full,
                fun_curve=fun_curve, var_curve=var_curve)


# ── Y2: which half of the kernel carries each concept ────────────────────────
def y2_channels(M, Xtr, Xte, concepts):
    Ntr, Ptr, _ = M.channels(Xtr)
    Nte, Pte, _ = M.channels(Xte)
    Fte = Nte * Pte
    banks = {                                   # freeze one channel at its mean
        "both": (Fte, (Ntr * Ptr).mean(0)),
        "align": (Nte * Ptr.mean(0), (Ntr * Ptr.mean(0)).mean(0)),
        "place": (Ntr.mean(0) * Pte, (Ntr.mean(0) * Ptr).mean(0)),
    }
    out = []
    for c in concepts:
        v = c["v"]
        s = (banks["both"][0] - banks["both"][1]) @ v
        r = {}
        for k in ("align", "place"):
            F, mu = banks[k]
            r[k] = abs(float(np.corrcoef((F - mu) @ v, s)[0, 1]))
        tot = r["align"] ** 2 + r["place"] ** 2 + 1e-30
        out.append(dict(j=c["j"], align_frac=r["align"] ** 2 / tot,
                        place_frac=r["place"] ** 2 / tot))
    return out


# ── Y3: the eps dial, and the admissibility audit ────────────────────────────
def y3_eps(M, Xtr, Xte, yte, mults=(1, 32, 1024, 1e4, 3e4, 1e5, 3e5)):
    _, _, d2 = M.channels(Xte)
    med = float(np.median(d2))
    rows = []
    for mult in mults:
        eps = M.eps * mult
        Ftr, Fte = M.features(Xtr, eps=eps), M.features(Xte, eps=eps)
        mu = Ftr.mean(0)
        acc = 100.0 * float((((Fte - mu) @ M.A + (mu @ M.A + M.bias)).argmax(1) == yte).mean())
        rows.append(dict(eps=eps, ratio=eps / med, acc=acc))
    return dict(trained_eps=M.eps, median_d2=med, ratio=M.eps / med,
                min_d2=float(d2.min()), sweep=rows,
                engages=bool(M.eps / med > 0.01))


# ── Y4: who carries a concept, and can we edit it there ──────────────────────
def y4_support(M, Zte, yte, base, mu, concepts, topk=(4, 16, 64)):
    pred0 = (Zte @ M.A + base).argmax(1)
    full = 100.0 * float((pred0 == yte).mean())

    def pair_damage(pred):
        P = np.zeros((10, 10))
        for t, p in zip(yte, pred):
            if t != p:
                P[min(t, p), max(t, p)] += 1
        return P
    P0 = pair_damage(pred0)
    out = []
    for c in concepts[:6]:
        v, rows = c["v"], c["top_units"]
        ent = []
        for k in topk:                          # the series' exact row edit
            A2 = M.A.copy()
            A2[np.argsort(-np.abs(v))[:k]] = 0
            base2 = mu @ A2 + M.bias          # the constant moves with the edit
            pred = (Zte @ A2 + base2).argmax(1)
            d = pair_damage(pred) - P0
            i, jj = np.unravel_index(np.argmax(d), d.shape)
            ent.append(dict(k=k, acc=100.0 * float((pred == yte).mean()),
                            worst=[CLS[i], CLS[jj]],
                            conc=float(d.max() / max(d[d > 0].sum(), 1))))
        # the logit-space cut of the same concept, for comparison
        P = np.eye(M.C) - np.outer(c["e"], c["e"])
        predc = ((Zte @ M.A) @ P + base).argmax(1)
        d = pair_damage(predc) - P0
        i, jj = np.unravel_index(np.argmax(d), d.shape)
        out.append(dict(j=c["j"], contrast=c["contrast"], support=c["support"],
                        row_edits=ent,
                        concept_cut=dict(acc=100.0 * float((predc == yte).mean()),
                                         worst=[CLS[i], CLS[jj]],
                                         conc=float(d.max() / max(d[d > 0].sum(), 1)))))
    return dict(full=full, concepts=out)


# ── Y5: the b dial (the term that buys universality) ─────────────────────────
def y5_b(M, Xtr, Xte, yte, mults=(0.0, 0.25, 1.0, 4.0, 16.0)):
    rows = []
    for mult in mults:
        b = M.b * mult
        Ftr, Fte = M.features(Xtr, b=b), M.features(Xte, b=b)
        mu = Ftr.mean(0)
        acc = 100.0 * float((((Fte - mu) @ M.A + (mu @ M.A + M.bias)).argmax(1) == yte).mean())
        rows.append(dict(b=b, mult=mult, acc=acc))
    return dict(trained_b=M.b, sweep=rows)


# ── H: can the data resolve these directions at all? ─────────────────────────
def hygiene(M, Zte, res, B=24, seed=0):
    """Bootstrap the measure; report each concept's subspace stability."""
    rng = np.random.default_rng(seed)
    E0, n = res["E"], len(Zte)
    angles = np.zeros(M.C)
    for _ in range(B):
        idx = rng.integers(0, n, n)
        Zb = Zte[idx]
        Cb = Zb.T @ Zb / n
        Sb = M.A.T @ Cb @ M.A
        _, Eb = np.linalg.eigh(Sb)
        Eb = Eb[:, ::-1]
        for j in range(M.C):
            angles[j] += abs(float(E0[:, j] @ Eb[:, j]))
    return (angles / B).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=os.path.join(HERE, "results", "kgl_blog-mercer-v1"))
    ap.add_argument("--arm", default="trained")
    ap.add_argument("--epoch", type=int, default=12)
    a = ap.parse_args()

    M = YatModel.load(a.bundle, a.arm, a.epoch)
    Xtr, ytr, Xte, yte = fmnist()
    Ftr, Fte = M.features(Xtr), M.features(Xte)
    mu = Ftr.mean(0)
    Ztr, Zte = Ftr - mu, Fte - mu
    base = mu @ M.A + M.bias

    print(f"=== Yat protocol :: arm={a.arm} m={M.m} b={M.b:.3f} eps={M.eps:.4f} ===\n")
    res = y1_concepts(M, Ztr, Zte, yte, base)
    ch = y2_channels(M, Xtr, Xte, res["concepts"])
    stab = hygiene(M, Zte, res)

    print(f"Y1  the network is {M.C} concepts (exact), not {M.m} units")
    print(f"    full model {res['full']:.2f}%   function basis: " +
          " ".join(f"k{k}={v:.1f}" for k, v in res["fun_curve"][:10]))
    print( "    variance basis (for contrast):     " +
          " ".join(f"k{k}={v:.1f}" for k, v in res["var_curve"]))
    print("\n    concept | gain  | contrast              | align | place | support | smooth | stable")
    for c, cc, st in zip(res["concepts"][:8], ch, stab):
        print(f"      {c['j']+1:2d}    |{c['gain']:6.0f} | {c['contrast'][0]:>8s} vs {c['contrast'][1]:<9s} |"
              f" {100*cc['align_frac']:4.0f}% | {100*cc['place_frac']:4.0f}% |"
              f"  {c['support']:5.0f}  | {100*c['smooth_frac']:4.0f}%  | {st:.3f}")

    e = y3_eps(M, Xtr, Xte, yte)
    print(f"\nY3  admissibility: eps={e['trained_eps']:.4f} vs median d^2={e['median_d2']:.1f}"
          f"  ->  ratio {e['ratio']:.2e}")
    print(f"    the softening {'ENGAGES' if e['engages'] else 'NEVER ENGAGES'}"
          f" (closest point in the test set sits at d^2={e['min_d2']:.1f})")
    for r in e["sweep"]:
        print(f"      eps {r['eps']:10.2f}  (x median d^2: {r['ratio']:7.3f})   acc {r['acc']:6.2f}%")

    bb = y5_b(M, Xtr, Xte, yte)
    print(f"\nY5  b dial (trained b={bb['trained_b']:.3f}): " +
          "  ".join(f"{r['mult']:g}x -> {r['acc']:.2f}%" for r in bb["sweep"]))

    sup = y4_support(M, Zte, yte, base, mu, res["concepts"])
    print(f"\nY4  the write path: delete a concept's top-support prototype rows")
    print( "    concept                 | support |  cut in logit space | delete 4 rows | 16 | 64")
    for s in sup["concepts"]:
        r = {x["k"]: x for x in s["row_edits"]}
        print(f"      {s['contrast'][0]:>8s} vs {s['contrast'][1]:<9s} |  {s['support']:5.0f}  |"
              f"   {s['concept_cut']['acc']:5.2f}% ({s['concept_cut']['worst'][0][:4]}/{s['concept_cut']['worst'][1][:4]})"
              f"  |  {r[4]['acc']:5.2f}%     | {r[16]['acc']:5.2f}% | {r[64]['acc']:5.2f}%")

    out = dict(arm=a.arm, m=M.m, b=M.b, eps=M.eps, full=res["full"],
               fun_curve=res["fun_curve"], var_curve=res["var_curve"],
               concepts=[{k: v for k, v in c.items() if k not in ("v", "pic", "e")}
                         for c in res["concepts"]],
               channels=ch, stability=stab, eps_dial=e, b_dial=bb, write_path=sup)
    p = os.path.join(HERE, "results", f"yat_protocol_{a.arm}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    np.savez_compressed(os.path.join(HERE, "results", f"yat_protocol_{a.arm}_pics.npz"),
                        pics=np.stack([c["pic"] for c in res["concepts"]]),
                        V=np.stack([c["v"] for c in res["concepts"]]))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
