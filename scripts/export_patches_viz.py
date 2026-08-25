"""Run the interrogation protocol on the patch networks, and export the post.

For each patch size the audit asks the three questions the whole-image model
answered badly:

  does eps engage?          eps against the distances between patches and
                            prototypes (whole images: 1e-4, the softening was
                            vestigial)
  which channel carries a   freeze alignment or proximity and re-score each
  concept?                  concept (whole images: fine concepts were ~100%
                            alignment)
  how narrow is a concept?  support size, the number that decides whether the
                            exact row edit can remove one

Plus the thing only mean pooling gives: the exact per-patch ballot, since
logits = mean_p (A^T phi(x_p)).

Writes public/patch-parts/{sizes,votes,parts}.json.
"""

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "public", "patch-parts")
os.makedirs(OUT, exist_ok=True)
CLS = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
       "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]


def bundle():
    c = sorted(g for g in glob.glob(os.path.join(HERE, "results", "kgl_blog-patches-*"))
               if "smoke" not in g)
    assert c, "no full patch bundle yet"
    return c[-1]


def load_data():
    import torchvision
    tr = torchvision.datasets.FashionMNIST("/tmp/fmnist", train=True, download=True)
    te = torchvision.datasets.FashionMNIST("/tmp/fmnist", train=False, download=True)
    f = lambda ds: (ds.data.numpy().astype(np.float32) / 255).reshape(-1, 28, 28)
    return f(tr), tr.targets.numpy(), f(te), te.targets.numpy()


def to_patches(X, P):
    n, g = X.shape[0], 28 // P
    return (X.reshape(n, g, P, g, P).transpose(0, 1, 3, 2, 4).reshape(n, g * g, P * P))


def channels(Xp, W, b, eps):
    """alignment and proximity, kept apart, over a flat stack of patches."""
    dot = Xp @ W.T
    d2 = np.maximum((Xp ** 2).sum(1, keepdims=True) + (W ** 2).sum(1) - 2 * dot, 0)
    return (dot + b) ** 2, 1.0 / (d2 + eps), d2


def audit(P, B, Xtr, ytr, Xte, yte, nsub=4000):
    z = np.load(os.path.join(B, f"patches_P{P}.npz"))
    W, A, bias = z["W"].astype(np.float64), z["A"].astype(np.float64), z["bias"].astype(np.float64)
    b, eps = float(z["b"]), float(z["eps"])
    g = 28 // P
    Ptr, Pte = to_patches(Xtr[:nsub], P), to_patches(Xte, P)
    npatch = Ptr.shape[1]

    def pooled(Pk):                                   # mean over patches
        flat = Pk.reshape(-1, P * P)
        N, Pi, d2 = channels(flat, W, b, eps)
        F = (N * Pi).reshape(len(Pk), npatch, -1).mean(1)
        return F, d2
    Ftr, _ = pooled(Ptr)
    Fte, d2te = pooled(Pte)
    mu = Ftr.mean(0)
    Ztr, Zte = Ftr - mu, Fte - mu
    base = mu @ A + bias
    acc = 100.0 * float(((Zte @ A + base).argmax(1) == yte).mean())

    # Y1: the concepts, from the logit covariance
    C = Ztr.T @ Ztr / len(Ztr)
    S = A.T @ C @ A
    sig, E = np.linalg.eigh(S)
    sig, E = np.maximum(sig[::-1], 0), E[:, ::-1]

    # Y2: the channel ledger, on the pooled features
    flat_tr, flat_te = Ptr.reshape(-1, P * P), Pte.reshape(-1, P * P)
    Ntr, Pitr, _ = channels(flat_tr, W, b, eps)
    Nte, Pite, _ = channels(flat_te, W, b, eps)
    mkpool = lambda X_, n_: X_.reshape(n_, npatch, -1).mean(1)
    banks = {
        "both": (mkpool(Nte * Pite, len(Pte)), mkpool(Ntr * Pitr, len(Ptr)).mean(0)),
        "align": (mkpool(Nte * Pitr.mean(0), len(Pte)), mkpool(Ntr * Pitr.mean(0), len(Ptr)).mean(0)),
        "place": (mkpool(Ntr.mean(0) * Pite, len(Pte)), mkpool(Ntr.mean(0) * Pitr, len(Ptr)).mean(0)),
    }
    concepts = []
    for j in range(min(6, A.shape[1])):
        v = A @ E[:, j]
        s = (banks["both"][0] - banks["both"][1]) @ v
        r = {}
        for k in ("align", "place"):
            F_, m_ = banks[k]
            r[k] = abs(float(np.corrcoef((F_ - m_) @ v, s)[0, 1]))
        tot = r["align"] ** 2 + r["place"] ** 2 + 1e-30
        support = float(np.abs(v).sum() ** 2 / max((v ** 2).sum(), 1e-30))
        e = E[:, j]
        order = np.argsort(-np.abs(v))
        cum = np.cumsum(v[order] ** 2) / max((v ** 2).sum(), 1e-30)
        concepts.append(dict(
            j=j, gain=float(sig[j]), support=support,
            n90=int(np.searchsorted(cum, 0.9) + 1),
            align=r["align"] ** 2 / tot, place=r["place"] ** 2 / tot,
            contrast=[CLS[int(np.argmax(e))], CLS[int(np.argmin(e))]],
            top_units=[int(i) for i in order[:12]]))
    med = float(np.median(d2te))
    return dict(P=P, npatch=int(npatch), grid=int(g), acc=acc, b=b, eps=eps,
                median_d2=med, min_d2=float(d2te.min()), eps_ratio=eps / med,
                concepts=concepts, m=int(W.shape[0])), (W, A, bias, b, eps, z)


def main():
    B = bundle()
    print("bundle:", B)
    D = json.load(open(os.path.join(B, "yat_patches.json")))
    Xtr, ytr, Xte, yte = load_data()
    sizes, keep = [], {}
    for P in D["sizes"]:
        r, mdl = audit(P, B, Xtr, ytr, Xte, yte)
        accs = [x["best_acc"] for x in D["rows"] if x["P"] == P]
        r["acc_mean"] = float(np.mean(accs))
        r["acc_std"] = float(np.std(accs))
        sizes.append(r)
        keep[P] = mdl
        print(f"P={P:2d} ({r['npatch']:2d} patches)  acc {r['acc_mean']:.2f}+-{r['acc_std']:.2f}"
              f"  eps={r['eps']:.4f}  eps/median d^2={r['eps_ratio']:.3e}"
              f"  concept1 place={100*r['concepts'][0]['place']:.0f}%"
              f"  support={r['concepts'][0]['support']:.0f}/{r['m']}")

    # the exact per-patch ballots, for the smallest useful patch size
    Pv = min(p for p in D["sizes"] if p <= 7) if any(p <= 7 for p in D["sizes"]) else D["sizes"][-1]
    z = keep[Pv][5]
    votes = z["votes"].astype(np.float32)          # (n, npatch, 10)
    labels = z["vote_labels"]
    g = 28 // Pv
    X14 = (Xte[:len(votes)].reshape(-1, 14, 2, 14, 2).mean((2, 4)) * 255)
    pick = list(range(24))
    json.dump(dict(P=Pv, grid=g, classes=CLS,
                   y=[int(labels[i]) for i in pick],
                   thumbs=[np.round(X14[i]).astype(int).reshape(-1).tolist() for i in pick],
                   votes=[np.round(votes[i], 3).tolist() for i in pick]),
              open(os.path.join(OUT, "votes.json"), "w"), separators=(",", ":"))

    # the parts themselves: prototype patches as pictures, per size
    parts = {}
    for P, (W, A, bias, b, eps, z) in keep.items():
        lo, hi = float(W.min()), float(W.max())
        parts[str(P)] = dict(
            P=P, protos=np.round((W[:48] - lo) / (hi - lo + 1e-12), 4).tolist(),
            contrast=[c["contrast"] for c in next(s for s in sizes if s["P"] == P)["concepts"]],
            top_units=[c["top_units"] for c in next(s for s in sizes if s["P"] == P)["concepts"]])
    json.dump(dict(sizes=sizes), open(os.path.join(OUT, "sizes.json"), "w"), separators=(",", ":"))
    json.dump(parts, open(os.path.join(OUT, "parts.json"), "w"), separators=(",", ":"))
    for f in sorted(os.listdir(OUT)):
        print(f"{f:12s} {os.path.getsize(os.path.join(OUT, f))/1e6:.2f} MB")


if __name__ == "__main__":
    main()
