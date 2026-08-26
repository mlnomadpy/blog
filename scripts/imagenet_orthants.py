r"""Does the readout's SIGN structure carry the taxonomy in models nobody here trained?

The section-and-orthant reading has two halves, and only one of them is Yat-
specific. "Sections are prototypes, so each unit is a picture" needs a kernel
network. But "the class score is a signed sum over units, so sign(A) assigns
every unit an orthant of logit space, and the orthant pattern is a fixed,
measure-independent object" needs nothing except a final linear layer, which is
what essentially every classifier ends in.

So the instrument travels, and it travels cheaply: the whole analysis reads ONE
MATRIX per model. No forward passes, no dataset, no dictionary to train. That
is the property that makes it a candidate for scale, where activation probing
and dictionary learning both need a corpus and a fit.

Here it is pointed at pretrained ImageNet classifiers, where the ground truth is
richer than anything in this repo: WordNet, a human-curated hierarchy over the
thousand classes, plus one anchor group that needs no external data at all
(indices 151-268 are the 118 dog breeds).

The question that matters is not whether structure exists. It is whether the
instrument TRACKS MODEL QUALITY, because the spectral naming gate did the
opposite: it ranked the worst network most interpretable. So several
architectures spanning 70 to 83 percent top-1 go through identical code.

Run: python scripts/imagenet_orthants.py
Writes results/imagenet_orthants.json
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DOGS = slice(151, 269)          # Chihuahua .. Mexican hairless, 118 breeds
DEPTH = 5                       # where to cut the WordNet hypernym tree


def readouts():
    """(name, top1, final linear weight [C, d]) for a spread of architectures."""
    import torch
    from torchvision import models as tvm
    spec = [
        ("resnet18", 69.76, tvm.resnet18, tvm.ResNet18_Weights.IMAGENET1K_V1, "fc"),
        ("efficientnet_b0", 77.69, tvm.efficientnet_b0,
         tvm.EfficientNet_B0_Weights.IMAGENET1K_V1, "classifier.1"),
        ("resnet50", 80.86, tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V2, "fc"),
        ("convnext_tiny", 82.52, tvm.convnext_tiny,
         tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1, "classifier.2"),
    ]
    out = []
    for name, top1, ctor, w, attr in spec:
        m = ctor(weights=w)
        mod = m
        for part in attr.split("."):
            mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
        A = mod.weight.detach().numpy().astype(np.float64)      # (1000, d)
        out.append((name, top1, A))
        print(f"  loaded {name:<16} top1 {top1:.2f}   readout {A.shape}")
    return out


def wordnet_groups(depth=DEPTH):
    from nltk.corpus import wordnet as wn
    from torchvision.models import ResNet50_Weights
    cats = ResNet50_Weights.IMAGENET1K_V2.meta["categories"]
    lab, miss = [], 0
    for c in cats:
        ss = wn.synsets(c.replace(" ", "_"), pos="n")
        if not ss:
            lab.append("__none__"); miss += 1; continue
        path = ss[0].hypernym_paths()[0]
        lab.append(path[min(depth, len(path) - 1)].name())
    u = {n: i for i, n in enumerate(sorted(set(lab)))}
    g = np.array([u[n] for n in lab])
    print(f"  WordNet at depth {depth}: {len(u)} groups, {miss} unresolved")
    return g, cats


def ari(a, b):
    ua, ub = np.unique(a), np.unique(b)
    tab = np.zeros((len(ua), len(ub)))
    for i, x in enumerate(ua):
        for j, y in enumerate(ub):
            tab[i, j] = np.sum((a == x) & (b == y))
    comb = lambda x: x * (x - 1) / 2
    sij, si, sj = comb(tab).sum(), comb(tab.sum(1)).sum(), comb(tab.sum(0)).sum()
    exp, mx = si * sj / comb(len(a)), (si + sj) / 2
    return float((sij - exp) / max(mx - exp, 1e-30))


def kmeans(X, k, rng, iters=40, restarts=4):
    best, bl = None, np.inf
    for _ in range(restarts):
        ctr = X[rng.choice(len(X), k, replace=False)].copy()
        for _ in range(iters):
            lab = ((X[:, None, :] - ctr[None]) ** 2).sum(-1).argmin(1)
            for j in range(k):
                if (lab == j).any():
                    ctr[j] = X[lab == j].mean(0)
        loss = ((X - ctr[lab]) ** 2).sum()
        if loss < bl:
            best, bl = lab.copy(), loss
    return best


def cohesion(S, mask):
    """Mean pairwise sign agreement inside a group minus outside it."""
    G = (S @ S.T) / S.shape[1]
    iu = np.triu_indices(len(S), 1)
    inside = mask[iu[0]] & mask[iu[1]]
    return float(G[iu][inside].mean() - G[iu][~inside].mean())


def main():
    print("loading pretrained readouts (one matrix each, no forward passes)\n")
    models = readouts()
    groups, cats = wordnet_groups()
    K = len(np.unique(groups))
    dogmask = np.zeros(1000, bool); dogmask[DOGS] = True
    rng = np.random.default_rng(0)
    rows = []

    print(f"\n  {'model':<16} {'top1':>6} {'dog cohesion':>14} {'null':>8} "
          f"{'wordnet ARI':>12} {'null':>8}")
    for name, top1, A in models:
        S = np.sign(A)                       # (1000, d): each class's orthant
        d = S.shape[1]
        coh = cohesion(S, dogmask)
        lab = kmeans(S / np.sqrt(d), K, rng)
        a = ari(lab, groups)

        cn, an = [], []
        for _ in range(8):
            Ar = A.copy()
            for j in range(d):
                rng.shuffle(Ar[:, j])        # break cross-class structure, keep balance
            Sr = np.sign(Ar)
            cn.append(cohesion(Sr, dogmask))
            an.append(ari(kmeans(Sr / np.sqrt(d), K, rng), groups))
        rows.append(dict(model=name, top1=top1, dim=int(d), dog=coh,
                         dog_null=float(np.mean(cn)), dog_null_sd=float(np.std(cn)),
                         ari=a, ari_null=float(np.mean(an)),
                         ari_null_sd=float(np.std(an))))
        print(f"  {name:<16} {top1:>6.2f} {coh:>14.4f} {np.mean(cn):>8.4f} "
              f"{a:>12.4f} {np.mean(an):>+8.4f}")

    t = np.array([r["top1"] for r in rows]); q = np.array([r["ari"] for r in rows])
    dd = np.array([r["dog"] for r in rows])
    rho = float(np.corrcoef(t, q)[0, 1]); rd = float(np.corrcoef(t, dd)[0, 1])
    print(f"\n  does the instrument track model quality?")
    print(f"    corr(top1, wordnet ARI)   {rho:+.3f}")
    print(f"    corr(top1, dog cohesion)  {rd:+.3f}")
    print(f"    -> {'TRACKS QUALITY' if min(rho, rd) > 0.5 else 'does not clearly track quality'}"
          f"   (contrast: the spectral naming gate ranked the WORST model first)")

    p = os.path.join(HERE, "results", "imagenet_orthants.json")
    json.dump(dict(depth=DEPTH, groups=int(K), rows=rows,
                   corr_ari=rho, corr_dog=rd), open(p, "w"), indent=1, default=float)
    print(f"\nwrote {os.path.relpath(p, HERE)}")


if __name__ == "__main__":
    main()
