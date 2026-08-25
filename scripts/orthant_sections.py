r"""Sections and orthants: does the readout's SIGN pattern carry the taxonomy?

The spectral line kept failing for one reason: the eigenbasis is chosen by the
measure, so it rotates, it needs a sample-size gate, and that gate turned out to
reward underfitting. The section basis has none of those problems. A kernel
section is k(., w) for a fixed w, the sections span the RKHS by the reproducing
property, and in a Yat network they ARE the prototypes: fixed by the parameters,
one picture each, invariant to the measure.

Because the Yat kernel is nonnegative, the class score y_c = sum_u A_uc phi_u(x)
has an unambiguous sign structure: with every phi_u >= 0, the sign of a
contribution is exactly the sign of A_uc. So each prototype carries a C-bit vote
pattern, sign(A[u]) in {-1,+1}^C, i.e. an ORTHANT of logit space, and the whole
readout is the matrix of those patterns.

The question that decides whether this is interpretable rather than merely
faithful: do the m vote patterns collapse into a few families a person can hold,
and does the ground-truth taxonomy fall out of them WITHOUT being told?

  families        effective number of distinct prototype vote-patterns
                  (participation ratio of the row-similarity spectrum)
  taxonomy        cluster the CLASS columns of sign(A) into 20 groups and score
                  against the real 20 superclasses by adjusted Rand index
  null            the same statistics after shuffling each column of A, which
                  destroys cross-class structure while preserving each class's
                  own sign balance

Run: python scripts/orthant_sections.py
Writes results/orthant_sections.json
"""

import json
import os

import numpy as np

from yat_audit import dataset, load_models

HERE = os.path.dirname(os.path.abspath(__file__))
C = 100


def participation(M):
    ev = np.linalg.eigvalsh(M)
    ev = np.maximum(ev, 0)
    return float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-30))


def ari(a, b):
    """Adjusted Rand index, written out so there is no sklearn dependency."""
    n = len(a)
    ua, ub = np.unique(a), np.unique(b)
    tab = np.zeros((len(ua), len(ub)))
    for i, x in enumerate(ua):
        for j, y in enumerate(ub):
            tab[i, j] = np.sum((a == x) & (b == y))
    comb = lambda x: x * (x - 1) / 2
    sij = comb(tab).sum()
    si = comb(tab.sum(1)).sum()
    sj = comb(tab.sum(0)).sum()
    exp = si * sj / comb(n)
    mx = (si + sj) / 2
    return float((sij - exp) / max(mx - exp, 1e-30))


def kmeans(X, k, rng, iters=60, restarts=8):
    best, bl = None, np.inf
    for _ in range(restarts):
        ctr = X[rng.choice(len(X), k, replace=False)]
        lab = np.zeros(len(X), int)
        for _ in range(iters):
            d = ((X[:, None, :] - ctr[None]) ** 2).sum(-1)
            lab = d.argmin(1)
            for j in range(k):
                if (lab == j).any():
                    ctr[j] = X[lab == j].mean(0)
        loss = ((X - ctr[lab]) ** 2).sum()
        if loss < bl:
            best, bl = lab.copy(), loss
    return best


def analyse(tag, A, f2c, rng, reps=16):
    S = np.sign(A)                                  # (m, C) vote patterns
    m = len(S)
    row = (S @ S.T) / S.shape[1]                    # prototype-to-prototype
    col = (S.T @ S) / m                             # class-to-class

    fam = participation(row)
    lab = kmeans(S.T / np.sqrt(m), 20, rng)
    a = ari(lab, f2c)

    # occupied orthants: how many prototypes share an identical pattern
    uniq = len(np.unique(S, axis=0))

    nulls_f, nulls_a = [], []
    for _ in range(reps):
        Ar = A.copy()
        for j in range(A.shape[1]):
            rng.shuffle(Ar[:, j])
        Sr = np.sign(Ar)
        nulls_f.append(participation((Sr @ Sr.T) / Sr.shape[1]))
        nulls_a.append(ari(kmeans(Sr.T / np.sqrt(m), 20, rng), f2c))

    print(f"\n  {tag}   (m={m} prototypes, {C} classes)")
    print(f"    distinct vote patterns      {uniq} of {m}")
    print(f"    effective families          {fam:8.1f}   null {np.mean(nulls_f):8.1f}")
    print(f"    taxonomy recovered, ARI     {a:8.3f}   null {np.mean(nulls_a):+8.3f}"
          f" +- {np.std(nulls_a):.3f}")
    verdict = ("TAXONOMY IS IN THE SIGNS" if a > np.mean(nulls_a) + 5 * np.std(nulls_a)
               else "no taxonomy structure beyond chance")
    print(f"    -> {verdict}")
    return dict(tag=tag, m=m, unique=uniq, families=fam,
                families_null=float(np.mean(nulls_f)), ari=a,
                ari_null=float(np.mean(nulls_a)), ari_null_sd=float(np.std(nulls_a)),
                verdict=verdict)


def main():
    Xtr, ytr, Xte, yte, names, coarse = dataset("cifar100g")
    f2c = np.zeros(C, int)
    for f_, c_ in zip(ytr, coarse[0]):
        f2c[f_] = c_
    rng = np.random.default_rng(0)
    out = []

    for seed, M in load_models(os.path.join(HERE, "results", "kgl_blog-cifar-v1"),
                               "trained", 12):
        out.append(analyse(f"flat bank, seed {seed}", M.A, f2c, rng))

    from export_mixer_viz import Mixer
    for s in (0, 1, 2):
        f = os.path.join(HERE, "results", "kgl_blog-mixer-v1", f"mixer_trained_s{s}.npz")
        if os.path.exists(f):
            out.append(analyse(f"mixer m=256, seed {s}", Mixer(f).A, f2c, rng))

    p = os.path.join(HERE, "results", "orthant_sections.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    print(f"\nwrote {os.path.relpath(p, HERE)}")


if __name__ == "__main__":
    main()
