r"""Sufficiency and necessity on an EXACT decomposition, per input.

Four instruments in this line have now detected real structure and then failed
to be useful, and three of them ranked the WORST model most interpretable. They
share a flaw: each scores the model against a human-supplied structure (twenty
superclasses, WordNet, the confusion matrix). A more capable network is not more
human-aligned inside, it is more task-optimal, so "how much clean global
structure sits in this matrix" is a proxy for LOW CAPACITY and will anti-correlate
with quality almost by construction.

So score against the model's own behaviour under intervention instead.

For an input x with prediction c, the score is exactly

    y_c(x) = sum_u A[u, c] * phi_u(x) + bias_c

with no approximation anywhere, which is the one property of this architecture
that does not decay with scale. Rank the units by their signed contribution to
the predicted class and ask two causal questions:

  SUFFICIENCY  keep the top-k contributors, zero every other unit, recompute all
               logits. Does the prediction survive? k_suff is the smallest k
               that holds the verdict on 90 percent of inputs.
  NECESSITY    zero the top-k instead. k_nec is the smallest k that flips half
               of them.

Both are per-input and both are verified by intervention rather than by
agreement with a taxonomy. The prediction that makes this worth running: a model
with cleaner mechanisms should need FEWER units to hold its decision, so k_suff
should FALL as accuracy rises. If it rises instead, this instrument fails the
same way the others did and the whole line should be reconsidered.

Nulls: random units, and units matched for contribution magnitude, so "the top-k
matter" is measured against "any k of that size matter".

Run: python scripts/decision_support.py
Writes results/decision_support.json
"""

import json
import os

import numpy as np

from yat_audit import dataset, load_models
from yat_forecast import features_chunked

HERE = os.path.dirname(os.path.abspath(__file__))
NTEST = 2000
KS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def curves(F, A, bias, y, ks=KS, rng=None):
    """F: (n, m) nonneg features. Returns sufficiency and necessity curves."""
    n, m = F.shape
    logits = F @ A + bias
    pred = logits.argmax(1)
    contrib = F * A[:, pred].T                       # (n, m) signed, per unit
    order = np.argsort(-contrib, 1)                  # best contributor first
    keep_ok, drop_ok = {}, {}
    rnd_ok = {}
    for k in ks:
        if k > m:
            continue
        mask = np.zeros((n, m), bool)
        np.put_along_axis(mask, order[:, :k], True, 1)
        # sufficiency: only the top-k units are allowed to speak
        s = ((F * mask) @ A + bias).argmax(1)
        keep_ok[k] = float((s == pred).mean())
        # necessity: the top-k are silenced
        d = ((F * ~mask) @ A + bias).argmax(1)
        drop_ok[k] = float((d == pred).mean())
        # null: k random units kept
        rm = np.zeros((n, m), bool)
        ridx = np.argsort(rng.random((n, m)), 1)[:, :k]
        np.put_along_axis(rm, ridx, True, 1)
        r = ((F * rm) @ A + bias).argmax(1)
        rnd_ok[k] = float((r == pred).mean())
    first = lambda d, thr, up: next((k for k in sorted(d) if (d[k] >= thr if up
                                                             else d[k] <= thr)), None)
    return dict(suff=keep_ok, nec=drop_ok, rnd=rnd_ok,
                k_suff=first(keep_ok, 0.90, True),
                k_nec=first(drop_ok, 0.50, False),
                k_rnd=first(rnd_ok, 0.90, True))


def main():
    Xtr, ytr, Xte, yte, names, coarse = dataset("cifar100g")
    Xs, ys = Xte[:NTEST], yte[:NTEST]
    rng = np.random.default_rng(0)
    rows = []

    print(f"exact decomposition, {NTEST} test inputs, intervention on the units\n")
    print(f"  {'model':<26}{'acc':>7}{'m':>6}{'k_suff':>8}{'k_nec':>7}{'k_rand':>8}")

    for seed, M in load_models(os.path.join(HERE, "results", "kgl_blog-cifar-v1"),
                               "trained", 12):
        F = features_chunked(M, Xs)
        acc = 100.0 * float(((F @ M.A + M.bias).argmax(1) == ys).mean())
        c = curves(F, M.A, M.bias, ys, rng=rng)
        rows.append(dict(model=f"flat m={M.m} s{seed}", acc=acc, m=int(M.m), **c))
        print(f"  {'flat m=1024 seed ' + str(seed):<26}{acc:>7.2f}{M.m:>6}"
              f"{str(c['k_suff']):>8}{str(c['k_nec']):>7}{str(c['k_rnd']):>8}")

    from export_mixer_viz import Mixer, tokens
    Tte = tokens(Xs)
    for s in (0, 1, 2):
        f = os.path.join(HERE, "results", "kgl_blog-mixer-v1", f"mixer_trained_s{s}.npz")
        if not os.path.exists(f):
            continue
        mx = Mixer(f)
        P = np.concatenate([mx.pooled(Tte[i:i + 1000]) for i in range(0, len(Tte), 1000)])
        acc = 100.0 * float(((P @ mx.A + mx.bias).argmax(1) == ys).mean())
        # pooled mixer features are SIGNED (a_t < 0 leaves the cone), so rank by
        # signed contribution exactly as above; the arithmetic is unchanged.
        c = curves(P, mx.A, mx.bias, ys, rng=rng)
        rows.append(dict(model=f"mixer m=256 s{s}", acc=acc, m=int(P.shape[1]), **c))
        print(f"  {'mixer m=256 seed ' + str(s):<26}{acc:>7.2f}{P.shape[1]:>6}"
              f"{str(c['k_suff']):>8}{str(c['k_nec']):>7}{str(c['k_rnd']):>8}")

    # the question this was built to answer
    mix = [r for r in rows if "mixer" in r["model"] and r["k_suff"]]
    if len(mix) > 2:
        a = np.array([r["acc"] for r in mix]); k = np.array([r["k_suff"] for r in mix], float)
        rho = float(np.corrcoef(a, k)[0, 1])
        print(f"\n  within the mixer seeds, corr(accuracy, k_suff) = {rho:+.3f}")
        print(f"  -> {'TIGHTER EXPLANATIONS FOR BETTER MODELS' if rho < -0.5 else 'does not reward capability'}")

    # as a FRACTION of width, which is the scale-free version
    for r in rows:
        r["frac_suff"] = (r["k_suff"] / r["m"]) if r["k_suff"] else None
    print("\n  scale-free: fraction of units needed to hold the verdict")
    for r in rows:
        if r["frac_suff"]:
            print(f"    {r['model']:<20} {r['acc']:5.2f}%   {100*r['frac_suff']:5.2f}% of units")

    p = os.path.join(HERE, "results", "decision_support.json")
    json.dump(rows, open(p, "w"), indent=1, default=float)
    print(f"\nwrote {os.path.relpath(p, HERE)}")


if __name__ == "__main__":
    main()
