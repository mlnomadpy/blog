r"""The same kernel-theory diagnosis, run on the PATCH MIXER, against the flat
whole-image bank as the control.

`yat_diagnose.py` asked why the flat model sits at 17 percent. That model is
one Yat bank on the whole 32x32 grayscale image. The architecture the series
actually moved to is the patch mixer: 8x8 patches, sixteen tokens, one
weight-tied block applied four times, mean pooling, linear readout, 21.4
percent at a tenth of the parameters.

So the question worth asking is not "why is the flat model stuck" but "did
patching and mixing change the DIAGNOSIS, or only the score". Four quantities
decide it, all read off frozen features:

  ridge ceiling     the best linear head those features admit, versus the head
                    SGD actually found: separates optimization from the rest
  spectral decay    lambda_k ~ k^-1/p. p near 1 means no useful decay: the
                    features spread label information thin instead of
                    concentrating it, and the KRR rate n^-2r/(2r+p) crawls
  alignment         centered kernel-target alignment: how much of the label
                    kernel this feature map can express at all
  effective dim     N(lam) = sum lambda_k/(lambda_k+lam)

Read at three depths of the mixer (patch embedding, after mixing, pooled) so
the contribution of each stage is priced separately.

Run: python scripts/mixer_diagnose.py
Writes results/mixer_diagnose.json
"""

import json
import os

import numpy as np

from export_mixer_viz import Mixer, tokens
from yat_audit import dataset, load_models
from yat_diagnose import cka, decay_exponent, eff_dim, ridge_acc
from yat_forecast import features_chunked

HERE = os.path.dirname(os.path.abspath(__file__))
MIX = os.path.join(HERE, "results", "kgl_blog-mixer-v1")
FLAT = os.path.join(HERE, "results", "kgl_blog-cifar-v1")
C = 100


def chunk_apply(fn, X, chunk=2000):
    return np.concatenate([fn(X[i:i + chunk]) for i in range(0, len(X), chunk)])


def report(tag, Ftr, ytr, Fte, yte, trained=None):
    acc, lam, ev = ridge_acc(Ftr, ytr, Fte, yte, C)
    slope, p = decay_exponent(ev)
    al = cka(Ftr, ytr, C)
    n = len(Ftr)
    row = dict(tag=tag, dim=int(Ftr.shape[1]), ridge=acc, lam=lam,
               decay=slope, p=p, cka=al,
               eff_dim_1e4=eff_dim(ev, 1e-4 * n),
               top15_share=float(ev[:15].sum() / ev.sum()),
               trained=trained)
    t = f"{trained:6.2f}%" if trained is not None else "     . "
    print(f"  {tag:<28} d={row['dim']:>5}  SGD {t}  ridge {acc:6.2f}%   "
          f"k^-{slope:4.2f} (p={p:4.2f})   CKA {al:.4f}   "
          f"N(1e-4)={row['eff_dim_1e4']:6.1f}   top15 {100*row['top15_share']:4.1f}%")
    return row


def main():
    Xtr, ytr, Xte, yte, names, coarse = dataset("cifar100g")
    rows = []
    print("CIFAR-100 grayscale. Ridge = the best linear head those features admit.\n")

    # ── the flat whole-image bank, as the control ──
    seed, M = load_models(FLAT, "trained", 12)[0]
    Ftr, Fte = features_chunked(M, Xtr), features_chunked(M, Xte)
    flat_acc = 100.0 * float((np.argmax(Fte @ M.A + M.bias, 1) == yte).mean())
    print("── control: one Yat bank on the whole image ──")
    rows.append(report("flat bank m=1024", Ftr, ytr, Fte, yte, flat_acc))
    rows.append(report("raw grayscale pixels", Xtr, ytr, Xte, yte))

    # ── the patch mixer, at three depths ──
    print("\n── the patch mixer: 8x8 patches, 16 tokens, weight-tied block x4 ──")
    Ttr, Tte = tokens(Xtr), tokens(Xte)
    for s in (0, 1, 2):
        f = os.path.join(MIX, f"mixer_trained_s{s}.npz")
        if not os.path.exists(f):
            continue
        mx = Mixer(f)
        pool = lambda X, m=mx: m.pooled(X)
        Ptr = chunk_apply(pool, Ttr)
        Pte = chunk_apply(pool, Tte)
        acc = 100.0 * float((np.argmax(Pte @ mx.A + mx.bias, 1) == yte).mean())
        rows.append(report(f"mixer seed {s}, pooled (r=4)", Ptr, ytr, Pte, yte, acc))
        if s == 0:
            # price each stage: patches before any mixing, then after
            emb = lambda X, m=mx: m.embed(X).mean(1)
            r0 = lambda X, m=mx: m.pooled(X, r=0)
            r1 = lambda X, m=mx: m.pooled(X, r=1)
            for lab, fn in (("  embed only, pooled", emb),
                            ("  r=0 (no mixing)", r0),
                            ("  r=1 (one round)", r1)):
                try:
                    A_ = chunk_apply(fn, Ttr); B_ = chunk_apply(fn, Tte)
                    rows.append(report(lab, A_, ytr, B_, yte))
                except Exception as e:
                    print(f"  {lab}: skipped ({e})")

    p = os.path.join(HERE, "results", "mixer_diagnose.json")
    json.dump(rows, open(p, "w"), indent=1, default=float)
    print(f"\nwrote {os.path.relpath(p, HERE)}")


if __name__ == "__main__":
    main()
