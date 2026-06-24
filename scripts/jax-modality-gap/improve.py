"""Method-improvement ablations on top of the recursive multi-task baseline.

Reuses demo.py (data, tokenizers, recursive block, training loop) and tests
concrete improvements against the SigLIP multi-task model, each over several
seeds so the comparison has error bars:

  base      : the current model (gate reads CLS_wav only; no input masking)
  ctxgate   : gate reads BOTH CLS tokens + the pooled signal context
              (reconstruction-informed, genuinely input-dependent)
  mask      : light input-token masking -> a true denoising-MAE hybrid
  ctx+mask  : both improvements together

Reports AMC accuracy at 4 and 12 dB and reconstruction MSE at 4 dB,
mean +/- std across seeds. Does NOT touch demo.py's outputs.

Run:  JAX_PLATFORMS=cpu python improve.py            (full: 4 variants x 3 seeds)
      JAX_PLATFORMS=cpu python improve.py --smoke    (fast sanity check)
"""
from __future__ import annotations
import sys
import numpy as np
import jax
import jax.numpy as jnp
import optax
import demo as D

DM, GH, DEPTH, M, NPAT, K = D.DMODEL, D.GATE_H, D.DEPTH, D.M, D.NPAT, D.K
HEADS, PONDER, MT_MASK = D.HEADS, D.PONDER, D.MT_MASK

SMOKE = "--smoke" in sys.argv
STEPS = 80 if SMOKE else D.STEPS
SEEDS = [0] if SMOKE else [0, 1, 2]
N_TRAIN, N_TEST = (1500, 600) if SMOKE else (D.N_TRAIN, D.N_TEST)


# ── core block compute with externally-supplied gates ───────────────────
def attn_ffn(blk, x, mask, g_attn, g_ffn):
    B, T, _ = x.shape
    y = D.ln(x, blk["g1"], blk["bn1"])
    q, k, v = jnp.split(y @ blk["Wqkv"], 3, axis=-1)
    dh = DM // HEADS
    hs = lambda t: t.reshape(B, T, HEADS, dh).transpose(0, 2, 1, 3)
    q, k, v = hs(q), hs(k), hs(v)
    scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(dh)
    if mask is not None:
        scores = jnp.where(mask, scores, -1e9)
    a = jax.nn.softmax(scores, axis=-1)
    o = (a @ v).transpose(0, 2, 1, 3).reshape(B, T, DM) @ blk["Wo"]
    x = x + g_attn * o
    y = D.ln(x, blk["g2"], blk["bn2"])
    return x + g_ffn * (jax.nn.gelu(y @ blk["ff1"] + blk["b1"]) @ blk["ff2"] + blk["b2"])


def gate_cls(blk, x):                       # baseline: CLS_wav only
    return jax.nn.sigmoid(
        jax.nn.gelu(x[:, 0] @ blk["gW1"] + blk["gb1"]) @ blk["gW2"] + blk["gb2"])


def gate_ctx(blk, x):                        # both CLS + pooled signal context
    ctx = jnp.concatenate([x[:, 0], x[:, 1], x[:, 2:].mean(1)], axis=-1)   # (B, 3*DM)
    return jax.nn.sigmoid(
        jax.nn.gelu(ctx @ blk["gWc1"] + blk["gb1"]) @ blk["gW2"] + blk["gb2"])


def recurse_g(blk, x, mask, gate_fn):
    gsum = 0.0
    for _ in range(DEPTH):
        g = gate_fn(blk, x)
        gsum = gsum + jnp.mean(g)
        ga, gf = g[:, 0].reshape(-1, 1, 1), g[:, 1].reshape(-1, 1, 1)
        x = attn_ffn(blk, x, mask, ga, gf)
    return x, gsum / DEPTH


def mt_forward_g(p, wav, con, gate_fn, key=None, mask_p=0.0):
    B = wav.shape[0]
    clsw = jnp.broadcast_to(p["clsA"], (B, 1, DM))
    clsc = jnp.broadcast_to(p["clsB"], (B, 1, DM))
    wt, ct = D.wav_tokens(p, wav), D.con_tokens(p, con)
    if mask_p > 0 and key is not None:                 # zero a fraction of input tokens
        kw, kc = jax.random.split(key)
        wt = wt * (jax.random.uniform(kw, (B, M, 1)) > mask_p)
        ct = ct * (jax.random.uniform(kc, (B, NPAT, 1)) > mask_p)
    x = jnp.concatenate([clsw, clsc, wt, ct], axis=1)
    x, gpen = recurse_g(p["blk"], x, MT_MASK, gate_fn)
    zA, zB = D.l2n(x[:, 0] @ p["outA"]), D.l2n(x[:, 1] @ p["outB"])
    rec = (x[:, 2:2 + M] @ p["recon"] + p["recon_b"]).reshape(B, 2 * M)
    pooled = jnp.concatenate([x[:, 0], x[:, 1], x[:, 2:].mean(1)], axis=1)
    return rec, zA, zB, pooled, gpen


def make_variant(key, ctx):
    p = D.make_multitask(key)
    if ctx:                                            # extra gate-input weights (3*DM -> GH)
        p["blk"]["gWc1"] = jax.random.normal(jax.random.split(key, 7)[6], (3 * DM, GH)) * 0.02
    return p


def loss_for(gate_fn, mask_p):
    def loss(p, wav, con, Z, key):
        rec, zA, zB, _, gpen = mt_forward_g(p, wav, con, gate_fn, key, mask_p)
        rec_loss = jnp.mean((rec - Z) ** 2)
        t = jnp.exp(p["logit_scale"])
        logits = t * (zA @ zB.T) + p["bias"]
        n = zA.shape[0]
        labels = 2.0 * jnp.eye(n) - 1.0
        cls_loss = jnp.mean(jax.nn.softplus(-labels * logits))
        return rec_loss + 0.1 * cls_loss + PONDER * gpen
    return loss


VARIANTS = {
    "base":     (gate_cls, 0.0, False),
    "ctxgate":  (gate_ctx, 0.0, True),
    "mask":     (gate_cls, 0.3, False),
    "ctx+mask": (gate_ctx, 0.3, True),
}


def run_seed(seed):
    wav_tr, con_tr, Z_tr, y_tr = D.generate(N_TRAIN, seed)
    out = {}
    # fixed eval probe sets at 4 and 12 dB
    probes = {}
    for snr in (4, 12):
        wtr, ctr, _, ytr = D.generate(2000, 300 + seed * 10 + snr, snr=float(snr))
        wte, cte, _, yte = D.generate(1000, 700 + seed * 10 + snr, snr=float(snr))
        probes[snr] = (wtr, ctr, ytr, wte, cte, yte)
    wav_te, con_te, Z_te, _ = D.generate(N_TEST, seed + 1, snr=4.0)

    for name, (gate_fn, mask_p, ctx) in VARIANTS.items():
        p = make_variant(jax.random.key(100 + seed), ctx)
        p = D.train(loss_for(gate_fn, mask_p), p, (wav_tr, con_tr, Z_tr),
                    STEPS, 1e-3, stochastic=True)
        pool = jax.jit(lambda w, c, p=p, gf=gate_fn: mt_forward_g(p, w, c, gf)[3])
        rec = jax.jit(lambda w, c, p=p, gf=gate_fn: mt_forward_g(p, w, c, gf)[0])
        dn = float(np.mean((D.batched(rec, wav_te, con_te) - Z_te) ** 2))
        accs = {}
        for snr, (wtr, ctr, ytr, wte, cte, yte) in probes.items():
            accs[snr] = D.class_probe(D.batched(pool, wtr, ctr), ytr,
                                      D.batched(pool, wte, cte), yte)
        out[name] = dict(denoise=dn, amc4=accs[4], amc12=accs[12])
        print(f"  seed {seed} {name:9s} denoise {dn:.3f}  AMC@4 {accs[4]*100:4.1f}%  AMC@12 {accs[12]*100:4.1f}%")
    return out


def main():
    print(f"Improvement ablations  (steps={STEPS}, seeds={SEEDS})")
    allres = {n: {"denoise": [], "amc4": [], "amc12": []} for n in VARIANTS}
    for s in SEEDS:
        r = run_seed(s)
        for n in VARIANTS:
            for k in ("denoise", "amc4", "amc12"):
                allres[n][k].append(r[n][k])
    print("\n===================== SUMMARY (mean +/- std) =====================")
    print(f"{'variant':10s} {'denoise MSE':>16s} {'AMC@4dB':>14s} {'AMC@12dB':>14s}")
    for n in VARIANTS:
        d = np.array(allres[n]["denoise"]); a4 = np.array(allres[n]["amc4"]) * 100
        a12 = np.array(allres[n]["amc12"]) * 100
        print(f"{n:10s} {d.mean():7.3f}+/-{d.std():.3f}   "
              f"{a4.mean():5.1f}+/-{a4.std():4.1f}   {a12.mean():5.1f}+/-{a12.std():4.1f}")
    print("==================================================================")


if __name__ == "__main__":
    import os
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    main()
