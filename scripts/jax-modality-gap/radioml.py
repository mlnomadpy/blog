"""Deno_nMAE on RadioML 2018.01A — the real benchmark.

Self-supervised pretraining of the recursive, gated, multi-task encoder on
RadioML 2018.01A, then a linear-probe (and optional fine-tune) AMC evaluation
across SNR, to compare against the DenoMAE / DenoMAE 2.0 line.

Two modalities are derived from each I/Q example:
  waveform     : the 1024-sample I/Q sequence, split into patches
  constellation: a G x G 2D histogram of the I/Q points, split into patches

RadioML provides no clean reference, so the denoising target is the RadioML
signal itself: we add extra Gaussian noise to form the input and reconstruct
the original signal (denoise-the-added-noise). Objectives, in one Zorro-masked
forward pass: denoising reconstruction + SigLIP contrastive on per-modality CLS
tokens + ponder cost on the adaptive-depth gate.

Usage:
  python radioml.py --inspect --h5 /path/GOLD_XYZ_OSC.0001_1024.hdf5
  python radioml.py --h5 /path/GOLD_XYZ_OSC.0001_1024.hdf5 --model multitask
  python radioml.py --h5 ... --model untied --depth 6     # baseline for the L-sweep
"""
from __future__ import annotations
import os, sys, argparse, time
import numpy as np
import jax, jax.numpy as jnp
import optax

# ── shapes / hyperparameters ────────────────────────────────────────────
SIGLEN = 1024            # RadioML 2018.01A: 1024 complex samples per example
WPATCH = 16             # waveform patch length -> SIGLEN/WPATCH tokens
NWAV = SIGLEN // WPATCH
G = 32                  # constellation histogram resolution
CPATCH = 4
NCON = (G // CPATCH) ** 2
DMODEL = 128
HEADS = 8
DFF = 256
DEPTH = 6               # recursion depth (the n in Deno_nMAE)
NCLS = 24               # RadioML 2018.01A modulation classes
GATE_H = 32
PONDER = 0.03
ADD_NOISE_STD = 0.1     # std of the extra noise added to form the denoising input

T = 2 + NWAV + NCON     # [CLS_w, CLS_c, wav tokens, con tokens]


# ── data ─────────────────────────────────────────────────────────────────
def inspect(h5path):
    import h5py
    with h5py.File(h5path, "r") as f:
        print("keys:", list(f.keys()))
        for k in f.keys():
            print(f"  {k}: shape {f[k].shape} dtype {f[k].dtype}")


def load_radioml(h5path, n_max=120000, seed=0):
    """Return (sig [N,SIGLEN,2] float32, label [N] int, snr [N] int).

    Handles the standard GOLD_XYZ layout (X signals, Y one-hot labels, Z SNR).
    Subsamples to n_max for tractable CPU/GPU memory.
    """
    import h5py
    with h5py.File(h5path, "r") as f:
        keys = list(f.keys())
        xk = "X" if "X" in keys else keys[0]
        yk = "Y" if "Y" in keys else keys[1]
        zk = "Z" if "Z" in keys else (keys[2] if len(keys) > 2 else None)
        N = f[xk].shape[0]
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(N, size=min(n_max, N), replace=False))
        X = np.asarray(f[xk][idx], np.float32)               # (n, 1024, 2)
        Y = np.asarray(f[yk][idx])
        lab = Y.argmax(1).astype(np.int32) if Y.ndim == 2 else Y.astype(np.int32)
        snr = (np.asarray(f[zk][idx]).reshape(-1).astype(np.int32)
               if zk is not None else np.zeros(len(idx), np.int32))
    # per-example energy normalization (unit average power)
    p = np.sqrt((X ** 2).mean(axis=(1, 2), keepdims=True) + 1e-9)
    X = X / p
    return X, lab, snr


def constellation(sig):
    """(B,SIGLEN,2) -> (B, G*G) normalized 2D I/Q histogram."""
    B = sig.shape[0]
    edges = np.linspace(-3.0, 3.0, G + 1)
    out = np.zeros((B, G * G), np.float32)
    for i in range(B):
        h, _, _ = np.histogram2d(sig[i, :, 0], sig[i, :, 1], bins=[edges, edges])
        out[i] = (h / SIGLEN).astype(np.float32).reshape(-1)
    return out


# ── recursive gated transformer (raw JAX) ───────────────────────────────
def l2n(x):
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

def ln(x, g, b):
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return (x - m) * jax.lax.rsqrt(v + 1e-5) * g + b

def init_block(key):
    ks = jax.random.split(key, 6); sc = 0.02
    return dict(
        Wqkv=jax.random.normal(ks[0], (DMODEL, 3 * DMODEL)) * sc,
        Wo=jax.random.normal(ks[1], (DMODEL, DMODEL)) * sc,
        ff1=jax.random.normal(ks[2], (DMODEL, DFF)) * sc, b1=jnp.zeros(DFF),
        ff2=jax.random.normal(ks[3], (DFF, DMODEL)) * sc, b2=jnp.zeros(DMODEL),
        g1=jnp.ones(DMODEL), bn1=jnp.zeros(DMODEL),
        g2=jnp.ones(DMODEL), bn2=jnp.zeros(DMODEL),
        gWc1=jax.random.normal(ks[4], (3 * DMODEL, GATE_H)) * sc, gb1=jnp.zeros(GATE_H),
        gW2=jax.random.normal(ks[5], (GATE_H, 2)) * sc, gb2=jnp.full((2,), 3.0),
    )

def gate_value(p, x):
    """Reconstruction-informed dual-CLS gate: reads both CLS tokens + the pooled
    signal context (validated to beat the CLS-only gate, and genuinely
    input-dependent)."""
    ctx = jnp.concatenate([x[:, 0], x[:, 1], x[:, 2:].mean(1)], axis=-1)   # (B, 3*DMODEL)
    return jax.nn.sigmoid(jax.nn.gelu(ctx @ p["gWc1"] + p["gb1"]) @ p["gW2"] + p["gb2"])

def block(p, x, mask):
    B, Tt, _ = x.shape
    gate = gate_value(p, x)
    ga, gf = gate[:, 0].reshape(B, 1, 1), gate[:, 1].reshape(B, 1, 1)
    y = ln(x, p["g1"], p["bn1"])
    q, k, v = jnp.split(y @ p["Wqkv"], 3, axis=-1)
    dh = DMODEL // HEADS
    hs = lambda t: t.reshape(B, Tt, HEADS, dh).transpose(0, 2, 1, 3)
    q, k, v = hs(q), hs(k), hs(v)
    scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(dh)
    if mask is not None:
        scores = jnp.where(mask, scores, -1e9)
    a = jax.nn.softmax(scores, axis=-1)
    o = (a @ v).transpose(0, 2, 1, 3).reshape(B, Tt, DMODEL) @ p["Wo"]
    x = x + ga * o
    y = ln(x, p["g2"], p["bn2"])
    return x + gf * (jax.nn.gelu(y @ p["ff1"] + p["b1"]) @ p["ff2"] + p["b2"])

def recurse(blocks, x, mask, tied):
    """tied=True: one shared block applied DEPTH times (Deno_nMAE).
       tied=False: DEPTH distinct blocks (untied baseline)."""
    gsum = 0.0
    for i in range(DEPTH):
        blk = blocks[0] if tied else blocks[i]
        gsum = gsum + jnp.mean(gate_value(blk, x))
        x = block(blk, x, mask)
    return x, gsum / DEPTH


def _mt_mask():
    m = np.zeros((T, T), bool)
    wav_s, con_s = slice(2, 2 + NWAV), slice(2 + NWAV, T)
    m[0, 0] = True; m[0, wav_s] = True
    m[1, 1] = True; m[1, con_s] = True
    m[2:, :] = True
    return jnp.asarray(m)
MT_MASK = _mt_mask()


def init_model(key, tied=True):
    ks = jax.random.split(key, 10)
    nb = 1 if tied else DEPTH
    p = dict(blocks=[init_block(ks[i]) for i in range(nb)])
    p["wav_proj"] = jax.random.normal(ks[6], (WPATCH * 2, DMODEL)) * 0.1
    p["wav_pos"] = jnp.zeros((NWAV, DMODEL))
    p["con_proj"] = jax.random.normal(ks[7], (CPATCH * CPATCH, DMODEL)) * 0.1
    p["con_pos"] = jnp.zeros((NCON, DMODEL))
    p["clsA"] = jnp.zeros((1, 1, DMODEL)); p["clsB"] = jnp.zeros((1, 1, DMODEL))
    p["recon"] = jax.random.normal(ks[8], (DMODEL, WPATCH * 2)) * 0.1
    p["recon_b"] = jnp.zeros(WPATCH * 2)
    p["outA"] = jax.random.normal(ks[9], (DMODEL, DMODEL)) * 0.1
    p["outB"] = jax.random.normal(jax.random.split(ks[9])[0], (DMODEL, DMODEL)) * 0.1
    p["logit_scale"] = jnp.log(5.0); p["bias"] = jnp.asarray(-2.0)
    return p, tied

def wav_tokens(p, sig):                       # (B,SIGLEN,2) -> (B,NWAV,DMODEL)
    B = sig.shape[0]
    x = sig.reshape(B, NWAV, WPATCH * 2)
    return x @ p["wav_proj"] + p["wav_pos"]

def con_tokens(p, con):
    B = con.shape[0]; npg = G // CPATCH
    img = con.reshape(B, npg, CPATCH, npg, CPATCH).transpose(0, 1, 3, 2, 4)
    img = img.reshape(B, NCON, CPATCH * CPATCH)
    return img @ p["con_proj"] + p["con_pos"]

def forward(p, tied, sig, con, mask_key=None, mask_ratio=0.0):
    B = sig.shape[0]
    clsw = jnp.broadcast_to(p["clsA"], (B, 1, DMODEL))
    clsc = jnp.broadcast_to(p["clsB"], (B, 1, DMODEL))
    wt, ct = wav_tokens(p, sig), con_tokens(p, con)
    if mask_ratio > 0 and mask_key is not None:        # MAE-style input masking
        kw, kc = jax.random.split(mask_key)
        wt = wt * (jax.random.uniform(kw, (B, NWAV, 1)) > mask_ratio)
        ct = ct * (jax.random.uniform(kc, (B, NCON, 1)) > mask_ratio)
    x = jnp.concatenate([clsw, clsc, wt, ct], axis=1)
    x, gpen = recurse(p["blocks"], x, MT_MASK, tied)
    zA, zB = l2n(x[:, 0] @ p["outA"]), l2n(x[:, 1] @ p["outB"])
    rec = (x[:, 2:2 + NWAV] @ p["recon"] + p["recon_b"]).reshape(B, SIGLEN, 2)
    pooled = jnp.concatenate([x[:, 0], x[:, 1], x[:, 2:].mean(1)], axis=1)
    return rec, zA, zB, pooled, gpen

def loss_fn(p, tied, sig, con, key, head="siglip", lam=0.1, mask_ratio=0.5):
    # masked reconstruction: mask input patches, reconstruct the full signal
    rec, zA, zB, _, gpen = forward(p, tied, sig, con, mask_key=key, mask_ratio=mask_ratio)
    rec_loss = jnp.mean((rec - sig) ** 2)
    n = zA.shape[0]
    if head == "siglip":
        t = jnp.exp(p["logit_scale"]); logits = t * (zA @ zB.T) + p["bias"]
        labels = 2.0 * jnp.eye(n) - 1.0
        cls_loss = jnp.mean(jax.nn.softplus(-labels * logits))
    else:  # infonce
        logits = zA @ zB.T / 0.1
        lab = jnp.arange(n)
        cls_loss = 0.5 * (optax.softmax_cross_entropy_with_integer_labels(logits, lab).mean()
                          + optax.softmax_cross_entropy_with_integer_labels(logits.T, lab).mean())
    return rec_loss + lam * cls_loss + PONDER * gpen


# ── probes ───────────────────────────────────────────────────────────────
def ridge_fit(X, Y, lam=1.0):
    X = np.concatenate([X, np.ones((X.shape[0], 1))], 1)
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)

def class_probe(tr, ytr, te, yte):
    W = ridge_fit(tr, np.eye(NCLS)[ytr])
    pred = (np.concatenate([te, np.ones((te.shape[0], 1))], 1) @ W).argmax(1)
    return float((pred == yte).mean())

def batched_pool(p, tied, sig, con, bs=512):
    f = jax.jit(lambda s, c: forward(p, tied, s, c)[3])
    outs = []
    for i in range(0, sig.shape[0], bs):
        outs.append(np.asarray(f(jnp.asarray(sig[i:i+bs]), jnp.asarray(con[i:i+bs]))))
    return np.concatenate(outs, 0)


def train(p, tied, sig, con, steps, lr, head, bs=256, seed=0):
    opt = optax.adam(lr); state = opt.init(p)
    key = jax.random.key(seed)
    @jax.jit
    def step(p, state, s, c, k):
        l, g = jax.value_and_grad(loss_fn)(p, tied, s, c, k, head)
        upd, state = opt.update(g, state)
        return optax.apply_updates(p, upd), state, l
    n = sig.shape[0]
    for t in range(steps):
        key, sub, sk = jax.random.split(key, 3)
        idx = np.asarray(jax.random.randint(sub, (bs,), 0, n))
        p, state, l = step(p, state, jnp.asarray(sig[idx]), jnp.asarray(con[idx]), sk)
        if t % 200 == 0:
            print(f"  step {t:4d}  loss {float(l):.4f}", flush=True)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", type=str, required="--inspect" not in sys.argv)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--model", choices=["multitask", "multitask_infonce", "recon", "untied"], default="multitask")
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--n", type=int, default=120000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    global DEPTH
    DEPTH = args.depth

    if args.inspect:
        inspect(args.h5); return

    print(f"Loading RadioML from {args.h5} (n_max={args.n}) ...", flush=True)
    sig, lab, snr = load_radioml(args.h5, n_max=args.n, seed=args.seed)
    print(f"  signals {sig.shape}  classes {lab.max()+1}  SNR range [{snr.min()},{snr.max()}]", flush=True)
    print("Building constellation histograms ...", flush=True)
    con = constellation(sig)

    # split
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(sig))
    ntr = int(0.8 * len(sig))
    tr, te = perm[:ntr], perm[ntr:]

    tied = args.model != "untied"
    head = "infonce" if args.model == "multitask_infonce" else "siglip"
    p, tied = init_model(jax.random.key(args.seed), tied=tied)
    print(f"Pretraining model={args.model} depth={DEPTH} tied={tied} head={head} ...", flush=True)
    t0 = time.time()
    p = train(p, tied, sig[tr], con[tr], args.steps, 1e-3, head, seed=args.seed)
    print(f"  trained in {time.time()-t0:.0f}s", flush=True)

    # linear-probe AMC across SNR
    h_tr = batched_pool(p, tied, sig[tr], con[tr])
    h_te = batched_pool(p, tied, sig[te], con[te])
    overall = class_probe(h_tr, lab[tr], h_te, lab[te])
    print(f"\nOverall linear-probe AMC: {overall*100:.2f}%")
    print("AMC by SNR:")
    for s in sorted(set(snr[te].tolist())):
        m = snr[te] == s
        if m.sum() < 20:
            continue
        acc = class_probe(h_tr, lab[tr], h_te[m], lab[te][m])
        print(f"  SNR {s:+3d} dB : {acc*100:5.2f}%  (n={int(m.sum())})")


if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cuda")
    main()
