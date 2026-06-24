"""Contrastive vs. joint reconstruction on complementary modalities.

A small, runnable demonstration of the thesis in
"When the Modality Gap Hurts". We synthesise DenoMAE-style radio modalities of
a hidden signal:

  Z        : a clean sequence of modulated symbols (the "signal", the Form)
  waveform : the noisy symbols in time order        (keeps time, loses cleanliness)
  constell : a 2D histogram of the noisy I/Q points (keeps geometry, loses time)

The two modalities are *complementary*: the waveform has order and the
constellation has clean geometry, and you need both to recover Z. We then train
two models on exactly the same data and compare what they recover:

  contrastive : two encoders aligned with InfoNCE  -> keeps the intersection
  joint MAE   : one encoder reconstructs Z from both -> keeps the union

Outputs (PNG, styled to the blog) go to ../../public/modality-gap/.

Run:  python demo.py
"""

from __future__ import annotations

import os
import numpy as np
import jax
import jax.numpy as jnp
import optax

SEED = 0
G = 24            # constellation histogram resolution (G x G)
PATCH = 4         # constellation patch size -> (G/PATCH)^2 patch tokens
M = 96            # symbols per sample (DenoMAE uses ~1024; we use 96 tokens)
SNR_DB = 4.0      # reference SNR for the denoise/recover-Z comparison
SNR_LO, SNR_HI = -2.0, 10.0      # mixed-SNR range used during representation training
SNR_GRID = [-4, -2, 0, 2, 4, 6, 8, 10, 12]   # test SNRs for the AMC-vs-SNR sweep

# Recursive (ALBERT-style) transformer: ONE block, applied DEPTH times.
DMODEL = 48
HEADS = 4
DFF = 128
DEPTH = 4         # recursion depth (the single shared block is reused this often)
D = DMODEL        # contrastive embedding dim

N_TRAIN = 7000
N_TEST = 2000
STEPS = 700
BATCH = 128

# ── constellations (unit average energy) ────────────────────────────────
def _norm(pts):
    pts = np.asarray(pts, dtype=np.complex64)
    pts /= np.sqrt(np.mean(np.abs(pts) ** 2))
    return pts

def _qam16():
    lv = np.array([-3, -1, 1, 3])
    return _norm([complex(i, q) for i in lv for q in lv])

def _qam(m):
    s = int(round(m ** 0.5)); lv = np.arange(-(s - 1), s, 2)
    return _norm([complex(i, q) for i in lv for q in lv])

SCHEMES = [
    _norm([1, -1]),                                                  # BPSK
    _norm([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]),                       # QPSK
    _norm([np.exp(1j * np.pi * k / 4) for k in range(8)]),           # 8PSK
    _qam(16),                                                        # 16QAM
    _qam(64),                                                        # 64QAM
    _norm([-3, -1, 1, 3]),                                           # 4PAM (real line)
    _norm([-7, -5, -3, -1, 1, 3, 5, 7]),                            # 8PAM (real line)
    _norm([np.exp(1j * np.pi * k / 8) for k in range(16)]),         # 16PSK
]
SCHEME_NAMES = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "4PAM", "8PAM", "16PSK"]
K = len(SCHEMES)


def generate(n, seed, snr=None):
    """Return (waveform, constellation, Z_clean, labels) as numpy arrays.

    snr=None draws a different SNR per sample from [SNR_LO, SNR_HI] (mixed,
    used for representation training); a float fixes the SNR (used for the
    per-SNR evaluation sweep).
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, K, n)
    wav = np.zeros((n, 2 * M), np.float32)     # ordered noisy I/Q
    con = np.zeros((n, G * G), np.float32)      # histogram of noisy I/Q (order-free)
    Z = np.zeros((n, 2 * M), np.float32)        # ordered clean I/Q  (the signal)
    edges = np.linspace(-2.0, 2.0, G + 1)
    for i in range(n):
        s = snr if snr is not None else rng.uniform(SNR_LO, SNR_HI)
        sigma = 10 ** (-s / 20.0)
        pts = SCHEMES[labels[i]]
        sym = pts[rng.integers(0, len(pts), M)]
        noise = (rng.standard_normal(M) + 1j * rng.standard_normal(M)) * sigma / np.sqrt(2)
        noisy = sym + noise
        wav[i, 0::2] = noisy.real; wav[i, 1::2] = noisy.imag
        Z[i, 0::2] = sym.real;     Z[i, 1::2] = sym.imag
        h, _, _ = np.histogram2d(noisy.real, noisy.imag, bins=[edges, edges])
        con[i] = (h / M).astype(np.float32).reshape(-1)
    return wav, con, Z, labels.astype(np.int32)


# ── recursive (ALBERT-style) transformer in raw JAX ─────────────────────
# ONE transformer block, whose weights are reused DEPTH times (recursion) and,
# within a model, shared across both modality towers. Tokens: each symbol is a
# token for the waveform; each PATCH x PATCH cell is a token for the
# constellation image.
NPAT = (G // PATCH) ** 2

def l2n(x):
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

def ln(x, g, b):
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return (x - m) * jax.lax.rsqrt(v + 1e-5) * g + b

GATE_H = 16   # hidden width of the CLS-conditioned gate MLP

def init_block(key):
    ks = jax.random.split(key, 6); sc = 0.02
    return dict(
        Wqkv=jax.random.normal(ks[0], (DMODEL, 3 * DMODEL)) * sc,
        Wo=jax.random.normal(ks[1], (DMODEL, DMODEL)) * sc,
        ff1=jax.random.normal(ks[2], (DMODEL, DFF)) * sc, b1=jnp.zeros(DFF),
        ff2=jax.random.normal(ks[3], (DFF, DMODEL)) * sc, b2=jnp.zeros(DMODEL),
        g1=jnp.ones(DMODEL), bn1=jnp.zeros(DMODEL),
        g2=jnp.ones(DMODEL), bn2=jnp.zeros(DMODEL),
        # CLS-conditioned gates: two scalars (attention, FFN), init OPEN (~0.95)
        gW1=jax.random.normal(ks[4], (DMODEL, GATE_H)) * sc, gb1=jnp.zeros(GATE_H),
        gW2=jax.random.normal(ks[5], (GATE_H, 2)) * sc, gb2=jnp.full((2,), 3.0),
    )

def block(p, x, mask=None):                       # x: (B, T, DMODEL)
    B, T, _ = x.shape
    # CLS-conditioned gate: one sigmoid scalar per sublayer, driven by the CLS
    # token. It scales each block's residual update, so the block can soft-skip
    # itself at this recursion step (adaptive effective depth).
    gate = jax.nn.sigmoid(
        jax.nn.gelu(x[:, 0] @ p["gW1"] + p["gb1"]) @ p["gW2"] + p["gb2"])   # (B, 2)
    g_attn = gate[:, 0].reshape(B, 1, 1); g_ffn = gate[:, 1].reshape(B, 1, 1)

    y = ln(x, p["g1"], p["bn1"])
    q, k, v = jnp.split(y @ p["Wqkv"], 3, axis=-1)
    dh = DMODEL // HEADS
    hs = lambda t: t.reshape(B, T, HEADS, dh).transpose(0, 2, 1, 3)
    q, k, v = hs(q), hs(k), hs(v)
    scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(dh)   # (B, H, T, T)
    if mask is not None:                          # mask (T, T): True = may attend
        scores = jnp.where(mask, scores, -1e9)
    a = jax.nn.softmax(scores, axis=-1)
    o = (a @ v).transpose(0, 2, 1, 3).reshape(B, T, DMODEL) @ p["Wo"]
    x = x + g_attn * o
    y = ln(x, p["g2"], p["bn2"])
    return x + g_ffn * (jax.nn.gelu(y @ p["ff1"] + p["b1"]) @ p["ff2"] + p["b2"])

def recurse(blk, x, mask=None):
    """Apply the shared block DEPTH times; also return the mean gate value
    (a ponder cost the loss penalizes lightly, so blocks close when unneeded)."""
    gsum = 0.0
    for _ in range(DEPTH):                        # the single block, reused
        g = jax.nn.sigmoid(
            jax.nn.gelu(x[:, 0] @ blk["gW1"] + blk["gb1"]) @ blk["gW2"] + blk["gb2"])
        gsum = gsum + jnp.mean(g)
        x = block(blk, x, mask)
    return x, gsum / DEPTH

PONDER = 0.03   # weight on the mean-gate ponder cost

def gates_over_depth(blk, x, mask=None):
    """Mean (attention, FFN) gate value at each recursion step. (DEPTH, 2)."""
    gs = []
    for _ in range(DEPTH):
        g = jax.nn.sigmoid(
            jax.nn.gelu(x[:, 0] @ blk["gW1"] + blk["gb1"]) @ blk["gW2"] + blk["gb2"])
        gs.append(jnp.mean(g, axis=0))
        x = block(blk, x, mask)
    return jnp.stack(gs)

def adapters(key):
    ks = jax.random.split(key, 6)
    return dict(
        wav_proj=jax.random.normal(ks[0], (2, DMODEL)) * 0.1, wav_pos=jnp.zeros((M, DMODEL)),
        con_proj=jax.random.normal(ks[1], (PATCH * PATCH, DMODEL)) * 0.1, con_pos=jnp.zeros((NPAT, DMODEL)),
        clsA=jnp.zeros((1, 1, DMODEL)), clsB=jnp.zeros((1, 1, DMODEL)),
    )

def wav_tokens(p, wav):                           # (B, 2M) -> (B, M, DMODEL)
    B = wav.shape[0]
    return wav.reshape(B, M, 2) @ p["wav_proj"] + p["wav_pos"]

def con_tokens(p, con):                           # (B, G*G) -> (B, NPAT, DMODEL)
    B = con.shape[0]; npg = G // PATCH
    img = con.reshape(B, npg, PATCH, npg, PATCH).transpose(0, 1, 3, 2, 4)
    img = img.reshape(B, NPAT, PATCH * PATCH)
    return img @ p["con_proj"] + p["con_pos"]

# ── contrastive (two towers, shared recursive block) ────────────────────
def make_contrastive(key):
    ks = jax.random.split(key, 3)
    p = dict(blk=init_block(ks[0]))
    p.update(adapters(ks[1]))
    p["outA"] = jax.random.normal(ks[2], (DMODEL, D)) * 0.1
    p["outB"] = jax.random.normal(jax.random.split(ks[2])[0], (DMODEL, D)) * 0.1
    return p

def embed_A(p, wav):
    B = wav.shape[0]
    cls = jnp.broadcast_to(p["clsA"], (B, 1, DMODEL))
    x, _ = recurse(p["blk"], jnp.concatenate([cls, wav_tokens(p, wav)], axis=1))
    return l2n(x[:, 0] @ p["outA"])

def embed_B(p, con):
    B = con.shape[0]
    cls = jnp.broadcast_to(p["clsB"], (B, 1, DMODEL))
    x, _ = recurse(p["blk"], jnp.concatenate([cls, con_tokens(p, con)], axis=1))
    return l2n(x[:, 0] @ p["outB"])

def contrastive_loss(params, wav, con, tau=0.1):
    zA, zB = embed_A(params, wav), embed_B(params, con)
    logits = zA @ zB.T / tau
    labels = jnp.arange(wav.shape[0])
    la = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    lb = optax.softmax_cross_entropy_with_integer_labels(logits.T, labels)
    return jnp.mean(la + lb) / 2

def make_siglip(key):
    base = make_contrastive(key)
    base["logit_scale"] = jnp.log(5.0)
    base["bias"] = jnp.asarray(-2.0)
    return base

def siglip_loss(params, wav, con):
    zA, zB = embed_A(params, wav), embed_B(params, con)
    t = jnp.exp(params["logit_scale"])
    logits = t * (zA @ zB.T) + params["bias"]
    n = zA.shape[0]
    labels = 2.0 * jnp.eye(n) - 1.0
    return jnp.mean(jax.nn.softplus(-labels * logits))

# ── joint denoising autoencoder (one recursive transformer over both) ───
def make_mae(key):
    ks = jax.random.split(key, 4)
    p = dict(blk=init_block(ks[0]))
    p.update(adapters(ks[1]))
    p["cls"] = jnp.zeros((1, 1, DMODEL))
    p["recon"] = jax.random.normal(ks[2], (DMODEL, 2)) * 0.1
    p["recon_b"] = jnp.zeros(2)
    return p

def mae_forward(p, wav, con):
    B = wav.shape[0]
    cls = jnp.broadcast_to(p["cls"], (B, 1, DMODEL))
    x = jnp.concatenate([cls, wav_tokens(p, wav), con_tokens(p, con)], axis=1)
    x, gpen = recurse(p["blk"], x)
    wav_out = x[:, 1:1 + M]                        # per-symbol token outputs
    rec = (wav_out @ p["recon"] + p["recon_b"]).reshape(B, 2 * M)
    pooled = jnp.concatenate([x[:, 0], x[:, 1:].mean(1)], axis=1)  # CLS + mean
    return rec, pooled, gpen

def mae_loss(params, wav, con, Z, key):
    rec, _, gpen = mae_forward(params, wav, con)
    return jnp.mean((rec - Z) ** 2) + PONDER * gpen

# ── multi-task: ONE masked pass = reconstruction + SigLIP on per-modality CLS ─
# Sequence: [CLS_w, CLS_c, wav tokens (M), con tokens (NPAT)]. Attention mask:
# CLS_w sees only itself + waveform; CLS_c sees only itself + constellation, so
# the two CLS embeddings stay modality-pure for the contrastive term. Every
# signal token sees everything, so the per-token reconstruction still fuses both
# modalities. One encoder, one pass, one CLS per modality.
def _mt_mask():
    T = 2 + M + NPAT
    m = np.zeros((T, T), bool)
    wav_s, con_s = slice(2, 2 + M), slice(2 + M, T)
    m[0, 0] = True; m[0, wav_s] = True            # CLS_w: self + waveform
    m[1, 1] = True; m[1, con_s] = True            # CLS_c: self + constellation
    m[2:, :] = True                               # signal tokens: full attention
    return jnp.asarray(m)

MT_MASK = _mt_mask()

def make_multitask(key):
    p = make_mae(key)                              # shares the recursive block
    ks = jax.random.split(key, 3)
    p["outA"] = jax.random.normal(ks[0], (DMODEL, D)) * 0.1
    p["outB"] = jax.random.normal(ks[1], (DMODEL, D)) * 0.1
    p["logit_scale"] = jnp.log(5.0)
    p["bias"] = jnp.asarray(-2.0)
    return p

def mt_forward(p, wav, con):
    B = wav.shape[0]
    clsw = jnp.broadcast_to(p["clsA"], (B, 1, DMODEL))
    clsc = jnp.broadcast_to(p["clsB"], (B, 1, DMODEL))
    x = jnp.concatenate([clsw, clsc, wav_tokens(p, wav), con_tokens(p, con)], axis=1)
    x, gpen = recurse(p["blk"], x, MT_MASK)
    zA = l2n(x[:, 0] @ p["outA"])                  # waveform CLS
    zB = l2n(x[:, 1] @ p["outB"])                  # constellation CLS
    rec = (x[:, 2:2 + M] @ p["recon"] + p["recon_b"]).reshape(B, 2 * M)
    pooled = jnp.concatenate([x[:, 0], x[:, 1], x[:, 2:].mean(1)], axis=1)
    return rec, zA, zB, pooled, gpen

def multitask_loss(params, wav, con, Z, key, lam_cls=0.1):
    """Multi-task with a SigLIP (pairwise sigmoid) contrastive head."""
    rec, zA, zB, _, gpen = mt_forward(params, wav, con)
    rec_loss = jnp.mean((rec - Z) ** 2)
    t = jnp.exp(params["logit_scale"])
    logits = t * (zA @ zB.T) + params["bias"]
    n = zA.shape[0]
    labels = 2.0 * jnp.eye(n) - 1.0
    cls_loss = jnp.mean(jax.nn.softplus(-labels * logits))
    return rec_loss + lam_cls * cls_loss + PONDER * gpen

def multitask_loss_infonce(params, wav, con, Z, key, lam_cls=0.1, tau=0.1):
    """Multi-task with an InfoNCE (softmax) contrastive head, same forward pass.
    Lets us ask whether reconstruction closes the modality gap for *both*
    contrastive families, not just the sigmoid one."""
    rec, zA, zB, _, gpen = mt_forward(params, wav, con)
    rec_loss = jnp.mean((rec - Z) ** 2)
    logits = zA @ zB.T / tau
    n = zA.shape[0]
    labels = jnp.arange(n)
    la = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    lb = optax.softmax_cross_entropy_with_integer_labels(logits.T, labels)
    cls_loss = jnp.mean(la + lb) / 2
    return rec_loss + lam_cls * cls_loss + PONDER * gpen


def train(loss_fn, params, data, steps, lr, stochastic=False):
    opt = optax.adam(lr)
    state = opt.init(params)
    key = jax.random.key(SEED)

    @jax.jit
    def step(params, state, batch, key):
        if stochastic:
            l, g = jax.value_and_grad(loss_fn)(params, *batch, key)
        else:
            l, g = jax.value_and_grad(loss_fn)(params, *batch)
        upd, state = opt.update(g, state)
        return optax.apply_updates(params, upd), state, l

    n = data[0].shape[0]
    for t in range(steps):
        key, sub = jax.random.split(key)
        idx = jax.random.randint(sub, (BATCH,), 0, n)
        batch = tuple(jnp.asarray(d)[idx] for d in data)
        key, sk = jax.random.split(key)
        params, state, l = step(params, state, batch, sk)
    return params


# ── probes / metrics ────────────────────────────────────────────────────
def ridge_fit(X, Y, lam=1.0):
    X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ Y)

def ridge_pred(W, X):
    X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    return X @ W

def class_probe(emb_tr, y_tr, emb_te, y_te):
    Y = np.eye(K)[y_tr]
    W = ridge_fit(emb_tr, Y)
    pred = ridge_pred(W, emb_te).argmax(1)
    return float((pred == y_te).mean())

def denoise_probe(emb_tr, Z_tr, emb_te, Z_te):
    W = ridge_fit(emb_tr, Z_tr)
    pred = ridge_pred(W, emb_te)
    return float(np.mean((pred - Z_te) ** 2)), pred

def batched(fn, *arrs, bs=512):
    """Run a jitted encoder over chunks to keep attention matrices small."""
    n = arrs[0].shape[0]; outs = []
    for i in range(0, n, bs):
        outs.append(np.asarray(fn(*[jnp.asarray(a[i:i + bs]) for a in arrs])))
    return np.concatenate(outs, 0)


def main():
    print("Generating complementary radio modalities (mixed-SNR training)...")
    wav_tr, con_tr, Z_tr, y_tr = generate(N_TRAIN, SEED)            # mixed SNR
    wav_te, con_te, Z_te, y_te = generate(N_TEST, SEED + 1, snr=SNR_DB)  # fixed 4 dB

    # baseline: how noisy is the input itself (MSE of noisy waveform vs clean)
    noisy_mse = float(np.mean((wav_te - Z_te) ** 2))

    # ── contrastive ──
    print("Training contrastive dual-encoder (InfoNCE)...")
    cp = make_contrastive(jax.random.key(SEED))
    cp = train(contrastive_loss, cp, (wav_tr, con_tr), STEPS, 1e-3)
    embA = jax.jit(lambda w: embed_A(cp, w)); embB = jax.jit(lambda c: embed_B(cp, c))
    zA_tr = batched(embA, wav_tr); zB_tr = batched(embB, con_tr)
    zA_te = batched(embA, wav_te); zB_te = batched(embB, con_te)

    # modality gap: distance between modality centroids on the sphere
    gap = float(np.linalg.norm(zA_te.mean(0) - zB_te.mean(0)))
    matched_cos = float(np.mean(np.sum(zA_te * zB_te, axis=1)))

    c_class = class_probe(zA_tr, y_tr, zA_te, y_te)
    c_denoise, _ = denoise_probe(zA_tr, Z_tr, zA_te, Z_te)

    # ── SigLIP (pairwise sigmoid, the "better" contrastive loss) ──
    print("Training contrastive dual-encoder (SigLIP)...")
    sp = make_siglip(jax.random.key(SEED + 5))
    sp = train(siglip_loss, sp, (wav_tr, con_tr), STEPS, 1e-3)
    sembA = jax.jit(lambda w: embed_A(sp, w)); sembB = jax.jit(lambda c: embed_B(sp, c))
    sA_tr = batched(sembA, wav_tr); sA_te = batched(sembA, wav_te); sB_te = batched(sembB, con_te)
    s_matched_cos = float(np.mean(np.sum(sA_te * sB_te, axis=1)))
    s_gap = float(np.linalg.norm(sA_te.mean(0) - sB_te.mean(0)))
    s_class = class_probe(sA_tr, y_tr, sA_te, y_te)
    s_denoise, _ = denoise_probe(sA_tr, Z_tr, sA_te, Z_te)

    # ── joint MAE ──
    print("Training joint masked autoencoder (reconstruct Z)...")
    mp = make_mae(jax.random.key(SEED + 2))
    mp = train(mae_loss, mp, (wav_tr, con_tr, Z_tr), STEPS, 1e-3, stochastic=True)
    rec_fn = jax.jit(lambda w, c: mae_forward(mp, w, c)[0])
    pool_fn = jax.jit(lambda w, c: mae_forward(mp, w, c)[1])
    h_tr = batched(pool_fn, wav_tr, con_tr); h_te = batched(pool_fn, wav_te, con_te)
    rec_te = batched(rec_fn, wav_te, con_te)
    m_denoise = float(np.mean((rec_te - Z_te) ** 2))
    m_class = class_probe(h_tr, y_tr, h_te, y_te)
    # The autoencoder has one shared space. We report a comparable "gap" by
    # encoding waveform-only and constellation-only inputs and measuring the
    # distance between their centroids.
    zc = np.zeros_like(con_te); zw = np.zeros_like(wav_te)
    hA = batched(pool_fn, wav_te, zc); hB = batched(pool_fn, zw, con_te)
    hA = hA / (np.linalg.norm(hA, axis=1, keepdims=True) + 1e-9)
    hB = hB / (np.linalg.norm(hB, axis=1, keepdims=True) + 1e-9)
    mae_gap = float(np.linalg.norm(hA.mean(0) - hB.mean(0)))

    # ── multi-task: reconstruction + SigLIP CLS contrastive ──
    print("Training multi-task (reconstruction + SigLIP)...")
    tp = make_multitask(jax.random.key(SEED + 9))
    tp = train(multitask_loss, tp, (wav_tr, con_tr, Z_tr), STEPS, 1e-3, stochastic=True)
    t_rec_fn = jax.jit(lambda w, c: mt_forward(tp, w, c)[0])
    t_pool_fn = jax.jit(lambda w, c: mt_forward(tp, w, c)[3])
    t_za_fn = jax.jit(lambda w, c: mt_forward(tp, w, c)[1])
    t_zb_fn = jax.jit(lambda w, c: mt_forward(tp, w, c)[2])
    t_rec_te = batched(t_rec_fn, wav_te, con_te)
    t_denoise = float(np.mean((t_rec_te - Z_te) ** 2))
    th_tr = batched(t_pool_fn, wav_tr, con_tr); th_te = batched(t_pool_fn, wav_te, con_te)
    t_class = class_probe(th_tr, y_tr, th_te, y_te)
    tzA_te = batched(t_za_fn, wav_te, con_te); tzB_te = batched(t_zb_fn, wav_te, con_te)
    t_matched_cos = float(np.mean(np.sum(tzA_te * tzB_te, axis=1)))
    t_gap = float(np.linalg.norm(tzA_te.mean(0) - tzB_te.mean(0)))

    # ── multi-task with the InfoNCE head (same forward pass, softmax instead of
    #    sigmoid): does reconstruction close the gap for BOTH contrastive families? ──
    print("Training multi-task (reconstruction + InfoNCE)...")
    tip = make_multitask(jax.random.key(SEED + 9))   # same init as SigLIP variant
    tip = train(multitask_loss_infonce, tip, (wav_tr, con_tr, Z_tr), STEPS, 1e-3, stochastic=True)
    ti_pool_fn = jax.jit(lambda w, c: mt_forward(tip, w, c)[3])
    ti_za_fn = jax.jit(lambda w, c: mt_forward(tip, w, c)[1])
    ti_zb_fn = jax.jit(lambda w, c: mt_forward(tip, w, c)[2])
    ti_rec_te = batched(jax.jit(lambda w, c: mt_forward(tip, w, c)[0]), wav_te, con_te)
    ti_denoise = float(np.mean((ti_rec_te - Z_te) ** 2))
    tih_tr = batched(ti_pool_fn, wav_tr, con_tr); tih_te = batched(ti_pool_fn, wav_te, con_te)
    ti_class = class_probe(tih_tr, y_tr, tih_te, y_te)
    tizA_te = batched(ti_za_fn, wav_te, con_te); tizB_te = batched(ti_zb_fn, wav_te, con_te)
    ti_matched_cos = float(np.mean(np.sum(tizA_te * tizB_te, axis=1)))
    ti_gap = float(np.linalg.norm(tizA_te.mean(0) - tizB_te.mean(0)))

    # gate trajectory across recursion depth (does the gate learn adaptive depth?)
    nb = 512
    mw, mc = jnp.asarray(wav_te[:nb]), jnp.asarray(con_te[:nb])
    mt_seq = jnp.concatenate([jnp.broadcast_to(tp["clsA"], (nb, 1, DMODEL)),
                              jnp.broadcast_to(tp["clsB"], (nb, 1, DMODEL)),
                              wav_tokens(tp, mw), con_tokens(tp, mc)], axis=1)
    gate_mt = np.asarray(gates_over_depth(tp["blk"], mt_seq, MT_MASK))     # (DEPTH, 2)
    mae_seq = jnp.concatenate([jnp.broadcast_to(mp["cls"], (nb, 1, DMODEL)),
                               wav_tokens(mp, mw), con_tokens(mp, mc)], axis=1)
    gate_mae = np.asarray(gates_over_depth(mp["blk"], mae_seq, None))      # (DEPTH, 2)
    print("gate (attn,ffn) by depth  MAE :", np.round(gate_mae, 2).tolist())
    print("gate (attn,ffn) by depth  Multi:", np.round(gate_mt, 2).tolist())

    # ── DenoMAE-style evaluation: linear-probe modulation classification (AMC)
    #    across a sweep of test SNRs. Each model uses BOTH modalities so the
    #    comparison is fair (contrastive concatenates its two encoder outputs;
    #    the MAE uses its shared latent). The probe is linear (ridge).
    print("Linear-probe AMC vs SNR sweep...")

    def feat_contrastive(eA, eB, w, c):
        return np.concatenate([batched(eA, w), batched(eB, c)], axis=1)

    amc = {"infonce": [], "siglip": [], "mae": [], "multitask": [], "multitask_infonce": []}
    for si, snr in enumerate(SNR_GRID):
        wtr, ctr, _, ytr = generate(2500, 100 + si, snr=float(snr))
        wte, cte, _, yte = generate(1200, 500 + si, snr=float(snr))
        amc["infonce"].append(class_probe(feat_contrastive(embA, embB, wtr, ctr), ytr,
                                          feat_contrastive(embA, embB, wte, cte), yte))
        amc["siglip"].append(class_probe(feat_contrastive(sembA, sembB, wtr, ctr), ytr,
                                         feat_contrastive(sembA, sembB, wte, cte), yte))
        amc["mae"].append(class_probe(batched(pool_fn, wtr, ctr), ytr,
                                      batched(pool_fn, wte, cte), yte))
        amc["multitask"].append(class_probe(batched(t_pool_fn, wtr, ctr), ytr,
                                            batched(t_pool_fn, wte, cte), yte))
        amc["multitask_infonce"].append(class_probe(batched(ti_pool_fn, wtr, ctr), ytr,
                                                    batched(ti_pool_fn, wte, cte), yte))
    print("  SNR      :", "  ".join(f"{s:>5}" for s in SNR_GRID))
    for k in ("infonce", "siglip", "mae", "multitask", "multitask_infonce"):
        print(f"  {k:17s}:", "  ".join(f"{a*100:4.0f}%" for a in amc[k]))

    results = dict(noisy_mse=noisy_mse, gap=gap, matched_cos=matched_cos,
                   c_class=c_class, c_denoise=c_denoise,
                   s_matched_cos=s_matched_cos, s_gap=s_gap, s_class=s_class, s_denoise=s_denoise,
                   m_class=m_class, m_denoise=m_denoise, mae_gap=mae_gap,
                   t_class=t_class, t_denoise=t_denoise, t_matched_cos=t_matched_cos, t_gap=t_gap,
                   ti_class=ti_class, ti_denoise=ti_denoise,
                   ti_matched_cos=ti_matched_cos, ti_gap=ti_gap,
                   snr_grid=np.array(SNR_GRID, float),
                   amc_infonce=np.array(amc["infonce"]),
                   amc_siglip=np.array(amc["siglip"]),
                   amc_mae=np.array(amc["mae"]),
                   amc_multitask=np.array(amc["multitask"]),
                   amc_multitask_infonce=np.array(amc["multitask_infonce"]),
                   gate_mt=gate_mt, gate_mae=gate_mae)

    print("\n==================== RESULTS ====================")
    print(f"input noise floor (noisy vs clean MSE)  : {noisy_mse:.3f}")
    print("                     standalone -> multi-task (recon closes the gap?)")
    print(f"matched-pair cos   InfoNCE {matched_cos:+.3f} -> {ti_matched_cos:+.3f}   "
          f"SigLIP {s_matched_cos:+.3f} -> {t_matched_cos:+.3f}")
    print(f"centroid gap       InfoNCE {gap:.3f} -> {ti_gap:.3f}   "
          f"SigLIP {s_gap:.3f} -> {t_gap:.3f}   (MAE {mae_gap:.3f})")
    print(f"modulation class   InfoNCE {c_class*100:4.1f}%->{ti_class*100:4.1f}%   "
          f"SigLIP {s_class*100:4.1f}%->{t_class*100:4.1f}%   MAE {m_class*100:4.1f}%")
    print(f"recover-Z MSE      InfoNCE {c_denoise:.3f}->{ti_denoise:.3f}   "
          f"SigLIP {s_denoise:.3f}->{t_denoise:.3f}   MAE {m_denoise:.3f}")
    print("================================================\n")

    # save raw numbers + a few arrays for plotting
    out = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "public", "modality-gap"))
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "results.npz"),
             zA=zA_te[:1500], zB=zB_te[:1500], y=y_te[:1500],
             sA=sA_te[:1500], sB=sB_te[:1500],
             hA=hA[:1500], hB=hB[:1500],
             Z_te=Z_te[:8], wav_te=wav_te[:8], rec_te=rec_te[:8],
             **{k: np.array(v) for k, v in results.items()})
    print("saved arrays ->", os.path.join(out, "results.npz"))
    return results


if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    main()
