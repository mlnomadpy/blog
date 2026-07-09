"""velocity_ledger.py — the experiment behind "Transformers With a Velocity Ledger" (Arc D2).

Four parameter-matched character-level GPTs on tinyshakespeare (same corpus, tokenizer,
steps, optimizer, lr schedule, and seeds), differing ONLY in the residual-stream update:

  A. plain        pre-norm transformer:  x += Attn(ln x); x += MLP(ln x)   (forward Euler)
  B. ledger       ONE persistent velocity stream shared across both sub-updates,
                  zero-initialized at the embedding, D1's exact convention:
                      v = mu*v + (1-mu)*F(ln x);  x = x + h*v      (mu=0.9, h=1, both fixed)
                  At mu=0, h=1 this is EXACTLY variant A. v is depth-state, not weights:
                  the ledger adds zero parameters (verified and reported in the bundle).
  C. ngpt_lite    an honest SIMPLIFIED nGPT (arXiv 2410.01131), named ngpt-lite: the
                  hidden state lives on the sphere of radius sqrt(D) (retract = L2
                  renormalize after the embedding and after every sub-update); NO
                  LayerNorms inside blocks (the state is already normalized); the eigen
                  learning rates reduced to a per-layer, per-dim learnable step scale
                  alpha (init 1.0) on each sub-update:  x = retract(x + alpha ⊙ F(x)).
                  NOT claimed: full nGPT (no weight-matrix normalization, no logit
                  scale s_z, no QK normalization).
  D. ngpt_ledger  the synthesis (the post's novel experiment): the heavy-ball ledger
                  living alongside the normalized state — update v in ambient space from
                  the block output, take the scaled step, then retract (momentum in the
                  tangent-ish sense with retraction):
                      v = mu*v + (1-mu)*F(x);  x = retract(x + h * alpha ⊙ v)

Parameter matching: A and B are identical. C/D drop 2 LayerNorms per block (4D params)
and add 2 alphas per block (2D params): a net −2·D·LAYERS params, ≈0.09% of the model,
well within the 1% budget; exact counts are written to the bundle. All variants share
the same final LayerNorm + untied unembedding head (matched readout).

Telemetry (one npz + one json bundle in ./results/, dense enough to feed all viz):
  per run — train-loss curve (every TRAIN_LOG steps), val-loss curve (every VAL_LOG
  steps, 4 fixed val batches), BEST (early-stopped) val loss + the step it occurred,
  final val loss + bpc, the blow-up ratio (final/best), exact param count, wall-clock,
  a generated-text sample; depth telemetry on a fixed probe batch at 4 checkpoints
  (init / ~10% / 50% / 100% of steps): per-sub-update residual-stream norm profile,
  segment (path) lengths, turning angles; ledger variants add the velocity-norm
  profile through depth; ngpt variants add the pre-retraction raw step size (the
  post-retraction displacement is the seglen array).

TRUSTWORTHY-COMPARISON FIXES (v2, over the v1 severely-overfit run):
  1. Headline metric is the BEST (early-stopped) val loss over the whole curve, plus
     the step it occurred — NOT the endpoint. Every comparison number quotes best-val.
  2. Overfitting tamed two ways: (a) dropout=0.15 on the embedding, each residual
     sub-update, and the attention matrix (weight decay kept at 0.01); (b) a LARGER
     corpus — the complete works of Shakespeare (~5.4M chars, ~5x tinyshakespeare),
     Gutenberg-normalized to a compact ASCII vocab, with tinyshakespeare fallback if
     the download fails. Same STEPS now cover far fewer epochs (~14 vs ~90).
  3. >=3 seeds per variant (was 2) for the headline comparison, so the tie / robustness
     ordering is a real effect and not one-seed noise.
  4. The over-training blow-up is kept as an HONEST SECONDARY signal, not the headline:
     per run we export best-val AND final-val AND the blow-up ratio (final/best), so the
     post can show "tied at best; they differ in how badly they over-train" clearly
     labeled as over-training dynamics, not model quality.
  5. Sanity gate: a healthy plain baseline should reach best-val well below random and
     its final-val should now sit much closer to best-val (dropout tames the blow-up).

Dropout is deterministic (off) for all eval / val-loss / depth-telemetry / generation
passes (model.eval()) and on only for the training forward pass (model.train()).

SMOKE=1: tiny config (2 layers, 40 steps, 1 seed), all 4 variants, full telemetry
schema including best-val/best-step/blow-up + a dropout-on training pass, plus a 20-step
full-config timing probe to project the real run's wall-clock.

Run on Kaggle (single P100) via kgl.py. Never locally.
"""
import warnings; warnings.filterwarnings('ignore')
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import nnx

SMOKE = os.environ.get('SMOKE') == '1'
RESULTS = Path('results'); RESULTS.mkdir(exist_ok=True)

# ── config ──────────────────────────────────────────────────────────────────
# STEPS unchanged (12000) but the corpus is ~5x larger, so this is ~14 epochs, not ~90.
FULL_CFG = dict(D=192, LAYERS=6, HEADS=4, T=128, FF=768, BATCH=64, STEPS=12000,
                TRAIN_LOG=50, VAL_LOG=250, SEEDS=(0, 1, 2), GEN_LEN=300, DROPOUT=0.15)
SMOKE_CFG = dict(D=64, LAYERS=2, HEADS=2, T=64, FF=128, BATCH=16, STEPS=40,
                 TRAIN_LOG=5, VAL_LOG=10, SEEDS=(0,), GEN_LEN=60, DROPOUT=0.15)
CFG = SMOKE_CFG if SMOKE else FULL_CFG
D, LAYERS, HEADS, T, FF = CFG['D'], CFG['LAYERS'], CFG['HEADS'], CFG['T'], CFG['FF']
BATCH, STEPS = CFG['BATCH'], CFG['STEPS']
TRAIN_LOG, VAL_LOG = CFG['TRAIN_LOG'], CFG['VAL_LOG']
SEEDS, GEN_LEN = CFG['SEEDS'], CFG['GEN_LEN']
DROPOUT = CFG['DROPOUT']

MU, H_STEP = 0.9, 1.0                       # fixed, as D1 (momentum_resnet.py)
PEAK_LR, END_LR, WD, CLIP = 1e-3, 1e-4, 0.01, 1.0
WARMUP = max(10, STEPS // 60)
CKPTS = sorted({0, max(1, STEPS // 10), STEPS // 2, STEPS})
VARIANTS = ('plain', 'ledger', 'ngpt_lite', 'ngpt_ledger')
VAL_BATCH_SEEDS = (900001, 900002, 900003, 900004)
PROBE_SEED, PROBE_BS = 4242, 8
GEN_SEED, GEN_TEMP = 7, 0.8

# ── data: complete works of Shakespeare, char level (~5.4M chars, ~5x tinyshakespeare)
# Same domain/vocab flavour as tinyshakespeare so best-val targets stay comparable, but
# large enough that 12000 steps is ~14 epochs, not the ~90 that memorized v1. Normalized
# to a compact ASCII vocab; falls back to tinyshakespeare if the Gutenberg download fails.
TINY_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
FULL_URL = 'https://www.gutenberg.org/cache/epub/100/pg100.txt'   # complete works
_UNI = {                                    # curly quotes / dashes / ligatures -> ASCII
    '‘': "'", '’': "'", '“': '"', '”': '"', '–': '-',
    '—': '-', '…': '...', 'æ': 'ae', 'œ': 'oe', 'Æ': 'AE',
    'Œ': 'OE', 'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
    'à': 'a', 'â': 'a', 'ä': 'a', 'î': 'i', 'ï': 'i',
    'ç': 'c', 'Ç': 'C', 'É': 'E', 'À': 'A', 'ô': 'o',
    'û': 'u', 'á': 'a', 'ñ': 'n', '\t': ' ', '\r': '',
}


def _normalize(t):
    for k, v in _UNI.items():
        t = t.replace(k, v)
    # keep printable ASCII + newline only; drop anything still exotic
    return ''.join(ch for ch in t if ch == '\n' or 32 <= ord(ch) < 127)


def _load_corpus():
    import urllib.request
    dst = Path('/tmp/velocity_corpus.txt')
    if dst.exists():
        return dst.read_text(), 'cached'
    try:                                    # prefer the larger corpus
        raw = urllib.request.urlopen(FULL_URL, timeout=60).read().decode('utf-8', 'ignore')
        s = raw.find('*** START OF THE PROJECT')
        e = raw.find('*** END OF THE PROJECT')
        body = raw[raw.find('\n', s) + 1:e] if (s >= 0 and e >= 0) else raw
        t = _normalize(body)
        if len(t) > 3_000_000:              # sanity: got the whole thing
            dst.write_text(t)
            return t, 'complete_shakespeare'
        raise RuntimeError(f'corpus too short: {len(t)}')
    except Exception as ex:
        print(f'[data] full corpus failed ({ex!r}); falling back to tinyshakespeare', flush=True)
        raw = urllib.request.urlopen(TINY_URL, timeout=60).read().decode('utf-8', 'ignore')
        t = _normalize(raw)
        dst.write_text(t)
        return t, 'tinyshakespeare'


text, CORPUS = _load_corpus()
chars = sorted(set(text)); V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for c, i in stoi.items()}
data = np.array([stoi[c] for c in text], dtype=np.int32)
n_split = int(0.9 * len(data)); train_data, val_data = data[:n_split], data[n_split:]
RANDOM_LOSS = math.log(V)                    # uniform-prediction baseline for the sanity gate
print(f'[data] corpus={CORPUS} chars={len(data):,} vocab={V} '
      f'epochs~={STEPS * BATCH * T / len(train_data):.1f} random_loss={RANDOM_LOSS:.3f}', flush=True)


def get_batch(split, bs, seed):
    d = train_data if split == 'train' else val_data
    r = np.random.RandomState(seed); ix = r.randint(0, len(d) - T - 1, size=bs)
    x = np.stack([d[i:i + T] for i in ix]); y = np.stack([d[i + 1:i + 1 + T] for i in ix])
    return jnp.asarray(x), jnp.asarray(y)


# ── model ────────────────────────────────────────────────────────────────────
def retract(z, radius):
    return z * (radius / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8))


class Block(nnx.Module):
    def __init__(s, variant, dm, ff, heads, drop, *, rngs):
        ngpt = variant.startswith('ngpt')
        if ngpt:
            s.alpha1 = nnx.Param(jnp.ones((dm,)))
            s.alpha2 = nnx.Param(jnp.ones((dm,)))
        else:
            s.ln1 = nnx.LayerNorm(dm, rngs=rngs)
            s.ln2 = nnx.LayerNorm(dm, rngs=rngs)
        s.attn = nnx.MultiHeadAttention(num_heads=heads, in_features=dm, decode=False,
                                        dropout_rate=drop, rngs=rngs)   # attention dropout
        s.fc1 = nnx.Linear(dm, ff, rngs=rngs)
        s.fc2 = nnx.Linear(ff, dm, rngs=rngs)
        s.drop1 = nnx.Dropout(drop, rngs=rngs)   # residual dropout after each sub-update
        s.drop2 = nnx.Dropout(drop, rngs=rngs)


class GPT(nnx.Module):
    def __init__(s, variant, drop=0.0, *, rngs, dm=None, ff=None, heads=None, layers=None, ctx=None):
        s.variant = variant
        s.dm = dm or D; s.ctx = ctx or T
        s.tok = nnx.Embed(V, s.dm, rngs=rngs)
        s.pos = nnx.Embed(s.ctx, s.dm, rngs=rngs)
        s.edrop = nnx.Dropout(drop, rngs=rngs)   # embedding dropout
        _blocks = [Block(variant, s.dm, ff or FF, heads or HEADS, drop, rngs=rngs)
                   for _ in range(layers or LAYERS)]
        # nnx.data(...) so strict-nnx (flax >=0.12) treats the block list as pytree data,
        # not a static attr; falls back to a plain list on older flax that lacks nnx.data.
        s.blocks = nnx.data(_blocks) if hasattr(nnx, 'data') else _blocks
        s.lnf = nnx.LayerNorm(s.dm, rngs=rngs)
        s.head = nnx.Linear(s.dm, V, use_bias=False, rngs=rngs)

    def _stream(s, idx, collect=False):
        ngpt = s.variant.startswith('ngpt')
        led = s.variant in ('ledger', 'ngpt_ledger')
        radius = math.sqrt(s.dm)
        Tn = idx.shape[1]
        x = s.edrop(s.tok(idx) + s.pos(jnp.arange(Tn)))
        if ngpt:
            x = retract(x, radius)
        mask = nnx.make_causal_mask(idx)
        v = jnp.zeros_like(x)
        rec = {'x': [x], 'v': [], 'raw': []}
        for b in s.blocks:
            for which in ('attn', 'mlp'):
                if ngpt:
                    inp = x
                    alpha = (b.alpha1 if which == 'attn' else b.alpha2).value
                else:
                    inp = (b.ln1 if which == 'attn' else b.ln2)(x)
                    alpha = None
                if which == 'attn':
                    f = b.drop1(b.attn(inp, mask=mask))
                else:
                    f = b.drop2(b.fc2(jax.nn.gelu(b.fc1(inp))))
                if led:
                    v = MU * v + (1.0 - MU) * f       # the velocity ledger (heavy-ball)
                    step = H_STEP * v
                else:
                    step = f
                if alpha is not None:
                    step = alpha * step               # ngpt per-dim step scale
                xn = x + step
                if ngpt:
                    xn = retract(xn, radius)          # retraction back to the sphere
                if collect:
                    rec['x'].append(xn)
                    rec['v'].append(v)
                    rec['raw'].append(step)
                x = xn
        return x, rec

    def __call__(s, idx):
        x, _ = s._stream(idx)
        return s.head(s.lnf(x))


@nnx.jit
def fwd_logits(m, idx):
    return m(idx)


# flax version compat: newer nnx.Optimizer takes wrt= and update(model, grads);
# older takes neither. Decide once from the signature.
import inspect
NEW_OPT = 'wrt' in inspect.signature(nnx.Optimizer.__init__).parameters


def make_opt(model, tx):
    return (nnx.Optimizer(model, tx, wrt=nnx.Param) if NEW_OPT
            else nnx.Optimizer(model, tx))


def opt_update(o, m, g):
    if NEW_OPT:
        o.update(m, g)
    else:
        o.update(g)


def param_count(model):
    return sum(int(np.prod(p.shape)) for p in jax.tree.leaves(nnx.state(model, nnx.Param)))


def dropout_is_off(model):
    """Confirm every Dropout module in the model is deterministic (eval mode)."""
    flags = [bool(getattr(m, 'deterministic', True))
             for _, m in nnx.iter_graph(model) if isinstance(m, nnx.Dropout)]
    return len(flags) > 0 and all(flags)


# ── depth telemetry (eager, on a fixed probe batch) ─────────────────────────
def depth_stats(model, probe):
    _, rec = model._stream(probe, collect=True)
    xs = np.stack([np.asarray(a) for a in rec['x']])          # [2L+1, B, T, D]
    seg = np.diff(xs, axis=0)                                  # [2L,  B, T, D]
    seglen = np.linalg.norm(seg, axis=-1)                      # [2L,  B, T]
    xnorm = np.linalg.norm(xs, axis=-1)                        # [2L+1,B, T]
    d1, d2 = seg[:-1], seg[1:]
    cos = ((d1 * d2).sum(-1)
           / (np.linalg.norm(d1, axis=-1) * np.linalg.norm(d2, axis=-1) + 1e-12))
    turn = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))      # [2L-1, B, T]
    out = {
        'xnorm': xnorm.mean((1, 2)).astype(np.float32),        # norm profile through depth
        'seglen': seglen.mean((1, 2)).astype(np.float32),      # per-sub-update path length
        'turn': turn.mean((1, 2)).astype(np.float32),          # turning angle per sub-update
    }
    if model.variant in ('ledger', 'ngpt_ledger'):
        vn = np.stack([np.asarray(a) for a in rec['v']])
        out['vnorm'] = np.linalg.norm(vn, axis=-1).mean((1, 2)).astype(np.float32)
    if model.variant.startswith('ngpt'):
        raw = np.stack([np.asarray(a) for a in rec['raw']])
        out['rawstep'] = np.linalg.norm(raw, axis=-1).mean((1, 2)).astype(np.float32)
        # post-retraction displacement IS out['seglen']; pre-retraction raw step above.
    return out


# ── generation (fixed sliding window of exactly T tokens -> one jit compile) ─
def generate(model, n=GEN_LEN, temp=GEN_TEMP, seed=GEN_SEED):
    model.eval()                                               # dropout off for sampling
    prompt = 'ROMEO:\n'
    prefix = text[n_split - (T - len(prompt)):n_split]         # real corpus chars as primer
    ctx = [stoi[c] for c in (prefix + prompt)][-T:]
    r = np.random.RandomState(seed); out = []
    for _ in range(n):
        logits = np.asarray(fwd_logits(model, jnp.asarray([ctx])))[0, -1] / temp
        p = np.exp(logits - logits.max()); p /= p.sum()
        t = int(r.choice(V, p=p)); out.append(t); ctx = ctx[1:] + [t]
    return ''.join(itos[t] for t in out)


# ── one training run ─────────────────────────────────────────────────────────
def run_one(variant, seed, arrays):
    key = f'{variant}_s{seed}'
    t0 = time.time()
    model = GPT(variant, DROPOUT, rngs=nnx.Rngs(seed))
    nparams = param_count(model)
    sched = optax.warmup_cosine_decay_schedule(0.0, PEAK_LR, WARMUP, STEPS, END_LR)
    tx = optax.chain(optax.clip_by_global_norm(CLIP), optax.adamw(sched, weight_decay=WD))
    opt = make_opt(model, tx)

    @nnx.jit
    def step(m, o, x, y):
        def loss(mm):
            return optax.softmax_cross_entropy_with_integer_labels(mm(x), y).mean()
        l, g = nnx.value_and_grad(loss)(m)
        opt_update(o, m, g)
        return l

    @nnx.jit
    def eval_loss(m, x, y):
        return optax.softmax_cross_entropy_with_integer_labels(m(x), y).mean()

    val_batches = [get_batch('val', BATCH, sd) for sd in VAL_BATCH_SEEDS]

    def val_loss():
        # dropout OFF for the val forward pass; restore train mode afterwards so the
        # jitted train `step` keeps its graphdef and is never re-traced mid-loop.
        model.eval()
        vl = float(np.mean([float(eval_loss(model, vx, vy)) for vx, vy in val_batches]))
        model.train()
        return vl

    probe, _ = get_batch('val', PROBE_BS, PROBE_SEED)

    wb = None
    if os.environ.get('WANDB') == '1' and os.environ.get('WANDB_API_KEY'):
        try:
            import wandb
            wb = wandb.init(project=os.environ.get('WANDB_PROJECT', 'blog-velocity-ledger'),
                            name=key, config=dict(CFG, variant=variant, seed=seed,
                                                  mu=MU, h=H_STEP, peak_lr=PEAK_LR),
                            reinit=True)
        except Exception as e:
            print(f'  [wandb off: {e!r}]', flush=True)

    train_steps, train_losses, val_steps, val_losses = [], [], [], []
    ck_i = 0

    def checkpoint(at_step):
        nonlocal ck_i
        model.eval()                                           # dropout off for telemetry
        st = depth_stats(model, probe)
        model.train()
        for name, arr in st.items():
            arrays[f'{key}/ck{ck_i}_{name}'] = arr
        ck_i += 1
        print(f'  [{key}] depth telemetry @ step {at_step} '
              f'(seglen mean {st["seglen"].mean():.3f}, turn mean {st["turn"].mean():.1f} deg)',
              flush=True)

    model.train()                                              # dropout ON for training
    dropout_on_train = not dropout_is_off(model)               # sanity: dropout path active
    checkpoint(0)                                              # init (eval inside)
    for it in range(1, STEPS + 1):
        x, y = get_batch('train', BATCH, seed * 1000003 + it)
        l = step(model, opt, x, y)
        if it % TRAIN_LOG == 0 or it == STEPS:
            train_steps.append(it); train_losses.append(float(l))
        if it % VAL_LOG == 0 or it == STEPS:
            vl = val_loss(); val_steps.append(it); val_losses.append(vl)
            if wb:
                try: wb.log({'train_loss': float(l), 'val_loss': vl, 'step': it})
                except Exception: pass
        if it in CKPTS:
            checkpoint(it)
        if it == 100 and not SMOKE:
            eta = (time.time() - t0) / it * STEPS
            print(f'  [{key}] {it} steps in {time.time()-t0:.0f}s -> projected {eta/60:.1f} min/run',
                  flush=True)

    # ── headline metric: BEST (early-stopped) val loss, not the endpoint ─────
    val_arr = np.asarray(val_losses, np.float32)
    best_i = int(np.argmin(val_arr))
    best_val = float(val_arr[best_i]); best_step = int(val_steps[best_i])
    best_bpc = best_val / math.log(2)
    final_val = float(val_losses[-1]); final_step = int(val_steps[-1])
    final_bpc = final_val / math.log(2)
    blowup_ratio = final_val / best_val if best_val > 0 else float('nan')   # secondary signal
    below_random = best_val < RANDOM_LOSS
    bpc = final_bpc                                            # kept for back-compat prints
    sample = generate(model)                                  # eval mode inside
    wall = time.time() - t0
    nan_free = bool(np.isfinite(train_losses).all() and np.isfinite(val_losses).all())

    arrays[f'{key}/train_steps'] = np.asarray(train_steps, np.int32)
    arrays[f'{key}/train_loss'] = np.asarray(train_losses, np.float32)
    arrays[f'{key}/val_steps'] = np.asarray(val_steps, np.int32)
    arrays[f'{key}/val_loss'] = val_arr

    if wb:
        try:
            wb.summary.update({'best_val_loss': best_val, 'best_step': best_step,
                               'best_bpc': best_bpc, 'final_val_loss': final_val,
                               'final_bpc': final_bpc, 'blowup_ratio': blowup_ratio,
                               'below_random': below_random, 'params': nparams, 'wall_s': wall})
            wb.finish()
        except Exception:
            pass

    print(f'  [{key}] done: BEST val {best_val:.4f} ({best_bpc:.4f} bpc) @ step {best_step} '
          f'| final {final_val:.4f} (blowup {blowup_ratio:.2f}x) | below_random={below_random} '
          f'| dropout_on_train={dropout_on_train} | {nparams:,} params, {wall:.0f}s, '
          f'nan_free={nan_free}', flush=True)
    return dict(variant=variant, seed=seed, params=nparams,
                best_val_loss=round(best_val, 5), best_step=best_step, best_bpc=round(best_bpc, 5),
                final_val_loss=round(final_val, 5), final_step=final_step,
                final_bpc=round(final_bpc, 5), blowup_ratio=round(blowup_ratio, 4),
                below_random=below_random, dropout_on_train=dropout_on_train,
                wall_s=round(wall, 1), nan_free=nan_free, sample=sample)


# ── smoke-only: full-config timing probe (projects the real run's wall-clock) ─
def timing_probe():
    F = FULL_CFG
    print('timing probe: full config, plain variant, 20 steps ...', flush=True)
    global D, LAYERS, HEADS, T, FF                              # temporarily full-size
    saved = (D, LAYERS, HEADS, T, FF)
    D, LAYERS, HEADS, T, FF = F['D'], F['LAYERS'], F['HEADS'], F['T'], F['FF']
    try:
        model = GPT('plain', rngs=nnx.Rngs(0))
        tx = optax.chain(optax.clip_by_global_norm(CLIP), optax.adamw(PEAK_LR, weight_decay=WD))
        opt = make_opt(model, tx)

        @nnx.jit
        def step(m, o, x, y):
            def loss(mm):
                return optax.softmax_cross_entropy_with_integer_labels(mm(x), y).mean()
            l, g = nnx.value_and_grad(loss)(m)
            opt_update(o, m, g)
            return l

        x, y = get_batch('train', F['BATCH'], 0)
        step(model, opt, x, y)                                   # compile
        t0 = time.time()
        for it in range(20):
            l = step(model, opt, x, y)
        float(l)
        sps = (time.time() - t0) / 20
        total = sps * F['STEPS'] * len(VARIANTS) * len(F['SEEDS'])
        print(f'timing probe: {sps*1000:.0f} ms/step at full config -> '
              f'projected {total/3600:.2f} h for {len(VARIANTS)*len(F["SEEDS"])} runs '
              f'of {F["STEPS"]} steps (+ telemetry overhead)', flush=True)
        return dict(ms_per_step=round(sps * 1000, 1), projected_hours=round(total / 3600, 2))
    finally:
        D, LAYERS, HEADS, T, FF = saved


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print(f'velocity_ledger: SMOKE={SMOKE}  device={jax.devices()[0]}', flush=True)
    print(f'config: d={D} layers={LAYERS} heads={HEADS} ctx={T} ff={FF} batch={BATCH} '
          f'steps={STEPS} seeds={list(SEEDS)} mu={MU} h={H_STEP} ckpts={CKPTS}', flush=True)

    arrays = {'ckpt_steps': np.asarray(CKPTS, np.int32)}   # CKPTS already includes 0 (init)
    runs = {}
    for variant in VARIANTS:
        for seed in SEEDS:
            print(f'=== {variant} seed {seed} ===', flush=True)
            runs[f'{variant}_s{seed}'] = run_one(variant, seed, arrays)

    # ── headline (best-val) + secondary (blow-up) comparison, per variant across seeds ─
    def _agg(vals):
        a = np.asarray(vals, np.float64)
        return dict(mean=round(float(a.mean()), 5), std=round(float(a.std()), 5),
                    min=round(float(a.min()), 5), max=round(float(a.max()), 5), n=len(a))
    comparison = {}
    for variant in VARIANTS:
        rs = [runs[f'{variant}_s{s}'] for s in SEEDS]
        comparison[variant] = {
            'best_val': _agg([r['best_val_loss'] for r in rs]),      # HEADLINE
            'best_bpc': _agg([r['best_bpc'] for r in rs]),
            'best_step': _agg([r['best_step'] for r in rs]),
            'final_val': _agg([r['final_val_loss'] for r in rs]),    # SECONDARY (over-trained)
            'blowup_ratio': _agg([r['blowup_ratio'] for r in rs]),   # SECONDARY dynamics
            'below_random_all_seeds': all(r['below_random'] for r in rs),
            'params': rs[0]['params'],
        }
    # honest ordering readouts for the post authors
    headline_order = sorted(VARIANTS, key=lambda v: comparison[v]['best_val']['mean'])
    blowup_order = sorted(VARIANTS, key=lambda v: comparison[v]['blowup_ratio']['mean'])

    summary = {
        'smoke': SMOKE,
        'corpus': CORPUS, 'corpus_chars': int(len(data)), 'random_loss': round(RANDOM_LOSS, 5),
        'headline_metric': 'best_val (early-stopped minimum val loss over training); '
                           'the final-val + blowup_ratio are a SECONDARY over-training '
                           'dynamics signal, NOT a quality measure',
        'headline_order_best_val': list(headline_order),
        'blowup_order_least_to_most': list(blowup_order),
        'config': dict(CFG, mu=MU, h=H_STEP, peak_lr=PEAK_LR, end_lr=END_LR,
                       warmup=WARMUP, weight_decay=WD, clip=CLIP, dropout=DROPOUT,
                       corpus=CORPUS, random_loss=round(RANDOM_LOSS, 5),
                       ckpt_steps=sorted({0, *CKPTS}), variants=list(VARIANTS),
                       seeds=list(SEEDS), vocab=V, val_batch_seeds=list(VAL_BATCH_SEEDS),
                       probe_seed=PROBE_SEED, probe_bs=PROBE_BS,
                       gen_seed=GEN_SEED, gen_temp=GEN_TEMP),
        'design_notes': {
            'ledger': 'one shared velocity across both sub-updates per layer, zero-init at '
                      'the embedding; v = mu*v + (1-mu)*F(ln x); x = x + h*v; mu=0.9, h=1 '
                      'fixed (D1 convention; mu=0 recovers plain exactly); adds 0 params',
            'ngpt_lite': 'state retracted to radius sqrt(D) after embedding and every '
                         'sub-update; no in-block LayerNorms; per-layer per-dim learnable '
                         'step scales alpha (init 1.0); shared final LN + head; NOT full '
                         'nGPT (no weight normalization, logit scale, or QK norm)',
            'ngpt_ledger': 'ambient-space heavy-ball ledger + step x + h*alpha*v, then '
                           'retraction to the sphere',
            'regularization': f'dropout={DROPOUT} on embedding + each residual sub-update + '
                              'attention matrix; weight_decay=0.01; dropout OFF (deterministic) '
                              'for all val/telemetry/generation passes',
            'telemetry': 'depth arrays indexed per SUB-update (2 per layer): xnorm [2L+1], '
                         'seglen/vnorm/rawstep [2L], turn [2L-1]; probe batch fixed '
                         '(val, seed 4242, bs 8); ngpt post-retraction displacement = seglen',
        },
        'comparison': comparison,
        'runs': runs,
    }
    if SMOKE:
        try:
            summary['timing_probe'] = timing_probe()
        except Exception as e:
            summary['timing_probe'] = {'error': repr(e)}
            print(f'timing probe failed: {e!r}', flush=True)

    summary['total_wall_s'] = round(time.time() - t0, 1)
    np.savez_compressed(RESULTS / 'velocity_ledger.npz', **arrays)
    (RESULTS / 'velocity_ledger_summary.json').write_text(json.dumps(summary, indent=1))
    print(f'wrote {RESULTS}/velocity_ledger.npz ({(RESULTS/"velocity_ledger.npz").stat().st_size//1024} KB) '
          f'+ velocity_ledger_summary.json', flush=True)
    print(f'HEADLINE (best-val mean, low=better): '
          + ', '.join(f'{v}={comparison[v]["best_val"]["mean"]:.3f}' for v in headline_order), flush=True)
    print(f'SECONDARY (blowup final/best, low=robuster): '
          + ', '.join(f'{v}={comparison[v]["blowup_ratio"]["mean"]:.2f}x' for v in blowup_order), flush=True)
    brief = {k: dict(best_val=r['best_val_loss'], best_step=r['best_step'],
                     final_val=r['final_val_loss'], blowup=r['blowup_ratio'],
                     below_random=r['below_random'], dropout_on=r['dropout_on_train'],
                     params=r['params'], nan_free=r['nan_free']) for k, r in runs.items()}
    print('VELOCITY_LEDGER_DONE', json.dumps({'smoke': SMOKE, 'corpus': CORPUS,
                                              'total_s': summary['total_wall_s'],
                                              'runs': brief}), flush=True)


if __name__ == '__main__':
    main()
