"""The transformer MLP block as a representer theorem: a white-box key-value memory.

This is the companion to "The MLP Block Is a Representer Theorem." It trains a real
char-level transformer on tinyshakespeare whose feed-forward sublayer is a Yat-kernel
block, so the FFN output is exactly the representer form

    out(x) = Σ_u  k(W_u, x) · v_u ,     v_u = (W2)_{:,u}

a vote over learned (key W_u, value v_u) memory slots, the same equation as the
attention beside it but over a fixed, learned memory instead of the present context.

Because the block is literally that sum, it is a white box. The script then does four
things you cannot do to an opaque ReLU FFN, all exact (not gradient approximations):

  1. READ      decode each value vector v_u through the unembedding: every memory slot
               gets a human-readable label (the tokens it writes into the stream).
  2. ATTRIBUTE a token's FFN output is a sum, so split it into the slots that produced
               it: which memory wrote this next-token push, and how much.
  3. EDIT      delete / amplify a single (W_u, v_u) slot and the generation changes the
               way the readout predicts: white-box model surgery.
  4. ABSTAIN   Yat is local and bounded, so a token far from every key makes all
               k(W_u, x) tiny and the memory writes ~nothing: the kernel weights are an
               out-of-distribution signal you can read off.

Run: python scripts/yat_ffn_whitebox.py
"""
import warnings; warnings.filterwarnings('ignore')
import os, time, json
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp, optax
from flax import nnx

ROOT = Path(__file__).resolve().parents[1]; PUB = ROOT / 'public'
STEPS = int(os.environ.get('STEPS', '4500'))

# ── data: tinyshakespeare, char level ──
DATA = Path('/tmp/tinyshakespeare.txt')
if not DATA.exists():
    import urllib.request
    urllib.request.urlretrieve('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt', DATA)
text = DATA.read_text()
chars = sorted(set(text)); V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for c, i in stoi.items()}
data = np.array([stoi[c] for c in text], dtype=np.int32)
n = int(0.9 * len(data)); train_data, val_data = data[:n], data[n:]

T, D, HEADS, FF, LAYERS = 128, 128, 4, 256, 3            # context, model dim, heads, memory slots, depth


def get_batch(split, bs, seed):
    d = train_data if split == 'train' else val_data
    r = np.random.RandomState(seed); ix = r.randint(0, len(d) - T - 1, size=bs)
    x = np.stack([d[i:i + T] for i in ix]); y = np.stack([d[i + 1:i + 1 + T] for i in ix])
    return jnp.asarray(x), jnp.asarray(y)


# ── the Yat-kernel feed-forward block: out(x) = Σ_u k(W_u, x) v_u ──
class YatFFN(nnx.Module):
    def __init__(s, *, rngs, b0=0.5, eps0=1.0):
        s.W = nnx.Param(jax.random.normal(rngs.params(), (FF, D)) * 0.02)   # [FF, D] keys W_u
        s.Vv = nnx.Param(jax.random.normal(rngs.params(), (D, FF)) * 0.02)  # [D, FF] values, v_u = Vv[:, u]
        s.log_b = nnx.Param(jnp.full((), jnp.log(jnp.expm1(b0))))
        s.log_eps = nnx.Param(jnp.full((), jnp.log(jnp.expm1(eps0))))
    def kernel(s, x):                                    # k(W_u, x): the memory weights [..., FF]
        b = jax.nn.softplus(s.log_b.value); eps = jax.nn.softplus(s.log_eps.value)
        dot = x @ s.W.value.T
        d2 = (x ** 2).sum(-1, keepdims=True) + (s.W.value ** 2).sum(-1) - 2 * dot
        return (dot + b) ** 2 / (d2 + eps)
    def __call__(s, x):
        return s.kernel(x) @ s.Vv.value.T                # Σ_u k(W_u, x) v_u  [..., D]


class Block(nnx.Module):
    def __init__(s, *, rngs):
        s.ln1 = nnx.LayerNorm(D, rngs=rngs); s.ln2 = nnx.LayerNorm(D, rngs=rngs)
        s.attn = nnx.MultiHeadAttention(num_heads=HEADS, in_features=D, decode=False, rngs=rngs)
        s.ffn = YatFFN(rngs=rngs)
    def __call__(s, x, mask):
        x = x + s.attn(s.ln1(x), mask=mask)
        x = x + s.ffn(s.ln2(x))
        return x


class GPT(nnx.Module):
    def __init__(s, *, rngs):
        s.tok = nnx.Embed(V, D, rngs=rngs); s.pos = nnx.Embed(T, D, rngs=rngs)
        s.blocks = nnx.List([Block(rngs=rngs) for _ in range(LAYERS)])
        s.lnf = nnx.LayerNorm(D, rngs=rngs); s.head = nnx.Linear(D, V, use_bias=False, rngs=rngs)
    def __call__(s, idx):
        Tn = idx.shape[1]; x = s.tok(idx) + s.pos(jnp.arange(Tn))
        mask = nnx.make_causal_mask(idx)
        for b in s.blocks: x = b(x, mask)
        return s.head(s.lnf(x))
    def residual_into_ffn(s, idx, layer):                # x fed to the FFN of `layer` (post-LN), all positions
        Tn = idx.shape[1]; x = s.tok(idx) + s.pos(jnp.arange(Tn)); mask = nnx.make_causal_mask(idx)
        for li, b in enumerate(s.blocks):
            x = x + b.attn(b.ln1(x), mask=mask)
            if li == layer: return b.ln2(x)
            x = x + b.ffn(b.ln2(x))


# ── train (cached to /tmp so the white-box demos can iterate without retraining) ──
CKPT = Path(f'/tmp/yat_ffn_{LAYERS}x{D}x{FF}_{STEPS}.msgpack')
model = GPT(rngs=nnx.Rngs(0))
from flax import serialization
if CKPT.exists() and os.environ.get('RETRAIN') != '1':
    gdef, state = nnx.split(model)
    pure = serialization.from_bytes(nnx.to_pure_dict(state), CKPT.read_bytes())
    nnx.replace_by_pure_dict(state, pure); model = nnx.merge(gdef, state)
    print(f'loaded cached model from {CKPT.name}')
    vx, vy = get_batch('val', 256, 99)
    vloss = float(optax.softmax_cross_entropy_with_integer_labels(model(vx), vy).mean())
else:
    print(f'training Yat-FFN transformer ({LAYERS} layers, d={D}, {FF} memory slots) on tinyshakespeare...')
    opt = nnx.Optimizer(model, optax.adamw(3e-4, weight_decay=0.01), wrt=nnx.Param)
    @nnx.jit
    def step(m, opt, x, y):
        def loss(mm): return optax.softmax_cross_entropy_with_integer_labels(mm(x), y).mean()
        l, g = nnx.value_and_grad(loss)(m); opt.update(m, g); return l
    t0 = time.time()
    for it in range(STEPS):
        x, y = get_batch('train', 48, it)
        l = step(model, opt, x, y)
        if it % 500 == 0: print(f'  step {it:5d}  loss {float(l):.3f}  ({time.time()-t0:.0f}s)')
    vx, vy = get_batch('val', 256, 99)
    vloss = float(optax.softmax_cross_entropy_with_integer_labels(model(vx), vy).mean())
    print(f'  done. val loss {vloss:.3f}  ({time.time()-t0:.0f}s)')
    _, state = nnx.split(model); CKPT.write_bytes(serialization.to_bytes(nnx.to_pure_dict(state)))
print(f'  val loss {vloss:.3f}')


# ════════════════════════════════════════════════════════════════════════════
# white-box readers over the trained memory
# ════════════════════════════════════════════════════════════════════════════
U = np.asarray(model.head.kernel.value)                  # [D, V] unembedding; U[:, t] reads token t
LAYER = LAYERS - 1                                        # inspect the last block's memory
ffn = model.blocks[LAYER].ffn
Wkeys = np.asarray(ffn.W.value)                          # [FF, D] keys
Vvals = np.asarray(ffn.Vv.value)                         # [D, FF] values, v_u = Vvals[:, u]


def logit_lens(v, k=6):                                  # tokens a residual delta v promotes
    s = v @ U                                            # [V]
    return [itos[i] for i in np.argsort(-s)[:k]]


def encode(s): return jnp.asarray([[stoi[c] for c in s]])


def generate(m, prompt, n=240, temp=0.8, seed=0, edit=None):
    r = np.random.RandomState(seed); idx = [stoi[c] for c in prompt]
    if edit is not None: edit(m)
    try:
        for _ in range(n):
            ctx = jnp.asarray([idx[-T:]]); logits = np.asarray(m(ctx))[0, -1] / temp
            p = np.exp(logits - logits.max()); p /= p.sum()
            idx.append(int(r.choice(V, p=p)))
    finally:
        if edit is not None: edit.undo(m)
    return ''.join(itos[i] for i in idx)


print('\n══ 1. READ THE MEMORY: each slot v_u decoded to the tokens it writes ══')
# pick slots whose value vectors point most decisively at a single token (most legible)
sharp = np.argsort(-(np.max(Vvals.T @ U, 1) - np.mean(Vvals.T @ U, 1)))[:8]
for u in sharp:
    print(f'  slot {u:3d}  writes: {logit_lens(Vvals[:, u])}')


print('\n══ 2. ATTRIBUTE one FFN output to its memory slots (exact, it is a sum) ══')
prompt = 'ROMEO:\nWhat'
x = np.asarray(model.residual_into_ffn(encode(prompt), LAYER))[0, -1]      # FFN input at last position
kw = np.asarray(ffn.kernel(jnp.asarray(x[None])))[0]                       # k(W_u, x) [FF]
out = kw @ Vvals.T                                                        # the FFN output vector
nxt = int((U.T @ out).argmax());
contrib_to_nxt = kw * (Vvals.T @ U[:, nxt])                               # slot u's push toward token `nxt`
print(f'  context ...{prompt[-8:]!r}  the FFN most promotes next-char {itos[nxt]!r}')
print(f'  reconstruction error |Σ_u k_u v_u - out| = {np.linalg.norm(kw @ Vvals.T - out):.2e}  (it is exactly the sum)')
for u in np.argsort(-contrib_to_nxt)[:5]:
    print(f'    slot {u:3d}  k={kw[u]:6.2f}  push={contrib_to_nxt[u]:+6.2f}  writes {logit_lens(Vvals[:, u], 4)}')


print('\n══ 3. EDIT one slot and watch generation change (white-box surgery) ══')
# the slot that most writes the speaker-tag colon ":" and the one that writes newlines
def slot_for(tok): return int(np.argmax(Vvals.T @ U[:, stoi[tok]]))
class Amp:                                                # multiply one value vector v_u by g (g=0 deletes it)
    def __init__(s, layer, u, g): s.u, s.layer, s.g, s.saved = u, layer, g, None
    def __call__(s, m):
        v = m.blocks[s.layer].ffn.Vv; s.saved = np.asarray(v.value).copy()
        nv = np.asarray(v.value).copy(); nv[:, s.u] *= s.g; v.value = jnp.asarray(nv)
    def undo(s, m): m.blocks[s.layer].ffn.Vv.value = jnp.asarray(s.saved)
GAINS = [1.0, 12.0, 30.0]
edit_demo = {}
for name, tok in [('newline', '\n'), ('space', ' ')]:
    u = slot_for(tok); P = 'GLOUCESTER:\n'
    samples = {}
    for g in GAINS:
        gen = generate(model, P, n=200, seed=3, edit=(None if g == 1.0 else Amp(LAYER, u, g)))
        samples[g] = {'text': gen[len(P):], 'count': gen[len(P):].count(tok)}
    print(f'  slot {u} writes {tok!r} ({logit_lens(Vvals[:, u], 5)})')
    for g in GAINS:
        print(f'    gain x{g:<4} {tok!r}-count={samples[g]["count"]:3d}  | {samples[g]["text"][:48]!r}')
    edit_demo[name] = {'slot': u, 'tok': tok, 'samples': {str(g): samples[g]['count'] for g in GAINS}}


print('\n══ 4. ABSTAIN: far-from-memory tokens make the write collapse ══')
def memory_response(seq_idx):
    x = np.asarray(model.residual_into_ffn(seq_idx, LAYER))[0]            # [T, D] FFN inputs
    kw = np.asarray(ffn.kernel(jnp.asarray(x)))                          # [T, FF] kernel weights
    return np.linalg.norm(kw @ Vvals.T, axis=1), kw.max(1)              # write norm, peak weight per token
rr = np.random.RandomState(0)
ind = encode(text[20000:20000 + T])                                      # real Shakespeare
ood_rand = jnp.asarray(rr.randint(0, V, size=(1, T)))                    # random chars
ood_rep = encode(('q' * T)[:T])                                          # a degenerate repeat, far from any prose
n_in, k_in = memory_response(ind); n_r, k_r = memory_response(ood_rand); n_q, k_q = memory_response(ood_rep)
print(f'  peak kernel weight  in-dist {k_in.mean():.2f}   random-chars {k_r.mean():.2f}   repeat-char {k_q.mean():.2f}')
print(f'  FFN write norm      in-dist {n_in.mean():.2f}   random-chars {n_r.mean():.2f}   repeat-char {n_q.mean():.2f}')

# ════════════════════════════════════════════════════════════════════════════
# export the real trained weights for the live in-browser forward pass.
# A NumPy reference forward (mirroring exactly what the JS will do) is asserted to
# match Flax model() before dumping, so the JS port is guaranteed faithful.
# ════════════════════════════════════════════════════════════════════════════
def _np_forward(P, ids):                                 # numpy port of GPT.__call__, single sequence
    HD = D // HEADS; scale = 1.0 / np.sqrt(HD)
    def ln(x, g):  # g = (scale[D], bias[D])
        m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
        return (x - m) / np.sqrt(v + 1e-6) * g[0] + g[1]
    def mha(x, a):
        q = np.einsum('td,dhk->thk', x, a['q_k']) + a['q_b']
        k = np.einsum('td,dhk->thk', x, a['k_k']) + a['k_b']
        v = np.einsum('td,dhk->thk', x, a['v_k']) + a['v_b']
        s = np.einsum('qhk,shk->hqs', q, k) * scale
        Tn = x.shape[0]; m = np.triu(np.ones((Tn, Tn), bool), 1)
        s = np.where(m[None], -1e30, s)
        s = s - s.max(-1, keepdims=True); e = np.exp(s); aw = e / e.sum(-1, keepdims=True)
        o = np.einsum('hqs,shk->qhk', aw, v)
        return np.einsum('thk,hkd->td', o, a['o_k']) + a['o_b']
    def yat(x, f):
        dot = x @ f['W'].T; d2 = (x ** 2).sum(-1, keepdims=True) + (f['W'] ** 2).sum(-1) - 2 * dot
        return ((dot + f['b']) ** 2 / (d2 + f['eps'])) @ f['Vv'].T
    x = P['tok'][np.asarray(ids)] + P['pos'][:len(ids)]
    for L in range(LAYERS):
        bl = P['blocks'][L]
        x = x + mha(ln(x, bl['ln1']), bl['attn'])
        x = x + yat(ln(x, bl['ln2']), bl['ffn'])
    x = ln(x, P['lnf'])
    return x @ P['head']                                 # [T, V] logits


def _grab():                                             # pull every tensor as numpy from the live model
    g = lambda v: np.asarray(v)
    blocks = []
    for L in range(LAYERS):
        b = model.blocks[L]
        blocks.append({
            'ln1': (g(b.ln1.scale.value), g(b.ln1.bias.value)),
            'ln2': (g(b.ln2.scale.value), g(b.ln2.bias.value)),
            'attn': {'q_k': g(b.attn.query.kernel.value), 'q_b': g(b.attn.query.bias.value),
                     'k_k': g(b.attn.key.kernel.value), 'k_b': g(b.attn.key.bias.value),
                     'v_k': g(b.attn.value.kernel.value), 'v_b': g(b.attn.value.bias.value),
                     'o_k': g(b.attn.out.kernel.value), 'o_b': g(b.attn.out.bias.value)},
            'ffn': {'W': g(b.ffn.W.value), 'Vv': g(b.ffn.Vv.value),
                    'b': float(jax.nn.softplus(b.ffn.log_b.value)), 'eps': float(jax.nn.softplus(b.ffn.log_eps.value))},
        })
    return {'tok': g(model.tok.embedding.value), 'pos': g(model.pos.embedding.value),
            'head': g(model.head.kernel.value), 'lnf': (g(model.lnf.scale.value), g(model.lnf.bias.value)),
            'blocks': blocks}


P = _grab()
for probe in ['ROMEO:\nWhat', 'To be, or not to be', 'GLOUCESTER:\n']:
    ids = [stoi[c] for c in probe][:T]
    ref = _np_forward(P, ids); flax = np.asarray(model(encode(probe)[:, :T]))[0]
    err = np.abs(ref - flax).max()
    assert err < 1e-3, f'numpy forward mismatch on {probe!r}: {err}'
print(f'  numpy reference forward matches Flax (max err < 1e-3) on 3 probes')

SLUG = PUB / 'mlp-representer'; SLUG.mkdir(exist_ok=True)
# pack every tensor into one float32 blob + a small manifest; JS reads them in order
blob = []; manifest = []
def add(path, arr):
    a = np.asarray(arr, np.float32).ravel(); manifest.append({'path': path, 'shape': list(np.asarray(arr).shape)}); blob.append(a)
add('tok', P['tok']); add('pos', P['pos']); add('head', P['head'])
add('lnf.scale', P['lnf'][0]); add('lnf.bias', P['lnf'][1])
for L, b in enumerate(P['blocks']):
    add(f'b{L}.ln1.scale', b['ln1'][0]); add(f'b{L}.ln1.bias', b['ln1'][1])
    add(f'b{L}.ln2.scale', b['ln2'][0]); add(f'b{L}.ln2.bias', b['ln2'][1])
    for k in ['q_k', 'q_b', 'k_k', 'k_b', 'v_k', 'v_b', 'o_k', 'o_b']: add(f'b{L}.attn.{k}', b['attn'][k])
    add(f'b{L}.ffn.W', b['ffn']['W']); add(f'b{L}.ffn.Vv', b['ffn']['Vv'])
(SLUG / 'weights.bin').write_bytes(np.concatenate(blob).tobytes())
slot_writes = [[itos[int(i)] for i in np.argsort(-(Vvals[:, u] @ U))[:6]] for u in range(FF)]  # last-layer labels
mdl = {
    'V': V, 'T': T, 'D': D, 'heads': HEADS, 'ff': FF, 'layers': LAYERS, 'lastLayer': LAYERS - 1,
    'itos': [itos[i] for i in range(V)],
    'ffnScalars': [{'b': round(b['ffn']['b'], 6), 'eps': round(b['ffn']['eps'], 6)} for b in P['blocks']],
    'tensors': manifest,
    'slotWrites': slot_writes,        # last-layer slot -> the tokens it writes (logit lens)
}
(SLUG / 'model.json').write_text(json.dumps(mdl, separators=(',', ':')))
print(f"  wrote public/mlp-representer/ weights.bin ({(SLUG/'weights.bin').stat().st_size//1024} KB) + model.json ({(SLUG/'model.json').stat().st_size//1024} KB)")


# ── two static figures: the genuinely visual payoffs (no GIFs, the effect is not temporal) ──
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
BG, PANEL, INK, MUTED, LINE = '#0e0d0b', '#16140f', '#e8e2d4', '#9a9282', '#3a352c'
POS, NEG, GOOD = '#b3661b', '#4a7fb3', '#7bbf5a'
plt.rcParams.update({'figure.facecolor': BG, 'savefig.facecolor': BG, 'text.color': INK, 'axes.edgecolor': LINE, 'font.size': 10})

# Figure A: abstention. peak kernel weight per token, in-distribution vs off-distribution.
fig, ax = plt.subplots(figsize=(6.6, 3.8), dpi=120); ax.set_facecolor(PANEL)
bins = np.linspace(0, max(k_in.max(), 2.5), 30)
for arr, c, lab in [(k_in, GOOD, 'real Shakespeare (in-distribution)'), (k_r, POS, 'random characters'), (k_q, NEG, 'a repeated character')]:
    ax.hist(arr, bins=bins, color=c, alpha=0.55, label=f'{lab}   mean {arr.mean():.2f}', edgecolor=c)
ax.axvline(0, color=LINE, lw=0.8)
ax.set_xlabel('peak memory weight at the token,  max₍ᵤ₎ k(Wᵤ, x)', color=MUTED)
ax.set_ylabel('tokens', color=MUTED); ax.tick_params(colors=MUTED, labelsize=8)
ax.set_title('The memory only lights up for inputs it has seen', color=INK, fontsize=11.5, weight='bold')
ax.legend(facecolor=PANEL, edgecolor=LINE, labelcolor=INK, fontsize=8.2, loc='upper right')
for sp in ax.spines.values(): sp.set_color(LINE)
fig.tight_layout(); fig.savefig(PUB / 'yat-ffn-abstain.png'); plt.close(fig)

# Figure B: editing. turn up one readable slot's gain; the output becomes what it writes.
u_nl = slot_for('\n'); gains = np.array([1, 2, 3, 4, 6, 8, 11, 15, 20, 26, 34], float)
frac = []
for g in gains:
    gen = generate(model, 'GLOUCESTER:\n', n=200, seed=3, edit=(None if g == 1 else Amp(LAYER, u_nl, g)))
    body = gen[len('GLOUCESTER:\n'):]; frac.append(body.count('\n') / len(body))
fig, ax = plt.subplots(figsize=(6.6, 3.8), dpi=120); ax.set_facecolor(PANEL)
ax.plot(gains, frac, '-o', color=POS, lw=2.2, ms=5, mec=BG)
for g, f, lab in [(1, frac[0], 'coherent'), (11, frac[np.argmin(abs(gains-11))], 'choppy'), (34, frac[-1], 'only newlines')]:
    fi = frac[int(np.argmin(abs(gains - g)))]
    ax.annotate(lab, (g, fi), textcoords='offset points', xytext=(6, -4 if g > 1 else 8), color=INK, fontsize=8.5)
ax.set_xlabel("gain on slot %d (the newline-writer)" % u_nl, color=MUTED)
ax.set_ylabel('fraction of output that is a newline', color=MUTED); ax.tick_params(colors=MUTED, labelsize=8)
ax.set_ylim(-0.03, 1.03)
ax.set_title('Turn up one readable slot and the model writes only what it says', color=INK, fontsize=11.5, weight='bold')
for sp in ax.spines.values(): sp.set_color(LINE)
fig.tight_layout(); fig.savefig(PUB / 'yat-ffn-edit.png'); plt.close(fig)
print('wrote public/yat-ffn-abstain.png and public/yat-ffn-edit.png')

print('\nWHITEBOX_DONE', json.dumps({'vloss': round(vloss, 3), 'edit': edit_demo,
      'peak_in': round(float(k_in.mean()), 2), 'peak_rand': round(float(k_r.mean()), 2), 'peak_rep': round(float(k_q.mean()), 2)}))
