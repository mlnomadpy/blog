"""Three teaching GIFs for the train-the-features JAX/Flax NNX companion.

Each animates a *process*, not a slideshow, and uses the series' physics: a
representation is a warm gas that crystallises into ordered class droplets as the
backbone trains, read the whole time by a head that is never trained.

  1. train-crystallize.gif    the test-feature cloud condenses, frame by frame
                              (Procrustes-tweened between epochs), from a tangle
                              into separated class droplets; a built head reads it.
  2. train-two-heads.gif      two accuracy curves draw themselves: the built head
                              is good immediately, the trained head climbs to catch
                              it, and ends only a point or two ahead.
  3. train-free-lunch.gif     a random, never-trained backbone, yet its features
                              pour through the built head and sort into the right
                              bins at 72.7% where chance is 10%.

Every number is from this run; same pipeline as scripts/jax_train_features.py.
Run: python scripts/render_train_features_gifs.py
"""
import warnings; warnings.filterwarnings('ignore')
import os
from pathlib import Path
import numpy as np, jax, jax.numpy as jnp, optax, torchvision
from flax import nnx
from sklearn.cluster import KMeans
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]; PUB = ROOT / 'public'
BG = '#0e0d0b'; PANEL = '#16140f'; INK = '#e8e2d4'; MUTED = '#9a9282'; LINE = '#3a352c'
CLS = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
COL = ['#b3661b', '#4a7fb3', '#3a8f5e', '#9a4f9c', '#c2553a', '#5a7d3a', '#2f8f8f', '#a06a2a', '#7a5fc0', '#c0892a']
plt.rcParams.update({'figure.facecolor': BG, 'savefig.facecolor': BG, 'text.color': INK,
                     'axes.edgecolor': LINE, 'font.size': 11})


def ease(t): return t * t * (3 - 2 * t)


def fig_rgba(fig):
    fig.canvas.draw(); rgba = np.asarray(fig.canvas.buffer_rgba()).copy(); plt.close(fig); return rgba


def save_gif(path, frames, fps, hold=16):
    Image.fromarray(frames[-1]).save(str(path).replace('.gif', '-preview.png'))
    imageio.mimsave(path, frames + [frames[-1]] * hold, duration=1 / fps, loop=0,
                    palettesize=128, subrectangles=True)
    print(f'wrote {path} ({os.path.getsize(path)//1024} KB)')


# ── model (matches the companion) ──
class Backbone(nnx.Module):
    def __init__(s, rngs):
        s.c1 = nnx.Conv(1, 16, (3, 3), padding='SAME', rngs=rngs); s.c2 = nnx.Conv(16, 32, (3, 3), padding='SAME', rngs=rngs)
        s.lin = nnx.Linear(32 * 7 * 7, 64, rngs=rngs)
    def __call__(s, x):
        x = nnx.max_pool(nnx.relu(s.c1(x)), (2, 2), (2, 2)); x = nnx.max_pool(nnx.relu(s.c2(x)), (2, 2), (2, 2))
        return nnx.relu(s.lin(x.reshape(x.shape[0], -1)))


class Net(nnx.Module):
    def __init__(s, rngs): s.feat = Backbone(rngs); s.head = nnx.Linear(64, 10, rngs=rngs)
    def __call__(s, x): return s.head(s.feat(x))


tr = torchvision.datasets.FashionMNIST('/tmp/fmnist', train=True, download=True)
te = torchvision.datasets.FashionMNIST('/tmp/fmnist', train=False, download=True)
Xtr = jnp.asarray(tr.data.numpy()[:20000], jnp.float32)[..., None] / 255.0; ytr = tr.targets.numpy()[:20000]
Xte = jnp.asarray(te.data.numpy(), jnp.float32)[..., None] / 255.0; yte = te.targets.numpy()
feats = lambda net, X: np.concatenate([np.asarray(net.feat(X[i:i + 2000])) for i in range(0, len(X), 2000)])
VIZ = np.random.RandomState(1).permutation(len(Xte))[:650]
labv = yte[VIZ]


def build_pred(Ftr, Fte):
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6; Ztr, Zte = (Ftr - mu) / sd, (Fte - mu) / sd
    W, vote = [], []
    for c in range(10):
        W += list(KMeans(20, n_init=2, random_state=0).fit(Ztr[ytr == c][:2500]).cluster_centers_); vote += [c] * 20
    W = np.array(W, 'float32'); vote = np.array(vote)
    dot = Zte @ W.T; d2 = (Zte ** 2).sum(1, keepdims=True) + (W ** 2).sum(1) - 2 * dot
    ker = (dot + 0.5) ** 2 / (d2 + np.median(d2) * 0.1)
    score = np.full((len(Zte), 10), -1e9)
    for c in range(10): score[:, c] = ker[:, vote == c].max(1)
    return score.argmax(1)


def pca2(F):
    Fc = F - F.mean(0); U, S, Vt = np.linalg.svd(Fc, full_matrices=False); return Fc @ Vt[:2].T


def procrustes(A, B):
    """Rotate/reflect B onto A (orthogonal Procrustes), return aligned B."""
    U, _, Vt = np.linalg.svd(A.T @ B); R = (U @ Vt).T; return B @ R


# ── train, snapshotting layouts + both accuracies per epoch ──
net = Net(nnx.Rngs(0)); opt = nnx.Optimizer(net, optax.adam(1e-3), wrt=nnx.Param)

@nnx.jit
def step(net, opt, xb, yb):
    def loss(net): return optax.softmax_cross_entropy_with_integer_labels(net(xb), yb).mean()
    l, g = nnx.value_and_grad(loss)(net); opt.update(net, g); return l


def trained_acc():
    pr = np.concatenate([np.asarray(net(Xte[i:i + 2000]).argmax(1)) for i in range(0, len(Xte), 2000)])
    return 100 * (pr == yte).mean()


EPOCHS = 6
layouts, built_acc, train_acc, built_pred_viz = [], [], [], []
rand_pred = None; Fte_final = None
for ep in range(EPOCHS + 1):
    Ftr, Fte = feats(net, Xtr), feats(net, Xte)
    bp = build_pred(Ftr, Fte)
    built_acc.append(100 * (bp == yte).mean()); train_acc.append(trained_acc())
    layouts.append(pca2(Fte[VIZ])); built_pred_viz.append(bp[VIZ].copy())
    if ep == 0:
        rand_pred = bp[VIZ].copy(); rand_feats2 = pca2(Fte[VIZ])
    if ep == EPOCHS: Fte_final = Fte
    print(f'  epoch {ep}: built {built_acc[-1]:.1f}%  trained {train_acc[-1]:.1f}%')
    if ep == EPOCHS: break
    perm = np.random.RandomState(ep).permutation(len(Xtr))
    for i in range(0, len(Xtr) - 256, 256):
        j = perm[i:i + 256]; step(net, opt, Xtr[j], jnp.asarray(ytr[j]))

# a real random (untrained) linear head on the SAME frozen trained features
_Wrand = np.random.RandomState(2).randn(Fte_final.shape[1], 10).astype('float32')
rand_head_acc = 100 * ((Fte_final @ _Wrand).argmax(1) == yte).mean()
print(f'  random linear head on trained features: {rand_head_acc:.1f}%')

# normalise + Procrustes-align every layout to the FINAL one, so tweens are smooth
ref = layouts[-1]; ref = ref / (np.abs(ref).max() + 1e-9)
aligned = []
for L in layouts:
    L = L / (np.abs(L).max() + 1e-9); aligned.append(procrustes(ref, L))
aligned[-1] = ref


# ═══════════════════════════════════════════════════════════════════════════
# GIF 1 — crystallisation: the cloud condenses from gas to class droplets
# ═══════════════════════════════════════════════════════════════════════════
def gif_crystallize():
    TW = 12                                            # tween frames per epoch hop
    frames = []
    for e in range(EPOCHS):
        A, Bnext = aligned[e], aligned[e + 1]
        aA, aB = built_acc[e], built_acc[e + 1]
        for t in range(TW):
            f = ease(t / TW); xy = (1 - f) * A + f * Bnext; acc = (1 - f) * aA + f * aB
            ep = e + f
            fig = plt.figure(figsize=(5.6, 5.7), dpi=104, facecolor=BG)
            fig.text(0.5, 0.95, 'Training the backbone crystallises the features',
                     ha='center', fontsize=13.5, weight='bold')
            fig.text(0.5, 0.905, 'a head that is never trained reads the same cloud the whole way',
                     ha='center', fontsize=9.2, color=MUTED)
            ax = fig.add_axes([0.04, 0.10, 0.74, 0.76]); ax.set_facecolor(PANEL)
            ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_xticks([]); ax.set_yticks([])
            for c in range(10):
                m = labv == c
                ax.scatter(xy[m, 0], xy[m, 1], s=11, c=COL[c], alpha=0.82, linewidths=0)
            for s in ax.spines.values(): s.set_color(LINE)
            ax.set_title(f'epoch {ep:.1f}', fontsize=11, color=MUTED, pad=5)
            # thermometer: the built head's accuracy
            axb = fig.add_axes([0.83, 0.10, 0.05, 0.76]); axb.set_xlim(0, 1); axb.set_ylim(0, 100)
            axb.set_facecolor(PANEL); axb.set_xticks([])
            axb.bar([0.5], [acc], width=1.0, color='#7bbf5a', align='center')
            axb.axhline(10, color=MUTED, lw=0.7, ls=':')
            axb.text(0.5, acc + 3, f'{acc:.0f}%', ha='center', fontsize=10, color='#7bbf5a', weight='bold')
            axb.set_yticks([10, 50, 90]); axb.tick_params(colors=MUTED, labelsize=7)
            axb.set_title('built\nhead', fontsize=8, color=MUTED, pad=3)
            for s in axb.spines.values(): s.set_color(LINE)
            frames.append(fig_rgba(fig))
    save_gif(PUB / 'train-crystallize.gif', frames, fps=13)


# ═══════════════════════════════════════════════════════════════════════════
# GIF 2 — two heads racing: built head good at once, trained head climbs to it
# ═══════════════════════════════════════════════════════════════════════════
def gif_two_heads():
    xs = np.arange(EPOCHS + 1)
    ba, ta = np.array(built_acc), np.array(train_acc)
    # dense interpolation for a smooth self-drawing curve
    dense = np.linspace(0, EPOCHS, 120)
    bd = np.interp(dense, xs, ba); td = np.interp(dense, xs, ta)
    frames = []
    NF = 84
    for fi in range(NF):
        k = int(ease(fi / (NF - 1)) * len(dense))
        k = max(2, k)
        fig = plt.figure(figsize=(6.6, 5.0), dpi=120, facecolor=BG)
        fig.text(0.5, 0.95, 'Two heads, read off the same backbone as it trains',
                 ha='center', fontsize=13.5, weight='bold')
        fig.text(0.5, 0.905, 'the built head is good immediately; training a head only ever buys a point or two more',
                 ha='center', fontsize=9.0, color=MUTED)
        ax = fig.add_axes([0.11, 0.12, 0.84, 0.74]); ax.set_facecolor(PANEL)
        ax.set_xlim(-0.15, EPOCHS + 0.15); ax.set_ylim(0, 100)
        ax.set_xlabel('epoch (backbone training)', color=MUTED, fontsize=10)
        ax.set_ylabel('test accuracy', color=MUTED, fontsize=10)
        ax.tick_params(colors=MUTED, labelsize=9)
        for s in ax.spines.values(): s.set_color(LINE)
        ax.axhline(10, color=MUTED, lw=0.7, ls=':'); ax.text(EPOCHS, 12, 'chance', color=MUTED, fontsize=8, ha='right')
        dd = dense[:k]
        # gap shading
        ax.fill_between(dd, bd[:k], td[:k], color='#7bbf5a', alpha=0.10)
        ax.plot(dd, td[:k], color='#c77d2a', lw=2.4, label='trained head')
        ax.plot(dd, bd[:k], color='#7bbf5a', lw=2.4, label='built head (never trained)')
        ax.scatter([dd[-1]], [td[k - 1]], s=44, color='#c77d2a', zorder=5, edgecolors=BG)
        ax.scatter([dd[-1]], [bd[k - 1]], s=44, color='#7bbf5a', zorder=5, edgecolors=BG)
        ax.text(dd[-1] + 0.04, td[k - 1], f'{td[k-1]:.0f}', color='#c77d2a', fontsize=9, va='center')
        ax.text(dd[-1] + 0.04, bd[k - 1] - 4, f'{bd[k-1]:.0f}', color='#7bbf5a', fontsize=9, va='center')
        ax.legend(loc='lower right', frameon=True, fontsize=9.5, facecolor=PANEL, edgecolor=LINE, labelcolor=INK)
        frames.append(fig_rgba(fig))
    save_gif(PUB / 'train-two-heads.gif', frames, fps=18, hold=20)


# ═══════════════════════════════════════════════════════════════════════════
# GIF 3 — the free lunch: random backbone features still sort into the right bins
# ═══════════════════════════════════════════════════════════════════════════
def gif_free_lunch():
    N = 240
    sel = np.random.RandomState(3).permutation(len(VIZ))[:N]
    true = labv[sel]; built = rand_pred[sel]
    chance = np.random.RandomState(5).randint(0, 10, N)   # the naive expectation
    # display the real full-test headline figures; the dots are an illustrative sample
    builtacc = built_acc[0]; chanceacc = 10.0
    # bin x-centres along the bottom; particles fall from a hopper into their predicted bin
    binx = np.linspace(0.06, 0.94, 10)
    order = np.argsort(true)                               # release roughly by class for a clean stream
    sel = sel[order]; true = true[order]; built = built[order]; chance = chance[order]

    def stack_targets(predbins):
        cnt = np.zeros(10, int); tx = np.zeros(N); ty = np.zeros(N)
        for i in range(N):
            b = predbins[i]; tx[i] = binx[b] + (np.random.RandomState(i).rand() - 0.5) * 0.06
            ty[i] = 0.05 + 0.011 * cnt[b]; cnt[b] += 1
        return tx, ty

    np.random.seed(0)
    txB, tyB = stack_targets(built); txC, tyC = stack_targets(chance)
    sx = np.random.RandomState(7).rand(N) * 0.88 + 0.06    # hopper spread at top

    NF = 88
    frames = []
    for fi in range(NF):
        prog = ease(fi / (NF - 1))
        nrel = int(prog * N * 1.25)
        fig = plt.figure(figsize=(8.8, 4.8), dpi=110, facecolor=BG)
        fig.text(0.5, 0.95, 'A backbone that never saw a gradient, and its features still sort',
                 ha='center', fontsize=13.5, weight='bold')
        fig.text(0.5, 0.905, 'the same random features poured through two heads: a coin (left) vs the built Yat head (right)',
                 ha='center', fontsize=9.0, color=MUTED)
        for side, (tx, ty, acc, ttl, who) in enumerate([
                (txC, tyC, chanceacc, 'a coin flip (what you’d expect)', chance),
                (txB, tyB, builtacc, 'the built head (the free lunch)', built)]):
            ax = fig.add_axes([0.04 + side * 0.50, 0.07, 0.44, 0.76]); ax.set_facecolor(PANEL)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_color(LINE)
            # bin floor labels
            for b in range(10):
                ax.add_patch(plt.Rectangle((binx[b] - 0.043, 0), 0.086, 0.035, color=COL[b], alpha=0.5))
            # falling / landed particles
            for i in range(min(nrel, N)):
                local = ease(np.clip((nrel - i) / 22, 0, 1))
                cx = sx[i] + (tx[i] - sx[i]) * local
                cy = 0.95 + (ty[i] - 0.95) * local
                correct = who[i] == true[i]
                ax.scatter([cx], [cy], s=18, color=COL[true[i]],
                           edgecolors='#7bbf5a' if correct and local > 0.9 else 'none',
                           linewidths=0.8, alpha=0.9, zorder=3)
            ax.set_title(ttl, fontsize=10, color=MUTED, pad=4)
            shown_acc = acc * ease(np.clip(nrel / N, 0, 1))
            ax.text(0.5, 0.92, f'{shown_acc:.0f}% in the right bin', ha='center', fontsize=12,
                    color='#7bbf5a' if side == 1 else MUTED, weight='bold')
        frames.append(fig_rgba(fig))
    save_gif(PUB / 'train-free-lunch.gif', frames, fps=16)


# ═══════════════════════════════════════════════════════════════════════════
# GIF 4 — the head is furniture: three heads on the SAME frozen trained features
# ═══════════════════════════════════════════════════════════════════════════
def gif_head_swap():
    heads = [('random linear head\n(untrained)', rand_head_acc, '#c2553a'),
             ('built Yat head\n(placed, not trained)', built_acc[-1], '#7bbf5a'),
             ('trained linear head\n(gradient-trained)', train_acc[-1], '#c77d2a')]
    HOLD, SWAP = 16, 12
    frames = []
    order = [0, 1, 2]
    cur = 0.0
    seq = []                                            # (from_idx, to_idx, t)
    for k in range(len(order)):
        for _ in range(HOLD): seq.append((order[k], order[k], 1.0))
        if k < len(order) - 1:
            for s in range(SWAP): seq.append((order[k], order[k + 1], ease((s + 1) / SWAP)))
    for (fi, (a, b, t)) in enumerate(seq):
        acc = heads[a][1] * (1 - t) + heads[b][1] * t
        lab = heads[b][0] if t > 0.5 else heads[a][0]
        col = heads[b][2] if t > 0.5 else heads[a][2]
        fig = plt.figure(figsize=(6.8, 5.0), dpi=120, facecolor=BG)
        fig.text(0.5, 0.95, 'The classifier is furniture you place', ha='center', fontsize=14, weight='bold')
        fig.text(0.5, 0.905, 'one frozen backbone, three heads bolted on: the features carry the accuracy, the head barely moves it',
                 ha='center', fontsize=8.8, color=MUTED)
        ax = fig.add_axes([0.13, 0.13, 0.82, 0.72]); ax.set_facecolor(PANEL)
        ax.set_xlim(0, 1); ax.set_ylim(0, 100); ax.set_xticks([])
        ax.set_ylabel('test accuracy', color=MUTED, fontsize=10); ax.tick_params(colors=MUTED, labelsize=9)
        for s in ax.spines.values(): s.set_color(LINE)
        ax.axhline(10, color=MUTED, lw=0.7, ls=':'); ax.text(0.98, 12, 'chance', color=MUTED, fontsize=8, ha='right')
        ax.bar([0.5], [acc], width=0.42, color=col, alpha=0.92)
        ax.text(0.5, acc + 3, f'{acc:.1f}%', ha='center', fontsize=15, color=col, weight='bold')
        ax.text(0.5, -8, lab, ha='center', fontsize=11, color=INK, clip_on=False)
        # small dots marking the three resting values for context
        for hi, (_, hv, hc) in enumerate(heads):
            ax.scatter([0.08 + hi * 0.02], [hv], s=24, color=hc, alpha=0.5)
        frames.append(fig_rgba(fig))
    save_gif(PUB / 'train-head-swap.gif', frames, fps=14, hold=18)


# ═══════════════════════════════════════════════════════════════════════════
# GIF 5 — errors melting: each test point colored by the REAL built-head verdict
#          as the backbone trains; red turns green, the head never touched
# ═══════════════════════════════════════════════════════════════════════════
def gif_errors_melt():
    TW = 12
    frames = []
    for e in range(EPOCHS):
        A, Bnext = aligned[e], aligned[e + 1]
        predA, predB = built_pred_viz[e], built_pred_viz[e + 1]
        nerrA = int((predA != labv).sum()); nerrB = int((predB != labv).sum())
        for t in range(TW):
            f = ease(t / TW); xy = (1 - f) * A + f * Bnext
            pred = predA if f < 0.5 else predB                 # snap colour at the midpoint
            nerr = nerrA if f < 0.5 else nerrB
            ep = e + f
            fig = plt.figure(figsize=(5.8, 5.9), dpi=106, facecolor=BG)
            fig.text(0.5, 0.95, 'Training the backbone melts the head’s mistakes', ha='center', fontsize=13.5, weight='bold')
            fig.text(0.5, 0.905, 'colour is the never-trained built head’s real verdict on each test image; red is a miss',
                     ha='center', fontsize=8.6, color=MUTED)
            ax = fig.add_axes([0.04, 0.06, 0.92, 0.80]); ax.set_facecolor(PANEL)
            ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_xticks([]); ax.set_yticks([])
            ok = pred == labv
            ax.scatter(xy[ok, 0], xy[ok, 1], s=12, color='#5a9e54', alpha=0.85, linewidths=0, zorder=2)
            ax.scatter(xy[~ok, 0], xy[~ok, 1], s=20, color='#d0563a', alpha=0.95,
                       linewidths=0.5, edgecolors='white', zorder=3)
            ax.set_title(f'epoch {ep:.1f}   misses: {nerr}/{len(labv)}', fontsize=11, color=INK, pad=6)
            for s in ax.spines.values(): s.set_color(LINE)
            frames.append(fig_rgba(fig))
    save_gif(PUB / 'train-errors-melt.gif', frames, fps=13)


if __name__ == '__main__':
    gif_crystallize()
    gif_two_heads()
    gif_free_lunch()
    gif_head_swap()
    gif_errors_melt()
    print(f'BUILT {[round(x,1) for x in built_acc]}')
    print(f'TRAINED {[round(x,1) for x in train_acc]}')
    print(f'RANDHEAD {round(rand_head_acc,1)}')
    print('TRAIN_GIFS_DONE')
