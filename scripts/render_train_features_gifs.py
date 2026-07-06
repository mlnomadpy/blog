"""Figures for the train-the-features JAX/Flax NNX companion.

Two of these are GIFs because motion carries information no single frame gives:
a real 2D feature cloud reorganising over real training epochs, read the whole
time by a head that is never trained. The rest are static figures from the same
real run, because their content is a result, not a process.

  GIFs (a real temporal process):
    train-crystallize.gif   the test-feature cloud condenses over training epochs
                            (Procrustes-tweened between them), from a tangle into
                            separated class droplets; a built head reads it.
    train-errors-melt.gif   the same cloud, each point coloured by the never-trained
                            built head's real verdict; misses melt as the backbone
                            sharpens (206 -> ~90).

  Static PNGs (a result, one real run):
    train-two-heads.png     both accuracy curves, built vs trained, across epochs.
    train-head-swap.png     three heads on one frozen backbone, side by side.
    train-free-lunch.png    random-backbone features sorted by a coin vs the built
                            head into ten class bins (10% vs 72.7%).

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
    imageio.mimsave(path, frames + [frames[-1]] * hold, duration=1 / fps, loop=0,
                    palettesize=128, subrectangles=True)
    print(f'wrote {path} ({os.path.getsize(path)//1024} KB)')


def save_png(path, fig):
    fig.savefig(path, dpi=fig.dpi, facecolor=BG)
    plt.close(fig)
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
# FIG 2 (static) — two heads, read off the same backbone across epochs. Both
#   curves at once: the built head starts high, the trained head climbs to it.
# ═══════════════════════════════════════════════════════════════════════════
def png_two_heads():
    xs = np.arange(EPOCHS + 1)
    ba, ta = np.array(built_acc), np.array(train_acc)
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
    ax.fill_between(xs, ba, ta, color='#7bbf5a', alpha=0.10)
    ax.plot(xs, ta, color='#c77d2a', lw=2.4, marker='o', ms=5, label='trained head')
    ax.plot(xs, ba, color='#7bbf5a', lw=2.4, marker='o', ms=5, label='built head (never trained)')
    ax.text(xs[-1] + 0.04, ta[-1], f'{ta[-1]:.0f}', color='#c77d2a', fontsize=9, va='center')
    ax.text(xs[-1] + 0.04, ba[-1] - 4, f'{ba[-1]:.0f}', color='#7bbf5a', fontsize=9, va='center')
    ax.text(0.04, ba[0] + 3, f'{ba[0]:.0f}% before any training', color='#7bbf5a', fontsize=8.5)
    ax.legend(loc='lower right', frameon=True, fontsize=9.5, facecolor=PANEL, edgecolor=LINE, labelcolor=INK)
    save_png(PUB / 'train-two-heads.png', fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIG 3 (static) — the free lunch: the same random-backbone features sorted into
#   ten class bins by a coin flip (left) vs the built Yat head (right). Each dot is
#   coloured by its true class, so a dot in a same-coloured column is a hit. The
#   headline % are the real full-test-set numbers.
# ═══════════════════════════════════════════════════════════════════════════
def png_free_lunch():
    N = 260
    sel = np.random.RandomState(3).permutation(len(VIZ))[:N]
    true = labv[sel]; built = rand_pred[sel]
    chance = np.random.RandomState(5).randint(0, 10, N)   # the naive expectation
    builtacc = built_acc[0]; chanceacc = 10.0             # real full-test-set figures
    binx = np.linspace(0.06, 0.94, 10)
    order = np.argsort(true)
    true = true[order]; built = built[order]; chance = chance[order]

    def stack(predbins):
        cnt = np.zeros(10, int); tx = np.zeros(N); ty = np.zeros(N)
        for i in range(N):
            b = predbins[i]; tx[i] = binx[b] + (np.random.RandomState(i).rand() - 0.5) * 0.055
            ty[i] = 0.07 + 0.0135 * cnt[b]; cnt[b] += 1
        return tx, ty

    txB, tyB = stack(built); txC, tyC = stack(chance)
    fig = plt.figure(figsize=(8.8, 4.8), dpi=110, facecolor=BG)
    fig.text(0.5, 0.95, 'A backbone that never saw a gradient, and its features still sort',
             ha='center', fontsize=13.5, weight='bold')
    fig.text(0.5, 0.905, 'the same random features sorted into ten class bins: a coin (left) vs the built Yat head (right)',
             ha='center', fontsize=9.0, color=MUTED)
    for side, (tx, ty, acc, ttl, who) in enumerate([
            (txC, tyC, chanceacc, 'a coin flip (what you’d expect)', chance),
            (txB, tyB, builtacc, 'the built head (the free lunch)', built)]):
        ax = fig.add_axes([0.04 + side * 0.50, 0.07, 0.44, 0.76]); ax.set_facecolor(PANEL)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_color(LINE)
        for b in range(10):
            ax.add_patch(plt.Rectangle((binx[b] - 0.043, 0), 0.086, 0.045, color=COL[b], alpha=0.5))
        for i in range(N):
            correct = who[i] == true[i]
            ax.scatter([tx[i]], [ty[i]], s=20, color=COL[true[i]],
                       edgecolors='#7bbf5a' if correct else 'none',
                       linewidths=0.9, alpha=0.9, zorder=3)
        ax.set_title(ttl, fontsize=10, color=MUTED, pad=4)
        ax.text(0.5, 0.93, f'{acc:.0f}% in the right bin', ha='center', fontsize=12,
                color='#7bbf5a' if side == 1 else MUTED, weight='bold')
    save_png(PUB / 'train-free-lunch.png', fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIG 4 (static) — the head is furniture: three heads on the SAME frozen trained
#   backbone, side by side. The features carry the accuracy; the head barely moves it.
# ═══════════════════════════════════════════════════════════════════════════
def png_head_swap():
    heads = [('random linear head\n(untrained)', rand_head_acc, '#c2553a'),
             ('built Yat head\n(placed, not trained)', built_acc[-1], '#7bbf5a'),
             ('trained linear head\n(gradient-trained)', train_acc[-1], '#c77d2a')]
    fig = plt.figure(figsize=(6.8, 5.0), dpi=120, facecolor=BG)
    fig.text(0.5, 0.95, 'The classifier is furniture you place', ha='center', fontsize=14, weight='bold')
    fig.text(0.5, 0.905, 'one frozen backbone, three heads bolted on: the features carry the accuracy, the head barely moves it',
             ha='center', fontsize=8.8, color=MUTED)
    ax = fig.add_axes([0.13, 0.18, 0.82, 0.67]); ax.set_facecolor(PANEL)
    xs = np.arange(3)
    ax.set_xlim(-0.6, 2.6); ax.set_ylim(0, 100); ax.set_xticks([])
    ax.set_ylabel('test accuracy', color=MUTED, fontsize=10); ax.tick_params(colors=MUTED, labelsize=9)
    for s in ax.spines.values(): s.set_color(LINE)
    ax.axhline(10, color=MUTED, lw=0.7, ls=':'); ax.text(2.55, 12, 'chance', color=MUTED, fontsize=8, ha='right')
    for i, (lab, acc, col) in enumerate(heads):
        ax.bar([i], [acc], width=0.62, color=col, alpha=0.92)
        ax.text(i, acc + 2.5, f'{acc:.1f}%', ha='center', fontsize=13.5, color=col, weight='bold')
        ax.text(i, -9, lab, ha='center', fontsize=9.5, color=INK, clip_on=False)
    save_png(PUB / 'train-head-swap.png', fig)


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
    gif_crystallize()      # KEEP: real cloud reorganising over training epochs
    gif_errors_melt()      # KEEP: real per-point verdicts over training epochs
    png_two_heads()        # static: a result (both curves), not a process
    png_free_lunch()       # static: a result (final sort into bins), not a process
    png_head_swap()        # static: three fixed values, side by side
    print(f'BUILT {[round(x,1) for x in built_acc]}')
    print(f'TRAINED {[round(x,1) for x in train_acc]}')
    print(f'RANDHEAD {round(rand_head_acc,1)}')
    print('TRAIN_GIFS_DONE')
