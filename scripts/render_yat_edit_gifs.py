"""Physics-flavoured GIFs for "Editing a Network by Hand, in JAX/Flax NNX".

The kernel's denominator 1/(r^2+eps) is a softened inverse-square law, so the
prototypes are masses and the input space is a potential landscape. We render
that landscape on real Fashion-MNIST, laid out by UMAP, and let the SOFT field
(sum of per-class 1/(r^2+eps) pulls) morph as masses fade in (teaching) and out
(forgetting). Because fields superpose, masses fade independently and the basins
reshape smoothly.

  edit-teach.gif   - classes taught one at a time; each drops its masses and a
                     new basin grows into the map. Title shows real accuracy.
  edit-forget.gif  - one class's masses fade to zero; its basin collapses and the
                     neighbours flood in to reclaim the ground, the rest frozen.
  edit-settle.gif  - test points released as particles, falling into the basin of
                     the class the real network assigns them: classification as
                     gravitational settling.

Run: python scripts/render_yat_edit_gifs.py   (writes public/edit-*.gif)
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, torchvision, umap
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.cluster import KMeans

ACCENT = '#c2553a'

CLS = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
COL = ['#b3661b', '#4a7fb3', '#3a8f5e', '#9a4f9c', '#c2553a', '#5a7d3a', '#2f8f8f', '#a06a2a', '#7a5fc0', '#c0892a']
B, EPS, PER, PAPER, FG, MUT = 0.5, 0.05, 20, '#fbf8f1', '#1a1a1a', '#6b6b6b'
EPS2, GRID = 0.05, 128
hexrgb = lambda h: np.array([int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)]) / 255
CRGB = np.array([hexrgb(c) for c in COL]); BGR = hexrgb(PAPER)

tr = torchvision.datasets.FashionMNIST('/tmp/fmnist', train=True, download=True)
te = torchvision.datasets.FashionMNIST('/tmp/fmnist', train=False, download=True)
X = tr.data.numpy().reshape(-1, 784).astype('float32') / 255.0; y = tr.targets.numpy()
Xte = te.data.numpy().reshape(-1, 784).astype('float32') / 255.0; yte = te.targets.numpy()

# build the prototype bank (the masses) and the real (784-D) nearest-prototype labels
protos = np.stack([KMeans(PER, n_init=3, random_state=0).fit(X[y == c][:2500]).cluster_centers_ for c in range(10)])
flat = protos.reshape(200, 784); vote = np.repeat(np.arange(10), PER)
onehot = np.eye(10)[vote]                                          # [200,10]


def predict(Xq, enabled):                                         # real nearest-prototype vote
    dot = Xq @ flat.T
    ker = (dot + B) ** 2 / ((Xq ** 2).sum(1, keepdims=True) + (flat ** 2).sum(1) - 2 * dot + EPS)
    sc = np.where(enabled[vote], ker, -1e9)
    cls = np.full((len(Xq), 10), -1e9)
    for c in range(10): cls[:, c] = sc[:, vote == c].max(1) if enabled[c] else -1e9
    return cls.argmax(1)


# UMAP layout of a test sample, prototypes transformed into it, then standardised
print('fitting UMAP...')
samp = np.random.RandomState(0).permutation(len(Xte))[:4000]
reducer = umap.UMAP(n_neighbors=15, min_dist=0.25, random_state=42).fit(Xte[samp])
emb = reducer.embedding_; lab = yte[samp]
P2 = reducer.transform(flat)
mu, sd = emb.mean(0), emb.std(0)
emb = (emb - mu) / sd; P2 = (P2 - mu) / sd
lo, hi = emb.min(0) - 0.6, emb.max(0) + 0.6
gx = np.linspace(lo[0], hi[0], GRID); gy = np.linspace(lo[1], hi[1], GRID)
GXX, GYY = np.meshgrid(gx, gy)
Gpts = np.stack([GXX.ravel(), GYY.ravel()], 1)                    # [G,2]
D2 = ((Gpts[:, None, :] - P2[None, :, :]) ** 2).sum(2)            # [G,200] grid-to-mass dist^2


def field_rgb(mass):                                              # mass:[200] in [0,1]; soft superposed field
    act = mass[None, :] / (D2 + EPS2)                             # [G,200] inverse-square pulls
    Fc = act @ onehot                                            # [G,10] per-class field (superposition)
    win = Fc.argmax(1); fmax = Fc.max(1)
    bright = np.clip(np.sqrt(fmax / (fmax + np.quantile(fmax, 0.6) + 1e-9)), 0, 1)
    t = (0.1 + 0.8 * bright)[:, None]
    rgb = BGR[None, :] * (1 - t) + CRGB[win] * t
    rgb = np.where((mass[None, :].sum() > 0), rgb, BGR[None, :])
    return rgb.reshape(GRID, GRID, 3)


def base_ax():
    fig = plt.figure(figsize=(5.6, 4.7), dpi=78); fig.patch.set_facecolor(PAPER)
    ax = fig.add_axes([0.02, 0.03, 0.74, 0.88]); ax.set_facecolor(PAPER)          # the map
    lax = fig.add_axes([0.78, 0.03, 0.21, 0.88]); lax.set_facecolor(PAPER)        # vertical legend
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    for s in ax.spines.values(): s.set_color('#d8d2c4')
    lax.axis('off')
    return fig, ax, lax


def draw_legend(lax, active, hi_c=None):
    lax.clear(); lax.axis('off'); lax.set_xlim(0, 1); lax.set_ylim(0, 10.4)
    lax.text(0.04, 10.1, 'classes', color=MUT, fontsize=8.5, va='center')
    for i in range(10):
        yy = 9.1 - i * 0.95; on = active[i]; a = 1.0 if on else 0.26
        lax.add_patch(Rectangle((0.05, yy - 0.17), 0.17, 0.34, color=COL[i], alpha=a))
        lax.text(0.30, yy, CLS[i], color=(FG if on else MUT), fontsize=8, va='center', alpha=a)
        if hi_c == i: lax.add_patch(Rectangle((0.01, yy - 0.40), 0.98, 0.80, fill=False, edgecolor=ACCENT, lw=1.4))


def draw_field(ax, lax, mass, title, active, hi_c=None, particles=None):
    ax.clear(); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.imshow(field_rgb(mass), extent=[lo[0], hi[0], lo[1], hi[1]], origin='lower', aspect='auto', interpolation='bilinear')
    ax.scatter(emb[:, 0], emb[:, 1], s=2, c=[COL[c] for c in lab], alpha=0.16, linewidths=0)
    on = mass > 0.5
    ax.scatter(P2[on, 0], P2[on, 1], s=44, c=[COL[c] for c in vote[on]], edgecolors='white', linewidths=0.8, zorder=5)
    off = (mass <= 0.5)
    if off.any():
        ax.scatter(P2[off, 0], P2[off, 1], s=38, facecolors='none', edgecolors=MUT, linewidths=1.0, marker='x', zorder=5)
    if particles is not None:
        ax.scatter(particles[:, 0], particles[:, 1], s=8, c=[COL[c] for c in particles[:, 2].astype(int)], alpha=0.9, edgecolors='white', linewidths=0.25, zorder=6)
    ax.set_title(title, color=FG, fontsize=12.5, pad=9)
    draw_legend(lax, active, hi_c)


# ── GIF 1: teaching, masses fade in one class at a time ──
order = [0, 1, 5, 7, 2, 4, 6, 3, 8, 9]
fig, ax, lax = base_ax(); seq = []
mass = np.zeros(200)
seq += [(mass.copy(), 0, None)] * 3
for k, c in enumerate(order):
    for t in np.linspace(0.34, 1, 3):                               # fade the new class's masses in
        m = mass.copy(); m[vote == c] = t; seq.append((m, k + 1, c))
    mass[vote == c] = 1
    seq += [(mass.copy(), k + 1, None)] * 3                         # pause so the basin registers
seq += [(mass.copy(), 10, None)] * 7

def upd1(s):
    m, k, cur = s; en = np.zeros(10, bool); [en.__setitem__(order[i], True) for i in range(k)]
    acc = (predict(Xte, en)[np.isin(yte, np.where(en)[0])] == yte[np.isin(yte, np.where(en)[0])]).mean() * 100 if k else 0
    draw_field(ax, lax, m, f'teaching, no training   ·   {k}/10 classes   ·   {acc:.0f}%', en, hi_c=cur)
    return []

print('rendering teach...'); FuncAnimation(fig, upd1, frames=seq, interval=130).save('public/edit-teach.gif', writer=PillowWriter(fps=7.5))
plt.close(fig); print('wrote public/edit-teach.gif')

# ── GIF 2: forgetting Sandal, its masses fade to zero, neighbours reclaim ──
fig, ax, lax = base_ax(); seq = []; full = np.ones(200)
seq += [(full.copy(), 'all')] * 8
for t in np.linspace(1, 0, 14):                                    # fade Sandal's masses out
    m = full.copy(); m[vote == 5] = t; seq.append((m, 'del'))
seq += [(seq[-1][0].copy(), 'del')] * 10

def upd2(s):
    m, tag = s; en = np.ones(10, bool); hi = None
    if tag == 'del' and (m[vote == 5] < 0.5).all():
        en[5] = False
        oth = [c for c in range(10) if c != 5]
        pr = predict(Xte, en); rec = np.mean([(pr[yte == c] == c).mean() for c in oth]) * 100
        draw_field(ax, lax, m, f'forgetting Sandal   ·   recall 0%   ·   others {rec:.0f}% (unchanged)', en)
    else:
        if tag == 'del': hi = 5
        pr = predict(Xte, en); rec = np.mean([(pr[yte == c] == c).mean() for c in range(10) if c != 5]) * 100
        draw_field(ax, lax, m, f'all 10 present   ·   others {rec:.0f}%', en, hi_c=hi)
    return []

print('rendering forget...'); FuncAnimation(fig, upd2, frames=seq, interval=130).save('public/edit-forget.gif', writer=PillowWriter(fps=7))
plt.close(fig); print('wrote public/edit-forget.gif')

# ── GIF 3: classification as settling — particles fall into their basin ──
fig, ax, lax = base_ax()
rng = np.random.RandomState(7)
NP = 430; pick = rng.permutation(len(emb))[:NP]
pcls = predict(Xte[samp[pick]], np.ones(10, bool))               # real predicted class
# each particle's target = nearest prototype of its predicted class, in 2D
targets = np.zeros((NP, 2))
for i in range(NP):
    cand = np.where(vote == pcls[i])[0]
    j = cand[np.argmin(((P2[cand] - emb[pick[i]]) ** 2).sum(1))]
    targets[i] = P2[j]
start = rng.uniform(lo, hi, (NP, 2))
pos = start.copy(); vel = np.zeros((NP, 2)); STEPS = 40
states = []
for f in range(STEPS):
    vel = (vel + (targets - pos) * 0.035) * 0.86; pos = pos + vel
    states.append(np.column_stack([pos.copy(), pcls]))
states = [states[0]] * 5 + states + [states[-1]] * 12          # pause at release and once settled

ALLON = np.ones(10, bool)
def upd3(p):
    draw_field(ax, lax, np.ones(200), 'classification is falling into a basin', ALLON, particles=p)
    return []

print('rendering settle...'); FuncAnimation(fig, upd3, frames=states, interval=110).save('public/edit-settle.gif', writer=PillowWriter(fps=11))
plt.close(fig); print('wrote public/edit-settle.gif')
