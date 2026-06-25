"""GIF for the train-the-features JAX companion: the representation untangling as
the Flax NNX backbone trains. Snapshots the 2D layout of test features and the
constructed-head accuracy at several epochs and animates the classes pulling
apart. Writes public/train-features-untangle.gif.

Run: python scripts/render_train_features_untangle_gif.py
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, jax, jax.numpy as jnp, optax, torchvision
from flax import nnx
from sklearn.cluster import KMeans
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.animation as animation

CLS = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
COL = ['#b3661b', '#4a7fb3', '#3a8f5e', '#9a4f9c', '#c2553a', '#5a7d3a', '#2f8f8f', '#a06a2a', '#7a5fc0', '#c0892a']


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
viz = np.random.RandomState(1).permutation(len(Xte))[:700]


def pca2(F):
    Fc = F - F.mean(0); U, S, Vt = np.linalg.svd(Fc, full_matrices=False); Y = Fc @ Vt[:2].T
    return Y / (np.abs(Y).max() + 1e-9)


def constructed_acc(net):
    Ftr, Fte = feats(net, Xtr), feats(net, Xte)
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6; Ztr, Zte = (Ftr - mu) / sd, (Fte - mu) / sd
    W, vote = [], []
    for c in range(10):
        W += list(KMeans(20, n_init=2, random_state=0).fit(Ztr[ytr == c][:2500]).cluster_centers_); vote += [c] * 20
    W = np.array(W, 'float32'); vote = np.array(vote)
    dot = Zte @ W.T; d2 = (Zte ** 2).sum(1, keepdims=True) + (W ** 2).sum(1) - 2 * dot
    ker = (dot + 0.5) ** 2 / (d2 + np.median(d2) * 0.1)
    score = np.full((len(Zte), 10), -1e9)
    for c in range(10): score[:, c] = ker[:, vote == c].max(1)
    return pca2(feats(net, Xte[viz])), 100 * (score.argmax(1) == yte).mean()


net = Net(nnx.Rngs(0)); opt = nnx.Optimizer(net, optax.adam(1e-3), wrt=nnx.Param)

@nnx.jit
def step(net, opt, xb, yb):
    def loss(net): return optax.softmax_cross_entropy_with_integer_labels(net(xb), yb).mean()
    l, g = nnx.value_and_grad(loss)(net); opt.update(net, g); return l

SNAPS = [0, 1, 2, 4, 6]; frames = []
for ep in range(SNAPS[-1] + 1):
    if ep in SNAPS:
        xy, acc = constructed_acc(net); frames.append((ep, xy, acc)); print(f"  snap epoch {ep}: constructed {acc:.1f}%")
    perm = np.random.RandomState(ep).permutation(len(Xtr))
    for i in range(0, len(Xtr) - 256, 256):
        j = perm[i:i + 256]; step(net, opt, Xtr[j], jnp.asarray(ytr[j]))

plt.rcParams.update({'figure.facecolor': '#0e0d0b', 'savefig.facecolor': '#0e0d0b', 'text.color': '#e8e2d4'})
fig, ax = plt.subplots(figsize=(5.2, 5.0), dpi=128); ax.set_xticks([]); ax.set_yticks([])
lab = yte[viz]


def draw(fi):
    ep, xy, acc = frames[fi % len(frames)]
    ax.clear(); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
    for c in range(10):
        m = lab == c; ax.scatter(xy[m, 0], xy[m, 1], s=10, c=COL[c], alpha=0.8, linewidths=0)
    ax.set_title(f'epoch {ep}: a built head reads these features at {acc:.0f}%\n(no head training; the backbone does the work)',
                 fontsize=11.5, color='#e8e2d4', pad=8)
    for s in ax.spines.values(): s.set_color('#3a352c')
    return []


order = [g for g in range(len(frames)) for _ in range(3)]
ani = animation.FuncAnimation(fig, lambda i: draw(order[i]), frames=len(order), interval=520, blit=False)
fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.03)
ani.save('public/train-features-untangle.gif', writer='pillow', fps=1.9)
import os; print(f"wrote public/train-features-untangle.gif ({os.path.getsize('public/train-features-untangle.gif')//1024} KB)")
