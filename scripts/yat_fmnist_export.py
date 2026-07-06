"""Provenance for the "Your Neuron Is a Picture" pair (C1).

Every headline number in `your-neuron-is-a-picture.mdx` and its JAX companion
`yat-mlp-fmnist-jax-flax-nnx.mdx` is printed by one of the sections below.

A) The shipped interactive model. The explainer's in-browser panels run the
   trained weights in `public/yat-fmnist/model.json` (K=50, prototypes seeded
   from training images). This section loads those weights and evaluates them
   on the full Fashion-MNIST test set, recomputes the Fashion-vs-MNIST
   max-match medians, and measures what the shipped abstention threshold tau
   (`oodThreshold` in `public/yat-fmnist/meta.json`) actually separates.
B) The companion's traced run. The exact inline config of the companion
   (K=48, 20k-image subset, 12 epochs, adam 4e-3) for the Yat-MLP, the
   same-shaped ReLU MLP, and the noise-seeded Yat-MLP, evaluated the way the
   companion evaluates them (test[:5000]), plus the traced model's
   Fashion-vs-MNIST max-match medians. Also a K=50 ReLU baseline under the
   same config evaluated on the full test set, to pair with section A.
C) The hand-built classifier: 20 k-means centroids per class, one-hot readout,
   zero gradient steps; accuracy under the linear sum-vote readout and under
   the nearest-prototype (per-class max) readout, in both the full-train
   config (the one behind `public/yat-fmnist/notrain.json` and the
   NoTrainClassifier panel, matching `scripts/yat_editable_fmnist.py`) and
   the companion's 20k-subset config.
D) Warm start, low data: 40 images per class; hand-built zero-shot score, then
   fine-tune from the hand-built start vs from a random start.
E) Warm start, imbalance: Pullover and Sneaker starved to 30 training images
   each; warm start (k-means centroids per class + one-hot readout) vs random
   start, mean over five seeds; rare-class recall and overall accuracy.

Run: python3 scripts/yat_fmnist_export.py
"""
import warnings; warnings.filterwarnings('ignore')
import json
from pathlib import Path

import numpy as np
import torchvision
from sklearn.cluster import KMeans
import jax, jax.numpy as jnp
import optax
from flax import nnx

ROOT = Path(__file__).resolve().parents[1]
META = json.load(open(ROOT / 'public/yat-fmnist/meta.json'))
MODEL = json.load(open(ROOT / 'public/yat-fmnist/model.json'))
CLS = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']

tr = torchvision.datasets.FashionMNIST('/tmp/fmnist', train=True, download=True)
te = torchvision.datasets.FashionMNIST('/tmp/fmnist', train=False, download=True)
X = tr.data.numpy().reshape(-1, 784).astype('float32') / 255.0; y = tr.targets.numpy()
Xte = te.data.numpy().reshape(-1, 784).astype('float32') / 255.0; yte = te.targets.numpy()
mnist = torchvision.datasets.MNIST('/tmp/mnist', train=True, download=True)
Xood = mnist.data.numpy().reshape(-1, 784).astype('float32') / 255.0


def yat_kernel(W, Xq, b, eps):
    dot = Xq @ W.T
    d2 = (Xq ** 2).sum(1, keepdims=True) + (W ** 2).sum(1) - 2 * dot
    return (dot + b) ** 2 / (d2 + eps)


# --- A) the shipped interactive model -------------------------------------
print('=== A) shipped interactive model (public/yat-fmnist/model.json) ===')
Wm = np.array(MODEL['W'], 'float32'); Wout = np.array(MODEL['Wout'], 'float32')
bout = np.array(MODEL['bout'], 'float32'); bm, epsm = MODEL['b'], MODEL['eps']
logits = yat_kernel(Wm, Xte, bm, epsm) @ Wout + bout
acc = (logits.argmax(1) == yte).mean() * 100
print(f"  Yat-MLP (K=50) on the full test set: {acc:.1f}%   (meta.json says {META['yatAcc']})")
fash_max = yat_kernel(Wm, Xte, bm, epsm).max(1)
ood_max = yat_kernel(Wm, Xood[:2000], bm, epsm).max(1)
tau = META['oodThreshold']
print(f"  max-match median: Fashion {np.median(fash_max):.1f}, MNIST {np.median(ood_max):.1f}")
print(f"  tau = {tau} (meta.json oodThreshold): Fashion above tau {100*(fash_max>=tau).mean():.1f}%, "
      f"MNIST below tau {100*(ood_max<tau).mean():.1f}%")
for p in (1, 2, 5, 10):
    print(f"    (Fashion max-match p{p}: {np.percentile(fash_max, p):.2f})")

# --- B) the companion's traced run -----------------------------------------
print('\n=== B) companion inline config (K=48, 20k subset, 12 epochs, adam 4e-3) ===')
Xs, ys = X[:20000], y[:20000]
K = 48
Winit = Xs[np.random.RandomState(1).permutation(len(Xs))[:K]]


class YatLayer(nnx.Module):
    def __init__(self, d_in, n_units, *, rngs: nnx.Rngs, Winit, b0=1.0, eps0=1.0):
        self.W = nnx.Param(jnp.asarray(Winit))
        self.log_b = nnx.Param(jnp.full((), jnp.log(jnp.expm1(b0))))
        self.log_eps = nnx.Param(jnp.full((), jnp.log(jnp.expm1(eps0))))

    def __call__(self, x):
        b, eps = jax.nn.softplus(self.log_b.value), jax.nn.softplus(self.log_eps.value)
        dot = x @ self.W.value.T
        dist2 = jnp.sum(x ** 2, -1, keepdims=True) + jnp.sum(self.W.value ** 2, -1) - 2 * dot
        return (dot + b) ** 2 / (dist2 + eps)


class YatMLP(nnx.Module):
    def __init__(self, d_in, n_units, d_out, *, rngs: nnx.Rngs, Winit):
        self.yat = YatLayer(d_in, n_units, rngs=rngs, Winit=Winit)
        self.readout = nnx.Linear(n_units, d_out, rngs=rngs)

    def __call__(self, x):
        return self.readout(self.yat(x))


class ReluMLP(nnx.Module):
    def __init__(self, d_in, n_units, d_out, *, rngs: nnx.Rngs):
        self.l1 = nnx.Linear(d_in, n_units, rngs=rngs)
        self.l2 = nnx.Linear(n_units, d_out, rngs=rngs)

    def __call__(self, x):
        return self.l2(jax.nn.relu(self.l1(x)))


@nnx.jit
def train_step(model, opt, xb, yb):
    def loss_fn(model):
        return optax.softmax_cross_entropy_with_integer_labels(model(xb), yb).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    opt.update(model, grads)
    return loss


def train(model, Xtr, ytr, epochs=12, B=256, lr=4e-3, seed0=0):
    opt = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)
    for epoch in range(epochs):
        perm = np.random.RandomState(seed0 + epoch).permutation(len(Xtr))
        for i in range(0, len(Xtr) - B, B):
            j = perm[i:i + B]
            train_step(model, opt, jnp.asarray(Xtr[j]), jnp.asarray(ytr[j]))
    return model


def accuracy(model, Xq, yq):
    return float((jnp.argmax(model(jnp.asarray(Xq)), -1) == yq).mean()) * 100


yat = train(YatMLP(784, K, 10, rngs=nnx.Rngs(0), Winit=Winit), Xs, ys)
print(f"  Yat-MLP  (K=48), test[:5000]: {accuracy(yat, Xte[:5000], yte[:5000]):.1f}%")
relu = train(ReluMLP(784, K, 10, rngs=nnx.Rngs(0)), Xs, ys)
print(f"  ReLU MLP (K=48), test[:5000]: {accuracy(relu, Xte[:5000], yte[:5000]):.1f}%")
mu, sd = Xs.mean(0), Xs.std(0)
Wnoise = np.clip(mu + sd * np.random.RandomState(1).randn(K, 784), 0, 1).astype('float32')
noisy = train(YatMLP(784, K, 10, rngs=nnx.Rngs(0), Winit=Wnoise), Xs, ys)
print(f"  Yat-MLP  (K=48) seeded from stats-matched noise, test[:5000]: {accuracy(noisy, Xte[:5000], yte[:5000]):.1f}%")
fash_mm = np.asarray(yat.yat(jnp.asarray(Xte[:2000]))).max(1)
ood_mm = np.asarray(yat.yat(jnp.asarray(Xood[:2000]))).max(1)
print(f"  traced model max-match median: Fashion {np.median(fash_mm):.1f}, MNIST {np.median(ood_mm):.1f}")
relu50 = train(ReluMLP(784, 50, 10, rngs=nnx.Rngs(0)), Xs, ys)
print(f"  ReLU MLP (K=50, same config) on the full test set: {accuracy(relu50, Xte, yte):.1f}%")

# --- C) hand-built, sum-vote vs nearest-prototype ---------------------------
print('\n=== C) hand-built classifier, 20 k-means centroids per class, no training ===')
B_HAND, EPS_HAND = 0.5, 0.05
for name, Xpool, ypool in [('full-train config (notrain.json / yat_editable_fmnist.py)', X, y),
                           ('companion 20k-subset config', Xs, ys)]:
    Wh, lab = [], []
    for c in range(10):
        cen = KMeans(20, n_init=3, random_state=0).fit(Xpool[ypool == c][:2500]).cluster_centers_
        Wh += list(cen); lab += [c] * 20
    Wh = np.array(Wh, 'float32'); lab = np.array(lab)
    Kh = yat_kernel(Wh, Xte, B_HAND, EPS_HAND)
    score_sum = np.stack([Kh[:, lab == c].sum(1) for c in range(10)], 1)
    score_max = np.stack([Kh[:, lab == c].max(1) for c in range(10)], 1)
    print(f"  {name}:")
    print(f"    sum-vote (linear readout)        : {(score_sum.argmax(1) == yte).mean() * 100:.1f}%")
    print(f"    nearest-prototype (per-class max): {(score_max.argmax(1) == yte).mean() * 100:.1f}%")


def hand_built(Xpool, ypool, per):
    W, lb = [], []
    for c in range(10):
        pool = Xpool[ypool == c]
        kk = min(per, len(pool))
        cen = KMeans(kk, n_init=3, random_state=0).fit(pool).cluster_centers_
        W += list(cen); lb += [c] * kk
    A = np.eye(10)[lb].astype('float32')
    return np.array(W, 'float32'), A


def warm_model(W0, A0, rngs):
    m = YatMLP(784, len(W0), 10, rngs=rngs, Winit=W0, )
    m.yat.log_b = nnx.Param(jnp.full((), jnp.log(jnp.expm1(0.5))))
    m.yat.log_eps = nnx.Param(jnp.full((), jnp.log(jnp.expm1(0.05))))
    m.readout.kernel = nnx.Param(jnp.asarray(A0))
    m.readout.bias = nnx.Param(jnp.zeros(10))
    return m


# --- D) warm start on low data ----------------------------------------------
print('\n=== D) warm start, 40 training images per class ===')
idx = np.concatenate([np.where(y == c)[0][:40] for c in range(10)])
Xlow, ylow = X[idx], y[idx]
W0, A0 = hand_built(Xlow, ylow, per=5)                    # 5 centroids/class -> K=50
warm = warm_model(W0, A0, nnx.Rngs(0))
print(f"  hand-built, zero gradient steps: {accuracy(warm, Xte, yte):.1f}%")
train(warm, Xlow, ylow, epochs=60, B=100)
print(f"  fine-tuned from the hand-built start: {accuracy(warm, Xte, yte):.1f}%")
Wr = Xlow[np.random.RandomState(0).permutation(len(Xlow))[:50]]
cold = train(YatMLP(784, 50, 10, rngs=nnx.Rngs(0), Winit=Wr), Xlow, ylow, epochs=60, B=100)
print(f"  same network from a random start:     {accuracy(cold, Xte, yte):.1f}%")

# --- E) warm start under imbalance ------------------------------------------
print('\n=== E) warm start, Pullover & Sneaker starved to 30 images (5 seeds) ===')
RARE = [2, 7]
keep = np.ones(len(Xs), bool)
for c in RARE:
    hits = np.where(ys == c)[0]
    keep[hits[30:]] = False
Xi, yi = Xs[keep], ys[keep]
rare_te = np.isin(yte, RARE)


def rare_recall(model):
    pred = np.asarray(jnp.argmax(model(jnp.asarray(Xte[rare_te])), -1))
    return float((pred == yte[rare_te]).mean()) * 100


res = {'warm': [], 'cold': []}
for seed in range(5):
    W0, A0 = hand_built(Xi, yi, per=5)
    warm = train(warm_model(W0, A0, nnx.Rngs(seed)), Xi, yi, seed0=100 * seed)
    Wr = Xi[np.random.RandomState(seed).permutation(len(Xi))[:50]]
    cold = train(YatMLP(784, 50, 10, rngs=nnx.Rngs(seed), Winit=Wr), Xi, yi, seed0=100 * seed)
    res['warm'].append((rare_recall(warm), accuracy(warm, Xte, yte)))
    res['cold'].append((rare_recall(cold), accuracy(cold, Xte, yte)))
for k in ('cold', 'warm'):
    r = np.array(res[k])
    print(f"  {k:4s} start: rare-class recall {r[:, 0].mean():.0f}%   overall {r[:, 1].mean():.1f}%   (mean of 5)")
