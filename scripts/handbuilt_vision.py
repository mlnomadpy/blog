"""Can you build the FEATURES by hand too, with no training?

A training-free image classifier: hand-engineered patch detectors (oriented
edges and corners, mixer-style: every channel is a named feature, pooled over a
grid of patches), then a CONSTRUCTED Yat head (k-means prototypes, nearest-
prototype vote) on top. Nothing is trained. The point is to find the accuracy a
fully hand-built, fully readable network reaches, against the raw-pixel head
(79%) and a trained CNN backbone (85.7%).

Run: python scripts/handbuilt_vision.py
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, torchvision
from scipy.ndimage import convolve
from sklearn.cluster import KMeans

CLS = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
tr = torchvision.datasets.FashionMNIST('/tmp/fmnist', train=True, download=True)
te = torchvision.datasets.FashionMNIST('/tmp/fmnist', train=False, download=True)
Xtr = tr.data.numpy().astype('float32') / 255.0; ytr = tr.targets.numpy()
Xte = te.data.numpy().astype('float32') / 255.0; yte = te.targets.numpy()

# ── hand-built patch detectors (no training) ──
# oriented edge templates: a Gabor-like bank at NB orientations, plus a corner
# (gradient energy in two directions) and a flat/intensity channel.
NB = 6                                                          # orientations, the "assigned dimensions"
GRID = 7                                                        # 7x7 = 49 patches over the 28x28 image
sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 'float32')  # Sobel
sy = sx.T


def features(X):
    n = len(X)
    gx = np.stack([convolve(im, sx, mode='nearest') for im in X])
    gy = np.stack([convolve(im, sy, mode='nearest') for im in X])
    mag = np.sqrt(gx ** 2 + gy ** 2) + 1e-6
    ang = (np.arctan2(gy, gx) % np.pi)                         # 0..pi (undirected edges)
    # soft-assign each pixel's gradient to the NB orientation bins (named channels)
    feats = []
    centers = np.linspace(0, np.pi, NB, endpoint=False)
    for b in range(NB):
        d = np.abs(((ang - centers[b] + np.pi / 2) % np.pi) - np.pi / 2)
        w = np.clip(1 - d / (np.pi / NB), 0, 1) * mag           # this orientation's energy map
        feats.append(w)
    feats.append(np.abs(gx) * np.abs(gy))                       # two-direction gradient = corner energy
    feats = np.stack(feats, 1)                                  # [n, NB+1, 28, 28]
    # pool over the GRID of patches (mixer token-pool): each patch x each channel = one dimension
    ps = 28 // GRID
    P = feats[:, :, :GRID * ps, :GRID * ps].reshape(n, NB + 1, GRID, ps, GRID, ps).mean((3, 5))  # [n,C,GRID,GRID]
    V = P.reshape(n, -1)
    # per-image L2 normalise (contrast invariance, the HOG trick), all hand-fixed
    return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-6)


print(f"building hand-engineered features: {NB} oriented-edge channels + 1 corner channel, pooled over {GRID}x{GRID} patches")
Ftr, Fte = features(Xtr[:20000]), features(Xte)
ytr_s = ytr[:20000]
print(f"feature dimension: {Ftr.shape[1]} (each = a named detector at a patch position)")

# ── constructed Yat head on the hand-built features (no training) ──
B, PER = 0.5, 20
mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
Ftr, Fte = (Ftr - mu) / sd, (Fte - mu) / sd
W, vote = [], []
for c in range(10):
    W += list(KMeans(PER, n_init=2, random_state=0).fit(Ftr[ytr_s == c][:2500]).cluster_centers_); vote += [c] * PER
W = np.array(W, 'float32'); vote = np.array(vote)
dot = Fte @ W.T
d2 = (Fte ** 2).sum(1, keepdims=True) + (W ** 2).sum(1) - 2 * dot
ker = (dot + B) ** 2 / (d2 + np.median(d2) * 0.1)
score = np.full((len(Fte), 10), -1e9)
for c in range(10): score[:, c] = ker[:, vote == c].max(1)
acc = 100 * (score.argmax(1) == yte).mean()
print(f"\nHAND-BUILT, NO TRAINING ANYWHERE: {acc:.1f}%")
print("  reference: raw-pixel constructed head 79%, trained CNN backbone 85.7%")

# nearest-centroid (1 prototype/class) too, the simplest possible head
cen = np.stack([Ftr[ytr_s == c].mean(0) for c in range(10)])
nc = 100 * (((Fte[:, None] - cen[None]) ** 2).sum(2).argmin(1) == yte).mean()
print(f"  with a single centroid per class (nearest-mean): {nc:.1f}%")
