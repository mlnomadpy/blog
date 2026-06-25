"""Can you build the FEATURES by hand too, with no training?

A training-free image classifier: hand-engineered patch detectors (oriented
edges and corners, mixer-style: every channel is a named feature, pooled over a
grid of patches), then a CONSTRUCTED Yat head (k-means prototypes, nearest-
prototype vote) on top. Nothing is trained. The point is to find the accuracy a
fully hand-built, fully readable network reaches, against the raw-pixel head
(79%) and a trained CNN backbone (85.7%).

Also dumps public/handbuilt/handbuilt.json + sprites for the explainer viz.

Run: python scripts/handbuilt_vision.py
"""
import warnings; warnings.filterwarnings('ignore')
import os, json, numpy as np, torchvision
from scipy.ndimage import convolve
from sklearn.cluster import KMeans
from PIL import Image

CLS = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
tr = torchvision.datasets.FashionMNIST('/tmp/fmnist', train=True, download=True)
te = torchvision.datasets.FashionMNIST('/tmp/fmnist', train=False, download=True)
Xtr = tr.data.numpy().astype('float32') / 255.0; ytr = tr.targets.numpy()
Xte = te.data.numpy().astype('float32') / 255.0; yte = te.targets.numpy()

# ── hand-built patch detectors (no training) ──
# oriented edge templates: a Gabor-like bank at NB orientations, plus a corner
# (gradient energy in two directions). Every channel is a named measurement.
NB = 6                                                          # orientations, the "assigned dimensions"
GRID = 7                                                        # 7x7 = 49 patches over the 28x28 image
NCH = NB + 1                                                    # 6 oriented edges + 1 corner
sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 'float32')  # Sobel
sy = sx.T
CENTERS = np.linspace(0, np.pi, NB, endpoint=False)            # orientation bin centres (radians)


def channel_maps(X):
    """[n, NCH, 28, 28]: per-pixel response of each named detector (pre-pool)."""
    gx = np.stack([convolve(im, sx, mode='nearest') for im in X])
    gy = np.stack([convolve(im, sy, mode='nearest') for im in X])
    mag = np.sqrt(gx ** 2 + gy ** 2) + 1e-6
    ang = (np.arctan2(gy, gx) % np.pi)                         # 0..pi (undirected edges)
    out = []
    for b in range(NB):
        d = np.abs(((ang - CENTERS[b] + np.pi / 2) % np.pi) - np.pi / 2)
        out.append(np.clip(1 - d / (np.pi / NB), 0, 1) * mag)  # this orientation's energy
    out.append(np.abs(gx) * np.abs(gy))                        # corner = two-direction gradient
    return np.stack(out, 1)


def pool(maps):
    """[n, NCH, GRID, GRID]: mean-pool each channel over the GRID of patches."""
    n = len(maps); ps = 28 // GRID
    return maps[:, :, :GRID * ps, :GRID * ps].reshape(n, NCH, GRID, ps, GRID, ps).mean((3, 5))


def features(X):
    V = pool(channel_maps(X)).reshape(len(X), -1)
    return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-6)  # per-image L2 norm (HOG contrast trick)


print(f"hand-engineered features: {NB} oriented-edge channels + 1 corner channel, pooled over {GRID}x{GRID} patches")
Ftr_raw, Fte_raw = features(Xtr[:20000]), features(Xte)
ytr_s = ytr[:20000]
print(f"feature dimension: {Ftr_raw.shape[1]} (each = a named detector at a patch position)")

# ── constructed Yat head on the hand-built features (no training) ──
B, PER = 0.5, 20
mu, sd = Ftr_raw.mean(0), Ftr_raw.std(0) + 1e-6
Ftr, Fte = (Ftr_raw - mu) / sd, (Fte_raw - mu) / sd
W, vote = [], []
for c in range(10):
    W += list(KMeans(PER, n_init=2, random_state=0).fit(Ftr[ytr_s == c][:2500]).cluster_centers_); vote += [c] * PER
W = np.array(W, 'float32'); vote = np.array(vote)
dot = Fte @ W.T
d2 = (Fte ** 2).sum(1, keepdims=True) + (W ** 2).sum(1) - 2 * dot
EPS = float(np.median(d2) * 0.1)
ker = (dot + B) ** 2 / (d2 + EPS)
score = np.full((len(Fte), 10), -1e9)
for c in range(10): score[:, c] = ker[:, vote == c].max(1)
pred = score.argmax(1)
acc = 100 * (pred == yte).mean()
print(f"\nHAND-BUILT, NO TRAINING ANYWHERE: {acc:.1f}%")
print("  reference: raw-pixel constructed head 79%, trained CNN backbone 85.7%")
cen = np.stack([Ftr[ytr_s == c].mean(0) for c in range(10)])
nc = 100 * (((Fte[:, None] - cen[None]) ** 2).sum(2).argmin(1) == yte).mean()
print(f"  with a single centroid per class (nearest-mean): {nc:.1f}%")

# ════════════════════════ dump viz assets ════════════════════════
os.makedirs('public/handbuilt', exist_ok=True)
rng = np.random.RandomState(3)


def sprite(tiles, cols, path):                                 # tiles: list of [h,w] in [0,1]
    n = len(tiles); rows = (n + cols - 1) // cols; h, w = tiles[0].shape
    im = np.zeros((rows * h, cols * w), 'uint8')
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols); im[r*h:r*h+h, c*w:c*w+w] = (np.clip(t, 0, 1) * 255).astype('uint8')
    Image.fromarray(im, 'L').save(path)


# sample images shown in the viz: a varied spread, balanced across classes
SAMP = 60
sel = np.concatenate([rng.permutation(np.where(yte == c)[0])[:SAMP // 10] for c in range(10)])
rng.shuffle(sel)
sample_imgs = [Xte[i] for i in sel]
sprite(sample_imgs, 10, 'public/handbuilt/samples.png')        # 28x28 raw tiles, 10 cols

# response maps for the first DECN samples (one per class, in class order) for DecomposeImage
DECN = 10
dec_sel = np.array([np.where(yte == c)[0][0] for c in range(10)])
dec_maps = channel_maps(Xte[dec_sel])                          # [10, NCH, 28, 28]
# normalise each image's maps by its own global max so channels stay comparable
map_tiles = []
for i in range(DECN):
    m = dec_maps[i]
    for ch in range(NCH):
        ch_map = m[ch]; cmx = ch_map.max() + 1e-6
        map_tiles.append((ch_map / cmx) ** 0.6)   # per-channel norm + gamma so every detector's map reads clearly
sprite(map_tiles, NCH, 'public/handbuilt/maps.png')            # 10 rows x NCH cols
sprite([Xte[i] for i in dec_sel], 1, 'public/handbuilt/decimg.png')  # the 10 decompose source images

# feature-exemplars: nearest real (train) image to each prototype, for the vote gallery
Wun = W * sd + mu                                              # back to raw feature space for NN search
exe = []
for u in range(len(W)):
    c = vote[u]; idx = np.where(ytr_s == c)[0]
    j = idx[((Ftr_raw[idx] - Wun[u]) ** 2).sum(1).argmin()]
    exe.append(Xtr[j])
sprite(exe, PER, 'public/handbuilt/exemplars.png')             # 20 cols x 10 rows

# pooled feature of each sample, reshaped [NCH, GRID, GRID] is what AssignedDimensions reads
samples = [{'label': int(yte[i]), 'pred': int(pred[k]),
            'feat': [round(float(v), 3) for v in Fte[i]],          # z-scored (the head's input)
            'featRaw': [round(float(v), 4) for v in Fte_raw[i]]}    # L2-normed pooled energy (non-negative, for display)
           for k, i in enumerate(sel)]
data = {
    'classes': CLS, 'NB': NB, 'GRID': GRID, 'nChan': NCH, 'dim': int(W.shape[1]),
    'centersDeg': [round(float(a) * 180 / np.pi) for a in CENTERS],
    'detectors': [f'edge {round(a*180/np.pi)}°' for a in CENTERS] + ['corner'],
    'B': B, 'eps': round(EPS, 4),
    'mu': [round(float(v), 5) for v in mu], 'sd': [round(float(v), 5) for v in sd],   # raw-feature z-score, to map a live (drawn) image into the prototype space
    'ladder': {'pixels': 79.0, 'handbuilt': round(acc, 1), 'nearestCentroid': round(nc, 1),
               'trainedConstructed': 83.2, 'trainedHead': 85.7, 'chance': 10.0},
    'samples': samples,
    'protos': [[round(float(v), 3) for v in w] for w in W],
    'vote': [int(v) for v in vote],
}
json.dump(data, open('public/handbuilt/handbuilt.json', 'w'))
print(f"\nwrote public/handbuilt/handbuilt.json ({len(samples)} samples, {len(W)} prototypes)")
print("wrote samples.png, maps.png, decimg.png, exemplars.png")
