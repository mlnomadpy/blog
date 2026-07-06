"""How far DOWN can construction reach? Build a SECOND hand-designed layer.

The hand-built post (scripts/handbuilt_vision.py) built ONE feature layer by
hand (6 oriented-edge channels + 1 corner, pooled over a 7x7 grid = 343 named
dimensions) and hit 83.3% on Fashion-MNIST with a constructed Yat head. This
script builds a LAYER 2 by hand on top of it: named combinations of layer-1
responses, every one nameable in a sentence:

  - junction   "a 30-degree edge and a 90-degree edge in the same cell"
  - continuation "a 0-degree edge that continues one cell along its own direction"
  - bend       "a 60-degree edge whose continuation has rotated one step (30 deg)"

computed on a 14x14 cell grid of the layer-1 maps, pooled coarser (4x4), then
fed to the exact same constructed Yat head (k-means prototypes per class,
nearest-prototype max readout, eps = median(d2)*0.1). Reports the layer-1
baseline (reproduces 83.3%), layer-2 alone, layer-1+2 concatenated, nearest-
centroid controls, a vocabulary-growth curve (names vs accuracy), and the
combinatorial count of POSSIBLE nameable pair-detectors vs the ones used.
Reference for the trained net: 85.7% (scripts/construct_vs_optimize.py).

Also dumps public/handbuilt-depth/depth.json + samples.png for the viz.

Run: python scripts/handbuilt_depth.py
"""
import warnings; warnings.filterwarnings('ignore')
import os, json, math, numpy as np, torchvision
from scipy.ndimage import convolve
from sklearn.cluster import KMeans
from PIL import Image

CLS = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
tr = torchvision.datasets.FashionMNIST('/tmp/fmnist', train=True, download=True)
te = torchvision.datasets.FashionMNIST('/tmp/fmnist', train=False, download=True)
Xtr = tr.data.numpy().astype('float32') / 255.0; ytr = tr.targets.numpy()
Xte = te.data.numpy().astype('float32') / 255.0; yte = te.targets.numpy()

# ════════ LAYER 1: verbatim from scripts/handbuilt_vision.py ════════
NB = 6                                                          # orientations
GRID = 7                                                        # layer-1 pool grid
NCH = NB + 1                                                    # 6 edges + 1 corner
sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 'float32')  # Sobel
sy = sx.T
CENTERS = np.linspace(0, np.pi, NB, endpoint=False)


def channel_maps(X):
    """[n, NCH, 28, 28]: per-pixel response of each named detector (pre-pool)."""
    gx = np.stack([convolve(im, sx, mode='nearest') for im in X])
    gy = np.stack([convolve(im, sy, mode='nearest') for im in X])
    mag = np.sqrt(gx ** 2 + gy ** 2) + 1e-6
    ang = (np.arctan2(gy, gx) % np.pi)
    out = []
    for b in range(NB):
        d = np.abs(((ang - CENTERS[b] + np.pi / 2) % np.pi) - np.pi / 2)
        out.append(np.clip(1 - d / (np.pi / NB), 0, 1) * mag)
    out.append(np.abs(gx) * np.abs(gy))
    return np.stack(out, 1)


def pool_l1(maps):
    n = len(maps); ps = 28 // GRID
    return maps[:, :, :GRID * ps, :GRID * ps].reshape(n, NCH, GRID, ps, GRID, ps).mean((3, 5))


def l2norm(V):
    return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-6)


# ════════ LAYER 2: hand-designed combinations of layer-1 responses ════════
CELL = 14                                                       # layer-2 works on a 14x14 cell grid
GRID2 = 4                                                       # coarser re-pool for layer-2 features
DEG = [round(a * 180 / np.pi) for a in CENTERS]
# contour direction of orientation bin b: the edge runs perpendicular to its
# gradient angle theta_b, so one step "along its own direction" is the unit
# vector (-sin theta, cos theta), applied at cell resolution with bilinear shift.
CONTOUR = [(-math.sin(a), math.cos(a)) for a in CENTERS]        # (dx, dy) per bin


def cells(maps):
    """[n, NCH, 14, 14]: layer-1 maps mean-pooled 2x2 to the cell grid."""
    n = len(maps)
    return maps.reshape(n, NCH, CELL, 2, CELL, 2).mean((3, 5))


def shift_int(A, dy, dx):
    """A[..., y, x] -> A[..., y+dy, x+dx], zero outside."""
    out = np.zeros_like(A); h, w = A.shape[-2:]
    y0, y1 = max(0, -dy), min(h, h - dy)
    x0, x1 = max(0, -dx), min(w, w - dx)
    if y1 > y0 and x1 > x0:
        out[..., y0:y1, x0:x1] = A[..., y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    return out


def shift_frac(A, dy, dx):
    """bilinear fractional shift: sample A at (y+dy, x+dx)."""
    y0, x0 = math.floor(dy), math.floor(dx); fy, fx = dy - y0, dx - x0
    return ((1 - fy) * (1 - fx) * shift_int(A, y0, x0) + (1 - fy) * fx * shift_int(A, y0, x0 + 1)
            + fy * (1 - fx) * shift_int(A, y0 + 1, x0) + fy * fx * shift_int(A, y0 + 1, x0 + 1))


def l2_specs(kinds=('junction', 'continuation', 'bend', 'stripe')):
    """The named layer-2 vocabulary. Every entry is one detector, one sentence."""
    specs = []
    if 'junction' in kinds:
        for b1 in range(NB):
            for b2 in range(b1 + 1, NB):
                specs.append(dict(kind='junction', b1=b1, b2=b2, short=f'{DEG[b1]}°+{DEG[b2]}° junction',
                                  name=f'a {DEG[b1]}° edge and a {DEG[b2]}° edge in the same cell, a junction where the two orientations meet'))
    if 'continuation' in kinds:
        for b in range(NB):
            specs.append(dict(kind='continuation', b1=b, b2=b, short=f'{DEG[b]}° continuation',
                              name=f'a {DEG[b]}° edge that continues one cell along its own direction, an elongated contour'))
    if 'bend' in kinds:
        for b in range(NB):
            specs.append(dict(kind='bend', b1=b, b2=-1, short=f'{DEG[b]}° bend',
                              name=f'a {DEG[b]}° edge whose continuation one cell along has rotated one step (30°), a gentle curve'))
    if 'stripe' in kinds:
        for b in range(NB):
            specs.append(dict(kind='stripe', b1=b, b2=b, short=f'{DEG[b]}° stripe',
                              name=f'a {DEG[b]}° edge with a parallel {DEG[b]}° edge two cells across, the two sides of a strip'))
    return specs


def l2_maps(C, specs, t=1.0, rule='min'):
    """[n, len(specs), 14, 14]: each named combination, evaluated on the cell grid."""
    comb = (np.minimum if rule == 'min' else lambda a, b: np.sqrt(a * b))
    E = C[:, :NB]
    out = []
    for s in specs:
        if s['kind'] == 'junction':
            m = comb(E[:, s['b1']], E[:, s['b2']])
        elif s['kind'] == 'continuation':
            dx, dy = CONTOUR[s['b1']]
            m = comb(E[:, s['b1']], shift_frac(E[:, s['b1']], dy * t, dx * t))
        elif s['kind'] == 'stripe':  # same orientation ACROSS the contour: two parallel sides
            b = s['b1']; gdx, gdy = math.cos(CENTERS[b]), math.sin(CENTERS[b])   # gradient dir = across the edge
            m = comb(E[:, b], np.maximum(shift_frac(E[:, b], gdy * 2, gdx * 2),
                                         shift_frac(E[:, b], -gdy * 2, -gdx * 2)))
        else:  # bend: the continuation arrives rotated one orientation step either way
            b = s['b1']; dx, dy = CONTOUR[b]
            nb = np.maximum(shift_frac(E[:, (b + 1) % NB], dy * t, dx * t),
                            shift_frac(E[:, (b - 1) % NB], dy * t, dx * t))
            m = comb(E[:, b], nb)
        out.append(m)
    return np.stack(out, 1)


def pool_l2(maps):
    """center-crop the 14x14 cell grid to 12x12 and mean-pool 3x3 -> GRID2 x GRID2."""
    n, k = maps.shape[:2]; ps = 3
    m = maps[:, :, 1:1 + GRID2 * ps, 1:1 + GRID2 * ps]
    return m.reshape(n, k, GRID2, ps, GRID2, ps).mean((3, 5))


# ════════ the constructed Yat head, exactly the C4 recipe ════════
B, PER = 0.5, 20


def build_and_eval(Ftr_raw, Fte_raw, ytr_s, yte, seed=0):
    mu, sd = Ftr_raw.mean(0), Ftr_raw.std(0) + 1e-6
    Ftr, Fte = (Ftr_raw - mu) / sd, (Fte_raw - mu) / sd
    W, vote = [], []
    for c in range(10):
        W += list(KMeans(PER, n_init=2, random_state=seed).fit(Ftr[ytr_s == c][:2500]).cluster_centers_); vote += [c] * PER
    W = np.array(W, 'float32'); vote = np.array(vote)
    dot = Fte @ W.T
    d2 = (Fte ** 2).sum(1, keepdims=True) + (W ** 2).sum(1) - 2 * dot
    EPS = float(np.median(d2) * 0.1)
    ker = (dot + B) ** 2 / (d2 + EPS)
    score = np.full((len(Fte), 10), -1e9)
    for c in range(10): score[:, c] = ker[:, vote == c].max(1)
    acc = 100 * (score.argmax(1) == yte).mean()
    cen = np.stack([Ftr[ytr_s == c].mean(0) for c in range(10)])
    nc = 100 * (((Fte[:, None] - cen[None]) ** 2).sum(2).argmin(1) == yte).mean()
    return acc, nc


# ════════ run ════════
NTR = 20000
print('extracting layer-1 maps (Sobel + orientation binning, the C4 recipe)...')
Mtr, Mte = channel_maps(Xtr[:NTR]), channel_maps(Xte)
ytr_s = ytr[:NTR]
F1tr, F1te = l2norm(pool_l1(Mtr).reshape(NTR, -1)), l2norm(pool_l1(Mte).reshape(len(Xte), -1))
Ctr, Cte = cells(Mtr), cells(Mte)
print(f'layer 1: {NCH} named channels x {GRID}x{GRID} patches = {F1tr.shape[1]} named dimensions')

acc1, nc1 = build_and_eval(F1tr, F1te, ytr_s, yte)
print(f'\nLAYER 1 ONLY (must reproduce the C4 number): {acc1:.1f}%   (nearest-centroid {nc1:.1f}%)')

# ---- design iterations on the layer-2 vocabulary ----
print('\n=== layer-2 design iterations (constructed head on layer-2 alone / concat) ===')
results = {}
for tag, kinds, t, rule in [
        ('V1 junctions only            ', ('junction',), 1.0, 'min'),
        ('V2 + continuations           ', ('junction', 'continuation'), 1.0, 'min'),
        ('V3 + bends                   ', ('junction', 'continuation', 'bend'), 1.0, 'min'),
        ('V4 + stripes (full)          ', ('junction', 'continuation', 'bend', 'stripe'), 1.0, 'min'),
        ('V4 geometric-mean AND        ', ('junction', 'continuation', 'bend', 'stripe'), 1.0, 'geo'),
        ('V4 two-cell reach            ', ('junction', 'continuation', 'bend', 'stripe'), 2.0, 'min'),
]:
    specs = l2_specs(kinds)
    F2tr = l2norm(pool_l2(l2_maps(Ctr, specs, t, rule)).reshape(NTR, -1))
    F2te = l2norm(pool_l2(l2_maps(Cte, specs, t, rule)).reshape(len(Xte), -1))
    a2, nc2 = build_and_eval(F2tr, F2te, ytr_s, yte)
    Fc_tr = np.concatenate([F1tr, F2tr], 1); Fc_te = np.concatenate([F1te, F2te], 1)
    ac, ncc = build_and_eval(Fc_tr, Fc_te, ytr_s, yte)
    results[tag.strip()] = dict(kinds=kinds, t=t, rule=rule, names=len(specs), dims=F2tr.shape[1],
                                l2only=a2, l2nc=nc2, concat=ac, concatnc=ncc)
    print(f'{tag} {len(specs):>2} names {F2tr.shape[1]:>4} dims | layer-2 alone {a2:5.1f}% (nc {nc2:4.1f}%) | layer-1+2 {ac:5.1f}% (nc {ncc:4.1f}%)')

# final design = the full min-AND one-cell vocabulary (V4)
FIN = results['V4 + stripes (full)']
specs = l2_specs(FIN['kinds'])
F2tr = l2norm(pool_l2(l2_maps(Ctr, specs, 1.0, 'min')).reshape(NTR, -1))
F2te = l2norm(pool_l2(l2_maps(Cte, specs, 1.0, 'min')).reshape(len(Xte), -1))

# ---- the vocabulary-growth curve: accuracy as named combinations are added ----
print('\n=== vocabulary growth: names vs accuracy (concat head) ===')
curve = [dict(stage='layer 1 alone', names=NCH, dims=F1tr.shape[1], acc=round(acc1, 1))]
for stage, kinds in [('+ 15 junctions', ('junction',)),
                     ('+ 6 continuations', ('junction', 'continuation')),
                     ('+ 6 bends', ('junction', 'continuation', 'bend')),
                     ('+ 6 stripes', ('junction', 'continuation', 'bend', 'stripe'))]:
    sp = l2_specs(kinds)
    A2tr = l2norm(pool_l2(l2_maps(Ctr, sp, 1.0, 'min')).reshape(NTR, -1))
    A2te = l2norm(pool_l2(l2_maps(Cte, sp, 1.0, 'min')).reshape(len(Xte), -1))
    a, _ = build_and_eval(np.concatenate([F1tr, A2tr], 1), np.concatenate([F1te, A2te], 1), ytr_s, yte)
    curve.append(dict(stage=stage, names=NCH + len(sp), dims=F1tr.shape[1] + A2tr.shape[1], acc=round(a, 1)))
    print(f'{stage:<20} {NCH + len(sp):>2} names {F1tr.shape[1] + A2tr.shape[1]:>4} dims -> {a:.1f}%')

# ---- the combinatorial wall: how many pair-detectors COULD be named ----
# a pairwise layer-2 type = (channel c1 here) AND (channel c2 at relative cell
# offset o), offsets in the 3x3 neighbourhood; (c1,c2,o) ~ (c2,c1,-o).
same = NCH * (NCH + 1) // 2                       # o = (0,0), unordered incl. self-pairs
shifted = NCH * NCH * 8 // 2                      # 8 nonzero offsets, halved by symmetry
pairs_possible = same + shifted
triples_possible = (NCH ** 3) * (9 ** 2) // 6     # order-of-magnitude, 3-way AND types
print(f'\ncombinatorial wall: layer 1 has {NCH} names; possible pairwise layer-2 types: {pairs_possible}'
      f' (we hand-picked {len(specs)}); possible 3-way types: ~{triples_possible:,}')

a2, nc2 = FIN['l2only'], FIN['l2nc']; acc_c, nc_c = FIN['concat'], FIN['concatnc']
best_tag = max(results, key=lambda k: results[k]['concat']); best_concat = results[best_tag]['concat']
print(f'\n════════ FINAL NUMBERS ════════')
print(f'layer 1 alone ({NCH} names, {F1tr.shape[1]} dims):          {acc1:.1f}%   nearest-centroid {nc1:.1f}%')
print(f'layer 2 alone ({len(specs)} names, {F2tr.shape[1]} dims):         {a2:.1f}%   nearest-centroid {nc2:.1f}%')
print(f'layer 1 + layer 2 ({NCH + len(specs)} names, {F1tr.shape[1] + F2tr.shape[1]} dims):    {acc_c:.1f}%   nearest-centroid {nc_c:.1f}%')
print(f'best design across all iterations: {best_tag} -> {best_concat:.1f}% (a {best_concat - acc1:+.1f} move on layer 1)')
print(f'reference: raw-pixel head 79.0%, trained backbone + built head 83.2%, fully trained 85.7% (scripts/construct_vs_optimize.py)')

# ════════ dump viz assets ════════
os.makedirs('public/handbuilt-depth', exist_ok=True)
rng = np.random.RandomState(7)


def sprite(tiles, cols, path):
    n = len(tiles); rows = (n + cols - 1) // cols; h, w = tiles[0].shape
    im = np.zeros((rows * h, cols * w), 'uint8')
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols); im[r*h:r*h+h, c*w:c*w+w] = (np.clip(t, 0, 1) * 255).astype('uint8')
    Image.fromarray(im, 'L').save(path)


SAMP = 60
sel = np.concatenate([rng.permutation(np.where(yte == c)[0])[:SAMP // 10] for c in range(10)])
rng.shuffle(sel)
sprite([Xte[i] for i in sel], 10, 'public/handbuilt-depth/samples.png')

data = {
    'classes': CLS, 'NB': NB, 'nChan': NCH, 'CELL': CELL, 'GRID2': GRID2,
    'centersDeg': DEG,
    'detectorsL1': [f'edge {d}°' for d in DEG] + ['corner'],
    'contour': [[round(dx, 4), round(dy, 4)] for dx, dy in CONTOUR],
    'specsL2': [dict(kind=s['kind'], b1=s['b1'], b2=s['b2'], short=s['short'], name=s['name']) for s in specs],
    'ladder': {'pixels': 79.0, 'l1': round(acc1, 1), 'l2only': round(float(a2), 1),
               'concat': round(float(acc_c), 1), 'bestConcat': round(float(best_concat), 1), 'bestTag': best_tag,
               'trainedConstructed': 83.2, 'trainedHead': 85.7,
               'nearestCentroidL1': round(nc1, 1), 'nearestCentroidConcat': round(float(nc_c), 1), 'chance': 10.0},
    'vocab': {'curve': curve, 'pairsPossible': int(pairs_possible), 'triplesPossible': int(triples_possible),
              'used': len(specs)},
    'iterations': [dict(tag=k, **{kk: (round(float(vv), 1) if isinstance(vv, float) else vv)
                                  for kk, vv in v.items() if kk not in ('kinds',)}) for k, v in results.items()],
    'samples': [{'label': int(yte[i])} for i in sel],
}
json.dump(data, open('public/handbuilt-depth/depth.json', 'w'))
print(f"\nwrote public/handbuilt-depth/depth.json + samples.png ({SAMP} samples)")
