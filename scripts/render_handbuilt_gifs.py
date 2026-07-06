"""Figures for the hand-built JAX/Flax NNX companion.

Only one of these is a GIF, because only one is a real temporal process: k-means
Lloyd iterations, where the prototypes actually move frame to frame and the
convergence is the point. Everything else is a fixed computation (a convolution,
a pooling, a kernel shape, a garment's ten class scores), so it is a static figure
from the same real run, not a motion faked onto a result.

  GIF (a real iterative process):
    handbuilt-kmeans.gif      real Lloyd iterations placing prototypes in the
                              feature clouds of three classes; the markers migrate.

  Static PNGs (a fixed computation, one real run):
    handbuilt-sobel.png       image -> gx, gy -> gradient field, the whole conv.
    handbuilt-orient-rose.png the garment's edges and its orientation histogram.
    handbuilt-coarsegrain.png full detector maps, their 7x7 pooled versions, and
                              the resulting 343-number feature vector.
    handbuilt-yat-well.png    four real garments and their real ten-class Yat-score
                              landscapes; the star sits in the deepest well.
    handbuilt-kernel-shape.png the Yat kernel at the model's real eps, as a 2D
                              field and a 1D cut, vs true 1/r^2 and a Gaussian.

Reuses public/handbuilt/handbuilt.json (prototypes, vote, mu, sd, eps) and pulls
a few raw Fashion-MNIST images for the gradient fields. Every number is from the
same pipeline as scripts/jax_handbuilt.py.

Run: python scripts/render_handbuilt_gifs.py
"""
import warnings; warnings.filterwarnings('ignore')
import json, os
from pathlib import Path
import numpy as np
from scipy.ndimage import convolve
import torchvision
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, FancyArrow
import imageio.v2 as imageio
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / 'public'
HB = json.load(open(PUB / 'handbuilt' / 'handbuilt.json'))

# ── palette (matches src/components/viz/handbuilt.js) ──
BG = '#0e0d0b'; PANEL = '#16140f'; INK = '#e8e2d4'; MUTED = '#9a9282'; LINE = '#3a352c'
CLASS_COL = ['#b3661b', '#4a7fb3', '#3a8f5e', '#9a4f9c', '#c2553a', '#5a7d3a', '#2f8f8f', '#a06a2a', '#7a5fc0', '#c0892a']
CHAN_COL = ['#c2553a', '#c77d2a', '#5a7d3a', '#2f8f8f', '#4a7fb3', '#7a5fc0', '#a06a2a']
CLASSES = HB['classes']
plt.rcParams.update({'figure.facecolor': BG, 'savefig.facecolor': BG, 'text.color': INK,
                     'axes.edgecolor': LINE, 'font.size': 11})

NB, GRID = 6, 7
CENTERS = np.linspace(0, np.pi, NB, endpoint=False)
SX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32); SY = SX.T


def ease(t):
    return t * t * (3 - 2 * t)


def pipeline(img):
    """One [28,28] image in [0,1] -> (gx, gy, mag, ang, channels[7,28,28], pooled[7,7,7])."""
    gx = convolve(img, SX, mode='nearest'); gy = convolve(img, SY, mode='nearest')
    mag = np.sqrt(gx ** 2 + gy ** 2) + 1e-6
    ang = np.mod(np.arctan2(gy, gx), np.pi)
    edges = []
    for c in CENTERS:
        d = np.abs(np.mod(ang - c + np.pi / 2, np.pi) - np.pi / 2)
        edges.append(np.clip(1 - d / (np.pi / NB), 0, 1) * mag)
    corner = np.abs(gx) * np.abs(gy)
    chans = np.stack(edges + [corner])           # [7,28,28]
    ps = 28 // GRID
    pooled = chans[:, :GRID * ps, :GRID * ps].reshape(7, GRID, ps, GRID, ps).mean((2, 4))
    return gx, gy, mag, ang, chans, pooled


# ── load a few clean test images per the same indices used elsewhere ──
te = torchvision.datasets.FashionMNIST('/tmp/fmnist', train=False, download=True)
TEX = te.data.numpy().astype(np.float32) / 255.0
TEY = te.targets.numpy()


def grab(label, k=0):
    idx = np.where(TEY == label)[0]
    return TEX[idx[k]]


def fig_rgba(fig):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba


def save_gif(path, frames, fps, hold=14):
    frames = frames + [frames[-1]] * hold
    imageio.mimsave(path, frames, duration=1 / fps, loop=0, palettesize=128, subrectangles=True)
    print(f'wrote {path} ({os.path.getsize(path)//1024} KB)')


def save_png(path, fig):
    fig.savefig(path, dpi=fig.dpi, facecolor=BG)
    plt.close(fig)
    print(f'wrote {path} ({os.path.getsize(path)//1024} KB)')


# ═══════════════════════════════════════════════════════════════════════════
# FIG 1 (static) — orientation: the garment's strong edges (left, drawn as oriented
#   strokes coloured by bin) and the orientation histogram they sum to (right).
# ═══════════════════════════════════════════════════════════════════════════
def png_orient_rose():
    img = grab(8, 0)                                   # a Bag: clean strong edges
    gx, gy, mag, ang, chans, pooled = pipeline(img)
    thr = np.quantile(mag, 0.985)
    ys, xs = np.where(mag >= thr)
    order = np.argsort(-mag[ys, xs])[:90]
    ys, xs = ys[order], xs[order]
    a = ang[ys, xs]
    binof = np.clip((a / (np.pi / NB)).astype(int), 0, NB - 1)
    sx0 = xs / 27.0; sy0 = 1 - ys / 27.0
    rose_cx, rose_cy = 0.5, 0.5

    fig = plt.figure(figsize=(8.6, 4.5), dpi=110, facecolor=BG)
    fig.text(0.5, 0.93, 'A hand-built detector: sorting edges by their orientation',
             ha='center', fontsize=14, weight='bold')
    fig.text(0.5, 0.865, 'each strong edge belongs to the bin nearest its angle; the rose on the right is the garment’s edge signature',
             ha='center', fontsize=9.5, color=MUTED)
    axL = fig.add_axes([0.04, 0.08, 0.40, 0.74]); axL.set_facecolor(PANEL)
    axR = fig.add_axes([0.52, 0.08, 0.44, 0.74]); axR.set_facecolor(PANEL); axR.set_aspect('equal')
    axL.imshow(img, cmap='gray', extent=[0, 1, 0, 1], vmin=0, vmax=1, alpha=0.62, zorder=0)
    axL.set_xlim(0, 1); axL.set_ylim(0, 1); axL.set_xticks([]); axL.set_yticks([])
    axL.set_title('the picture + its edges', fontsize=10, color=MUTED, pad=4)
    axR.set_xlim(0, 1); axR.set_ylim(0, 1); axR.set_xticks([]); axR.set_yticks([])
    axR.set_title('orientation rose (6 bins)', fontsize=10, color=MUTED, pad=4)
    for s in list(axL.spines.values()) + list(axR.spines.values()): s.set_color(LINE)

    # the oriented edge strokes on the garment, coloured by their bin
    for i in range(len(a)):
        dx = 0.02 * np.cos(a[i]); dy = 0.02 * np.sin(a[i])
        axL.plot([sx0[i] - dx, sx0[i] + dx], [sy0[i] - dy, sy0[i] + dy],
                 color=CHAN_COL[binof[i]], lw=1.3, alpha=0.9, solid_capstyle='round', zorder=3)

    # rose guide wedges + tick labels
    for b in range(NB):
        th0 = np.degrees(CENTERS[b]); th1 = np.degrees(CENTERS[b] + np.pi / NB)
        axR.add_patch(Wedge((rose_cx, rose_cy), 0.40, th0, th1, width=0.40,
                            facecolor=CHAN_COL[b], alpha=0.07, edgecolor=LINE, lw=0.5, zorder=0))
        lab_th = CENTERS[b] + np.pi / (2 * NB)
        axR.text(rose_cx + 0.45 * np.cos(lab_th), rose_cy + 0.45 * np.sin(lab_th),
                 f'{int(round(np.degrees(CENTERS[b])))}°', ha='center', va='center',
                 fontsize=8, color=CHAN_COL[b])

    # the histogram petals: real per-bin edge mass
    tot = np.array([(binof == b).sum() for b in range(NB)], float)
    tot = tot / (tot.max() + 1e-9)
    for b in range(NB):
        th0 = np.degrees(CENTERS[b]); th1 = np.degrees(CENTERS[b] + np.pi / NB)
        axR.add_patch(Wedge((rose_cx, rose_cy), 0.10 + 0.30 * tot[b], th0, th1,
                            facecolor=CHAN_COL[b], alpha=0.55, edgecolor=INK, lw=0.6, zorder=2))
    save_png(PUB / 'handbuilt-orient-rose.png', fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIG 2 (static) — coarse-graining: the full-resolution detector maps (top), their
#   7x7 pooled versions (middle), and the 343-number feature vector they flatten to.
# ═══════════════════════════════════════════════════════════════════════════
def png_coarsegrain():
    img = grab(0, 1)                                   # a T-shirt
    gx, gy, mag, ang, chans, pooled = pipeline(img)
    ps = 28 // GRID                                    # 4
    full = chans[:, :GRID * ps, :GRID * ps]
    full = full / (full.reshape(7, -1).max(1)[:, None, None] + 1e-9)
    poolN = pooled / (pooled.reshape(7, -1).max(1)[:, None, None] + 1e-9)
    block = np.repeat(np.repeat(poolN, ps, 1), ps, 2)  # [7,28,28] blocky
    names = HB['detectors']

    fig = plt.figure(figsize=(8.8, 5.4), dpi=110, facecolor=BG)
    fig.text(0.5, 0.95, 'Coarse-graining: from full-resolution maps to 343 named numbers',
             ha='center', fontsize=14, weight='bold')
    fig.text(0.5, 0.905, 'each detector map (top) averages over a 7×7 grid of patches (middle); the 7×7×7 cells flatten into one feature vector',
             ha='center', fontsize=9.5, color=MUTED)
    fig.text(0.022, 0.735, 'full', ha='center', va='center', rotation=90, fontsize=8.5, color=MUTED)
    fig.text(0.022, 0.515, '7×7\npooled', ha='center', va='center', rotation=90, fontsize=8.5, color=MUTED)
    for ci in range(7):
        # top row: full-resolution maps
        axt = fig.add_axes([0.05 + ci * 0.133, 0.64, 0.118, 0.19]); axt.set_facecolor(PANEL)
        axt.imshow(full[ci], cmap='magma', vmin=0, vmax=1, interpolation='bilinear')
        axt.set_xticks([]); axt.set_yticks([])
        for s in axt.spines.values(): s.set_color(CHAN_COL[ci]); s.set_linewidth(1.4)
        axt.set_title(names[ci], fontsize=7.5, color=CHAN_COL[ci], pad=2)
        # middle row: the 7x7 pooled (blocky) map
        axm = fig.add_axes([0.05 + ci * 0.133, 0.42, 0.118, 0.19]); axm.set_facecolor(PANEL)
        axm.imshow(block[ci], cmap='magma', vmin=0, vmax=1, interpolation='nearest')
        axm.set_xticks([]); axm.set_yticks([])
        for s in axm.spines.values(): s.set_color(CHAN_COL[ci]); s.set_linewidth(1.0)
        for g in range(1, GRID):
            axm.axhline(g * ps - 0.5, color=BG, lw=0.6); axm.axvline(g * ps - 0.5, color=BG, lw=0.6)

    # bottom: the 343-vector bar, real pooled values
    axv = fig.add_axes([0.05, 0.10, 0.90, 0.22]); axv.set_facecolor(PANEL)
    axv.set_xlim(0, 343); axv.set_ylim(0, 1); axv.set_yticks([])
    vals = poolN.reshape(7, GRID * GRID)               # [7,49]
    for ci in range(7):
        base = ci * 49
        axv.bar(base + np.arange(49), vals[ci], width=1.0, color=CHAN_COL[ci], alpha=0.9, align='edge')
    for ci in range(1, 7):
        axv.axvline(ci * 49, color=LINE, lw=0.8)
    axv.set_xticks([24 + ci * 49 for ci in range(7)])
    axv.set_xticklabels(names, fontsize=7.5, color=MUTED)
    axv.set_title('the feature vector φ(x): 343 numbers, one per (detector, patch)',
                  fontsize=9.5, color=MUTED, pad=3)
    for s in axv.spines.values(): s.set_color(LINE)
    save_png(PUB / 'handbuilt-coarsegrain.png', fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIG 3 (static) — the Yat well: four real garments, each with the ten-class
#   Yat-score landscape its features fix. The star sits in the deepest well.
#   Three clean hits + one honest miss, all real scores.
# ═══════════════════════════════════════════════════════════════════════════
def png_yat_well():
    protos = np.array(HB['protos'], np.float32)        # [200,343] z-scored
    vote = np.array(HB['vote']); eps = HB['eps']; B = HB['B']
    samples = HB['samples']
    Z = np.array([s['feat'] for s in samples], np.float32)   # z-scored test feats
    lab = np.array([s['label'] for s in samples])

    def class_scores(z):                                # real Yat head scores, per class
        dot = protos @ z + B
        d2 = ((protos - z) ** 2).sum(1) + eps
        k = dot ** 2 / d2
        return np.array([k[vote == c].max() for c in range(10)])

    WSIG = 0.34
    xs = np.arange(10.0)
    xgrid = np.linspace(-0.7, 9.7, 600)

    def profile(depth):
        bumps = depth[None, :] * np.exp(-((xgrid[:, None] - xs[None, :]) / WSIG) ** 2 / 2)
        return -bumps.max(1)

    # choose demos: real garments, three clean hits + one honest miss, deterministic
    demos = []; used = set()
    for i in range(len(samples)):
        if lab[i] in used: continue
        used.add(lab[i])
        s = class_scores(Z[i]); pred = int(np.argmax(s))
        demos.append((i, s, pred))
    hits = [d for d in demos if d[2] == lab[d[0]]]
    miss = [d for d in demos if d[2] != lab[d[0]]]
    chosen = hits[:3] + (miss[:1] if miss else hits[3:4])

    fig = plt.figure(figsize=(9.6, 6.6), dpi=110, facecolor=BG)
    fig.text(0.5, 0.965, 'The head is a landscape of ten Yat wells', ha='center', fontsize=14, weight='bold')
    fig.text(0.5, 0.925, 'each well’s depth is the garment’s real Yat score for that class; its features place the star in the deepest one',
             ha='center', fontsize=8.8, color=MUTED)
    cellw, cellh = 0.465, 0.375
    origins = [(0.045, 0.50), (0.525, 0.50), (0.045, 0.055), (0.525, 0.055)]
    for di, (si, s, pred) in enumerate(chosen):
        d = (s - s.min()) / (s.max() - s.min() + 1e-9)
        depth = 0.12 + 0.88 * d
        V = profile(depth)
        img = grab(int(lab[si]), di)
        ox, oy = origins[di]
        ax = fig.add_axes([ox, oy + 0.045, cellw - 0.11, cellh - 0.10]); ax.set_facecolor(PANEL)
        ax.set_xlim(-0.7, 9.7); ax.set_ylim(-1.18, 0.16)
        ax.plot(xgrid, V, color=MUTED, lw=1.5, zorder=2)
        ax.fill_between(xgrid, V, -1.18, color='#1d1a14', zorder=1)
        ax.axhline(0, color=LINE, lw=0.8)
        for c in range(10):
            win = c == pred
            ax.scatter([xs[c]], [-depth[c]], s=95 if win else 38,
                       color=CLASS_COL[c], edgecolors='white' if win else BG,
                       linewidths=1.4 if win else 0.6, zorder=4)
        ax.scatter([xs[pred]], [-depth[pred]], s=180, marker='*', color='white',
                   edgecolors=BG, linewidths=1.3, zorder=5)
        ax.set_xticks(xs); ax.set_xticklabels(CLASSES, rotation=40, ha='right', fontsize=6.5, color=MUTED)
        ax.set_yticks([]); ax.set_ylabel('Yat score', color=MUTED, fontsize=8)
        for sp in ax.spines.values(): sp.set_color(LINE)
        # garment thumbnail
        axt = fig.add_axes([ox + cellw - 0.12, oy + cellh - 0.14, 0.085, 0.11])
        axt.imshow(img, cmap='gray', vmin=0, vmax=1); axt.set_xticks([]); axt.set_yticks([])
        for sp in axt.spines.values(): sp.set_color(LINE)
        axt.set_title(f'true:\n{CLASSES[int(lab[si])]}', fontsize=7, color=INK, pad=2)
        ok = pred == lab[si]
        msg = f'deepest: {CLASSES[pred]}  ' + ('✓ correct' if ok else '✗ a miss (the honest 83.3%)')
        fig.text(ox + (cellw - 0.11) / 2, oy, msg, ha='center', fontsize=9,
                 color='#7bbf5a' if ok else '#e08a6a', weight='bold')
    save_png(PUB / 'handbuilt-yat-well.png', fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIG 4 (static) — the Sobel convolution: image, gx, gy, and the gradient field
#   (magnitude + angle arrows). One fixed computation, shown whole.
# ═══════════════════════════════════════════════════════════════════════════
def png_sobel_gradient():
    img = grab(0, 2)                                    # a T-shirt
    gx = convolve(img, SX, mode='nearest'); gy = convolve(img, SY, mode='nearest')
    mag = np.sqrt(gx ** 2 + gy ** 2); ang = np.mod(np.arctan2(gy, gx), np.pi)
    H = 28; step = 2
    fig = plt.figure(figsize=(8.8, 3.6), dpi=110, facecolor=BG)
    fig.text(0.5, 0.92, 'A hand-built detector starts as a convolution', ha='center', fontsize=14, weight='bold')
    fig.text(0.5, 0.83, 'two Sobel kernels give the gradient gₓ, g_y at every pixel, then its magnitude and angle',
             ha='center', fontsize=9, color=MUTED)
    titles = ['image', 'gₓ (Sobel-x)', 'g_y (Sobel-y)', 'gradient field']
    for pi in range(4):
        ax = fig.add_axes([0.035 + pi * 0.245, 0.10, 0.20, 0.62]); ax.set_facecolor(PANEL)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_color(LINE)
        ax.set_title(titles[pi], fontsize=9, color=MUTED, pad=3)
        if pi == 0:
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        elif pi == 1:
            ax.imshow(gx, cmap='coolwarm', vmin=-4, vmax=4)
        elif pi == 2:
            ax.imshow(gy, cmap='coolwarm', vmin=-4, vmax=4)
        else:
            ax.imshow(mag, cmap='magma', vmin=0, vmax=4)
            ys, xs = np.mgrid[0:H:step, 0:H:step]
            u = np.cos(ang[::step, ::step]); v = -np.sin(ang[::step, ::step])
            m = mag[::step, ::step] > 0.8
            ax.quiver(xs[m], ys[m], u[m], v[m], color='#36d6c4', scale=34, width=0.006, alpha=0.8)
            ax.set_xlim(-0.5, H - 0.5); ax.set_ylim(H - 0.5, -0.5)
    save_png(PUB / 'handbuilt-sobel.png', fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIG 5 (static) — the Yat kernel: a softened inverse-square well as a 2D field
#   and a 1D cut, at one representative ε, against true 1/r² and a Gaussian.
# ═══════════════════════════════════════════════════════════════════════════
def png_kernel_shape():
    B = 0.5
    mu = np.array([1.0, 0.0])                           # a prototype off-origin
    eps = 0.15                                          # one representative softening
    gN = 260
    gv = np.linspace(-1.2, 2.6, gN); gw = np.linspace(-1.9, 1.9, gN)
    GX, GY = np.meshgrid(gv, gw); P = np.stack([GX.ravel(), GY.ravel()], 1)
    dot = P @ mu + B; dist2 = ((P - mu) ** 2).sum(1)
    t = np.linspace(-1.9, 1.9, 400)
    cdot = (mu @ mu) + B                                # constant along the perpendicular line
    yat = (dot ** 2 / (dist2 + eps)).reshape(gN, gN)
    lyat = cdot ** 2 / (t ** 2 + eps)                          # softened inverse-square
    linv = cdot ** 2 / (t ** 2 + 1e-3)                         # true inverse-square (blows up)
    lrbf = (cdot ** 2 / eps) * np.exp(-t ** 2 / (2 * (eps ** 0.5) ** 2))   # matched-peak Gaussian
    ymax = cdot ** 2 / eps * 1.15
    fig = plt.figure(figsize=(8.6, 4.2), dpi=110, facecolor=BG)
    fig.text(0.5, 0.93, 'The Yat kernel: a softened inverse-square well', ha='center', fontsize=14, weight='bold')
    fig.text(0.5, 0.85, 'k(z, μ) = (z·μ + b)² / (‖z − μ‖² + ε): peaked at the prototype, finite there because ε softens the singularity, heavy-tailed where a Gaussian dies',
             ha='center', fontsize=8.0, color=MUTED)
    axH = fig.add_axes([0.04, 0.09, 0.42, 0.68]); axH.set_facecolor(PANEL)
    axH.imshow(yat, extent=[-1.2, 2.6, -1.9, 1.9], origin='lower', cmap='magma',
               vmin=0, vmax=np.percentile(yat, 99.0))
    axH.scatter([mu[0]], [mu[1]], s=70, color='#36d6c4', edgecolors='white', linewidths=1.2, zorder=3)
    axH.text(mu[0], mu[1] + 0.26, 'prototype μ', color='#36d6c4', fontsize=8.5, ha='center')
    axH.axvline(mu[0], color='#36d6c4', lw=0.8, ls=':', alpha=0.6)   # the cross-section line
    axH.set_xticks([]); axH.set_yticks([]); axH.set_title('k(z, μ) over the plane', fontsize=9, color=MUTED, pad=3)
    for s in axH.spines.values(): s.set_color(LINE)
    axC = fig.add_axes([0.57, 0.16, 0.39, 0.58]); axC.set_facecolor(PANEL)
    axC.plot(t, np.minimum(linv, ymax * 1.5), color='#c2553a', lw=1.3, ls=':', label='true 1/r²  (blows up)')
    axC.plot(t, lrbf, color='#5a9fd0', lw=1.6, ls='--', label='Gaussian (thin tails)')
    axC.plot(t, lyat, color='#e0a45a', lw=2.6, label='Yat (finite peak, heavy tails)')
    axC.axvline(0, color='#36d6c4', lw=0.7, ls=':')
    axC.set_ylim(0, ymax); axC.set_xticks([])
    axC.set_xlabel('distance from μ (perpendicular cut)', color=MUTED, fontsize=8.5)
    axC.set_ylabel('kernel value', color=MUTED, fontsize=8.5); axC.tick_params(colors=MUTED, labelsize=7)
    axC.legend(loc='upper right', fontsize=7.2, facecolor=PANEL, edgecolor=LINE, labelcolor=INK)
    axC.set_title(f'ε = {eps:.2f}   (the placed softening)', fontsize=9, color=INK, pad=3)
    for s in axC.spines.values(): s.set_color(LINE)
    save_png(PUB / 'handbuilt-kernel-shape.png', fig)


# ═══════════════════════════════════════════════════════════════════════════
# GIF 6 — k-means places the prototypes: real Lloyd iterations on real features
#          for the confusable trio (Pullover, Coat, Shirt). No gradients.
# ═══════════════════════════════════════════════════════════════════════════
def feats343(imgs):
    out = []
    for im in imgs:
        _, _, _, _, _, pooled = pipeline(im)
        v = pooled.reshape(-1); v = v / (np.linalg.norm(v) + 1e-6)
        out.append(v)
    return np.array(out)


def gif_kmeans_prototypes():
    mu = np.array(HB['mu']); sd = np.array(HB['sd'])
    trio = [1, 7, 8]                                    # Trouser, Sneaker, Bag: well separated
    cols = [CLASS_COL[c] for c in trio]
    feats, labs = [], []
    for c in trio:
        idx = np.where(TEY == c)[0][:220]
        F = (feats343(TEX[idx]) - mu) / sd
        feats.append(F); labs += [c] * len(F)
    F = np.vstack(feats); labs = np.array(labs)
    # shared 2D PCA for display (fit on the trio)
    Fc = F - F.mean(0); U, S, Vt = np.linalg.svd(Fc, full_matrices=False); ax2 = Vt[:2]
    P2 = (F - F.mean(0)) @ ax2.T
    sc = np.abs(P2).max(); P2 /= sc
    PER = 12
    # real Lloyd per class, capturing centroids each iteration (in 343-d, projected for display)
    rng = np.random.RandomState(0); ITER = 9
    history = []                                        # list of [centroids2d, assign] per frame
    cent343 = {}; assign = {}
    for c in trio:
        Fc_ = F[labs == c]; init = Fc_[rng.choice(len(Fc_), PER, replace=False)]
        cent343[c] = init.copy()
    snaps = []
    for it in range(ITER):
        snap = {}
        for c in trio:
            Fc_ = F[labs == c]; C = cent343[c]
            d = ((Fc_[:, None] - C[None]) ** 2).sum(2); a = d.argmin(1)
            cen2 = ((C - F.mean(0)) @ ax2.T) / sc
            snap[c] = (cen2.copy(), a.copy())
            newC = np.array([Fc_[a == k].mean(0) if (a == k).any() else C[k] for k in range(PER)])
            cent343[c] = newC
        snaps.append(snap)
    frames = []
    HOLDN = 4
    for it in range(ITER):
        for rep in range(HOLDN if it in (0, ITER - 1) else 2):
            fig = plt.figure(figsize=(6.6, 5.6), dpi=110, facecolor=BG)
            fig.text(0.5, 0.95, 'Placing the prototypes with k-means (no gradients)', ha='center', fontsize=13.5, weight='bold')
            fig.text(0.5, 0.905, 'the one step that looks like learning: clustering moves each prototype to the centre of nearby features',
                     ha='center', fontsize=8.8, color=MUTED)
            ax = fig.add_axes([0.05, 0.06, 0.90, 0.80]); ax.set_facecolor(PANEL)
            ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_xticks([]); ax.set_yticks([])
            for ci, c in enumerate(trio):
                m = labs == c
                ax.scatter(P2[m, 0], P2[m, 1], s=15, color=cols[ci], alpha=0.42, linewidths=0)
                cen2, _ = snaps[it][c]
                ax.scatter(cen2[:, 0], cen2[:, 1], s=95, marker='X', color=cols[ci],
                           edgecolors='white', linewidths=1.3, zorder=4)
            ax.text(0.03, 0.95, f'k-means iteration {it + 1}/{ITER}', transform=ax.transAxes,
                    ha='left', fontsize=10.5, color=INK, weight='bold')
            for ci, c in enumerate(trio):
                ax.scatter([], [], marker='X', color=cols[ci], s=70, label=CLASSES[c], edgecolors='white')
            ax.legend(loc='lower right', fontsize=8.5, facecolor=PANEL, edgecolor=LINE, labelcolor=INK)
            for s in ax.spines.values(): s.set_color(LINE)
            frames.append(fig_rgba(fig))
    save_gif(PUB / 'handbuilt-kmeans.gif', frames, fps=4, hold=8)


if __name__ == '__main__':
    png_orient_rose()          # static: a fixed histogram, not a process
    png_coarsegrain()          # static: a fixed pooling, not a process
    png_yat_well()             # static: fixed per-class scores, not a process
    png_sobel_gradient()       # static: a fixed convolution, not a process
    png_kernel_shape()         # static: a math plot at the placed ε
    gif_kmeans_prototypes()    # KEEP: real Lloyd iterations, prototypes migrate
    print('HANDBUILT_GIFS_DONE')
