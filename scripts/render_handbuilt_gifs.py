"""Three teaching GIFs for the hand-built JAX/Flax NNX companion.

Each animates a *process*, not a slideshow, and reuses the series' one physical
world (the Yat denominator is a softened inverse-square well; classification is
falling into the deepest basin).

  1. handbuilt-orient-rose.gif   edges peel off the image and sort, by angle,
                                 into six orientation bins (a filling rose).
  2. handbuilt-coarsegrain.gif   the full-resolution detector maps cool into a
                                 7x7 block grid, then flatten into the 343 vector.
  3. handbuilt-yat-well.gif      a test garment's feature-particle rolls down the
                                 Yat potential into the nearest prototype well.

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
    Image.fromarray(frames[-1]).save(str(path).replace('.gif', '-preview.png'))
    frames = frames + [frames[-1]] * hold
    imageio.mimsave(path, frames, duration=1 / fps, loop=0, palettesize=128, subrectangles=True)
    print(f'wrote {path} ({os.path.getsize(path)//1024} KB)')


# ═══════════════════════════════════════════════════════════════════════════
# GIF 1 — orientation rose: edges sort by angle into six bins
# ═══════════════════════════════════════════════════════════════════════════
def gif_orient_rose():
    img = grab(8, 0)                                   # a Bag: clean strong edges
    gx, gy, mag, ang, chans, pooled = pipeline(img)
    # pick the strongest edge pixels as the migrating particles
    thr = np.quantile(mag, 0.985)
    ys, xs = np.where(mag >= thr)
    order = np.argsort(-mag[ys, xs])[:90]
    ys, xs = ys[order], xs[order]
    a = ang[ys, xs]; m = mag[ys, xs]
    binof = np.clip((a / (np.pi / NB)).astype(int), 0, NB - 1)
    # image coords -> left-panel axes coords (0..1), y flipped
    sx0 = xs / 27.0; sy0 = 1 - ys / 27.0
    # target: stack inside each bin's wedge on the right rose, at increasing radius
    rose_cx, rose_cy = 0.5, 0.5
    counts = np.zeros(NB, int)
    tx = np.zeros(len(a)); ty = np.zeros(len(a)); tr = np.zeros(len(a)); tth = np.zeros(len(a))
    for i in range(len(a)):
        b = binof[i]; rank = counts[b]; counts[b] += 1
        th = CENTERS[b] + (np.pi / NB) * 0.5                # wedge centre angle (0..pi)
        rad = 0.10 + 0.030 * rank                           # stack outward
        # mirror to full circle for a symmetric rose (edges are undirected)
        tth[i] = th; tr[i] = rad
        tx[i] = rose_cx + rad * np.cos(th); ty[i] = rose_cy + rad * np.sin(th)

    N_FLY, N_GROW, N_INTRO = 30, 18, 8
    total = N_INTRO + N_FLY + N_GROW
    frames = []
    for fi in range(total):
        fig = plt.figure(figsize=(8.6, 4.5), dpi=110, facecolor=BG)
        fig.text(0.5, 0.93, 'A hand-built detector: sorting edges by their orientation',
                 ha='center', fontsize=14, weight='bold')
        fig.text(0.5, 0.865, 'every strong edge in the picture flies to the bin nearest its angle; the rose is the garment’s edge signature',
                 ha='center', fontsize=9.5, color=MUTED)
        axL = fig.add_axes([0.04, 0.08, 0.40, 0.74]); axL.set_facecolor(PANEL)
        axR = fig.add_axes([0.52, 0.08, 0.44, 0.74]); axR.set_facecolor(PANEL); axR.set_aspect('equal')
        axL.imshow(img, cmap='gray', extent=[0, 1, 0, 1], vmin=0, vmax=1, alpha=0.62, zorder=0)
        axL.set_xlim(0, 1); axL.set_ylim(0, 1); axL.set_xticks([]); axL.set_yticks([])
        axL.set_title('the picture + its edges', fontsize=10, color=MUTED, pad=4)
        axR.set_xlim(0, 1); axR.set_ylim(0, 1); axR.set_xticks([]); axR.set_yticks([])
        axR.set_title('orientation rose (6 bins)', fontsize=10, color=MUTED, pad=4)
        for s in list(axL.spines.values()) + list(axR.spines.values()): s.set_color(LINE)

        # rose guide wedges + tick labels
        for b in range(NB):
            th0 = np.degrees(CENTERS[b]); th1 = np.degrees(CENTERS[b] + np.pi / NB)
            axR.add_patch(Wedge((rose_cx, rose_cy), 0.40, th0, th1, width=0.40,
                                facecolor=CHAN_COL[b], alpha=0.07, edgecolor=LINE, lw=0.5, zorder=0))
            lab_th = CENTERS[b] + np.pi / (2 * NB)
            axR.text(rose_cx + 0.45 * np.cos(lab_th), rose_cy + 0.45 * np.sin(lab_th),
                     f'{int(round(np.degrees(CENTERS[b])))}°', ha='center', va='center',
                     fontsize=8, color=CHAN_COL[b])

        # phase progress
        if fi < N_INTRO:
            pf = 0.0; gf = 0.0
        elif fi < N_INTRO + N_FLY:
            pf = ease((fi - N_INTRO) / N_FLY); gf = 0.0
        else:
            pf = 1.0; gf = ease((fi - N_INTRO - N_FLY) / N_GROW)

        # draw migrating edge segments (left -> right), colored by bin
        for i in range(len(a)):
            # stagger departures across the fly phase for a flowing look
            local = np.clip(pf * 1.6 - (i / len(a)) * 0.6, 0, 1)
            local = ease(local)
            cx = sx0[i] + (tx[i] - sx0[i]) * local
            cy = sy0[i] + (ty[i] - sy0[i]) * local
            col = CHAN_COL[binof[i]]
            if local < 0.5:                                  # still an oriented edge stroke
                dx = 0.025 * np.cos(a[i]); dy = 0.025 * np.sin(a[i])
                ax = axL if local < 0.04 else axR
                # draw on the panel it currently sits in (approx: left until it crosses)
                axR.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color=col,
                         lw=1.3, alpha=0.30 + 0.5 * local, solid_capstyle='round', zorder=3)
            else:
                axR.scatter([cx], [cy], s=12, color=col, alpha=0.5 * (1 - gf), linewidths=0, zorder=1)
        # also keep the edge strokes on the left image until they fly
        for i in range(len(a)):
            if pf < 0.5:
                dx = 0.02 * np.cos(a[i]); dy = 0.02 * np.sin(a[i])
                axL.plot([sx0[i] - dx, sx0[i] + dx], [sy0[i] - dy, sy0[i] + dy],
                         color=CHAN_COL[binof[i]], lw=1.2, alpha=0.85 * (1 - pf),
                         solid_capstyle='round', zorder=3)

        # grown petals (final histogram) once edges have landed
        if gf > 0:
            tot = np.array([(binof == b).sum() for b in range(NB)], float)
            tot = tot / (tot.max() + 1e-9)
            for b in range(NB):
                th0 = np.degrees(CENTERS[b]); th1 = np.degrees(CENTERS[b] + np.pi / NB)
                axR.add_patch(Wedge((rose_cx, rose_cy), 0.10 + 0.30 * tot[b] * gf, th0, th1,
                                    facecolor=CHAN_COL[b], alpha=0.55, edgecolor=INK, lw=0.6, zorder=2))
        frames.append(fig_rgba(fig))
    save_gif(PUB / 'handbuilt-orient-rose.gif', frames, fps=11)


# ═══════════════════════════════════════════════════════════════════════════
# GIF 2 — coarse-graining: full maps cool into a 7x7 grid, flatten to 343
# ═══════════════════════════════════════════════════════════════════════════
def gif_coarsegrain():
    img = grab(0, 1)                                   # a T-shirt
    gx, gy, mag, ang, chans, pooled = pipeline(img)
    ps = 28 // GRID                                    # 4
    # per-channel normalised full map and a block-expanded pooled map
    full = chans[:, :GRID * ps, :GRID * ps]
    full = full / (full.reshape(7, -1).max(1)[:, None, None] + 1e-9)
    poolN = pooled / (pooled.reshape(7, -1).max(1)[:, None, None] + 1e-9)
    block = np.repeat(np.repeat(poolN, ps, 1), ps, 2)  # [7,28,28] blocky
    names = HB['detectors']

    N_COOL, N_FLAT = 26, 22
    total = 8 + N_COOL + N_FLAT
    frames = []
    for fi in range(total):
        fig = plt.figure(figsize=(8.8, 4.6), dpi=110, facecolor=BG)
        fig.text(0.5, 0.93, 'Coarse-graining: from full-resolution maps to 343 named numbers',
                 ha='center', fontsize=14, weight='bold')
        fig.text(0.5, 0.865, 'each detector map cools into a 7×7 grid of patch averages; the 7×7×7 cells flatten into one feature vector',
                 ha='center', fontsize=9.5, color=MUTED)
        if fi < 8:
            cool = 0.0; flat = 0.0
        elif fi < 8 + N_COOL:
            cool = ease((fi - 8) / N_COOL); flat = 0.0
        else:
            cool = 1.0; flat = ease((fi - 8 - N_COOL) / N_FLAT)

        # top row: the 7 maps interpolating full -> blocky
        cur = (1 - cool) * full + cool * block
        for ci in range(7):
            ax = fig.add_axes([0.045 + ci * 0.133, 0.46, 0.118, 0.30]); ax.set_facecolor(PANEL)
            ax.imshow(cur[ci], cmap='magma', vmin=0, vmax=1, interpolation='nearest' if cool > 0.5 else 'bilinear')
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_color(CHAN_COL[ci]); s.set_linewidth(1.4)
            ax.set_title(names[ci], fontsize=7.5, color=CHAN_COL[ci], pad=2)
            if cool > 0.25:                                 # grid lines appear as it cools
                for g in range(1, GRID):
                    ax.axhline(g * ps - 0.5, color=BG, lw=0.6, alpha=cool)
                    ax.axvline(g * ps - 0.5, color=BG, lw=0.6, alpha=cool)

        # bottom: the 343-vector bar, filling in as cells flatten
        axv = fig.add_axes([0.045, 0.10, 0.91, 0.22]); axv.set_facecolor(PANEL)
        axv.set_xlim(0, 343); axv.set_ylim(0, 1); axv.set_yticks([])
        vals = poolN.reshape(7, GRID * GRID)               # [7,49]
        nshow = int(flat * 343)
        for ci in range(7):
            base = ci * 49
            xs = np.arange(49)
            show = np.clip(nshow - base, 0, 49)
            if show > 0:
                axv.bar(base + xs[:show], vals[ci, :show], width=1.0,
                        color=CHAN_COL[ci], alpha=0.9, align='edge')
        for ci in range(1, 7):
            axv.axvline(ci * 49, color=LINE, lw=0.8)
        axv.set_xticks([24 + ci * 49 for ci in range(7)])
        axv.set_xticklabels(names, fontsize=7.5, color=MUTED)
        axv.set_title('the feature vector φ(x): 343 numbers, one per (detector, patch)',
                      fontsize=9.5, color=MUTED, pad=3)
        for s in axv.spines.values(): s.set_color(LINE)
        frames.append(fig_rgba(fig))
    save_gif(PUB / 'handbuilt-coarsegrain.gif', frames, fps=12)


# ═══════════════════════════════════════════════════════════════════════════
# GIF 3 — the Yat well: ten wells along the class axis, each well's depth is the
#          garment's REAL Yat score for that class. The feature vector already
#          fixes all ten depths, so the garment sits in its deepest well from the
#          start (no fake trajectory). The only motion is the real scores forming.
# ═══════════════════════════════════════════════════════════════════════════
def gif_yat_well():
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

    WSIG = 0.34                                         # half-width of a drawn well
    xs = np.arange(10.0)                                # one well per class on the x-axis
    xgrid = np.linspace(-0.7, 9.7, 600)

    def profile(depth):                                 # smooth multi-well potential V(x) <= 0
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

    GROW, HOLD = 22, 12
    frames = []
    for di, (si, s, pred) in enumerate(chosen):
        # real per-class scores -> relative well depths (ordering and gaps are real)
        d = (s - s.min()) / (s.max() - s.min() + 1e-9)
        depthF = 0.12 + 0.88 * d                        # floor so the shallow wells stay visible
        img = grab(int(lab[si]), di)
        for fi in range(GROW + HOLD):
            g = ease(min(fi, GROW) / GROW)              # scores resolve from a shallow baseline
            depth = (0.12 + (depthF - 0.12) * g)
            V = profile(depth)
            fig = plt.figure(figsize=(7.8, 5.2), dpi=110, facecolor=BG)
            fig.text(0.5, 0.95, 'The head is a landscape of ten Yat wells', ha='center', fontsize=14, weight='bold')
            fig.text(0.5, 0.905, 'each well’s depth is this garment’s real Yat score for that class; its features already place it in the deepest one',
                     ha='center', fontsize=8.6, color=MUTED)
            ax = fig.add_axes([0.07, 0.16, 0.74, 0.70]); ax.set_facecolor(PANEL)
            ax.set_xlim(-0.7, 9.7); ax.set_ylim(-1.18, 0.16)
            ax.plot(xgrid, V, color=MUTED, lw=1.6, zorder=2)
            ax.fill_between(xgrid, V, 0.16, color='#000000', alpha=0.0)
            ax.fill_between(xgrid, V, -1.18, color='#1d1a14', zorder=1)
            ax.axhline(0, color=LINE, lw=0.8)
            for c in range(10):
                win = c == pred
                ax.scatter([xs[c]], [-depth[c]], s=120 if win else 48,
                           color=CLASS_COL[c], edgecolors='white' if win else BG,
                           linewidths=1.6 if win else 0.6, zorder=4)
            # the garment: a single point that already sits at the bottom of its deepest well
            ax.scatter([xs[pred]], [-depth[pred]], s=210, marker='*', color='white',
                       edgecolors=BG, linewidths=1.4, zorder=5)
            ax.set_xticks(xs); ax.set_xticklabels(CLASSES, rotation=40, ha='right', fontsize=8, color=MUTED)
            ax.set_yticks([]); ax.set_ylabel('Yat score (well depth)', color=MUTED, fontsize=9)
            for sp in ax.spines.values(): sp.set_color(LINE)
            # garment thumbnail
            axt = fig.add_axes([0.83, 0.55, 0.14, 0.21]); axt.imshow(img, cmap='gray', vmin=0, vmax=1)
            axt.set_xticks([]); axt.set_yticks([])
            for sp in axt.spines.values(): sp.set_color(LINE)
            axt.set_title(f'true:\n{CLASSES[int(lab[si])]}', fontsize=8.5, color=INK, pad=2)
            ok = pred == lab[si]
            msg = f'deepest well: {CLASSES[pred]}  ' + ('✓ correct' if ok else '✗ a miss (the honest 83.3%)')
            fig.text(0.5, 0.03, msg, ha='center', fontsize=11.5,
                     color='#7bbf5a' if ok else '#e08a6a', weight='bold')
            frames.append(fig_rgba(fig))
    save_gif(PUB / 'handbuilt-yat-well.gif', frames, fps=13, hold=12)


# ═══════════════════════════════════════════════════════════════════════════
# GIF 4 — the Sobel convolution: a real 3x3 kernel sweeps the image and builds
#          the gradient field (gx, gy -> magnitude + angle). All real convolution.
# ═══════════════════════════════════════════════════════════════════════════
def gif_sobel_gradient():
    img = grab(0, 2)                                    # a T-shirt
    gx = convolve(img, SX, mode='nearest'); gy = convolve(img, SY, mode='nearest')
    mag = np.sqrt(gx ** 2 + gy ** 2); ang = np.mod(np.arctan2(gy, gx), np.pi)
    H = 28
    # sweep order: row-major kernel positions
    order = [(r, c) for r in range(H) for c in range(H)]
    NSTEP = 40                                          # reveal in chunks
    frames = []
    # arrow subsample for the field
    step = 2
    for fi in range(NSTEP + 14):
        prog = ease(min(fi, NSTEP) / NSTEP)
        nrev = int(prog * len(order))
        revealed = np.zeros((H, H), bool)
        for (r, c) in order[:nrev]: revealed[r, c] = True
        fig = plt.figure(figsize=(8.8, 3.6), dpi=110, facecolor=BG)
        fig.text(0.5, 0.92, 'A hand-built detector starts as a convolution', ha='center', fontsize=14, weight='bold')
        fig.text(0.5, 0.83, 'two Sobel kernels sweep the image to give the gradient gₓ, g_y at every pixel, then its magnitude and angle',
                 ha='center', fontsize=9, color=MUTED)
        titles = ['image', 'gₓ (Sobel-x)', 'g_y (Sobel-y)', 'gradient field']
        for pi in range(4):
            ax = fig.add_axes([0.035 + pi * 0.245, 0.10, 0.20, 0.62]); ax.set_facecolor(PANEL)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_color(LINE)
            ax.set_title(titles[pi], fontsize=9, color=MUTED, pad=3)
            if pi == 0:
                ax.imshow(img, cmap='gray', vmin=0, vmax=1)
                # the sweeping 3x3 kernel window
                if nrev < len(order):
                    r, c = order[min(nrev, len(order) - 1)]
                    ax.add_patch(plt.Rectangle((c - 1.5, r - 1.5), 3, 3, fill=False, ec='#36d6c4', lw=1.6))
            elif pi == 1:
                ax.imshow(np.where(revealed, gx, np.nan), cmap='coolwarm', vmin=-4, vmax=4)
            elif pi == 2:
                ax.imshow(np.where(revealed, gy, np.nan), cmap='coolwarm', vmin=-4, vmax=4)
            else:
                ax.imshow(np.where(revealed, mag, np.nan), cmap='magma', vmin=0, vmax=4)
                ys, xs = np.mgrid[0:H:step, 0:H:step]
                u = np.cos(ang[::step, ::step]); v = -np.sin(ang[::step, ::step])
                m = (mag[::step, ::step] > 0.8) & revealed[::step, ::step]
                ax.quiver(xs[m], ys[m], u[m], v[m], color='#36d6c4', scale=34, width=0.006, alpha=0.8)
                ax.set_xlim(-0.5, H - 0.5); ax.set_ylim(H - 0.5, -0.5)
        frames.append(fig_rgba(fig))
    save_gif(PUB / 'handbuilt-sobel.gif', frames, fps=13)


# ═══════════════════════════════════════════════════════════════════════════
# GIF 5 — the Yat kernel itself: a softened inverse-square well, bounded and
#          peaked, against a Gaussian and an unbounded dot product. Real formula.
# ═══════════════════════════════════════════════════════════════════════════
def gif_kernel_shape():
    B = 0.5
    mu = np.array([1.0, 0.0])                           # a prototype off-origin
    gN = 220
    gv = np.linspace(-1.2, 2.6, gN); gw = np.linspace(-1.9, 1.9, gN)
    GX, GY = np.meshgrid(gv, gw); P = np.stack([GX.ravel(), GY.ravel()], 1)
    dot = P @ mu + B; dist2 = ((P - mu) ** 2).sum(1)
    # 1D cross-section PERPENDICULAR to mu (x = mu_x): z.mu is constant, so this
    # isolates the distance term and the kernel is a clean radial well.
    t = np.linspace(-1.9, 1.9, 400)
    cdot = (mu @ mu) + B                                # constant along the perpendicular line
    EPS_SEQ = np.geomspace(0.9, 0.05, 19)              # softening shrinks
    frames = []
    for fi, eps in enumerate(EPS_SEQ):
        yat = (dot ** 2 / (dist2 + eps)).reshape(gN, gN)
        lyat = cdot ** 2 / (t ** 2 + eps)                          # softened inverse-square
        linv = cdot ** 2 / (t ** 2 + 1e-3)                         # true inverse-square (blows up)
        lrbf = (cdot ** 2 / eps) * np.exp(-t ** 2 / (2 * (eps ** 0.5) ** 2))   # matched-peak Gaussian
        ymax = cdot ** 2 / eps * 1.15
        fig = plt.figure(figsize=(8.6, 4.2), dpi=98, facecolor=BG)
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
        axC.set_title(f'ε = {eps:.2f}   (softening knob)', fontsize=9, color=INK, pad=3)
        for s in axC.spines.values(): s.set_color(LINE)
        frames.append(fig_rgba(fig))
    frames = frames + frames[::-1]                     # breathe in and out
    save_gif(PUB / 'handbuilt-kernel-shape.gif', frames, fps=16, hold=2)


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
    gif_orient_rose()
    gif_coarsegrain()
    gif_yat_well()
    gif_sobel_gradient()
    gif_kernel_shape()
    gif_kmeans_prototypes()
    print('HANDBUILT_GIFS_DONE')
