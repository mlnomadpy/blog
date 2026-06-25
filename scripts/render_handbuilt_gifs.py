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
# GIF 3 — the Yat well: ten basins, the deepest set by the real per-class score,
#          and the garment's feature-particle rolls into it
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

    # ten wells on a ring; their DEPTH is the real per-class score for this garment
    ang = np.linspace(0, 2 * np.pi, 10, endpoint=False) - np.pi / 2
    WELL = np.stack([0.78 * np.cos(ang), 0.78 * np.sin(ang)], 1)   # [10,2] ring
    SIG = 0.30

    def field(p, depth):                                # smooth potential, deeper = more negative
        d2 = ((WELL - p) ** 2).sum(1)
        return -(depth * np.exp(-d2 / (2 * SIG ** 2))).sum()

    def grad(p, depth):
        e = 1e-3
        gx = (field(p + np.array([e, 0]), depth) - field(p - np.array([e, 0]), depth)) / (2 * e)
        gy = (field(p + np.array([0, e]), depth) - field(p - np.array([0, e]), depth)) / (2 * e)
        return np.array([gx, gy])

    gN = 150
    gxv = np.linspace(-1.25, 1.25, gN); gyv = np.linspace(-1.25, 1.25, gN)
    GXX, GYY = np.meshgrid(gxv, gyv); flat = np.stack([GXX.ravel(), GYY.ravel()], 1)

    # choose demos: one per class, mostly hits + keep an honest miss, deterministic
    demos = []; used = set()
    for i in range(len(samples)):
        if lab[i] in used: continue
        used.add(lab[i])
        s = class_scores(Z[i]); pred = int(np.argmax(s))
        demos.append((i, s, pred));
        if len(demos) >= 12: break
    hits = [d for d in demos if d[2] == d[0] and lab[d[0]] == d[2]]
    hits = [d for d in demos if d[2] == lab[d[0]]]
    miss = [d for d in demos if d[2] != lab[d[0]]]
    chosen = hits[:3] + (miss[:1] if miss else [])     # 3 clean hits, 1 honest miss

    FRAMES_PER, HOLD_BETWEEN = 34, 8
    frames = []
    for di, (si, s, pred) in enumerate(chosen):
        depth = (s - s.min()) / (s.max() - s.min() + 1e-9)   # 0..1 relative well depth
        depth = 0.25 + 0.95 * depth                           # floor so all wells visible
        Vfield = np.array([field(p, depth) for p in flat]).reshape(gN, gN)
        # roll the particle from the centre into the deepest basin
        p = np.array([0.02, 0.04]); v = np.zeros(2); traj = [p.copy()]
        for _ in range(80):
            g = grad(p, depth); v = 0.84 * v - 0.012 * g; p = p + v; traj.append(p.copy())
        traj = np.array(traj)
        img = grab(int(lab[si]), di)
        for fi in range(FRAMES_PER + HOLD_BETWEEN):
            t = min(fi, FRAMES_PER - 1); k = int(ease(t / (FRAMES_PER - 1)) * (len(traj) - 1))
            here = traj[k]
            fig = plt.figure(figsize=(7.4, 5.6), dpi=110, facecolor=BG)
            fig.text(0.5, 0.95, 'Classification as falling into the deepest Yat well',
                     ha='center', fontsize=14, weight='bold')
            fig.text(0.5, 0.905, 'each class digs a basin whose depth is its real Yat score for this garment; the feature-particle rolls into the deepest',
                     ha='center', fontsize=8.8, color=MUTED)
            ax = fig.add_axes([0.06, 0.05, 0.88, 0.82]); ax.set_facecolor(PANEL)
            ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
            ax.contourf(GXX, GYY, Vfield, levels=26, cmap='magma', alpha=0.95)
            # label each well with its class, brighten the winner
            for c in range(10):
                win = c == pred
                ax.scatter([WELL[c, 0]], [WELL[c, 1]], s=70 if win else 34,
                           color=CLASS_COL[c], edgecolors='white' if win else BG,
                           linewidths=1.4 if win else 0.5, zorder=3)
                lx, ly = WELL[c] * 1.30
                ax.text(lx, ly, CLASSES[c], ha='center', va='center', fontsize=8.5,
                        color=CLASS_COL[c], weight='bold' if win else 'normal')
            tr = traj[:k + 1]
            ax.plot(tr[:, 0], tr[:, 1], color='#36d6c4', lw=2.2, alpha=0.9, zorder=4)
            ax.scatter([here[0]], [here[1]], s=170, color='white', edgecolors=BG,
                       linewidths=1.8, zorder=5)
            axt = fig.add_axes([0.075, 0.69, 0.155, 0.155]); axt.imshow(img, cmap='gray', vmin=0, vmax=1)
            axt.set_xticks([]); axt.set_yticks([])
            for sp in axt.spines.values(): sp.set_color(LINE)
            axt.set_title(f'true: {CLASSES[int(lab[si])]}', fontsize=8.5, color=INK, pad=2)
            done = k >= len(traj) - 1
            if done:
                ok = pred == lab[si]
                msg = f'fell into: {CLASSES[pred]}  ' + ('✓ correct' if ok else '✗ a miss (the honest 83.3%)')
                fig.text(0.5, 0.015, msg, ha='center', fontsize=11.5,
                         color='#7bbf5a' if ok else '#e08a6a', weight='bold')
            else:
                fig.text(0.5, 0.015, 'rolling downhill…', ha='center', fontsize=10, color=MUTED)
            for sp in ax.spines.values(): sp.set_color(LINE)
            frames.append(fig_rgba(fig))
    save_gif(PUB / 'handbuilt-yat-well.gif', frames, fps=15, hold=10)


if __name__ == '__main__':
    gif_orient_rose()
    gif_coarsegrain()
    gif_yat_well()
    print('HANDBUILT_GIFS_DONE')
