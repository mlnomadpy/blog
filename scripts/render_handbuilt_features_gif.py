"""GIF for the hand-built-features JAX companion: a few garments, each shown with
its seven hand-built detector maps and the Yat-kernel vote. Reuses the dumped
prototypes in public/handbuilt/handbuilt.json so it is fast and exactly matches
the post. Writes public/handbuilt-detectors.gif.

Run: python scripts/render_handbuilt_features_gif.py
"""
import warnings; warnings.filterwarnings('ignore')
import json, numpy as np, torchvision
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.animation as animation
from scipy.ndimage import convolve

D = json.load(open('public/handbuilt/handbuilt.json'))
CLS, NB, GRID, NCH = D['classes'], D['NB'], D['GRID'], D['nChan']
W = np.array(D['protos'], 'float32'); vote = np.array(D['vote']); MU = np.array(D['mu']); SD = np.array(D['sd'])
B, EPS = D['B'], D['eps']
SX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 'float32'); SY = SX.T
CENTERS = np.linspace(0, np.pi, NB, endpoint=False)
COL = ['#c2553a', '#c77d2a', '#5a7d3a', '#2f8f8f', '#4a7fb3', '#7a5fc0', '#a06a2a']
NAMES = [f'{round(a*180/np.pi)}°' for a in CENTERS] + ['corner']

te = torchvision.datasets.FashionMNIST('/tmp/fmnist', train=False, download=True)
Xte = te.data.numpy().astype('float32') / 255.0; yte = te.targets.numpy()


def channel_maps(im):
    gx, gy = convolve(im, SX, mode='nearest'), convolve(im, SY, mode='nearest')
    mag = np.sqrt(gx ** 2 + gy ** 2) + 1e-6; ang = np.arctan2(gy, gx) % np.pi
    out = []
    for b in range(NB):
        d = np.abs(((ang - CENTERS[b] + np.pi / 2) % np.pi) - np.pi / 2)
        out.append(np.clip(1 - d / (np.pi / NB), 0, 1) * mag)
    out.append(np.abs(gx) * np.abs(gy)); return np.stack(out)


def feat_and_pred(im):
    m = channel_maps(im); ps = 28 // GRID
    pooled = m[:, :GRID * ps, :GRID * ps].reshape(NCH, GRID, ps, GRID, ps).mean((2, 4)).reshape(-1)
    v = pooled / (np.linalg.norm(pooled) + 1e-6); z = (v - MU) / SD
    dot = W @ z; d2 = (z ** 2).sum() + (W ** 2).sum(1) - 2 * dot
    ker = (dot + B) ** 2 / (d2 + EPS)
    score = np.array([ker[vote == c].max() for c in range(10)])
    return m, int(score.argmax())


# one clean garment per class, in class order
rng = np.random.RandomState(2)
sel = [np.where(yte == c)[0][rng.randint(50)] for c in range(10)]
frames_data = []
for i in sel:
    m, pred = feat_and_pred(Xte[i])
    frames_data.append((Xte[i], m, int(yte[i]), pred))

plt.rcParams.update({'figure.facecolor': '#0e0d0b', 'savefig.facecolor': '#0e0d0b', 'text.color': '#e8e2d4'})
fig = plt.figure(figsize=(7.6, 2.5), dpi=128)
gs = fig.add_gridspec(1, NCH + 2, width_ratios=[1.25, 0.25] + [1] * NCH, wspace=0.12)
ax_src = fig.add_subplot(gs[0, 0]); ax_maps = [fig.add_subplot(gs[0, 2 + k]) for k in range(NCH)]
for a in [ax_src] + ax_maps: a.set_xticks([]); a.set_yticks([])


def tint(gray, hexc):
    c = np.array([int(hexc[1:3], 16), int(hexc[3:5], 16), int(hexc[5:7], 16)]) / 255
    bg = np.array([24, 22, 20]) / 255; t = (gray / (gray.max() + 1e-6)) ** 0.6
    return bg[None, None] + (c - bg)[None, None] * t[..., None]


def draw(fi):
    src, m, true_c, pred = frames_data[fi % len(frames_data)]
    ax_src.clear(); ax_src.set_xticks([]); ax_src.set_yticks([])
    ax_src.imshow(src, cmap='gray', vmin=0, vmax=1)
    ok = pred == true_c
    ax_src.set_title(f'vote: {CLS[pred]}  {"✓" if ok else "(true " + CLS[true_c] + ")"}',
                     fontsize=11, color=('#5a9d5a' if ok else '#c2553a'), pad=6)
    ax_src.set_xlabel(CLS[true_c], fontsize=9, color='#9c968a')
    for k, a in enumerate(ax_maps):
        a.clear(); a.set_xticks([]); a.set_yticks([])
        a.imshow(tint(m[k], COL[k])); a.set_xlabel(NAMES[k], fontsize=8.5, color=COL[k])
        for s in a.spines.values(): s.set_color(COL[k]); s.set_linewidth(1.3)
    fig.suptitle('every garment as seven hand-built detector maps, then a Yat-kernel vote (nothing trained)',
                 fontsize=10.5, color='#9c968a', y=0.99)
    return []


# 3 duplicate frames per garment for a slow, readable pace
order = [g for g in range(len(frames_data)) for _ in range(3)]
ani = animation.FuncAnimation(fig, lambda i: draw(order[i]), frames=len(order), interval=470, blit=False)
fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.14)
ani.save('public/handbuilt-detectors.gif', writer='pillow', fps=2.1)
import os; print(f"wrote public/handbuilt-detectors.gif ({os.path.getsize('public/handbuilt-detectors.gif')//1024} KB)")
