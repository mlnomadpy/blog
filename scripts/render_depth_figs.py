"""Six static teaching figures for the depth-by-construction JAX/Flax NNX companion.

This post is a one-shot construction with NO temporal process: the layer-1 and
layer-2 feature maps are deterministic (no training), the accuracy ladder is a
final result, the vocabulary wall is a final scatter, and the synonym Gram is a
final correlation matrix. So each point is drawn as the clearest single-frame
STATIC figure, not a chart disguised as an animation. Every number comes from
public/handbuilt-depth/depth.json (the verified run of scripts/handbuilt_depth.py)
and every feature map is the real deterministic layer-1/layer-2 computation
re-derived locally.

  1. depth-junction-assembly.png  3-panel: edge map A, edge map B, and their
                                  pointwise min-AND junction, on a real garment.
  2. depth-vocab-wall.png         static scatter of accuracy vs vocabulary size
                                  with the 224 and 4,630 combinatorial walls marked.
  3. depth-ladder.png             static horizontal bar chart: built rungs near 83
                                  (blue), trained rungs (orange), zoomed to the
                                  photo finish.
  4. depth-layer2-alone.png       2-panel: the six layer-1 edge maps vs the
                                  hand-built layer-2 relation maps, with the 78.8%
                                  layer-2-alone number stated.
  5. depth-synonym-gram.png       static layer-2 x layer-1 correlation matrix with
                                  the continuation diagonal highlighted.
  6. depth-six-designs.png        static bar chart of the six design iterations,
                                  each near 83, best highlighted.

Run: python scripts/render_depth_figs.py
"""
import warnings; warnings.filterwarnings('ignore')
import json, os, math
from pathlib import Path
import numpy as np
from scipy.ndimage import convolve
import torchvision
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / 'public'
D = json.load(open(PUB / 'handbuilt-depth' / 'depth.json'))

# palette (matches the depth viz components)
BG = '#0e0d0b'; PANEL = '#16140f'; INK = '#e8e2d4'; MUTED = '#9a9282'; LINE = '#3a352c'
BUILT = '#4a7fb3'; TRAINED = '#c77d2a'; ACCENT = '#36d6c4'
CHAN_COL = ['#c2553a', '#c77d2a', '#5a7d3a', '#2f8f8f', '#4a7fb3', '#7a5fc0', '#a06a2a']
CLASSES = D['classes']
plt.rcParams.update({'figure.facecolor': BG, 'savefig.facecolor': BG, 'text.color': INK,
                     'axes.edgecolor': LINE, 'font.size': 11})

NB = D['NB']; NCH = D['nChan']; CELL = D['CELL']
CENTERS = np.linspace(0, np.pi, NB, endpoint=False)
DEG = D['centersDeg']
CONTOUR = [(-math.sin(a), math.cos(a)) for a in CENTERS]
SX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32); SY = SX.T


# layer-1 maps (verbatim recipe)
def channel_maps(img):
    gx = convolve(img, SX, mode='nearest'); gy = convolve(img, SY, mode='nearest')
    mag = np.sqrt(gx ** 2 + gy ** 2) + 1e-6
    ang = np.mod(np.arctan2(gy, gx), np.pi)
    out = []
    for b in range(NB):
        d = np.abs(np.mod(ang - CENTERS[b] + np.pi / 2, np.pi) - np.pi / 2)
        out.append(np.clip(1 - d / (np.pi / NB), 0, 1) * mag)
    out.append(np.abs(gx) * np.abs(gy))
    return np.stack(out)                                  # [7,28,28]


def cells(maps):
    return maps.reshape(NCH, CELL, 2, CELL, 2).mean((2, 4))   # [7,14,14]


te = torchvision.datasets.FashionMNIST('/tmp/fmnist', train=False, download=True)
TEX = te.data.numpy().astype(np.float32) / 255.0
TEY = te.targets.numpy()


def grab(label, k=0):
    idx = np.where(TEY == label)[0]
    return TEX[idx[k]]


def nrm(a):
    return a / (a.max() + 1e-9)


def _shift(A, dy, dx):
    y0, x0 = int(math.floor(dy)), int(math.floor(dx))
    out = np.zeros_like(A); h, w = A.shape
    ys, xs = max(0, -y0), max(0, -x0)
    ye, xe = min(h, h - y0), min(w, w - x0)
    if ye > ys and xe > xs:
        out[ys:ye, xs:xe] = A[ys + y0:ye + y0, xs + x0:xe + x0]
    return out


def _continuation(C, b):
    dx, dy = CONTOUR[b]
    e = C[b]
    return nrm(np.minimum(nrm(e), nrm(_shift(e, dy, dx))))


def save_fig(fig, name):
    p = PUB / name
    fig.savefig(str(p), dpi=110, facecolor=BG)
    plt.close(fig)
    print(f'wrote {p} ({os.path.getsize(p)//1024} KB)')


# ═══════════════════════════════════════════════════════════════════════════
# FIG 1 — junction assembly: edge map A, edge map B, their pointwise min-AND
# ═══════════════════════════════════════════════════════════════════════════
def fig_junction_assembly():
    img = grab(8, 3)                                      # a Bag: strong corners
    C = cells(channel_maps(img))
    b1, b2 = 1, 3                                         # 30 deg + 90 deg junction
    e1 = nrm(C[b1]); e2 = nrm(C[b2]); j = nrm(np.minimum(e1, e2))
    fig = plt.figure(figsize=(8.6, 3.9), dpi=110, facecolor=BG)
    fig.text(0.5, 0.94, 'A layer-2 junction is an AND of two layer-1 edges',
             ha='center', fontsize=14, weight='bold')
    fig.text(0.5, 0.865,
             f'the pointwise minimum of the {DEG[b1]}° and {DEG[b2]}° edge maps fires only where BOTH orientations are present: a corner',
             ha='center', fontsize=9, color=MUTED)
    # garment
    axg = fig.add_axes([0.02, 0.15, 0.19, 0.56]); axg.set_facecolor(PANEL)
    axg.imshow(img, cmap='gray', vmin=0, vmax=1); axg.set_xticks([]); axg.set_yticks([])
    axg.set_title('garment (a bag)', fontsize=8.5, color=MUTED, pad=3)
    for s in axg.spines.values(): s.set_color(LINE)
    # edge map 1
    ax1 = fig.add_axes([0.255, 0.15, 0.17, 0.56]); ax1.set_facecolor(PANEL)
    ax1.imshow(e1, cmap='magma', vmin=0, vmax=1); ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_title(f'edge {DEG[b1]}°', fontsize=9, color=CHAN_COL[b1], pad=3)
    for s in ax1.spines.values(): s.set_color(CHAN_COL[b1]); s.set_linewidth(1.4)
    # AND sign
    fig.text(0.44, 0.42, 'min', ha='center', va='center', fontsize=12, color=ACCENT, weight='bold')
    # edge map 2
    ax2 = fig.add_axes([0.47, 0.15, 0.17, 0.56]); ax2.set_facecolor(PANEL)
    ax2.imshow(e2, cmap='magma', vmin=0, vmax=1); ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title(f'edge {DEG[b2]}°', fontsize=9, color=CHAN_COL[b2], pad=3)
    for s in ax2.spines.values(): s.set_color(CHAN_COL[b2]); s.set_linewidth(1.4)
    # equals sign
    fig.text(0.665, 0.42, '=', ha='center', va='center', fontsize=15, color=ACCENT, weight='bold')
    # junction result
    axj = fig.add_axes([0.70, 0.15, 0.27, 0.56]); axj.set_facecolor(PANEL)
    axj.imshow(j, cmap='magma', vmin=0, vmax=1); axj.set_xticks([]); axj.set_yticks([])
    axj.set_title(f'{DEG[b1]}°+{DEG[b2]}° junction', fontsize=9.5, color=ACCENT, pad=3)
    for s in axj.spines.values(): s.set_color(ACCENT); s.set_linewidth(1.6)
    fig.text(0.5, 0.045, 'min(e₁, e₂): the cheapest honest AND of two non-negative energies, no parameter fit',
             ha='center', fontsize=8, color=MUTED)
    save_fig(fig, 'depth-junction-assembly.png')


# ═══════════════════════════════════════════════════════════════════════════
# FIG 2 — the vocabulary wall: static scatter of accuracy vs vocabulary size,
#          with the 224 / 4,630 combinatorial walls marked far to the right.
# ═══════════════════════════════════════════════════════════════════════════
def fig_vocab_wall():
    curve = D['vocab']['curve']
    names = [c['names'] for c in curve]
    accs = [c['acc'] for c in curve]
    labels = [c['stage'] for c in curve]
    pairs = D['vocab']['pairsPossible']; triples = D['vocab']['triplesPossible']
    trained = D['ladder']['trainedHead']
    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=110, facecolor=BG)
    ax.set_facecolor(PANEL)
    fig.suptitle('Naming does not run out. Knowing which names matter does.',
                 fontsize=13.5, weight='bold', y=0.97)
    ax.set_title('accuracy stays flat as the hand-built vocabulary grows; the design space runs to thousands',
                 fontsize=9, color=MUTED, pad=6)
    ax.set_xscale('log'); ax.set_xlim(5.5, 9000); ax.set_ylim(74, 87)
    ax.axhline(trained, color=TRAINED, lw=1.4, ls='--', alpha=0.9)
    ax.text(6.2, trained + 0.25, f'fully trained {trained:.1f}%', color=TRAINED, fontsize=8.5)
    ax.plot(names, accs, color=BUILT, lw=2.4, zorder=3, marker='o', ms=7, mfc=BUILT, mec='white', mew=0.9)
    ax.annotate('the five built networks\nstay flat at 83', (names[2], accs[2]),
                textcoords='offset points', xytext=(4, -34), fontsize=8, color=BUILT)
    ax.annotate(f'{labels[0]}\n{accs[0]:.1f}%', (names[0], accs[0]),
                textcoords='offset points', xytext=(-4, 10), fontsize=7.5, color=INK, ha='right')
    ax.annotate(f'{labels[-1]}\n{accs[-1]:.1f}%', (names[-1], accs[-1]),
                textcoords='offset points', xytext=(8, 8), fontsize=7.5, color=INK)
    # the two combinatorial walls, marked where they actually fall
    ax.axvline(pairs, color=MUTED, lw=1.3, ls=':')
    ax.text(pairs, 74.6, f'{pairs}\npairwise\ntypes', ha='center', fontsize=8, color=MUTED)
    ax.axvline(triples, color='#c2553a', lw=1.3, ls=':')
    ax.text(triples, 74.6, f'~{triples:,}\nthree-way\ntypes', ha='center', fontsize=8, color='#c2553a')
    # shade the region we could never hand-pick
    ax.axvspan(names[-1], 9000, color=MUTED, alpha=0.06)
    ax.set_xlabel('number of named detectors (log scale)', color=MUTED, fontsize=9)
    ax.set_ylabel('test accuracy (%)', color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values(): s.set_color(LINE)
    fig.subplots_adjust(bottom=0.13, top=0.86)
    save_fig(fig, 'depth-vocab-wall.png')


# ═══════════════════════════════════════════════════════════════════════════
# FIG 3 — the accuracy ladder, static horizontal bars zoomed to the photo finish
# ═══════════════════════════════════════════════════════════════════════════
def fig_ladder():
    L = D['ladder']
    rungs = [
        ('raw pixels',               L['pixels'],             BUILT,   False),
        ('layer 1 by hand',          L['l1'],                 BUILT,   True),
        ('layer 2 alone',            L['l2only'],             BUILT,   True),
        ('layer 1 + 2 by hand',      L['concat'],             BUILT,   True),
        ('best of six designs',      L['bestConcat'],         BUILT,   True),
        ('trained head, built feat', L['trainedConstructed'], TRAINED, False),
        ('fully trained network',    L['trainedHead'],        TRAINED, False),
    ]
    labels = [r[0] for r in rungs]; targets = [r[1] for r in rungs]; cols = [r[2] for r in rungs]
    y = np.arange(len(rungs))[::-1]
    fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=110, facecolor=BG)
    ax.set_facecolor(PANEL)
    fig.suptitle('The construction ladder: built (blue) vs trained (orange)',
                 fontsize=13.5, weight='bold', y=0.97)
    ax.set_title('the hand-built layers cluster near 83; only full training reaches 85.7',
                 fontsize=8.8, color=MUTED, pad=6)
    ax.barh(y, targets, color=cols, alpha=0.85, height=0.62, edgecolor=LINE)
    # dashed reference at the layer-1 by-hand rung
    ax.axvline(L['l1'], color=INK, lw=0.9, ls=':', alpha=0.55)
    ax.text(L['l1'], y.max() + 0.55, f'layer 1 by hand {L["l1"]:.1f}%', ha='center',
            fontsize=7.5, color=INK)
    for yi, t in zip(y, targets):
        ax.text(t + 0.12, yi, f'{t:.1f}%', va='center', ha='left', fontsize=9, color=INK, weight='bold')
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.5, color=INK)
    ax.set_xlim(78, 87.2); ax.set_xlabel('test accuracy (%)', color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values(): s.set_color(LINE)
    fig.subplots_adjust(left=0.30, top=0.85, right=0.96)
    save_fig(fig, 'depth-ladder.png')


# ═══════════════════════════════════════════════════════════════════════════
# FIG 4 — layer 2 alone: the six layer-1 edge maps vs the layer-2 relation maps,
#          with the measured 78.8% layer-2-alone accuracy stated.
# ═══════════════════════════════════════════════════════════════════════════
def fig_layer2_alone():
    img = grab(1, 4)                                      # a Trouser: clean contours
    C = cells(channel_maps(img))
    specs = [('30°+90° junction', np.minimum(nrm(C[1]), nrm(C[3]))),
             ('90° continuation', _continuation(C, 3)),
             ('0°+90° junction', np.minimum(nrm(C[0]), nrm(C[3])))]
    edges = [nrm(C[b]) for b in range(NB)]
    l2only = D['ladder']['l2only']; pixels = D['ladder']['pixels']
    fig = plt.figure(figsize=(8.6, 4.3), dpi=110, facecolor=BG)
    fig.text(0.5, 0.94, 'Classify from relations only: throw the edges away, keep the junctions',
             ha='center', fontsize=13, weight='bold')
    fig.text(0.5, 0.865,
             f'the 343 raw-edge dimensions on the left are discarded; the hand-built layer-2 detectors alone reach {l2only:.1f}%',
             ha='center', fontsize=9, color=MUTED)
    # left panel: the six edge maps, dimmed to signal they are discarded
    for b in range(NB):
        r, cc = divmod(b, 3)
        ax = fig.add_axes([0.03 + cc * 0.115, 0.44 - r * 0.30, 0.10, 0.26]); ax.set_facecolor(PANEL)
        ax.imshow(edges[b], cmap='magma', vmin=0, vmax=1, alpha=0.28)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'edge {DEG[b]}°', fontsize=6.5, color=CHAN_COL[b], pad=1)
        for s in ax.spines.values(): s.set_color(CHAN_COL[b]); s.set_linewidth(0.9)
    fig.text(0.19, 0.045, 'layer-1 edges (discarded)', ha='center', fontsize=8, color=MUTED)
    # middle: the layer-2 detectors, kept
    for k, (nm, m) in enumerate(specs):
        ax = fig.add_axes([0.40, 0.50 - k * 0.24, 0.17, 0.20]); ax.set_facecolor(PANEL)
        ax.imshow(nrm(m), cmap='magma', vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(nm, fontsize=7.5, color=ACCENT, pad=1)
        for s in ax.spines.values(): s.set_color(ACCENT); s.set_linewidth(1.1)
    fig.text(0.485, 0.045, 'layer-2 relations (kept)', ha='center', fontsize=8, color=ACCENT)
    # right: static accuracy readout as a filled gauge
    axd = fig.add_axes([0.63, 0.14, 0.34, 0.58], projection='polar'); axd.set_facecolor(PANEL)
    axd.set_theta_zero_location('W'); axd.set_theta_direction(-1)
    axd.set_thetamin(0); axd.set_thetamax(180)
    axd.set_rticks([]); axd.set_xticks([])
    th = np.linspace(0, np.pi, 100)
    axd.plot(th, np.ones_like(th), color=LINE, lw=6, solid_capstyle='round')
    frac = l2only / 100.0
    thf = np.linspace(0, np.pi * frac, 60)
    axd.plot(thf, np.ones_like(thf), color=BUILT, lw=6, solid_capstyle='round')
    # mark the raw-pixel baseline on the arc for reference
    thp = np.pi * (pixels / 100.0)
    axd.plot([thp, thp], [0.78, 1.22], color=MUTED, lw=1.4, ls=':')
    axd.text(thp, 1.32, f'pixels {pixels:.0f}', ha='center', va='center', fontsize=6.5, color=MUTED)
    axd.set_ylim(0, 1.4)
    axd.text(np.pi / 2, 0.12, f'{l2only:.1f}%', ha='center', va='center', fontsize=18, color=INK, weight='bold')
    axd.text(np.pi / 2, 1.34, 'layer-2-alone accuracy', ha='center', va='center', fontsize=8.5, color=MUTED)
    axd.text(0, 1.14, '0', color=MUTED, fontsize=7); axd.text(np.pi, 1.14, '100', color=MUTED, fontsize=7)
    save_fig(fig, 'depth-layer2-alone.png')


# ═══════════════════════════════════════════════════════════════════════════
# FIG 5 — the synonym Gram: static layer-2 x layer-1 correlation matrix,
#          continuation diagonal highlighted. Correlations from real features.
# ═══════════════════════════════════════════════════════════════════════════
def fig_synonym_gram():
    idx = np.concatenate([np.where(TEY == c)[0][:40] for c in range(10)])
    L1 = []; L2 = []
    cont_bins = list(range(NB))
    junc_pairs = [(0, 3), (1, 3), (1, 4), (2, 5), (0, 2), (3, 5)]
    for i in idx:
        C = cells(channel_maps(TEX[i]))
        L1.append(C.mean((1, 2)))                          # 7 layer-1 channel energies
        row = []
        for b in cont_bins:                                # 6 continuations
            dx, dy = CONTOUR[b]
            row.append(np.minimum(C[b], _shift(C[b], dy, dx)).mean())
        for (b1, b2) in junc_pairs:                        # 6 junctions
            row.append(np.minimum(C[b1], C[b2]).mean())
        L2.append(row)
    L1 = np.array(L1); L2 = np.array(L2)
    L1 = (L1 - L1.mean(0)) / (L1.std(0) + 1e-9)
    L2 = (L2 - L2.mean(0)) / (L2.std(0) + 1e-9)
    Corr = (L2.T @ L1) / len(idx)                          # [12 layer-2, 7 layer-1]
    l1names = [f'{d}°' for d in DEG] + ['corner']
    l2names = [f'{d}° cont' for d in DEG] + ['0+90 j', '30+90 j', '30+120 j', '60+150 j', '0+60 j', '90+150 j']
    fig, ax = plt.subplots(figsize=(7.4, 5.0), dpi=110, facecolor=BG)
    ax.set_facecolor(PANEL)
    fig.suptitle('Every layer-2 detector is a synonym for layer-1 coordinates',
                 fontsize=12.5, weight='bold', y=0.97)
    ax.set_title('correlation of each hand-built layer-2 dim with the layer-1 edges it is built from',
                 fontsize=8.3, color=MUTED, pad=6)
    im = ax.imshow(Corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    # highlight the continuation diagonal: each continuation row correlates with its own edge
    for b in range(NB):
        ax.add_patch(plt.Rectangle((b - 0.5, b - 0.5), 1, 1, fill=False,
                                   edgecolor=ACCENT, lw=2.0, zorder=5))
    ax.set_xticks(range(len(l1names))); ax.set_xticklabels(l1names, fontsize=7.5, color=MUTED, rotation=30, ha='right')
    ax.set_yticks(range(len(l2names))); ax.set_yticklabels(l2names, fontsize=7.5, color=INK)
    for s in ax.spines.values(): s.set_color(LINE)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label('correlation', color=MUTED, fontsize=8); cb.ax.tick_params(colors=MUTED, labelsize=7)
    fig.text(0.5, 0.035,
             'the teal diagonal: each continuation correlates almost perfectly with its own edge orientation; junction rows light up under both their edges',
             ha='center', fontsize=7.6, color=MUTED)
    fig.subplots_adjust(bottom=0.17, top=0.88)
    save_fig(fig, 'depth-synonym-gram.png')


# ═══════════════════════════════════════════════════════════════════════════
# FIG 6 — the six design iterations, static bar chart, best highlighted.
# ═══════════════════════════════════════════════════════════════════════════
def fig_six_designs():
    its = D['iterations']
    tags = [i['tag'] for i in its]
    concat = [i['concat'] for i in its]
    l1 = D['ladder']['l1']
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=110, facecolor=BG)
    ax.set_facecolor(PANEL)
    fig.suptitle('Six designs, six theories of what was broken: each lands near 83',
                 fontsize=13, weight='bold', y=0.97)
    ax.set_title('the researcher\'s sweep over the layer-2 vocabulary; no theory rises, the head shrugs',
                 fontsize=8.6, color=MUTED, pad=6)
    x = np.arange(len(its))
    best = int(np.argmax(concat))
    colors = [ACCENT if k == best else BUILT for k in range(len(its))]
    ax.axhline(l1, color=INK, lw=1.2, ls='--', alpha=0.7)
    ax.text(len(its) - 0.5, l1 + 0.10, f'layer 1 alone {l1:.1f}%', ha='right', fontsize=8.5, color=INK)
    ax.bar(x, concat, color=colors, alpha=0.85, width=0.6, edgecolor=LINE)
    for k in range(len(its)):
        ax.text(x[k], concat[k] + 0.10, f'{concat[k]:.1f}', ha='center', fontsize=9, color=INK, weight='bold')
    ax.text(x[best], 80.25, 'best', ha='center', fontsize=8, color=ACCENT, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace('V4 ', '').replace('V1 ', '').replace('V2 ', '').replace('V3 ', '') for t in tags],
                       fontsize=7.5, color=MUTED, rotation=18, ha='right')
    ax.set_ylim(80, 84.5); ax.set_ylabel('layer 1 + 2 accuracy (%)', color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values(): s.set_color(LINE)
    fig.subplots_adjust(bottom=0.16, top=0.86)
    save_fig(fig, 'depth-six-designs.png')


if __name__ == '__main__':
    fig_junction_assembly()
    fig_vocab_wall()
    fig_ladder()
    fig_layer2_alone()
    fig_synonym_gram()
    fig_six_designs()
    print('DEPTH_FIGS_DONE')
