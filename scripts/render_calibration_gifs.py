"""Figures for the calibration JAX/Flax NNX companion.

Every number here is a real value from the calibration run
(`scripts/yat_calibration.py`), read back from the exported JSON in
public/yat-calibration/. Nothing is trained here and nothing is faked.

Only ONE figure earns motion, because only one shows something no single frame
holds: a parameter sweep that transforms the object.

  1. calib-temperature.gif  KEPT AS GIF. One dial turning: every logit divided
                            by T, the Yat reliability curve genuinely rotating
                            toward the diagonal while the test NLL draws down to
                            its minimum near the fitted T. The sweep transforms
                            the curve, so the motion carries the information.

The rest are canonical static results; a static figure holds everything the
final frame of an animation would, so they are rendered as PNGs:

  2. calib-staircase.png    the reliability diagram: Yat bars sagging below the
                            diagonal with overconfidence strips, ReLU hugging it,
                            ECE labels. The canonical static calibration figure.
  3. calib-saturation.png   the softmax-vs-lead curve with each net's real median
                            top-two gap and top-logit magnitude marked. A curve.
  4. calib-channels.png     the two OOD histograms (field magnitude separating,
                            softmax confidence smearing) with their AUROCs.
  5. calib-seeds.png        the three real retrainings as a grouped dot chart:
                            the softmax channel's spread against the field's tight
                            cluster, all three seeds shown at once.

Run: python3 scripts/render_calibration_gifs.py
"""
import warnings; warnings.filterwarnings('ignore')
import json, os
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / 'public'
CAL = PUB / 'yat-calibration'
BINS = json.load(open(CAL / 'bins.json'))
HIST = json.load(open(CAL / 'hist.json'))
LOGITS = json.load(open(CAL / 'logits.json'))
CHAN = json.load(open(CAL / 'channels.json'))
SUMM = json.load(open(CAL / 'summary.json'))

# ── palette (warm-dark, matches the series) ──
BG = '#0e0d0b'; PANEL = '#16140f'; INK = '#e8e2d4'; MUTED = '#9a9282'; LINE = '#3a352c'
YAT = '#e0a45a'; RELU = '#5a9fd0'; DIAG = '#8f8877'
FIELD = '#3a8f5e'; SOFT = '#c2553a'; ACCENT = '#36d6c4'
plt.rcParams.update({'figure.facecolor': BG, 'savefig.facecolor': BG, 'text.color': INK,
                     'axes.edgecolor': LINE, 'font.size': 11})


def softmax(z):
    z = np.asarray(z, float)
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def reliability(z, y, n_bins=15):
    p = softmax(z)
    conf = p.max(1); pred = p.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    count = np.zeros(n_bins); mconf = np.zeros(n_bins); macc = np.zeros(n_bins)
    for b in range(n_bins):
        m = idx == b
        count[b] = m.sum()
        if count[b] > 0:
            mconf[b] = conf[m].mean(); macc[b] = correct[m].mean()
    ece = float((count / len(y) * np.abs(macc - mconf)).sum())
    return count, mconf, macc, ece


def nll_of(z, y):
    p = softmax(z)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())


def auroc(pos, neg):
    s = np.concatenate([pos, neg]); order = np.argsort(s, kind='mergesort')
    ranks = np.empty(len(s)); sr = s[order]; i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fig_rgba(fig):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba


def save_gif(path, frames, fps, hold=14, palettesize=96):
    frames = frames + [frames[-1]] * hold
    imageio.mimsave(path, frames, duration=1 / fps, loop=0,
                    palettesize=palettesize, subrectangles=True)
    print(f'wrote {path} ({os.path.getsize(path)//1024} KB)')


def save_png(path, fig):
    fig.savefig(path, dpi=110, facecolor=BG)
    plt.close(fig)
    print(f'wrote {path} ({os.path.getsize(path)//1024} KB)')


# shared: seed-0 logits + labels for both nets
Y = np.array(LOGITS['labels'])
ZY = np.array(LOGITS['yat']['test'], float)
ZR = np.array(LOGITS['relu']['test'], float)
TY = LOGITS['yat']['T']; TR = LOGITS['relu']['T']
NB = BINS['nBins']


# ═══════════════════════════════════════════════════════════════════════════
# PNG — the reliability diagram (the canonical static calibration figure).
#        Yat bars sag below the diagonal with overconfidence strips; ReLU hugs
#        it. ECE labels are the real seed-0 values.
# ═══════════════════════════════════════════════════════════════════════════
def png_staircase():
    edges = np.linspace(0, 1, NB + 1); ctr = (edges[:-1] + edges[1:]) / 2
    w = 1.0 / NB

    def panel(ax, z, col, title):
        count, mconf, macc, ece = reliability(z, Y, NB)
        ax.set_facecolor(PANEL)
        ax.plot([0, 1], [0, 1], color=DIAG, lw=1.4, ls='--', zorder=1)
        for b in range(NB):
            if count[b] == 0:
                continue
            ax.bar(ctr[b], macc[b], width=w * 0.86, color=col, alpha=0.9,
                   edgecolor=BG, lw=0.4, zorder=3)
            ax.plot([ctr[b] - w * 0.43, ctr[b] + w * 0.43], [mconf[b], mconf[b]],
                    color=INK, lw=1.7, zorder=5)
            lo, hi = sorted([macc[b], mconf[b]])
            ax.fill_between([ctr[b] - w * 0.43, ctr[b] + w * 0.43], lo, hi,
                            color=SOFT if mconf[b] > macc[b] else FIELD,
                            alpha=0.4, zorder=2)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel('claimed confidence', color=MUTED, fontsize=9)
        ax.set_title(f'{title}    ECE {ece*100:.1f}%', fontsize=11, color=col, pad=5)
        ax.tick_params(colors=MUTED, labelsize=7.5)
        for s in ax.spines.values(): s.set_color(LINE)

    fig = plt.figure(figsize=(8.8, 4.5), dpi=110, facecolor=BG)
    fig.text(0.5, 0.94, 'The reliability diagram: claimed confidence vs delivered accuracy',
             ha='center', fontsize=13.5, weight='bold')
    fig.text(0.5, 0.875, 'each test prediction drops into its confidence bin; the bar is delivered accuracy, the tick is the claim, the strip is the gap',
             ha='center', fontsize=9, color=MUTED)
    axL = fig.add_axes([0.07, 0.16, 0.40, 0.63])
    axR = fig.add_axes([0.56, 0.16, 0.40, 0.63])
    panel(axL, ZY, YAT, 'Yat MLP')
    panel(axR, ZR, RELU, 'ReLU MLP')
    axL.set_ylabel('delivered accuracy', color=MUTED, fontsize=9)
    fig.text(0.5, 0.035, 'seed-0 test set, 15 bins   '
             '(bar below tick means overconfident: claims more than it delivers)',
             ha='center', fontsize=8.5, color=MUTED)
    save_png(PUB / 'calib-staircase.png', fig)


# ═══════════════════════════════════════════════════════════════════════════
# GIF (KEPT) — temperature dial: divide logits by T, the reliability curve
#        genuinely rotates toward the diagonal and the test NLL draws to its min
#        near the fitted T. A real transforming sweep, so it earns the motion.
# ═══════════════════════════════════════════════════════════════════════════
def gif_temperature():
    Tsweep = np.linspace(1.0, 3.0, 40)
    nll_curve = np.array([nll_of(ZY / t, Y) for t in Tsweep])
    Tstar_idx = int(np.argmin(np.abs(Tsweep - TY)))

    frames = []
    for fi, Tcur in enumerate(Tsweep):
        count, mconf, macc, ece = reliability(ZY / Tcur, Y, NB)
        fig = plt.figure(figsize=(8.8, 4.6), dpi=110, facecolor=BG)
        fig.text(0.5, 0.94, 'One knob: dividing the Yat logits by a temperature T',
                 ha='center', fontsize=14, weight='bold')
        fig.text(0.5, 0.875, 'softening the logits rotates the overconfident curve toward the diagonal; the fitted T is where held-out NLL bottoms out',
                 ha='center', fontsize=9, color=MUTED)
        # left: reliability curve rotating
        axL = fig.add_axes([0.07, 0.13, 0.40, 0.66]); axL.set_facecolor(PANEL)
        axL.plot([0, 1], [0, 1], color=DIAG, lw=1.4, ls='--', zorder=1)
        good = count > 0
        axL.plot(mconf[good], macc[good], color=YAT, lw=2.2, marker='o', ms=3.5, zorder=4)
        for b in range(NB):
            if count[b] == 0:
                continue
            lo, hi = sorted([macc[b], mconf[b]])
            axL.fill_between([mconf[b] - 0.006, mconf[b] + 0.006], lo, hi,
                             color=SOFT, alpha=0.3, zorder=2)
        axL.set_xlim(0, 1); axL.set_ylim(0, 1)
        axL.set_xlabel('claimed confidence', color=MUTED, fontsize=9)
        axL.set_ylabel('delivered accuracy', color=MUTED, fontsize=9)
        axL.set_title(f'Yat reliability   T = {Tcur:.2f}   ECE {ece*100:.1f}%',
                      fontsize=10.5, color=YAT, pad=5)
        axL.tick_params(colors=MUTED, labelsize=7.5)
        for s in axL.spines.values(): s.set_color(LINE)
        # right: NLL vs T with the current T marked, min drawn
        axR = fig.add_axes([0.57, 0.13, 0.39, 0.66]); axR.set_facecolor(PANEL)
        axR.plot(Tsweep[:fi + 1], nll_curve[:fi + 1], color=ACCENT, lw=2.2, zorder=3)
        axR.scatter([Tcur], [nll_curve[fi]], s=55, color=ACCENT,
                    edgecolors='white', linewidths=1.2, zorder=5)
        axR.axvline(TY, color=YAT, lw=1.3, ls=':', zorder=2)
        axR.text(TY + 0.03, nll_curve.max() - 0.02 * (nll_curve.max() - nll_curve.min()),
                 f'fitted T = {TY:.2f}', color=YAT, fontsize=8.5, va='top')
        axR.set_xlim(1.0, 3.0)
        axR.set_ylim(nll_curve.min() - 0.02, nll_curve.max() + 0.02)
        axR.set_xlabel('temperature T', color=MUTED, fontsize=9)
        axR.set_ylabel('test NLL', color=MUTED, fontsize=9)
        axR.set_title('the loss the knob is minimizing', fontsize=10.5, color=ACCENT, pad=5)
        axR.tick_params(colors=MUTED, labelsize=7.5)
        for s in axR.spines.values(): s.set_color(LINE)
        frames.append(fig_rgba(fig))
    # hold longest on the fitted-T frame
    frames = frames + [frames[Tstar_idx]] * 10
    save_gif(PUB / 'calib-temperature.gif', frames, fps=11, hold=10)


# ═══════════════════════════════════════════════════════════════════════════
# PNG — softmax saturation: the real function softmax([gap, 0])[0], with each
#        net's real median top-two logit gap and median top-logit magnitude
#        marked. A static curve.
# ═══════════════════════════════════════════════════════════════════════════
def png_saturation():
    gap = np.linspace(0, 24, 300)
    p_win = 1 / (1 + np.exp(-gap))
    gapsY = ZY.max(1) - np.sort(ZY, 1)[:, -2]
    gapsR = ZR.max(1) - np.sort(ZR, 1)[:, -2]
    gY = float(np.median(gapsY)); gR = float(np.median(gapsR))
    topY = float(np.median(ZY.max(1))); topR = float(np.median(ZR.max(1)))
    fracY = float((softmax(ZY).max(1) > 0.99).mean())
    fracR = float((softmax(ZR).max(1) > 0.99).mean())

    fig = plt.figure(figsize=(8.6, 4.6), dpi=110, facecolor=BG)
    fig.text(0.5, 0.94, 'Why the bounded net brags: the softmax reads gaps, then saturates',
             ha='center', fontsize=13.2, weight='bold')
    fig.text(0.5, 0.875, 'confidence in the winner as its lead over the runner-up grows; past a lead of a few units the curve is flat at 1',
             ha='center', fontsize=9, color=MUTED)
    ax = fig.add_axes([0.10, 0.16, 0.85, 0.62]); ax.set_facecolor(PANEL)
    ax.plot(gap, p_win, color=INK, lw=2.6, zorder=3)
    ax.axhline(0.99, color=MUTED, lw=0.8, ls=':')
    ax.text(0.4, 0.986, 'confidence 0.99', color=MUTED, fontsize=7.5, va='top')
    pr = 1 / (1 + np.exp(-gR))
    ax.scatter([gR], [pr], s=95, color=RELU, edgecolors='white', linewidths=1.3, zorder=6)
    ax.annotate(f'ReLU: median gap {gR:.1f}\n(top logit ~{topR:.0f}, {fracR*100:.0f}% above 0.99)',
                (gR, pr), (gR + 2.2, 0.55), color=RELU, fontsize=8.2,
                arrowprops=dict(arrowstyle='->', color=RELU, lw=1.1))
    py = 1 / (1 + np.exp(-gY))
    ax.scatter([gY], [py], s=95, color=YAT, edgecolors='white', linewidths=1.3, zorder=6)
    ax.annotate(f'Yat: median gap {gY:.1f}\n(top logit ~{topY:.0f}, {fracY*100:.0f}% above 0.99)',
                (gY, py), (gY + 2.2, 0.30), color=YAT, fontsize=8.2,
                arrowprops=dict(arrowstyle='->', color=YAT, lw=1.1))
    ax.set_xlim(0, 24); ax.set_ylim(0, 1.05)
    ax.set_xlabel('lead of the winning logit over the runner-up', color=MUTED, fontsize=9.5)
    ax.set_ylabel('softmax confidence in the winner', color=MUTED, fontsize=9.5)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values(): s.set_color(LINE)
    fig.text(0.5, 0.035, 'the Yat net sits deeper on the flat: its huge top logits push more answers past 0.99, so it sounds certain more often',
             ha='center', fontsize=8.3, color=MUTED)
    save_png(PUB / 'calib-saturation.png', fig)


# ═══════════════════════════════════════════════════════════════════════════
# PNG — two OOD channels: the field-magnitude histogram separates Fashion from
#        MNIST while the softmax-confidence histogram smears them. Static, with
#        the real seed-0 AUROCs.
# ═══════════════════════════════════════════════════════════════════════════
def png_channels():
    kmT = np.array(LOGITS['yat']['kmaxTest'], float)
    kmO = np.array(LOGITS['yat']['kmaxOod'], float)
    cfT = softmax(ZY).max(1)                               # Fashion softmax conf
    cfO = softmax(np.array(LOGITS['yat']['ood'], float)).max(1)   # MNIST softmax conf
    au_field = auroc(kmT, kmO); au_soft = auroc(cfT, cfO)

    lkmT = np.log10(np.clip(kmT, 1e-3, None)); lkmO = np.log10(np.clip(kmO, 1e-3, None))
    fed = np.linspace(0.4, np.log10(700), 30)
    sed = np.linspace(0, 1, 30)

    def hist(ax, vt, vo, edges, xlab):
        ax.set_facecolor(PANEL)
        hT, _ = np.histogram(vt, bins=edges); hO, _ = np.histogram(vo, bins=edges)
        ctrs = (edges[:-1] + edges[1:]) / 2; wbar = (edges[1] - edges[0])
        peakT = max(1, hT.max()); peakO = max(1, hO.max())
        ax.bar(ctrs, hT / peakT, width=wbar, color=RELU, alpha=0.6,
               label='Fashion (in-dist.)', align='center', edgecolor=BG, lw=0.3)
        ax.bar(ctrs, hO / peakO, width=wbar, color=SOFT, alpha=0.6,
               label='MNIST (OOD)', align='center', edgecolor=BG, lw=0.3)
        ax.step(ctrs, hO / peakO, color=SOFT, lw=1.5, where='mid')
        ax.set_xlim(edges[0], edges[-1]); ax.set_ylim(0, 1.15)
        ax.set_xlabel(xlab, color=MUTED, fontsize=9); ax.set_yticks([])
        ax.tick_params(colors=MUTED, labelsize=7.5)
        for s in ax.spines.values(): s.set_color(LINE)
        ax.legend(loc='upper center', fontsize=7.5, facecolor=PANEL, edgecolor=LINE,
                  labelcolor=INK, framealpha=0.9)

    ft = [np.log10(v) for v in (10, 30, 100, 300)]
    ftl = ['10', '30', '100', '300']
    fig = plt.figure(figsize=(8.8, 4.4), dpi=110, facecolor=BG)
    fig.text(0.5, 0.94, 'One forward pass, two readings: field magnitude vs softmax confidence',
             ha='center', fontsize=13.2, weight='bold')
    fig.text(0.5, 0.875, 'the same Yat net scores Fashion and never-seen MNIST digits; the pre-softmax field separates them, the softmax smears them',
             ha='center', fontsize=9, color=MUTED)
    axL = fig.add_axes([0.06, 0.15, 0.41, 0.62])
    axR = fig.add_axes([0.56, 0.15, 0.41, 0.62])
    hist(axL, lkmT, lkmO, fed, 'strongest kernel match  (log scale)')
    axL.set_xticks(ft); axL.set_xticklabels(ftl)
    axL.set_title('field magnitude   AUROC %.3f' % au_field, fontsize=10.5, color=FIELD, pad=4)
    hist(axR, cfT, cfO, sed, 'softmax confidence')
    axR.set_title('softmax confidence   AUROC %.3f' % au_soft, fontsize=10.5, color=SOFT, pad=4)
    fig.text(0.5, 0.035, 'seed-0 scores   '
             '(the field sees "nothing here"; the softmax cannot)', ha='center',
             fontsize=8.5, color=MUTED)
    save_png(PUB / 'calib-channels.png', fig)


# ═══════════════════════════════════════════════════════════════════════════
# PNG — three real retrainings as a grouped dot chart. The softmax channel's
#        three seeds spread wide; the field channel's cluster tight. Showing all
#        three at once makes the instability plainer than a needle ever could.
# ═══════════════════════════════════════════════════════════════════════════
def png_seeds():
    seeds = CHAN['seeds']
    field_au, soft_au = [], []
    for s in seeds:
        d = CHAN[f's{s}']
        field_au.append(auroc(np.array(d['kmaxTest'], float), np.array(d['kmaxOod'], float)))
        soft_au.append(auroc(np.array(d['confTest'], float), np.array(d['confOod'], float)))
    field_au = np.array(field_au); soft_au = np.array(soft_au)

    fig = plt.figure(figsize=(8.4, 4.4), dpi=110, facecolor=BG)
    fig.text(0.5, 0.93, 'Retrain three times: which channel is stable?',
             ha='center', fontsize=13.5, weight='bold')
    fig.text(0.5, 0.865, 'Fashion-vs-MNIST separation (AUROC) of the same Yat net, one point per retraining',
             ha='center', fontsize=9, color=MUTED)
    ax = fig.add_axes([0.13, 0.14, 0.80, 0.64]); ax.set_facecolor(PANEL)

    cols = {'field': FIELD, 'soft': SOFT}
    xs = {'field': 0.0, 'soft': 1.0}
    jit = np.array([-0.12, 0.0, 0.12])
    for key, vals, lab in [('field', field_au, 'field magnitude'),
                           ('soft', soft_au, 'softmax confidence')]:
        x0 = xs[key]; col = cols[key]
        # spread band (min to max) behind the points
        ax.plot([x0, x0], [vals.min(), vals.max()], color=col, lw=2.0, alpha=0.35, zorder=1)
        ax.scatter(x0 + jit, vals, s=140, color=col, edgecolors='white',
                   linewidths=1.2, zorder=4)
        for sd, v, dx in zip(seeds, vals, jit):
            ax.text(x0 + dx, v + 0.006, f'{v:.3f}', color=INK, fontsize=7.5,
                    ha='center', va='bottom')
        # mean marker
        ax.plot([x0 - 0.22, x0 + 0.22], [vals.mean(), vals.mean()],
                color=col, lw=1.6, ls='--', zorder=3)
        ax.text(x0, vals.min() - 0.018,
                f'spread {vals.max() - vals.min():.3f}', color=col, fontsize=8.5,
                ha='center', va='top', weight='bold')

    ax.axhline(SUMM['auroc_agg']['relu']['maxlogit']['mean'], color=RELU, lw=1.0, ls=':', zorder=2)
    ax.text(1.42, SUMM['auroc_agg']['relu']['maxlogit']['mean'], 'ReLU max-logit\nbaseline',
            color=RELU, fontsize=7.5, va='center', ha='left')
    ax.set_xticks([0.0, 1.0]); ax.set_xticklabels(['field magnitude', 'softmax confidence'])
    ax.set_xlim(-0.5, 1.8); ax.set_ylim(0.79, 0.945)
    ax.set_ylabel('Fashion-vs-MNIST AUROC', color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for s in ax.spines.values(): s.set_color(LINE)
    fig.text(0.5, 0.035, 'the softmax channel lurches seed to seed; the field channel holds   '
             '(same network, same question, one answer you can deploy)',
             ha='center', fontsize=8.3, color=MUTED)
    save_png(PUB / 'calib-seeds.png', fig)


if __name__ == '__main__':
    png_staircase()
    gif_temperature()
    png_saturation()
    png_channels()
    png_seeds()
    print('CALIBRATION_FIGS_DONE')
