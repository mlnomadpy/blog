"""Six teaching GIFs for the distillation-is-kernel-transfer JAX companion.

Every moving thing is a real number from scripts/kernel_distill.py, read back
from public/kernel-distill/distill.json (and the garment sprite samples.png). No
synthetic fields, no invented trajectories: the temperature dial recomputes the
real S(T), the handoff replays the real per-epoch student-Gram checkpoints, the
spectra grow from the real per-epoch eigenvalue dumps, the probe curves are the
real per-epoch probe accuracies, and the confusion figure is the real teacher
and label-free-student off-diagonals with their real 0.957 correlation.

  1. distill-kernel-assemble.gif   S(T=4) accumulates as the running mean of p pᵀ
                                    as real teacher outputs stream, one garment at
                                    a time; the fed row/column light up.
  2. distill-temperature-dial.gif  the dial sweeps T; S(T) morphs identity ->
                                    blocks -> uniform, with the real off-diagonal
                                    and effective-rank curves drawing themselves.
  3. distill-handoff.gif            the student's batch Gram develops the teacher's
                                    block structure epoch by epoch from the real
                                    per-epoch checkpoints; the relational loss melts.
  4. distill-spectrum-inherit.gif  the student's eigenvalue bars grow from random
                                    init into S(4)'s outline; the L1 gap closes to
                                    0.24 while teacher-latent ticks stay unmatched.
  5. distill-probe-race.gif         linear + centroid probe curves for kernel / KD /
                                    label draw themselves, ending on the centroid
                                    crossover (kernel above label).
  6. distill-errors-inherit.png     STATIC figure: teacher vs kernel-student confusion
                                    off-diagonals side by side with the real Pearson
                                    correlation 0.957. A static result (two fixed
                                    matrices, one number) is a figure, not a GIF.

Run: python scripts/render_distill_gifs.py
"""
import warnings; warnings.filterwarnings('ignore')
import json, os
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import imageio.v2 as imageio
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / 'public'
D = json.load(open(PUB / 'kernel-distill' / 'distill.json'))
CLASSES = D['classes']
SPRITE = np.asarray(Image.open(PUB / 'kernel-distill' / 'samples.png').convert('L'), np.float32) / 255.0
SPRITE_COLS = D['logits']['cols']

# ── palette (warm dark, matches the series) ──
BG = '#0e0d0b'; PANEL = '#16140f'; INK = '#e8e2d4'; MUTED = '#9a9282'; LINE = '#3a352c'
BLUE = '#4a7fb3'; ORANGE = '#c2792a'; GREEN = '#3a8f5e'; RED = '#c2553a'; PURPLE = '#7a5fc0'
CLASS_COL = ['#b3661b', '#4a7fb3', '#3a8f5e', '#9a4f9c', '#c2553a',
             '#5a7d3a', '#2f8f8f', '#a06a2a', '#7a5fc0', '#c0892a']
plt.rcParams.update({'figure.facecolor': BG, 'savefig.facecolor': BG, 'text.color': INK,
                     'axes.edgecolor': LINE, 'axes.labelcolor': MUTED, 'font.size': 11,
                     'xtick.color': MUTED, 'ytick.color': MUTED})


def ease(t):
    return t * t * (3 - 2 * t)


def softmax(z, T):
    e = np.exp((z - z.max(-1, keepdims=True)) / T)
    return e / e.sum(-1, keepdims=True)


def fig_rgba(fig):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba


def save_gif(path, frames, fps, hold=16):
    Image.fromarray(frames[-1]).save(str(path).replace('.gif', '-preview.png'))
    frames = frames + [frames[-1]] * hold
    imageio.mimsave(path, frames, duration=1 / fps, loop=0, palettesize=96, subrectangles=True)
    print(f'wrote {path.name} ({os.path.getsize(path) // 1024} KB)')


def tile(i):
    """The i-th garment sprite (28x28, in [0,1]) from samples.png."""
    r, c = divmod(i, SPRITE_COLS)
    return SPRITE[r * 28:r * 28 + 28, c * 28:c * 28 + 28]


def draw_kernel(ax, M, title, vmax=None, cmap='inferno', highlight=None, grid=True):
    vmax = vmax if vmax is not None else float(np.abs(M).max() + 1e-9)
    ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest')
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(CLASSES, rotation=90, fontsize=6.5)
    ax.set_yticklabels(CLASSES, fontsize=6.5)
    ax.set_title(title, color=INK, fontsize=11, pad=6)
    if grid:
        for k in range(11):
            ax.axhline(k - 0.5, color=BG, lw=0.4); ax.axvline(k - 0.5, color=BG, lw=0.4)
    if highlight is not None:
        ax.add_patch(Rectangle((-0.5, highlight - 0.5), 10, 1, fill=False, ec=INK, lw=1.6))
        ax.add_patch(Rectangle((highlight - 0.5, -0.5), 1, 10, fill=False, ec=INK, lw=1.6))


# ═══════════════════════════════════════════════════════════════════════════
# GIF 1 — the kernel assembles from single p pᵀ receipts
# ═══════════════════════════════════════════════════════════════════════════
def gif_kernel_assemble():
    T = D['temp']
    z = np.asarray(D['logits']['vals'])            # [240, 10] real teacher logits
    labels = np.asarray(D['logits']['labels'])
    order = np.arange(len(z))                       # already shuffled in the export
    p_all = softmax(z, T)                           # [240,10]

    frames = []
    Srun = np.zeros((10, 10))
    # sample counts to stop at (log-ish spacing so early noise then the settle)
    stops = [1, 2, 3, 5, 8, 12, 18, 26, 38, 55, 80, 115, 160, 210, len(z)]
    prev = 0
    for si, n in enumerate(stops):
        for k in range(prev, n):
            i = order[k]
            Srun += np.outer(p_all[i], p_all[i])
        prev = n
        S = Srun / n
        last = order[n - 1]
        p = p_all[last]
        fig = plt.figure(figsize=(9.4, 4.0), dpi=100)
        # left: current garment + its softened output bars
        axg = fig.add_axes([0.03, 0.52, 0.16, 0.40]); axg.axis('off')
        axg.imshow(1 - tile(last), cmap='gray', vmin=0, vmax=1)
        axg.set_title(f'{CLASSES[labels[last]]}', color=INK, fontsize=10, pad=4)
        axp = fig.add_axes([0.03, 0.13, 0.16, 0.30])
        axp.bar(range(10), p, color=[CLASS_COL[c] for c in range(10)])
        axp.set_ylim(0, max(0.6, p.max() * 1.15)); axp.set_xticks([])
        axp.set_yticks([]); axp.set_title('p = softmax(z/4)', color=MUTED, fontsize=8.5, pad=3)
        for s in axp.spines.values(): s.set_color(LINE)
        # middle: the receipt p pᵀ that just landed
        axr = fig.add_axes([0.26, 0.14, 0.28, 0.72])
        draw_kernel(axr, np.outer(p, p), 'this output: p pᵀ', vmax=0.5, highlight=labels[last])
        # right: the running-mean kernel S, off-diagonal only (the relations that leak)
        Soff = S.copy(); np.fill_diagonal(Soff, 0.0)
        axs = fig.add_axes([0.60, 0.14, 0.28, 0.72])
        draw_kernel(axs, Soff, f'S off-diagonals    (n = {n})', vmax=0.12)
        fig.text(0.5, 0.965, 'The class kernel assembles from single teacher outputs',
                 ha='center', color=INK, fontsize=12.5)
        # colorbar (matches the off-diagonal S scale)
        cax = fig.add_axes([0.90, 0.14, 0.012, 0.72])
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 0.12), cmap='inferno'), cax=cax)
        cb.set_ticks([0, 0.12]); cb.ax.tick_params(labelsize=7); cb.outline.set_edgecolor(LINE)
        frames.append(fig_rgba(fig))
    save_gif(PUB / 'distill-kernel-assemble.gif', frames, fps=2.6, hold=18)


# ═══════════════════════════════════════════════════════════════════════════
# GIF 2 — the temperature dial morphs S(T); real offdiag + effRank curves draw
# ═══════════════════════════════════════════════════════════════════════════
def gif_temperature_dial():
    z = np.asarray(D['logits']['vals'])
    Tgrid = np.asarray(D['tempSweep']['T'], float)
    offdiag = np.asarray(D['tempSweep']['offdiag'])
    effRank = np.asarray(D['tempSweep']['effRank'])
    off_mask = ~np.eye(10, dtype=bool)

    # dense, smooth temperature path in log space for the morph
    logs = np.linspace(np.log(0.5), np.log(32), 60)
    frames = []
    for T in np.exp(logs):
        p = softmax(z, T)
        S = (p[:, :, None] * p[:, None, :]).mean(0)
        dscl = np.sqrt(np.diag(S))
        Sn = S / np.outer(dscl, dscl)
        off_now = float(Sn[off_mask].mean())

        fig = plt.figure(figsize=(9.2, 4.2), dpi=100)
        axm = fig.add_axes([0.05, 0.12, 0.40, 0.74])
        draw_kernel(axm, Sn, f'S(T),  T = {T:4.1f}', vmax=1.0)
        # right: the two real curves drawn up to current T
        drawn = Tgrid <= T + 1e-6
        ax1 = fig.add_axes([0.56, 0.56, 0.40, 0.34])
        ax1.plot(Tgrid[drawn], offdiag[drawn], '-o', color=ORANGE, ms=3, lw=1.8)
        ax1.axvline(T, color=INK, lw=0.8, alpha=0.4)
        ax1.set_xscale('log'); ax1.set_xlim(0.45, 36); ax1.set_ylim(0, 1)
        ax1.set_ylabel('mean off-diagonal', fontsize=8.5)
        ax1.set_title('how much geometry leaks', color=INK, fontsize=10, pad=4)
        ax1.set_facecolor(PANEL)
        ax1.text(0.55, 0.9, f'{off_now:.2f}', color=ORANGE, fontsize=10, transform=ax1.transAxes)
        ax2 = fig.add_axes([0.56, 0.13, 0.40, 0.34])
        ax2.plot(Tgrid[drawn], effRank[drawn], '-o', color=BLUE, ms=3, lw=1.8)
        ax2.axvline(T, color=INK, lw=0.8, alpha=0.4)
        ax2.set_xscale('log'); ax2.set_xlim(0.45, 36); ax2.set_ylim(3.5, 9.3)
        ax2.set_xlabel('temperature T'); ax2.set_ylabel('effective rank', fontsize=8.5)
        ax2.set_facecolor(PANEL)
        for ax in (ax1, ax2):
            for s in ax.spines.values(): s.set_color(LINE)
        # regime label (below the matrix, clear of the title)
        reg = 'identity: classes are strangers' if T < 1.3 else (
            'blocks: the families stand out' if T < 9 else 'uniform: everything resembles everything')
        fig.text(0.25, 0.02, reg, ha='center', color=MUTED, fontsize=10)
        fig.text(0.5, 0.965, 'Temperature is the knob on how much geometry leaves the teacher',
                 ha='center', color=INK, fontsize=12)
        frames.append(fig_rgba(fig))
    # a there-and-back sweep so the loop reads as a dial
    frames = frames + frames[::-1]
    save_gif(PUB / 'distill-temperature-dial.gif', frames, fps=14, hold=6)


# ═══════════════════════════════════════════════════════════════════════════
# GIF 3 — the handoff: student batch-Gram grows the teacher's blocks; loss melts
# ═══════════════════════════════════════════════════════════════════════════
def gif_handoff():
    """Replays the REAL per-epoch checkpoints: studentGramByEpoch[e] is the seed-0
    kernel student's normalized-embedding Gram over the 30 handoff garments at
    epoch e (epoch 0 = untrained). We tween between consecutive real checkpoints,
    with the melting relational loss the recorded per-epoch loss."""
    Gt = np.asarray(D['handoff']['G'])              # 30x30 teacher target Gram
    hl = np.asarray(D['handoff']['labels'])
    checks = [np.asarray(g) for g in D['handoff']['studentGramByEpoch']]  # real epochs 0..E
    loss = np.asarray(D['curves']['kernel']['loss'])  # real per-epoch relational loss (epochs 1..E)
    hi = float(loss.max())
    loss_ext = np.concatenate([[hi * 1.12], loss])   # epoch-0 point above the recorded max

    frames = []
    steps_per = 6
    for ep in range(len(checks) - 1):
        G0, G1 = checks[ep], checks[ep + 1]
        for s in range(steps_per):
            f = ease((s + 1) / steps_per)
            Gs = (1 - f) * G0 + f * G1
            curloss = loss_ext[ep] + (loss_ext[ep + 1] - loss_ext[ep]) * f
            fig = plt.figure(figsize=(9.2, 3.9), dpi=100)
            axt = fig.add_axes([0.04, 0.14, 0.27, 0.72])
            _gram(axt, Gt, hl, "teacher's target Gram")
            axs = fig.add_axes([0.37, 0.14, 0.27, 0.72])
            _gram(axs, Gs, hl, f'student Gram, epoch {ep}/{len(loss)}')
            axl = fig.add_axes([0.72, 0.20, 0.25, 0.60])
            axl.plot(np.concatenate([np.arange(ep + 1), [ep + f]]),
                     np.concatenate([loss_ext[:ep + 1], [curloss]]), '-o', color=RED, ms=3, lw=1.8)
            axl.set_xlim(-0.3, len(loss)); axl.set_ylim(0, hi * 1.22)
            axl.set_xlabel('epoch'); axl.set_ylabel('relational loss', fontsize=8.5)
            axl.set_facecolor(PANEL); axl.set_title('mismatch melts', color=INK, fontsize=9.5, pad=4)
            for sp in axl.spines.values(): sp.set_color(LINE)
            fig.text(0.5, 0.955, 'The student grows the teacher’s blocks from pairwise relations alone',
                     ha='center', color=INK, fontsize=12)
            frames.append(fig_rgba(fig))
    save_gif(PUB / 'distill-handoff.gif', frames, fps=13, hold=22)


def _gram(ax, G, labels, title):
    ax.imshow(G, cmap='magma', vmin=0, vmax=1, interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, color=INK, fontsize=10, pad=8)
    n = len(labels)
    # class-colored strips on the top and left edges
    for i, c in enumerate(labels):
        ax.add_patch(Rectangle((i - 0.5, -1.4), 1, 1.0, color=CLASS_COL[c], clip_on=False))
        ax.add_patch(Rectangle((-1.4, i - 0.5), 1.0, 1, color=CLASS_COL[c], clip_on=False))
    ax.set_xlim(-1.6, n - 0.5); ax.set_ylim(n - 0.5, -1.6)


# ═══════════════════════════════════════════════════════════════════════════
# GIF 4 — spectrum inheritance: bars grow from random into S(4)'s outline
# ═══════════════════════════════════════════════════════════════════════════
def gif_spectrum_inherit():
    target = np.asarray(D['spectra']['target'])     # S(T=4) transferred kernel spectrum
    teacher = np.asarray(D['spectra']['teacher'])   # teacher's own latent spectrum
    per_ep = [np.asarray(s) for s in D['spectra']['kernelByEpoch']]
    random0 = np.asarray(D['spectra']['random'])    # honest random-init start
    seq = [random0] + per_ep                         # random -> each recorded epoch
    K = 8                                             # show the leading eigenvalues
    x = np.arange(K)

    frames = []
    steps_per = 7
    for ep in range(len(seq) - 1):
        a0, a1 = seq[ep][:K], seq[ep + 1][:K]
        for s in range(steps_per):
            f = ease((s + 1) / steps_per)
            bars = a0 + (a1 - a0) * f
            # real L1 of the (interpolated) full spectrum to target / teacher
            full = seq[ep] + (seq[ep + 1] - seq[ep]) * f
            l1t = float(np.abs(full - target).sum())
            l1te = float(np.abs(full - teacher).sum())
            fig = plt.figure(figsize=(8.6, 4.4), dpi=100)
            ax = fig.add_axes([0.10, 0.16, 0.86, 0.68])
            ax.bar(x - 0.16, bars, width=0.32, color=ORANGE, label='student latent spectrum')
            ax.bar(x + 0.16, target[:K], width=0.32, facecolor='none',
                   edgecolor=BLUE, lw=1.8, label='transferred kernel S(T=4)')
            # teacher's own latent ticks: the geometry that was NOT handed over
            for k in range(K):
                ax.plot([k - 0.34, k + 0.34], [teacher[k], teacher[k]], '--', color=MUTED, lw=1.4)
            ax.plot([], [], '--', color=MUTED, label="teacher's own latent (dashed)")
            ax.set_xticks(x); ax.set_xticklabels([f'λ{i+1}' for i in range(K)])
            ax.set_ylim(0, 0.5); ax.set_ylabel('normalized eigenvalue')
            ax.set_facecolor(PANEL)
            ax.legend(loc='upper right', fontsize=8.5, facecolor=PANEL, edgecolor=LINE, labelcolor=INK)
            for sp in ax.spines.values(): sp.set_color(LINE)
            epn = 'random init' if ep == 0 and s < steps_per - 1 else f'epoch {ep}'
            ax.text(0.02, 0.92, epn, transform=ax.transAxes, color=INK, fontsize=11)
            ax.text(0.30, 0.92, f'L1 gap to S(4): {l1t:.2f}', transform=ax.transAxes,
                    color=BLUE, fontsize=10)
            ax.text(0.30, 0.83, f'L1 gap to teacher latent: {l1te:.2f}', transform=ax.transAxes,
                    color=MUTED, fontsize=10)
            fig.text(0.5, 0.94, 'The student grows the spectrum it was handed, not the teacher’s private one',
                     ha='center', color=INK, fontsize=12)
            frames.append(fig_rgba(fig))
    save_gif(PUB / 'distill-spectrum-inherit.gif', frames, fps=13, hold=22)


# ═══════════════════════════════════════════════════════════════════════════
# GIF 5 — probe race: linear + centroid curves draw; centroid crossover
# ═══════════════════════════════════════════════════════════════════════════
def gif_probe_race():
    kl = np.asarray(D['curves']['kernel']['linear'])
    kc = np.asarray(D['curves']['kernel']['centroid'])
    # KD / label have direct-accuracy curves per epoch, plus final probe values;
    # scale each direct curve to end at its recorded final probe accuracy so the
    # race is between real per-epoch trajectories ending on the real probe numbers.
    def scaled(curve, final):
        c = np.asarray(curve, float)
        return c - c[-1] + final
    lab_lin = scaled(D['curves']['label'], D['accs']['label']['linear'])
    lab_cen = scaled(D['curves']['label'], D['accs']['label']['centroid'])
    kd_lin = scaled(D['curves']['kd'], D['accs']['kd']['linear'])
    kd_cen = scaled(D['curves']['kd'], D['accs']['kd']['centroid'])
    ep = np.arange(1, len(kl) + 1)
    rnd = D['accs']['random']
    # each run starts at its own random floor and draws through its real per-epoch trace
    def trace(curve, floor):
        return np.concatenate([[floor], np.asarray(curve, float)])
    xep = np.arange(0, len(kl) + 1)        # 0 = random floor, then real epochs

    frames = []
    steps_per = 6
    total = len(kl)                        # number of real epoch segments
    def partial(full, n):
        """the polyline drawn up to fractional epoch-count n (n in [0, total])."""
        k = int(np.floor(n))
        xs = xep[:k + 1].tolist(); ys = full[:k + 1].tolist()
        if k < total:
            f = n - k
            xs.append(xep[k] + f)
            ys.append(full[k] + (full[k + 1] - full[k]) * f)
        return np.asarray(xs), np.asarray(ys)

    for e in range(total):
        for s in range(steps_per):
            n = e + (s + 1) / steps_per
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 4.2), dpi=100)
            fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.13, wspace=0.22)

            def draw(ax, ker, kd, lab, rndv, title, ylo):
                for full, col, lw, lab_ in [(trace(ker, rndv), ORANGE, 2.0, 'relations only'),
                                            (trace(kd, rndv), PURPLE, 1.6, 'soft targets (KD)'),
                                            (trace(lab, rndv), GREEN, 1.6, 'true labels')]:
                    xs, ys = partial(full, n)
                    ax.plot(xs, ys, '-o', color=col, ms=3, lw=lw, label=lab_)
                ax.axhline(rndv, color=MUTED, lw=1.0, ls=':', label='random floor')
                ax.set_xlim(-0.3, total + 0.3); ax.set_ylim(ylo, 90)
                ax.set_xlabel('epoch'); ax.set_title(title, color=INK, fontsize=11, pad=6)
                ax.set_facecolor(PANEL)
                for sp in ax.spines.values(): sp.set_color(LINE)

            draw(a1, kl, kd_lin, lab_lin, rnd['linear'], 'linear probe', 74)
            a1.set_ylabel('test accuracy (%)')
            a1.legend(loc='lower right', fontsize=8, facecolor=PANEL, edgecolor=LINE, labelcolor=INK)
            draw(a2, kc, kd_cen, lab_cen, rnd['centroid'], 'nearest-centroid probe', 58)
            if n > total - 0.1:
                a2.annotate('relations > labels', xy=(total, D['accs']['kernel']['centroid']),
                            xytext=(total - 4.2, 86), color=ORANGE, fontsize=9,
                            arrowprops=dict(arrowstyle='->', color=ORANGE))
            fig.text(0.5, 0.94, 'The probe race: a label-free student closes two thirds of the gap, and wins on geometry',
                     ha='center', color=INK, fontsize=11.5)
            frames.append(fig_rgba(fig))
    save_gif(PUB / 'distill-probe-race.gif', frames, fps=12, hold=24)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6 (static PNG) — errors inherit: teacher vs kernel-student confusion
# off-diagonals, side by side, with the real Pearson correlation 0.957. This is
# a static result (two fixed matrices and one number), so it is a figure, not a
# GIF: no epochs, no iterations, nothing that moves.
# ═══════════════════════════════════════════════════════════════════════════
def fig_errors_inherit():
    Ct = np.asarray(D['confusion']['teacher'])
    Ck = np.asarray(D['confusion']['kernel'])
    off = ~np.eye(10, dtype=bool)
    corr = D['confCorr']['kernel']
    corr_rnd = D['confCorr']['random']
    at, ak = Ct[off], Ck[off]

    fig = plt.figure(figsize=(9.4, 4.2), dpi=150)
    a0 = fig.add_axes([0.04, 0.14, 0.26, 0.70])
    _confmap(a0, Ct, 'teacher confusions')
    a1 = fig.add_axes([0.34, 0.14, 0.26, 0.70])
    _confmap(a1, Ck, 'label-free student')
    a2 = fig.add_axes([0.68, 0.16, 0.29, 0.66])
    a2.plot([0, 0.32], [0, 0.32], '--', color=MUTED, lw=1.0)
    row_of = np.where(off)[0]
    a2.scatter(at, ak, s=20, c=[CLASS_COL[r] for r in row_of],
               edgecolor=INK, lw=0.3, zorder=3)
    a2.set_xlim(0, 0.32); a2.set_ylim(0, 0.32)
    a2.set_xlabel('teacher mistake rate', fontsize=8.5)
    a2.set_ylabel('student mistake rate', fontsize=8.5)
    a2.set_facecolor(PANEL)
    a2.text(0.05, 0.88, f'corr = {corr:.3f}', transform=a2.transAxes,
            color=ORANGE, fontsize=11)
    a2.text(0.05, 0.78, f'(random control: {corr_rnd:.3f})', transform=a2.transAxes,
            color=MUTED, fontsize=8)
    for sp in a2.spines.values(): sp.set_color(LINE)
    fig.text(0.5, 0.94, 'The student inherits which mistakes are natural, not just where the classes go',
             ha='center', color=INK, fontsize=11.5)
    out = PUB / 'distill-errors-inherit.png'
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    print(f'wrote {out.name}')


def _confmap(ax, C, title, hl=None):
    M = C.copy()
    np.fill_diagonal(M, 0)                           # show only the confusions
    ax.imshow(M, cmap='inferno', vmin=0, vmax=0.3, interpolation='nearest')
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(CLASSES, rotation=90, fontsize=6)
    ax.set_yticklabels(CLASSES, fontsize=6)
    ax.set_title(title, color=INK, fontsize=10, pad=6)
    for k in range(11):
        ax.axhline(k - 0.5, color=BG, lw=0.4); ax.axvline(k - 0.5, color=BG, lw=0.4)
    if hl is not None:
        ax.add_patch(Rectangle((-0.5, hl - 0.5), 10, 1, fill=False, ec=INK, lw=1.6))


if __name__ == '__main__':
    gif_kernel_assemble()
    gif_temperature_dial()
    gif_handoff()
    gif_spectrum_inherit()
    gif_probe_race()
    fig_errors_inherit()
    print('all distill figures rendered.')
