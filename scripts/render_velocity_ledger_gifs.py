"""Figures for the "Transformers With a Velocity Ledger" JAX companion (Arc D2).

Rendered locally from the results bundle (no training, no Kaggle):
  scripts/results/kgl_blog-velocity-ledger-v2/velocity_ledger.npz
  scripts/results/kgl_blog-velocity-ledger-v2/velocity_ledger_summary.json

GIF (real motion only, one):
  velocity-ledger-straighten.gif   the residual-stream 2D path (real segment
                                   lengths + real turning angles) FORMING over
                                   the four training checkpoints, plain vs ledger
                                   side by side. The plain path grows into a long
                                   wander; the ledger's stays short and straight.
                                   This is a real temporal process (the path
                                   genuinely straightens as training proceeds), so
                                   it earns motion.

STATIC PNGs (categorical scoreboards / final profiles, no process to animate):
  velocity-ledger-tie.png          best-val, 3 seeds, 4 variants: the honest tie.
  velocity-ledger-pathbars.png     path length + turn angle bars (the real signal).
  velocity-ledger-steps.png        per-sub-update step size + velocity-stream norm.
  velocity-ledger-curves.png       train + val loss curves (seed 0), all 4 variants.

Run: python3 scripts/render_velocity_ledger_gifs.py [straighten|tie|pathbars|steps|curves|all]
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / 'public' / 'blog'
PUB.mkdir(parents=True, exist_ok=True)
SRC = ROOT / 'scripts' / 'results' / 'kgl_blog-velocity-ledger-v2'
NPZ = np.load(SRC / 'velocity_ledger.npz')
SUM = json.load(open(SRC / 'velocity_ledger_summary.json'))

BG = '#0e0d0b'; PANEL = '#16140f'; INK = '#e8e2d4'; MUTED = '#9a9282'; LINE = '#3a352c'
VCOL = {'plain': '#c2553a', 'ngpt_lite': '#b06fb0', 'ledger': '#5a9fd0', 'ngpt_ledger': '#4fc48f'}
VLAB = {'plain': 'plain', 'ngpt_lite': 'ngpt-lite', 'ledger': 'ledger', 'ngpt_ledger': 'ngpt-ledger'}
ORDER = ['plain', 'ngpt_lite', 'ledger', 'ngpt_ledger']
CKPTS = SUM['config']['ckpt_steps']            # [0, 1200, 6000, 12000]
plt.rcParams.update({'figure.facecolor': BG, 'savefig.facecolor': BG,
                     'text.color': INK, 'axes.edgecolor': LINE, 'font.size': 11})


def comp(v, field):
    return SUM['comparison'][v][field]


def path_len(v, seed, ck):
    return float(NPZ[f'{v}_s{seed}/ck{ck}_seglen'].sum())


def seed_mean(v, ck):
    return float(np.mean([path_len(v, s, ck) for s in range(3)]))


def polyline(seglen, turn):
    """2D shadow with real segment lengths and real turning angles (matches the
    web export exactly)."""
    heading, x, y, sign = 0.0, 0.0, 0.0, 1.0
    nodes = [(0.0, 0.0)]
    for i, L in enumerate(seglen):
        x += L * math.cos(heading); y += L * math.sin(heading)
        nodes.append((x, y))
        if i < len(turn):
            heading += sign * math.radians(turn[i]); sign = -sign
    return np.array(nodes)


def fig_rgba(fig):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba


def save_gif(path, frames, fps, hold=18, palettesize=96):
    Image.fromarray(frames[-1]).save(str(path).replace('.gif', '-preview.png'))
    dur = [1000.0 / fps] * len(frames)
    dur[-1] += hold * 1000.0 / fps
    imageio.mimsave(path, frames, duration=dur, loop=0, palettesize=palettesize, subrectangles=True)
    im = Image.open(path); im.seek(0)
    print(f'wrote {path} ({os.path.getsize(path)//1024} KB, {im.n_frames} frames)')


def style(ax):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values(): s.set_color(LINE)
    ax.tick_params(colors=MUTED)


def ease(t):
    t = np.clip(t, 0, 1); return t * t * (3 - 2 * t)


# ── GIF: the residual-stream path forming over training (plain vs ledger) ─────
def gif_straighten():
    # precompute the 2D polyline at each checkpoint for both variants (seed 0)
    paths = {}
    for v in ('plain', 'ledger'):
        paths[v] = [polyline(NPZ[f'{v}_s0/ck{ck}_seglen'], NPZ[f'{v}_s0/ck{ck}_turn'])
                    for ck in range(len(CKPTS))]
    # shared bound so lengths compare honestly
    B = max(np.abs(np.concatenate([p for v in paths for p in paths[v]])).max() for _ in [0]) * 1.08

    frames = []
    NPER = 20          # frames to draw each checkpoint's path segment by segment
    NHOLD = 8          # hold frames between checkpoints
    for ci in range(len(CKPTS)):
        for f in range(NPER + NHOLD):
            fig, axs = plt.subplots(1, 2, figsize=(8.4, 4.5), dpi=112)
            step = CKPTS[ci]
            fig.suptitle(f'residual-stream path forming through training  ·  step {"init" if step==0 else step}',
                         color=INK, fontsize=12, y=0.98)
            for j, v in enumerate(('plain', 'ledger')):
                ax = axs[j]; style(ax)
                nodes = paths[v][ci]
                pl = float(NPZ[f'{v}_s0/ck{ci}_seglen'].sum())
                # ghost the previous checkpoints faintly for a sense of history
                for pc in range(ci):
                    pn = paths[v][pc]
                    ax.plot(pn[:, 0], pn[:, 1], color=VCOL[v], alpha=0.12, lw=1)
                # draw the current path up to an eased fraction of its segments
                frac = ease(min(1.0, f / NPER))
                nseg = len(nodes) - 1
                upto = 1 + int(round(frac * nseg))
                ax.plot(nodes[:upto, 0], nodes[:upto, 1], color=VCOL[v], lw=2.4)
                ax.plot(nodes[upto - 1, 0], nodes[upto - 1, 1], 'o', color=VCOL[v], ms=6)
                ax.plot(0, 0, 'o', color=MUTED, ms=4)
                ax.text(0.03, 0.96, VLAB[v], color=VCOL[v], fontsize=12, fontweight='bold',
                        transform=ax.transAxes, va='top')
                ax.text(0.03, 0.06, f'path length {pl:.0f}', color=MUTED, fontsize=10,
                        transform=ax.transAxes, va='bottom')
                ax.set_xlim(-B, B); ax.set_ylim(-B, B); ax.set_aspect('equal')
                ax.set_xticks([]); ax.set_yticks([])
            fig.text(0.5, 0.02,
                     'a 2D shadow: every segment length and bend is the real measured number (probe batch)',
                     color=MUTED, fontsize=8.5, ha='center')
            fig.tight_layout(rect=[0, 0.04, 1, 0.94])
            frames.append(fig_rgba(fig))
    save_gif(PUB / 'velocity-ledger-straighten.gif', frames, fps=12)


# ── STATIC: the honest best-val tie ───────────────────────────────────────────
def png_tie():
    fig, ax = plt.subplots(figsize=(7.6, 3.6), dpi=130); style(ax)
    ys = list(range(len(ORDER)))[::-1]
    lo = min(comp(v, 'best_val')['min'] for v in ORDER)
    hi = max(comp(v, 'best_val')['max'] for v in ORDER)
    pad = (hi - lo) * 0.35
    for y, v in zip(ys, ORDER):
        bv = comp(v, 'best_val')
        ax.plot([bv['min'], bv['max']], [y, y], color=VCOL[v], lw=3, alpha=0.6, solid_capstyle='round')
        ax.plot(bv['mean'], y, 'o', color=VCOL[v], ms=9)
        ax.text(bv['max'] + (hi - lo) * 0.03, y, f"{bv['mean']:.3f}", color=INK, va='center', fontsize=10, fontweight='bold')
    ax.set_yticks(ys); ax.set_yticklabels([VLAB[v] for v in ORDER], fontsize=11)
    for t, v in zip(ax.get_yticklabels(), ORDER): t.set_color(VCOL[v])
    ax.set_xlim(lo - pad, hi + pad + (hi - lo) * 0.12)
    ax.set_xlabel('best validation loss (lower = better), 3 seeds', color=MUTED)
    ax.set_title('On quality the four variants tie: best-val lands inside seed noise', color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(PUB / 'velocity-ledger-tie.png'); plt.close(fig)
    print('wrote velocity-ledger-tie.png')


# ── STATIC: path length + turn angle bars ─────────────────────────────────────
def png_pathbars():
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.7), dpi=130)
    for ax, (field, title, unit) in zip(axs, [('path', 'total path length', ''),
                                              ('turn', 'mean turning angle', '°')]):
        style(ax)
        ys = list(range(len(ORDER)))[::-1]
        for y, v in zip(ys, ORDER):
            if field == 'path':
                val = seed_mean(v, len(CKPTS) - 1)
            else:
                val = float(np.mean([NPZ[f'{v}_s{s}/ck{len(CKPTS)-1}_turn'].mean() for s in range(3)]))
            ax.barh(y, val, color=VCOL[v], alpha=0.85, height=0.55)
            ax.text(val, y, f' {val:.0f}{unit}', color=INK, va='center', fontsize=10, fontweight='bold')
        ax.set_yticks(ys); ax.set_yticklabels([VLAB[v] for v in ORDER], fontsize=10)
        for t, v in zip(ax.get_yticklabels(), ORDER): t.set_color(VCOL[v])
        ax.set_title(title, color=INK, fontsize=11)
    fig.suptitle('The ledger shortens and straightens the depth path (final checkpoint, 3 seeds)',
                 color=INK, fontsize=12, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(PUB / 'velocity-ledger-pathbars.png'); plt.close(fig)
    print('wrote velocity-ledger-pathbars.png')


# ── STATIC: per-sub-update step size + velocity-stream norm ───────────────────
def png_steps():
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.8), dpi=130)
    ax = axs[0]; style(ax)
    for v in ORDER:
        s = NPZ[f'{v}_s0/ck{len(CKPTS)-1}_seglen']
        ax.plot(range(1, len(s) + 1), s, '-o', color=VCOL[v], ms=3, lw=2, label=VLAB[v])
    ax.set_xlabel('sub-update (embedding → readout)', color=MUTED)
    ax.set_ylabel('displacement', color=MUTED)
    ax.set_title('step size per sub-update', color=INK, fontsize=11)
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=LINE, labelcolor=INK)
    ax = axs[1]; style(ax)
    for v in ('ledger', 'ngpt_ledger'):
        vn = NPZ[f'{v}_s0/ck{len(CKPTS)-1}_vnorm']
        ax.plot(range(1, len(vn) + 1), vn, '-o', color=VCOL[v], ms=3, lw=2, label=VLAB[v])
    ax.set_xlabel('sub-update (embedding → readout)', color=MUTED)
    ax.set_ylabel('velocity-stream norm', color=MUTED)
    ax.set_title('the ledger fills from zero', color=INK, fontsize=11)
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=LINE, labelcolor=INK)
    fig.suptitle('The plain step balloons toward the readout; the ledger keeps each step small',
                 color=INK, fontsize=12, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(PUB / 'velocity-ledger-steps.png'); plt.close(fig)
    print('wrote velocity-ledger-steps.png')


# ── STATIC: loss curves (seed 0), all four variants ──────────────────────────
def png_curves():
    fig, ax = plt.subplots(figsize=(7.8, 4.0), dpi=130); style(ax)
    for v in ORDER:
        ts = NPZ[f'{v}_s0/train_steps']; tl = NPZ[f'{v}_s0/train_loss']
        vs = NPZ[f'{v}_s0/val_steps']; vl = NPZ[f'{v}_s0/val_loss']
        ax.plot(ts, tl, color=VCOL[v], alpha=0.35, lw=1)
        ax.plot(vs, vl, color=VCOL[v], lw=2, label=VLAB[v])
    ax.set_xlabel('training step', color=MUTED); ax.set_ylabel('loss', color=MUTED)
    ax.set_ylim(1.3, 2.2)
    ax.set_title('Train (faint) and val (bold) loss, seed 0: the four converge together',
                 color=INK, fontsize=12)
    ax.legend(fontsize=9, facecolor=PANEL, edgecolor=LINE, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(PUB / 'velocity-ledger-curves.png'); plt.close(fig)
    print('wrote velocity-ledger-curves.png')


JOBS = {'straighten': gif_straighten, 'tie': png_tie, 'pathbars': png_pathbars,
        'steps': png_steps, 'curves': png_curves}
if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    for name, fn in JOBS.items():
        if which in ('all', name):
            fn()
