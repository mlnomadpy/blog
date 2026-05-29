"""Render captured frames to a GIF, styled to match the blog.

Matplotlib draws each frame; imageio encodes the sequence. Colours and fonts
are pulled from the blog's design tokens so the GIFs look like they belong in
the post rather than like generic matplotlib output.
"""

from __future__ import annotations

import glob
import os

import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from data import PALETTE  # noqa: E402

# Blog palette (src/styles/global.css + LossExplorer.astro).
BG = "#fbf8f1"
PANEL = "#ffffff"
BORDER = "#e4e1d6"
FG = "#1a1a1a"
MUTED = "#5a5f66"
ACCENT = "#b3661b"


def _register_lora() -> str:
    """Register the bundled Lora serif if present; return the family name."""
    here = os.path.dirname(__file__)
    candidates = glob.glob(
        os.path.join(here, "..", "..", "node_modules", "@fontsource",
                     "lora", "files", "lora-latin-*-normal.woff")
    )
    # matplotlib's freetype can't read .woff; look for a .ttf fallback too.
    ttfs = glob.glob(
        os.path.join(here, "..", "..", "node_modules", "@fontsource",
                     "lora", "files", "*.ttf")
    )
    for path in ttfs:
        try:
            fm.fontManager.addfont(path)
            return fm.FontProperties(fname=path).get_name()
        except Exception:
            pass
    return "DejaVu Serif"  # safe fallback; ships with matplotlib


SERIF = _register_lora()


def _draw_frame(z, loss, acc, labels, *, title, on_sphere, size_px, dpi):
    fig_in = size_px / dpi
    fig = plt.figure(figsize=(fig_in, fig_in), dpi=dpi)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.92])
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(BORDER)
    ax.set_xticks([])
    ax.set_yticks([])

    lim = 1.25 if on_sphere else 2.6
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")

    # Axes through the origin + (for sphere losses) the unit circle.
    ax.axhline(0, color=BORDER, lw=1, zorder=0)
    ax.axvline(0, color=BORDER, lw=1, zorder=0)
    if on_sphere:
        ax.add_patch(Circle((0, 0), 1.0, fill=False, ec=BORDER,
                            ls=(0, (2, 3)), lw=1, zorder=1))

    colors = [PALETTE[int(l) % len(PALETTE)] for l in labels]
    ax.scatter(z[:, 0], z[:, 1], c=colors, s=46, edgecolors="white",
               linewidths=0.7, zorder=3)

    # Per-class centroid crosses (the nearest-centroid metric, made visible).
    for c in np.unique(labels):
        cz = z[labels == c].mean(axis=0)
        ax.plot(cz[0], cz[1], marker="+", ms=10, mew=2.2,
                color=PALETTE[int(c) % len(PALETTE)], zorder=4)

    ax.text(0.03, 0.965, title, transform=ax.transAxes, ha="left", va="top",
            family=SERIF, fontsize=12.5, fontweight="bold", color=FG)
    readout = (f"acc {acc * 100:.0f}%"
               + ("" if np.isnan(loss) else f"  ·  L {loss:.3f}"))
    ax.text(0.03, 0.905, readout, transform=ax.transAxes, ha="left", va="top",
            family="monospace", fontsize=9.5, color=MUTED)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def render_gif(frames, labels, out_path, *, title, on_sphere,
               fps=20, size_px=480, dpi=100, hold=1.2):
    """Render the captured ``(z, loss, acc)`` frames to ``out_path``."""
    imgs = [
        _draw_frame(z, loss, acc, labels, title=title, on_sphere=on_sphere,
                    size_px=size_px, dpi=dpi)
        for (z, loss, acc) in frames
    ]
    # Hold the final, converged frame so the geometry is readable on loop.
    imgs += [imgs[-1]] * int(hold * fps)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    imageio.mimsave(out_path, imgs, fps=fps, loop=0, subrectangles=True)
    return os.path.getsize(out_path)


def render_grid_gif(per_loss_frames, labels_by_loss, titles, out_path, *,
                    on_sphere_by_loss, fps=20, cell_px=300, dpi=100, hold=1.5):
    """Render a 2x3 grid GIF: all six losses on a shared step counter."""
    n = len(per_loss_frames)
    cols, rows = 3, 2
    length = max(len(f) for f in per_loss_frames)
    fig_w = cols * cell_px / dpi
    fig_h = rows * cell_px / dpi

    imgs = []
    for t in range(length):
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        fig.patch.set_facecolor(BG)
        for i in range(n):
            frames = per_loss_frames[i]
            z, loss, acc = frames[min(t, len(frames) - 1)]
            labels = labels_by_loss[i]
            on_sphere = on_sphere_by_loss[i]
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.set_facecolor(PANEL)
            for s in ax.spines.values():
                s.set_color(BORDER)
            ax.set_xticks([])
            ax.set_yticks([])
            lim = 1.25 if on_sphere else 2.6
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            if on_sphere:
                ax.add_patch(Circle((0, 0), 1.0, fill=False, ec=BORDER,
                                    ls=(0, (2, 3)), lw=0.8, zorder=1))
            colors = [PALETTE[int(l) % len(PALETTE)] for l in labels]
            ax.scatter(z[:, 0], z[:, 1], c=colors, s=22, edgecolors="white",
                       linewidths=0.5, zorder=3)
            ax.set_title(titles[i], family=SERIF, fontsize=9.5,
                         color=FG, pad=3)
            ax.text(0.03, 0.04, f"acc {acc * 100:.0f}%", transform=ax.transAxes,
                    ha="left", va="bottom", family="monospace", fontsize=7.5,
                    color=MUTED)
        fig.tight_layout(pad=0.6)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        imgs.append(buf)

    imgs += [imgs[-1]] * int(hold * fps)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    imageio.mimsave(out_path, imgs, fps=fps, loop=0, subrectangles=True)
    return os.path.getsize(out_path)
