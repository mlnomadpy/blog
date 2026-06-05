#!/usr/bin/env python3
"""Render a GIF showing how the activation over hidden units decides the readout
regime. The same scores, read out through softmax / relu / gelu / identity, land
in different geometric regimes; only softmax keeps y inside the prototype hull."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
OUT_GIF = ROOT / "public" / "convex-readout-regimes.gif"
OUT_PREVIEW = ROOT / "public" / "convex-readout-regimes-preview.png"

W, H = 1100, 470
FPS = 14
FRAMES = 70

BG = "#fbfaf6"
PANEL = "#ffffff"
INK = "#181818"
MUTED = "#666a70"
BORDER = "#ded9cb"
BLUE = "#4a7fb3"
GREEN = "#3a8f5e"
REDO = "#c2553a"
PALETTE = ["#b3661b", "#4a7fb3", "#3a8f5e", "#9a4f9c", "#c2553a"]

R = jnp.array([[0.66, 0.22], [0.22, 0.80], [-0.62, 0.50], [-0.66, -0.36], [0.32, -0.72]])
K = R.shape[0]
ACTS = [("softmax(s)", jax.nn.softmax), ("relu(s)", jax.nn.relu), ("gelu(s)", jax.nn.gelu), ("identity s", lambda s: s)]


def convex_hull(pts: np.ndarray) -> np.ndarray:
    pts = np.array(sorted(map(tuple, pts)))

    def half(points):
        h = []
        for p in points:
            while len(h) >= 2 and np.cross(h[-1] - h[-2], p - h[-2]) <= 0:
                h.pop()
            h.append(p)
        return h[:-1]

    return np.array(half(pts) + half(pts[::-1]))


HULL = convex_hull(np.asarray(R))


def scores(phase: float) -> jnp.ndarray:
    bias = 0.15
    return jnp.array(
        [
            0.9 * np.cos(phase) + bias,
            0.95 * np.sin(phase * 0.85 + 0.6) + bias,
            0.8 * np.cos(phase * 1.25 + 1.2) + bias,
            0.9 * np.sin(phase * 1.1 - 0.5) + bias,
            0.85 * np.cos(phase * 0.7 - 1.1) + bias,
        ]
    )


def verdict(a: np.ndarray):
    if (a < -1e-6).any():
        return "linear", REDO
    if abs(float(a.sum()) - 1.0) < 1e-3:
        return "convex", GREEN
    return "conic", BLUE


def draw_frame(frame: int) -> np.ndarray:
    s = scores(2 * np.pi * frame / FRAMES)

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    fig.text(0.5, 0.93, "The activation over hidden units chooses the readout regime", ha="center", color=INK, fontsize=17, weight="bold")
    fig.text(0.5, 0.875, "same scores s, four nonlinearities — only softmax makes y a convex combination inside the hull", ha="center", color=MUTED, fontsize=11)

    Rnp = np.asarray(R)
    for col, (name, fn) in enumerate(ACTS):
        a = np.asarray(fn(s))
        y = a @ Rnp
        vname, vcol = verdict(a)

        x0 = 0.045 + col * 0.238
        axh = fig.add_axes([x0, 0.40, 0.205, 0.34])
        axh.set_xlim(-1.15, 1.15)
        axh.set_ylim(-1.15, 1.15)
        axh.set_aspect("equal")
        axh.set_xticks([])
        axh.set_yticks([])
        axh.set_facecolor(PANEL)
        for spine in axh.spines.values():
            spine.set_color(BORDER)
        axh.add_patch(plt.Polygon(HULL, closed=True, facecolor=vcol, alpha=0.10, edgecolor=vcol, lw=1.2))
        axh.scatter(Rnp[:, 0], Rnp[:, 1], s=18, color=PALETTE, zorder=3)
        axh.scatter([y[0]], [y[1]], s=70, color=INK, zorder=5, edgecolors=PANEL, linewidths=1.4)
        axh.set_title(name, color=INK, fontsize=12, weight="bold", pad=6)
        axh.text(0.5, 1.005, "", transform=axh.transAxes)

        # verdict chip
        fig.text(x0 + 0.102, 0.355, vname, ha="center", color=vcol, fontsize=12, weight="bold")

        # coefficient bars (signed)
        axb = fig.add_axes([x0, 0.13, 0.205, 0.17])
        axb.set_facecolor(PANEL)
        scale = max(0.8, float(np.max(np.abs(a))))
        axb.bar(np.arange(K), a / scale, color=[PALETTE[i] if a[i] >= 0 else BLUE for i in range(K)], alpha=0.9)
        axb.axhline(0, color=BORDER, lw=1)
        axb.set_ylim(-1.05, 1.05)
        axb.set_xticks([])
        axb.set_yticks([])
        for spine in axb.spines.values():
            spine.set_color(BORDER)
        axb.text(0.02, 0.86, f"Σa={float(a.sum()):.2f}", transform=axb.transAxes, color=MUTED, fontsize=8.5)

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba


def main() -> None:
    frames = []
    for frame in range(FRAMES):
        frames.append(draw_frame(frame))
        if frame == FRAMES // 3:
            Image.fromarray(frames[-1]).save(OUT_PREVIEW)
        if (frame + 1) % 20 == 0:
            print(f"rendered {frame + 1}/{FRAMES} frames")
    imageio.mimsave(OUT_GIF, frames, duration=1 / FPS, loop=0, palettesize=128, subrectangles=True)
    print(f"wrote {OUT_GIF}")
    print(f"wrote {OUT_PREVIEW}")


if __name__ == "__main__":
    main()
