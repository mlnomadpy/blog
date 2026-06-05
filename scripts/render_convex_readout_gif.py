#!/usr/bin/env python3
"""Render a GIF of a Nadaraya-Watson kernel readout: a query x roams the input
space, its normalized kernel similarities to input prototypes become convex
coefficients, and the readout y = sum a_u r_u traces the interior of the output
prototype hull, never leaving it."""

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
OUT_GIF = ROOT / "public" / "convex-readout-field.gif"
OUT_PREVIEW = ROOT / "public" / "convex-readout-field-preview.png"

W, H = 1100, 560
FPS = 18
FRAMES = 90
BW = 0.55

BG = "#fbfaf6"
PANEL = "#ffffff"
INK = "#181818"
MUTED = "#666a70"
BORDER = "#ded9cb"
ACCENT = "#b3661b"
PALETTE = ["#b3661b", "#4a7fb3", "#3a8f5e", "#9a4f9c", "#c2553a"]

P = jnp.array([[-0.72, 0.58], [0.70, 0.66], [0.86, -0.42], [0.02, -0.86], [-0.86, -0.28]])  # input
R = jnp.array([[0.05, 0.82], [0.76, 0.18], [0.46, -0.70], [-0.56, -0.62], [-0.82, 0.36]])  # output
K = P.shape[0]


def convex_hull(pts: np.ndarray) -> np.ndarray:
    pts = sorted(map(tuple, pts))
    pts = np.array(pts)

    def half(points):
        h = []
        for p in points:
            while len(h) >= 2 and np.cross(h[-1] - h[-2], p - h[-2]) <= 0:
                h.pop()
            h.append(p)
        return h[:-1]

    return np.array(half(pts) + half(pts[::-1]))


HULL = convex_hull(np.asarray(R))


def readout(x: jnp.ndarray):
    d2 = jnp.sum((x[None, :] - P) ** 2, axis=-1)
    w = jnp.exp(-d2 / (2.0 * BW**2))
    a = w / w.sum()
    y = a @ R
    return np.asarray(a), np.asarray(y)


def path(frame: int) -> jnp.ndarray:
    t = 2 * np.pi * frame / FRAMES
    return jnp.array([0.92 * np.cos(t), 0.86 * np.sin(2 * t + 0.6)])


TRAIL: list[np.ndarray] = []


def disc(ax, title: str) -> None:
    ax.set_facecolor(PANEL)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axhline(0, color=BORDER, lw=0.8)
    ax.axvline(0, color=BORDER, lw=0.8)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.set_title(title, color=INK, fontsize=11.5, weight="bold", pad=8)


def draw_frame(frame: int) -> np.ndarray:
    x = path(frame)
    a, y = readout(x)
    xnp = np.asarray(x)
    TRAIL.append(y.copy())
    if len(TRAIL) > 70:
        TRAIL.pop(0)

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    fig.text(0.5, 0.95, "Kernel coefficients keep the readout inside the prototype hull", ha="center", color=INK, fontsize=17, weight="bold")
    fig.text(0.5, 0.905, r"$a_u(x)=\kappa(x,p_u)/\sum_v \kappa(x,p_v)\geq 0,\quad \sum_u a_u = 1,\quad y=\sum_u a_u r_u$", ha="center", color=MUTED, fontsize=11.5)

    axL = fig.add_axes([0.055, 0.26, 0.40, 0.56])
    axR = fig.add_axes([0.475, 0.26, 0.40, 0.56])
    axB = fig.add_axes([0.90, 0.26, 0.085, 0.56])

    disc(axL, "input space — query x")
    Pnp = np.asarray(P)
    for u in range(K):
        axL.plot([xnp[0], Pnp[u, 0]], [xnp[1], Pnp[u, 1]], color=PALETTE[u], lw=0.6 + 6 * a[u], alpha=0.2 + 0.6 * a[u])
    for u in range(K):
        axL.scatter([Pnp[u, 0]], [Pnp[u, 1]], s=40 + 360 * a[u], color=PALETTE[u], zorder=3)
        axL.text(Pnp[u, 0] + 0.06, Pnp[u, 1] + 0.06, f"p{u+1}", color=MUTED, fontsize=8.5)
    axL.scatter([xnp[0]], [xnp[1]], s=90, color=INK, zorder=5, edgecolors=PANEL, linewidths=1.5)
    axL.text(xnp[0] + 0.07, xnp[1] - 0.12, "x", color=INK, fontsize=12, weight="bold")

    disc(axR, "residual stream — y stays in the hull")
    axR.add_patch(plt.Polygon(HULL, closed=True, facecolor=ACCENT, alpha=0.08, edgecolor=ACCENT, lw=1.2))
    if len(TRAIL) > 1:
        tr = np.array(TRAIL)
        axR.plot(tr[:, 0], tr[:, 1], color=ACCENT, lw=1.0, alpha=0.35)
    Rnp = np.asarray(R)
    for u in range(K):
        axR.plot([Rnp[u, 0], y[0]], [Rnp[u, 1], y[1]], color=PALETTE[u], lw=0.5 + 4 * a[u], alpha=0.15 + 0.6 * a[u])
    for u in range(K):
        axR.scatter([Rnp[u, 0]], [Rnp[u, 1]], s=40 + 320 * a[u], color=PALETTE[u], zorder=3)
        axR.text(Rnp[u, 0] + 0.06, Rnp[u, 1] + 0.06, f"r{u+1}", color=MUTED, fontsize=8.5)
    axR.scatter([y[0]], [y[1]], s=110, color=INK, zorder=5, edgecolors=PANEL, linewidths=1.6)
    axR.text(y[0] + 0.07, y[1] - 0.12, "y", color=INK, fontsize=12, weight="bold")

    axB.set_facecolor(PANEL)
    axB.bar(np.arange(K), a, color=PALETTE, alpha=0.9)
    axB.set_ylim(0, 1)
    axB.set_xticks(np.arange(K))
    axB.set_xticklabels([f"{u+1}" for u in range(K)], color=MUTED, fontsize=8)
    axB.set_yticks([])
    for spine in axB.spines.values():
        spine.set_color(BORDER)
    axB.set_title("a_u", color=INK, fontsize=10, weight="bold", pad=6)

    fig.text(0.5, 0.075, rf"$\sum a_u = {float(a.sum()):.3f}$,  every coefficient $\geq 0$ — convexity is built into how the weights are formed, not checked afterward", ha="center", color=MUTED, fontsize=10.5)

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba


def main() -> None:
    frames = []
    for frame in range(FRAMES):
        frames.append(draw_frame(frame))
        if frame == FRAMES // 2:
            Image.fromarray(frames[-1]).save(OUT_PREVIEW)
        if (frame + 1) % 20 == 0:
            print(f"rendered {frame + 1}/{FRAMES} frames")
    imageio.mimsave(OUT_GIF, frames, duration=1 / FPS, loop=0, palettesize=128, subrectangles=True)
    print(f"wrote {OUT_GIF}")
    print(f"wrote {OUT_PREVIEW}")


if __name__ == "__main__":
    main()
