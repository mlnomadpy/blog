#!/usr/bin/env python3
"""Render a GIF of the non-identifiability of a convex decomposition: the same
output y, held provably fixed, is rewritten each frame as a different valid
convex combination of the same prototypes. The coefficient bars morph through
the whole family of decompositions (its endpoints are the sparse Caratheodory
ones) while a live readout shows max |y(t) - y(0)| staying at numerical zero.
Uses exactly the prototypes and decompositions from the post's code block."""

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
OUT_GIF = ROOT / "public" / "convex-readout-nonid.gif"
OUT_PREVIEW = ROOT / "public" / "convex-readout-nonid-preview.png"

W, H = 1100, 520
FPS = 14
FRAMES = 112

BG = "#fbfaf6"
PANEL = "#ffffff"
INK = "#181818"
MUTED = "#666a70"
BORDER = "#ded9cb"
ACCENT = "#b3661b"
PALETTE = ["#b3661b", "#4a7fb3", "#3a8f5e", "#9a4f9c"]

# The post's exact numbers: 4 prototypes in the plane, one fixed output y,
# and two exact convex decompositions of it.
R = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
Y0 = jnp.array([0.1, 0.1])
A1 = jnp.array([0.5, 0.1, 0.4, 0.0])
A2 = jnp.array([0.1, 0.5, 0.0, 0.4])
K = R.shape[0]

# a(t) = (1-t) a1 + t a2 stays a valid convex decomposition of y for every
# t in [0, 1]: the set of valid decompositions is convex, and here t = 0 and
# t = 1 are its sparse endpoints (only three prototypes each).
assert bool(jnp.allclose(A1 @ R, Y0)) and bool(jnp.allclose(A2 @ R, Y0))

HULL = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 0.0]])


def decomposition(t: float):
    a = (1.0 - t) * A1 + t * A2
    y = a @ R
    # every frame is a checked, exact convex decomposition of the same y
    assert bool((a >= 0.0).all())
    assert abs(float(a.sum()) - 1.0) < 1e-12
    dev = float(jnp.abs(y - Y0).max())
    assert dev < 1e-12
    return np.asarray(a), np.asarray(y), dev


def panel(ax, title: str) -> None:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.set_title(title, color=INK, fontsize=11.5, weight="bold", pad=8)


def draw_frame(frame: int, max_dev: float) -> tuple[np.ndarray, float]:
    t = 0.5 * (1.0 - np.cos(2 * np.pi * frame / FRAMES))
    a, y, dev = decomposition(float(t))
    max_dev = max(max_dev, dev)

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    fig.text(0.5, 0.945, "Same output, many convex decompositions", ha="center", color=INK, fontsize=17, weight="bold")
    fig.text(0.5, 0.895, r"every frame satisfies $a_u \geq 0$, $\sum_u a_u = 1$, $\sum_u a_u r_u = y$: the shares change, the output cannot", ha="center", color=MUTED, fontsize=11.5)

    axL = fig.add_axes([0.065, 0.20, 0.40, 0.60])
    axR = fig.add_axes([0.545, 0.20, 0.40, 0.60])

    panel(axL, "residual stream: y never moves")
    axL.set_xlim(-1.25, 1.25)
    axL.set_ylim(-1.25, 1.25)
    axL.set_aspect("equal")
    axL.set_xticks([])
    axL.set_yticks([])
    axL.plot(HULL[:, 0], HULL[:, 1], color=ACCENT, lw=1.2)
    axL.fill(HULL[:, 0], HULL[:, 1], color=ACCENT, alpha=0.06)
    Rnp = np.asarray(R)
    for u in range(K):
        axL.plot([Rnp[u, 0], y[0]], [Rnp[u, 1], y[1]], color=PALETTE[u], lw=0.5 + 7 * a[u], alpha=0.15 + 0.65 * a[u])
    label_off = [(0.09, 0.05), (0.09, 0.05), (0.05, 0.09), (0.09, -0.16)]
    for u in range(K):
        axL.scatter([Rnp[u, 0]], [Rnp[u, 1]], s=50 + 420 * a[u], color=PALETTE[u], zorder=3)
        axL.text(Rnp[u, 0] + label_off[u][0], Rnp[u, 1] + label_off[u][1], f"r{u+1}", color=MUTED, fontsize=9.5)
    axL.scatter([y[0]], [y[1]], s=130, color=INK, zorder=5, edgecolors=PANEL, linewidths=1.6)
    axL.text(y[0] + 0.08, y[1] - 0.13, "y", color=INK, fontsize=13, weight="bold")

    panel(axR, "the shares a_u, morphing through valid decompositions")
    axR.set_ylim(0, 0.62)
    axR.set_xlim(-0.6, K - 0.4)
    idx = np.arange(K)
    a1np, a2np = np.asarray(A1), np.asarray(A2)
    axR.bar(idx, a1np, width=0.78, fill=False, edgecolor=BORDER, lw=1.2, linestyle=(0, (3, 2)))
    axR.bar(idx, a2np, width=0.78, fill=False, edgecolor=MUTED, lw=1.0, linestyle=(0, (1, 2)))
    axR.bar(idx, a, width=0.62, color=PALETTE, alpha=0.9, zorder=3)
    for u in range(K):
        axR.text(u, a[u] + 0.015, f"{a[u]:.2f}", ha="center", color=INK, fontsize=9.5)
    axR.set_xticks(idx)
    axR.set_xticklabels([f"r{u+1}" for u in range(K)], color=MUTED, fontsize=10)
    axR.set_yticks([])

    fig.text(0.5, 0.065, f"Σ a = {float(a.sum()):.3f}     min a = {float(a.min()):+.3f}     max |y(t) - y(0)| over the run so far: {max_dev:.1e}", ha="center", color=MUTED, fontsize=11.5)

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba, max_dev


def main() -> None:
    frames = []
    max_dev = 0.0
    for frame in range(FRAMES):
        rgba, max_dev = draw_frame(frame, max_dev)
        frames.append(rgba)
        if frame == FRAMES // 4:
            Image.fromarray(frames[-1]).save(OUT_PREVIEW)
        if (frame + 1) % 28 == 0:
            print(f"rendered {frame + 1}/{FRAMES} frames")
    print(f"max |y(t) - y(0)| across the whole run: {max_dev:.3e}")
    imageio.mimsave(OUT_GIF, frames, duration=1 / FPS, loop=0, palettesize=128, subrectangles=True)
    print(f"wrote {OUT_GIF} ({OUT_GIF.stat().st_size / 1e6:.2f} MB)")
    print(f"wrote {OUT_PREVIEW}")


if __name__ == "__main__":
    main()
