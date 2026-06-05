#!/usr/bin/env python3
"""Render a GIF of the bilinear score splitting into a symmetric metric and an
antisymmetric directed part: B = S + A, scores s_ij = x_i^T B x_j."""

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
OUT_GIF = ROOT / "public" / "qk-bilinear-decomp.gif"
OUT_PREVIEW = ROOT / "public" / "qk-bilinear-decomp-preview.png"

W, H = 1100, 540
FPS = 16
FRAMES = 64

BG = "#fbfaf6"
PANEL = "#ffffff"
INK = "#181818"
MUTED = "#666a70"
BORDER = "#ded9cb"
BLUE = "#4a7fb3"
ACCENT = "#b3661b"

TOKENS = ["a", "b", "c", "d", "e", "f"]
X = jnp.array(
    [[-0.82, 0.18], [-0.30, 0.74], [0.10, -0.50], [0.66, 0.58], [0.86, -0.22], [-0.48, -0.66]]
)


def sym_metric(beta: float) -> jnp.ndarray:
    c, s = jnp.cos(beta), jnp.sin(beta)
    l1, l2 = 1.25, 0.45
    return jnp.array(
        [[l1 * c * c + l2 * s * s, (l1 - l2) * c * s], [(l1 - l2) * c * s, l1 * s * s + l2 * c * c]]
    )


S = sym_metric(0.5)
J = jnp.array([[0.0, -1.0], [1.0, 0.0]])


def scores(alpha: float):
    msym = X @ S @ X.T
    manti = alpha * (X @ J @ X.T)
    return np.asarray(msym), np.asarray(manti), np.asarray(msym + manti)


def panel(ax, title: str) -> None:
    ax.set_facecolor(PANEL)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.text(0.5, 1.05, title, transform=ax.transAxes, ha="center", va="bottom", color=INK, fontsize=11, weight="bold")


def heat(ax, mat: np.ndarray, vmax: float) -> None:
    ax.imshow(mat, cmap="BrBG", vmin=-vmax, vmax=vmax, aspect="equal")
    n = mat.shape[0]
    for i in range(n):  # outline the diagonal
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, lw=1.0))


def draw_frame(frame: int) -> np.ndarray:
    alpha = float(1.4 * np.sin(2 * np.pi * frame / FRAMES))
    msym, manti, mtot = scores(alpha)
    vmax = max(1e-6, np.max(np.abs(mtot)), np.max(np.abs(msym)))
    asym = float(np.max(np.abs(mtot - mtot.T)))

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    fig.text(0.5, 0.95, "A score splits into a symmetric metric and an antisymmetric direction", ha="center", color=INK, fontsize=18, weight="bold")
    fig.text(0.5, 0.905, r"$s_{ij}=x_i^\top B\,x_j = x_i^\top S\,x_j + x_i^\top A\,x_j,\qquad B = S + \alpha\,A$", ha="center", color=MUTED, fontsize=12)

    axc = fig.add_axes([0.04, 0.18, 0.26, 0.56])
    a1 = fig.add_axes([0.345, 0.30, 0.185, 0.40])
    a2 = fig.add_axes([0.555, 0.30, 0.185, 0.40])
    a3 = fig.add_axes([0.765, 0.30, 0.185, 0.40])

    # token cloud + metric ellipse (fixed in alpha)
    panel(axc, "tokens + metric  S")
    axc.set_xlim(-1.25, 1.25)
    axc.set_ylim(-1.25, 1.25)
    th = np.linspace(0, 2 * np.pi, 80)
    Snp = np.asarray(S)
    rr = np.array([1.0 / np.sqrt(max(1e-3, np.array([np.cos(t), np.sin(t)]) @ Snp @ np.array([np.cos(t), np.sin(t)]))) for t in th])
    axc.plot(rr * np.cos(th), rr * np.sin(th), color=ACCENT, lw=1.4, alpha=0.7)
    Xnp = np.asarray(X)
    axc.scatter(Xnp[:, 0], Xnp[:, 1], s=55, color=INK, zorder=3)
    for i, tok in enumerate(TOKENS):
        axc.text(Xnp[i, 0], Xnp[i, 1] + 0.13, tok, ha="center", color=MUTED, fontsize=9)
    axc.axhline(0, color=BORDER, lw=0.8)
    axc.axvline(0, color=BORDER, lw=0.8)

    panel(a1, r"symmetric  $x^\top S x$")
    heat(a1, msym, vmax)
    panel(a2, r"antisym  $\alpha\, x^\top A x$")
    heat(a2, manti, vmax)
    panel(a3, r"total  $x^\top B x$")
    heat(a3, mtot, vmax)

    fig.text(0.5, 0.135, rf"$\alpha = {alpha:+.2f}$    max$|s_{{ij}}-s_{{ji}}| = {asym:.2f}$", ha="center", color=ACCENT, fontsize=12, weight="bold")
    fig.text(0.5, 0.075, "the antisymmetric part is empty on the diagonal — the boxed self-scores never move, only the off-diagonal asymmetry grows with " + r"$\alpha$", ha="center", color=MUTED, fontsize=10.5)

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba


def main() -> None:
    frames = []
    for frame in range(FRAMES):
        frames.append(draw_frame(frame))
        if frame == FRAMES // 4:  # a frame with visible asymmetry for the static preview
            Image.fromarray(frames[-1]).save(OUT_PREVIEW)
        if (frame + 1) % 16 == 0:
            print(f"rendered {frame + 1}/{FRAMES} frames")
    imageio.mimsave(OUT_GIF, frames, duration=1 / FPS, loop=0, palettesize=128, subrectangles=True)
    print(f"wrote {OUT_GIF}")
    print(f"wrote {OUT_PREVIEW}")


if __name__ == "__main__":
    main()
