#!/usr/bin/env python3
"""Render a GIF of the linear-attention implementation path."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
OUT_GIF = ROOT / "public" / "linear-attention-pipeline.gif"
OUT_PREVIEW = ROOT / "public" / "linear-attention-pipeline-preview.png"

W, H = 1100, 560
FPS = 14
FRAMES = 56
N = 40
D = 8
M = 12
DV = 4

BG = "#fbfaf6"
PANEL = "#ffffff"
INK = "#181818"
MUTED = "#666a70"
BORDER = "#ded9cb"
BLUE = "#4a7fb3"
ACCENT = "#b3661b"
GREEN = "#3a8f5e"


def make_sequence() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    t = jnp.arange(N, dtype=jnp.float64)
    x = jnp.stack(
        [
            jnp.sin(t * 0.17),
            jnp.cos(t * 0.11),
            jnp.sin(t * 0.07 + 1.3),
            jnp.cos(t * 0.19 - 0.6),
            jnp.sin(t * 0.23),
            jnp.cos(t * 0.29),
            jnp.sin(t * 0.31 + 0.2),
            jnp.cos(t * 0.37 - 0.1),
        ],
        axis=1,
    )
    q = x * 0.9 + 0.05
    k = jnp.roll(x, 1, axis=0) * 0.9 - 0.03
    v = jnp.stack([jnp.sin(t * 0.2), jnp.cos(t * 0.16), jnp.sin(t * 0.09 + 1.0), jnp.cos(t * 0.13 - 0.4)], axis=1)
    j = jnp.arange(M, dtype=jnp.float64)
    omega = jnp.stack(
        [
            jnp.sin(j * 0.7),
            jnp.cos(j * 0.9),
            jnp.sin(j * 1.1 + 0.2),
            jnp.cos(j * 1.3 - 0.1),
            jnp.sin(j * 0.5 + 0.7),
            jnp.cos(j * 1.7),
            jnp.sin(j * 1.9 - 0.4),
            jnp.cos(j * 0.3 + 0.6),
        ],
        axis=1,
    )
    return q, k, v, omega


Q, K, V, OMEGA = make_sequence()


def positive_features(x: jnp.ndarray) -> jnp.ndarray:
    proj = x @ OMEGA.T
    norm = 0.5 * jnp.sum(x * x, axis=-1, keepdims=True)
    return jnp.exp(proj - norm) / jnp.sqrt(M)


PHI_Q = positive_features(Q)
PHI_K = positive_features(K)


def state_at(step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    upto = step + 1
    phi_k = PHI_K[:upto]
    values = V[:upto]
    s = jnp.einsum("tm,td->md", phi_k, values)
    z = jnp.sum(phi_k, axis=0)
    phi_q = PHI_Q[step]
    y = (phi_q @ s) / (phi_q @ z + 1e-6)
    return np.asarray(phi_q), np.asarray(PHI_K[step]), np.asarray(s), np.asarray(y)


def panel(ax, title: str, subtitle: str) -> None:
    ax.set_facecolor(PANEL)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.text(0.5, 1.035, title, transform=ax.transAxes, ha="center", va="bottom", color=INK, fontsize=12.5, weight="bold")
    if subtitle:
        ax.text(0.5, 0.94, subtitle, transform=ax.transAxes, ha="center", va="center", color=MUTED, fontsize=9.0)


def bars(ax, vals: np.ndarray, color: str, ylim: tuple[float, float] | None = None) -> None:
    xs = np.arange(len(vals))
    ax.bar(xs, vals, color=color, alpha=0.85)
    ax.axhline(0, color=BORDER, lw=1)
    ax.set_xlim(-0.6, len(vals) - 0.4)
    if ylim:
        ax.set_ylim(*ylim)


def draw_frame(frame: int) -> np.ndarray:
    step = frame % N
    phi_q, phi_k, s, y = state_at(step)

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    fig.text(0.5, 0.958, "Linear attention as a streaming recurrence", ha="center", va="center", color=INK, fontsize=21, weight="bold")
    fig.text(0.5, 0.918, r"Each token writes $\phi(k_i)v_i^T$ into S and $\phi(k_i)$ into z; the query reads $\phi(q_i)^T S / \phi(q_i)^T z$.", ha="center", va="center", color=MUTED, fontsize=11.5)

    ax0 = fig.add_axes([0.055, 0.55, 0.40, 0.24])
    ax1 = fig.add_axes([0.545, 0.55, 0.18, 0.24])
    ax2 = fig.add_axes([0.765, 0.55, 0.18, 0.24])
    ax3 = fig.add_axes([0.055, 0.13, 0.40, 0.26])
    ax4 = fig.add_axes([0.545, 0.13, 0.40, 0.26])

    panel(ax0, "sequence stream", "the selected N is fixed; the pointer advances")
    ax0.set_xlim(-1, N)
    ax0.set_ylim(-1, 1)
    for i in range(N):
        color = GREEN if i < step else MUTED
        alpha = 0.85 if i <= step else 0.23
        ax0.scatter([i], [0], s=34, color=color, alpha=alpha, zorder=3)
    ax0.scatter([step], [0], s=130, facecolors="none", edgecolors=ACCENT, linewidths=2.0, zorder=5)
    ax0.text(step, 0.34, f"i={step}", ha="center", va="bottom", color=ACCENT, fontsize=10, weight="bold")
    ax0.text(0.5, 0.18, r"$S_i=\sum_{j\leq i}\phi(k_j)v_j^T,\quad z_i=\sum_{j\leq i}\phi(k_j)$", transform=ax0.transAxes, ha="center", color=MUTED, fontsize=10)

    panel(ax1, r"key features", r"$\phi(k_i)$")
    bars(ax1, phi_k, GREEN, (0, max(0.8, phi_k.max() * 1.15)))

    panel(ax2, r"query features", r"$\phi(q_i)$")
    bars(ax2, phi_q, BLUE, (0, max(0.8, phi_q.max() * 1.15)))

    panel(ax3, "fixed-size state", r"$S_i \in R^{m \times d_v}$")
    vmax = max(1e-6, np.max(np.abs(s)))
    ax3.imshow(s.T, cmap="BrBG", aspect="auto", vmin=-vmax, vmax=vmax)
    ax3.set_xlabel("features m", color=MUTED, fontsize=9)
    ax3.set_ylabel("value dims", color=MUTED, fontsize=9)

    panel(ax4, "readout", "")
    bars(ax4, y, ACCENT, (-1.1, 1.1))
    ax4.text(0.5, -0.20, f"cost through step i: exact {(step + 1) ** 2:,} scores vs linear {(step + 1) * M:,} feature touches", transform=ax4.transAxes, ha="center", color=MUTED, fontsize=10)

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba


def main() -> None:
    frames = []
    for frame in range(FRAMES):
        frames.append(draw_frame(frame))
        if frame == 0:
            Image.fromarray(frames[-1]).save(OUT_PREVIEW)
        if (frame + 1) % 8 == 0:
            print(f"rendered {frame + 1}/{FRAMES} frames")
    imageio.mimsave(OUT_GIF, frames, duration=1 / FPS, loop=0, palettesize=128, subrectangles=True)
    print(f"wrote {OUT_GIF}")
    print(f"wrote {OUT_PREVIEW}")


if __name__ == "__main__":
    main()
