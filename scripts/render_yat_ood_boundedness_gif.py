#!/usr/bin/env python3
"""Render a GIF contrasting off-distribution behaviour of a ReLU unit and a Yat unit.

A 1-D line plot. A ReLU ramp relu(w·x + b) keeps climbing without bound as we
leave the data band, while a Yat kernel unit (w·x + b)²/((x − c)² + eps) stays
bounded — sharpening its peak as eps shrinks but never escaping the top of the
axis. We sweep eps from wide to narrow on a log schedule across the frames.
"""

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

ROOT = Path(__file__).resolve().parents[1]
OUT_GIF = ROOT / "public" / "yat-ood-boundedness.gif"
OUT_PREVIEW = ROOT / "public" / "yat-ood-boundedness-preview.png"

W, H = 1100, 420
FPS = 12
END_HOLD = 12

BG = "#fbfaf6"
PANEL = "#ffffff"
INK = "#181818"
MUTED = "#666a70"
BORDER = "#ded9cb"
ACCENT = "#b3661b"
BLUE = "#4a7fb3"

# scene parameters
W_PARAM, B_PARAM, C_PARAM = 1.0, 0.6, 0.0
XMIN, XMAX, M = -6.0, 6.0, 260
BAND = 2.0
YMAX = 5.0
N_FRAMES = 40
EPS_WIDE, EPS_NARROW = 3.0, 0.08

X = jnp.linspace(XMIN, XMAX, M)
EPS_SCHEDULE = jnp.exp(jnp.linspace(jnp.log(EPS_WIDE), jnp.log(EPS_NARROW), N_FRAMES))


@jax.jit
def relu_curve(x):
    return jax.nn.relu(W_PARAM * x + B_PARAM)


@jax.jit
def yat_curve(x, eps):
    return (W_PARAM * x + B_PARAM) ** 2 / ((x - C_PARAM) ** 2 + eps)


RELU = np.asarray(relu_curve(X))
X_NP = np.asarray(X)


def draw_frame(idx: int) -> np.ndarray:
    eps = float(EPS_SCHEDULE[idx])
    yat = np.asarray(yat_curve(X, eps))

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    fig.text(0.5, 0.945,
             "Off-distribution: ReLU is unbounded, the Yat unit is bounded",
             ha="center", color=INK, fontsize=16, weight="bold")
    fig.text(0.5, 0.885,
             "leave the data band and ReLU keeps climbing; the kernel unit's response stays bounded",
             ha="center", color=MUTED, fontsize=10.5)

    ax = fig.add_axes([0.08, 0.20, 0.88, 0.58])
    ax.set_facecolor(PANEL)

    # in-distribution band
    ax.axvspan(-BAND, BAND, color=ACCENT, alpha=0.07, lw=0)
    ax.text(0.0, YMAX * 0.96, "in-distribution  |x| ≤ 2", ha="center",
            va="top", color=MUTED, fontsize=9)

    # curves
    ax.plot(X_NP, RELU, color=BLUE, lw=2.0, label="ReLU σ(w·x+b)")
    ax.plot(X_NP, yat, color=ACCENT, lw=2.0, label="Yat unit")

    # ReLU unbounded annotation near where it exits the top
    x_exit = (YMAX - B_PARAM) / W_PARAM
    if XMIN <= x_exit <= XMAX:
        ax.annotate("ReLU → ∞ (unbounded)", xy=(x_exit, YMAX),
                    xytext=(x_exit - 3.6, YMAX * 0.80), color=BLUE, fontsize=9.5,
                    weight="bold",
                    arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.3))

    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(0, YMAX)
    ax.set_xlabel("input  x", color=MUTED, fontsize=10)
    ax.set_ylabel("unit response", color=MUTED, fontsize=10)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(BORDER)

    # eps readout
    ax.text(0.975, 0.93, f"eps = {eps:.3f}", transform=ax.transAxes, ha="right",
            color=ACCENT, fontsize=10.5, family="monospace", weight="bold")

    leg = ax.legend(loc="upper left", frameon=True, fontsize=9.5)
    leg.get_frame().set_edgecolor(BORDER)
    leg.get_frame().set_facecolor(PANEL)

    fig.text(0.5, 0.06,
             "unbounded extrapolation vs a bounded response — the difference between a half-plane and a kernel",
             ha="center", color=MUTED, fontsize=10)

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba


def main() -> None:
    frames = []
    for idx in range(N_FRAMES):
        frames.append(draw_frame(idx))
        if (idx + 1) % 10 == 0:
            print(f"rendered {idx + 1}/{N_FRAMES} frames")
    Image.fromarray(frames[-1]).save(OUT_PREVIEW)
    frames.extend([frames[-1]] * END_HOLD)
    imageio.mimsave(OUT_GIF, frames, duration=1 / FPS, loop=0,
                    palettesize=128, subrectangles=True)
    print(f"wrote {OUT_GIF}")
    print(f"wrote {OUT_PREVIEW}")


if __name__ == "__main__":
    main()
