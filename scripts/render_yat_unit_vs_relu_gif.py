#!/usr/bin/env python3
"""Render a GIF contrasting a ReLU unit with a Yat (kernel) unit.

The same weight vector w drives two fields over the 2D input plane. The ReLU
unit σ(w·x+b) lights up a whole half-plane: it carries only a direction. The
Yat unit (w·x+b)²/(‖x−w‖²+ε) lights up a neighbourhood of w: it carries a
centre in input space. We move w along a circle and watch both fields follow —
the ReLU boundary swings; the Yat blob travels with w.
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

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
OUT_GIF = ROOT / "public" / "yat-unit-vs-relu.gif"
OUT_PREVIEW = ROOT / "public" / "yat-unit-vs-relu-preview.png"

W, H = 1100, 560
FPS = 12
N_FRAMES = 40
END_HOLD = 8

BG = "#fbfaf6"
PANEL = "#ffffff"
INK = "#181818"
MUTED = "#666a70"
BORDER = "#ded9cb"
ACCENT = "#b3661b"

GRID = 64
EXTENT = 2.5
B = 0.5
EPS = 0.3
RADIUS = 1.3
GAMMA = 0.6


def _hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float64) / 255.0


BG_RGB = _hex_to_rgb(BG)
ACCENT_RGB = _hex_to_rgb(ACCENT)

# input-space grid (shared by both panels)
_axis = jnp.linspace(-EXTENT, EXTENT, GRID)
GX, GY = jnp.meshgrid(_axis, _axis)  # origin='lower' => row 0 is y=-EXTENT
PTS = jnp.stack([GX.ravel(), GY.ravel()], axis=-1)  # (GRID*GRID, 2)


def relu_field(w):
    """relu(w·x + b) over the grid -> (GRID, GRID)."""
    pre = PTS @ w + B
    return jax.nn.relu(pre).reshape(GRID, GRID)


def yat_field(w):
    """(w·x + b)² / (‖x − w‖² + eps) over the grid -> (GRID, GRID)."""
    pre = PTS @ w + B
    d2 = jnp.sum((PTS - w[None, :]) ** 2, axis=-1)
    return (pre ** 2 / (d2 + EPS)).reshape(GRID, GRID)


def normalize_to_rgb(field: np.ndarray) -> np.ndarray:
    """Normalize a field to [0,1] with gamma, lerp BG->ACCENT -> RGB image."""
    f = np.asarray(field, dtype=np.float64)
    lo, hi = float(f.min()), float(f.max())
    if hi - lo < 1e-12:
        t = np.zeros_like(f)
    else:
        t = (f - lo) / (hi - lo)
    t = np.power(t, GAMMA)
    rgb = BG_RGB[None, None, :] + t[:, :, None] * (ACCENT_RGB - BG_RGB)[None, None, :]
    return np.clip(rgb, 0.0, 1.0)


# precompute w trajectory and fields
ANGLES = np.linspace(0.0, 2.0 * np.pi, N_FRAMES, endpoint=False)
WS = [jnp.array([RADIUS * np.cos(a), RADIUS * np.sin(a)]) for a in ANGLES]
RELU_RGB = [normalize_to_rgb(relu_field(w)) for w in WS]
YAT_RGB = [normalize_to_rgb(yat_field(w)) for w in WS]


def _style_panel(ax, title):
    ax.set_xlim(-EXTENT, EXTENT)
    ax.set_ylim(-EXTENT, EXTENT)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.set_title(title, color=INK, fontsize=11.5, weight="bold", pad=8)
    ax.set_aspect("equal")


def draw_frame(idx: int) -> np.ndarray:
    w = np.asarray(WS[idx])

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    fig.text(0.5, 0.945, "A direction versus a point", ha="center",
             color=INK, fontsize=18, weight="bold")
    fig.text(0.5, 0.895,
             "the same weight w: ReLU activates a whole half-plane; the Yat unit, a neighbourhood of w",
             ha="center", color=MUTED, fontsize=11.5)

    # ── left: ReLU field ──
    axl = fig.add_axes([0.075, 0.155, 0.40, 0.62])
    axl.imshow(RELU_RGB[idx], extent=[-EXTENT, EXTENT, -EXTENT, EXTENT],
               origin="lower", aspect="equal", interpolation="bilinear")
    axl.plot(w[0], w[1], "o", mfc=PANEL, mec=INK, mew=1.8, ms=11, zorder=5)
    _style_panel(axl, "ReLU unit   σ(w·x+b)")

    # ── right: Yat field ──
    axr = fig.add_axes([0.525, 0.155, 0.40, 0.62])
    axr.imshow(YAT_RGB[idx], extent=[-EXTENT, EXTENT, -EXTENT, EXTENT],
               origin="lower", aspect="equal", interpolation="bilinear")
    axr.plot(w[0], w[1], "o", mfc=PANEL, mec=INK, mew=1.8, ms=11, zorder=5)
    _style_panel(axr, "Yat unit   (w·x+b)²/(‖x−w‖²+ε)")

    fig.text(0.5, 0.055,
             "the kernel unit carries a centre in input space; the ReLU unit carries only a direction",
             ha="center", color=MUTED, fontsize=10.5)

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
    # clearest frame: w on the +x axis (idx 0), blob fully inside the plane
    Image.fromarray(frames[0]).save(OUT_PREVIEW)
    frames.extend([frames[-1]] * END_HOLD)
    imageio.mimsave(OUT_GIF, frames, duration=1 / FPS, loop=0,
                    palettesize=128, subrectangles=True)
    print(f"wrote {OUT_GIF}")
    print(f"wrote {OUT_PREVIEW}")


if __name__ == "__main__":
    main()
