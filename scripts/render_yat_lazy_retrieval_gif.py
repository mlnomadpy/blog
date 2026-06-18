#!/usr/bin/env python3
"""Render a GIF of "lazy loading" sparse activation in the Yat kernel layer.

A query point sweeps a smooth Lissajous path over a field of 18 prototypes.
Each frame computes the Yat kernel k(W, x) = (W·x + b)^2 / (||x - W||^2 + eps)
from the query to every prototype (vectorized in jax). Only the prototypes
near the query "fire" (kernel value >= 0.18 of the per-frame max); the rest
stay dormant. Where a ReLU layer would light up ~half its units everywhere,
the kernel layer lights up only a handful — a content-addressed, lazy lookup.
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
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_GIF = ROOT / "public" / "yat-lazy-retrieval.gif"
OUT_PREVIEW = ROOT / "public" / "yat-lazy-retrieval-preview.png"

W, H = 900, 620
FPS = 12
END_HOLD = 10

BG = "#fbfaf6"
PANEL = "#ffffff"
INK = "#181818"
MUTED = "#666a70"
BORDER = "#ded9cb"
ACCENT = "#b3661b"

K = 18           # number of prototypes
N_FRAMES = 52
B = 0.5          # kernel bias term, b >= 0
EPS = 0.3        # kernel epsilon, eps > 0
EXTENT = 2.2     # domain is [-EXTENT, EXTENT]^2
GRID_N = 60      # heatmap resolution
ACTIVE_FRAC = 0.18

# colormap for the faint background field: BG -> ACCENT
FIELD_CMAP = LinearSegmentedColormap.from_list(
    "bg_accent", [to_rgb(BG), to_rgb(ACCENT)]
)


def yat_kernel(W, x):
    """Yat kernel from query x (..., 2) to prototypes W (K, 2) -> (..., K)."""
    dot = jnp.sum(W * x[..., None, :], axis=-1) + B          # (..., K)
    dist2 = jnp.sum((x[..., None, :] - W) ** 2, axis=-1)     # (..., K)
    return dot ** 2 / (dist2 + EPS)


def make_prototypes():
    """18 prototypes on a jittered grid over [-EXTENT, EXTENT]^2."""
    key = jax.random.key(7)
    # a 6x3 grid (=18) jittered a little
    xs = jnp.linspace(-EXTENT * 0.82, EXTENT * 0.82, 6)
    ys = jnp.linspace(-EXTENT * 0.78, EXTENT * 0.78, 3)
    gx, gy = jnp.meshgrid(xs, ys)
    grid = jnp.stack([gx.ravel(), gy.ravel()], axis=-1)      # (18, 2)
    jitter = jax.random.uniform(key, grid.shape, minval=-0.32, maxval=0.32)
    return grid + jitter


def query_path():
    """Smooth closed Lissajous path over N_FRAMES frames."""
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, N_FRAMES, endpoint=False)
    qx = 1.6 * jnp.cos(theta)
    qy = 1.25 * jnp.sin(2.0 * theta)
    return jnp.stack([qx, qy], axis=-1)                      # (N_FRAMES, 2)


PROTOS = make_prototypes()
PATH = query_path()

# Precompute the background field: max-over-prototypes kernel on a grid.
_gx = jnp.linspace(-EXTENT, EXTENT, GRID_N)
_gy = jnp.linspace(-EXTENT, EXTENT, GRID_N)
_mx, _my = jnp.meshgrid(_gx, _gy)
_grid_pts = jnp.stack([_mx.ravel(), _my.ravel()], axis=-1)   # (GRID_N^2, 2)
_field = yat_kernel(PROTOS, _grid_pts)                        # (GRID_N^2, K)
FIELD = jnp.max(_field, axis=-1).reshape(GRID_N, GRID_N)
FIELD_NP = np.asarray(FIELD)
# gamma + normalize for the faint heatmap
_fnorm = (FIELD_NP - FIELD_NP.min()) / (FIELD_NP.max() - FIELD_NP.min() + 1e-9)
FIELD_DISP = _fnorm ** 0.6

# Precompute per-frame kernel values (vectorized over the whole path).
KVALS = np.asarray(yat_kernel(PROTOS, PATH))                 # (N_FRAMES, K)
PROTOS_NP = np.asarray(PROTOS)
PATH_NP = np.asarray(PATH)


def draw_frame(idx: int) -> np.ndarray:
    kv = KVALS[idx]
    kmax = float(kv.max())
    thresh = ACTIVE_FRAC * kmax
    active = kv >= thresh
    n_active = int(active.sum())
    qx, qy = PATH_NP[idx]

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    fig.text(0.5, 0.945, "Lazy loading: only the nearby prototypes fire",
             ha="center", color=INK, fontsize=17, weight="bold")
    fig.text(0.5, 0.90,
             "a ReLU layer would fire ~50% of its units; the kernel layer fires a handful",
             ha="center", color=MUTED, fontsize=11)

    # centered square panel
    ax = fig.add_axes([0.215, 0.10, 0.57, 0.74])
    ax.set_facecolor(PANEL)
    ax.set_xlim(-EXTENT, EXTENT)
    ax.set_ylim(-EXTENT, EXTENT)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(BORDER)

    # faint background heatmap of the max-over-prototypes kernel field
    ax.imshow(FIELD_DISP, cmap=FIELD_CMAP, origin="lower",
              extent=[-EXTENT, EXTENT, -EXTENT, EXTENT],
              alpha=0.35, vmin=0.0, vmax=1.0, aspect="equal",
              interpolation="bilinear", zorder=0)

    # connecting lines from query to each active prototype (drawn under dots)
    for j in range(K):
        if active[j]:
            ax.plot([qx, PROTOS_NP[j, 0]], [qy, PROTOS_NP[j, 1]],
                    color=ACCENT, lw=0.9, alpha=0.45, zorder=2)

    # prototypes
    for j in range(K):
        px, py = PROTOS_NP[j]
        if active[j]:
            frac = kv[j] / (kmax + 1e-9)
            size = 60 + 320 * frac
            ax.scatter([px], [py], s=size, color=ACCENT,
                       edgecolors="white", linewidths=1.1, zorder=4)
        else:
            ax.scatter([px], [py], s=26, color=MUTED, alpha=0.45, zorder=3)

    # query marker
    ax.scatter([qx], [qy], s=140, color=INK, edgecolors="white",
               linewidths=1.6, zorder=6)

    # readout
    ax.text(0.035, 0.955, f"active: {n_active} / {K}",
            transform=ax.transAxes, ha="left", va="top",
            color=ACCENT, fontsize=12.5, weight="bold", family="monospace",
            bbox=dict(boxstyle="round,pad=0.32", fc=PANEL, ec=BORDER, lw=1.0))

    fig.text(0.5, 0.045,
             "k(W,x) = (W·x + b)² / (‖x − W‖² + ε)   "
             "— content-addressed: the query retrieves only what it is near",
             ha="center", color=MUTED, fontsize=10)

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba, n_active


def main() -> None:
    frames = []
    active_counts = []
    for idx in range(N_FRAMES):
        rgba, n_active = draw_frame(idx)
        frames.append(rgba)
        active_counts.append(n_active)
        if (idx + 1) % 10 == 0:
            print(f"rendered {idx + 1}/{N_FRAMES} frames")
    Image.fromarray(frames[-1]).save(OUT_PREVIEW)
    frames.extend([frames[-1]] * END_HOLD)  # hold at the end
    imageio.mimsave(OUT_GIF, frames, duration=1 / FPS, loop=0,
                    palettesize=128, subrectangles=True)
    ac = np.asarray(active_counts)
    print(f"active count: min={ac.min()} max={ac.max()} "
          f"mean={ac.mean():.1f} median={int(np.median(ac))}")
    print(f"wrote {OUT_GIF}")
    print(f"wrote {OUT_PREVIEW}")


if __name__ == "__main__":
    main()
