#!/usr/bin/env python3
"""Render a GIF of the attention temperature acting as a kernel bandwidth.

A companion figure for "Self-Attention as Kernel Regression in JAX/Flax NNX".
The 1/sqrt(d) scale on the scores is the bandwidth of the kernel smoother. Sweep
it: at a wide bandwidth (small scale) the weights are near-uniform and the output
is the average of all values; at a narrow bandwidth (large scale) the weights
collapse to one token and the output is a hard copy of the nearest value. Left:
the attention row morphing from a flat bar chart to a spike, with its entropy.
Right: the output sliding from the centroid of the values to a single vertex.
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
OUT_GIF = ROOT / "public" / "attention-bandwidth.gif"
OUT_PREVIEW = ROOT / "public" / "attention-bandwidth-preview.png"

W, H = 1100, 540
FPS = 13
END_HOLD = 14

BG = "#fbfaf6"
PANEL = "#ffffff"
INK = "#181818"
MUTED = "#666a70"
BORDER = "#ded9cb"
HULL = "#cdbf9a"
ACCENT = "#b3661b"

N, D = 6, 8
FRAMES = 56
D_OPER = 8  # the 1/sqrt(d) operating point lands inside the sweep


def setup():
    th = np.linspace(0, 2 * np.pi, N, endpoint=False) + 0.4
    Vv = np.stack([1.55 * np.cos(th), 1.05 * np.sin(th)], axis=1)
    q = jax.random.normal(jax.random.key(11), (D,))
    k = jax.random.normal(jax.random.key(12), (N, D))
    scores = np.asarray(k @ q)             # raw q.k per token
    return Vv, scores


VNP, SCORES = setup()
COLS = plt.cm.Oranges(np.linspace(0.4, 0.95, N))


def scale_at(frame):
    # geometric sweep from very wide (~0) to narrow bandwidth
    lo, hi = -2.2, 1.4
    return 10 ** (lo + (hi - lo) * frame / (FRAMES - 1))


def weights_at(scale):
    return np.asarray(jax.nn.softmax(jnp.asarray(SCORES) * scale))


def entropy(w):
    return float(-(w * np.log(w + 1e-12)).sum())


def draw_frame(frame: int) -> np.ndarray:
    scale = scale_at(frame)
    w = weights_at(scale)
    ent = entropy(w)
    y = w @ VNP
    near_oper = abs(np.log10(scale) - np.log10(1 / np.sqrt(D_OPER))) < 0.06

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    fig.text(0.5, 0.95, "The √d is a bandwidth", ha="center",
             color=INK, fontsize=18, weight="bold")
    fig.text(0.5, 0.905,
             "widen it and attention averages every value; narrow it and attention copies one",
             ha="center", color=MUTED, fontsize=11.5)

    # ── left: the attention row ──
    axb = fig.add_axes([0.08, 0.16, 0.40, 0.60])
    axb.set_facecolor(PANEL)
    axb.bar(np.arange(N), w, color=COLS, alpha=0.95)
    axb.axhline(1 / N, color=MUTED, ls=":", lw=1.0)
    axb.text(N - 0.55, 1 / N + 0.02, "uniform", color=MUTED, fontsize=8.5, ha="right")
    axb.set_xticks(range(N))
    axb.set_xticklabels([f"v{j}" for j in range(N)], color=MUTED, fontsize=9)
    axb.set_ylim(0, 1.05); axb.set_yticks([])
    for spine in axb.spines.values():
        spine.set_color(BORDER)
    axb.set_title("attention row  α = softmax(scale · q·k)", color=INK,
                  fontsize=11, weight="bold", pad=8)
    # entropy gauge
    axb.text(0.03, 0.93, f"entropy {ent:.2f}  /  log N = {np.log(N):.2f}",
             transform=axb.transAxes, color=ACCENT, fontsize=10.5,
             family="monospace", weight="bold")

    # ── right: output sliding through the value hull ──
    axv = fig.add_axes([0.56, 0.16, 0.36, 0.60])
    axv.set_facecolor(PANEL)
    axv.add_patch(plt.Polygon(VNP, closed=True, facecolor=HULL, alpha=0.3,
                              edgecolor=HULL, lw=1.5))
    for j in range(N):
        axv.scatter([VNP[j, 0]], [VNP[j, 1]], s=70, color=COLS[j], zorder=3)
        axv.annotate(f"v{j}", (VNP[j, 0], VNP[j, 1]), textcoords="offset points",
                     xytext=(6, 4), color=MUTED, fontsize=8)
    cen = VNP.mean(0)
    axv.scatter([cen[0]], [cen[1]], s=45, color=MUTED, marker="+", zorder=4)
    axv.scatter([y[0]], [y[1]], s=150, color=ACCENT, zorder=5,
                edgecolor="white", lw=1.4)
    axv.set_xlim(-2.2, 2.2); axv.set_ylim(-1.9, 1.9)
    axv.set_aspect("equal"); axv.set_xticks([]); axv.set_yticks([])
    for spine in axv.spines.values():
        spine.set_color(BORDER)
    axv.set_title("output  y = Σ αⱼ vⱼ", color=INK, fontsize=11, weight="bold", pad=8)

    band = "wide kernel · averaging" if scale < 0.5 else ("narrow kernel · copying" if scale > 2 else "")
    tag = f"scale = {scale:.2f}"
    if near_oper:
        tag += "   ← 1/√d"
    axv.text(0.5, -0.13, tag + ("   " + band if band else ""), transform=axv.transAxes,
             ha="center", color=INK if near_oper else MUTED,
             fontsize=10.5, family="monospace", weight="bold" if near_oper else "normal")

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba


def main() -> None:
    frames = []
    for frame in range(FRAMES):
        frames.append(draw_frame(frame))
        if (frame + 1) % 14 == 0:
            print(f"rendered {frame + 1}/{FRAMES} frames")
        if frame == FRAMES // 2:
            Image.fromarray(frames[-1]).save(OUT_PREVIEW)
    frames.extend([frames[-1]] * END_HOLD)
    imageio.mimsave(OUT_GIF, frames, duration=1 / FPS, loop=0,
                    palettesize=128, subrectangles=True)
    print(f"wrote {OUT_GIF}")
    print(f"wrote {OUT_PREVIEW}")


if __name__ == "__main__":
    main()
