#!/usr/bin/env python3
"""Render a GIF of four kernels sharing one Nadaraya-Watson normalizer.

A companion figure for "Self-Attention as Kernel Regression in JAX/Flax NNX".
The attention output is a weighted sum of value vectors, y = sum_j w_j v_j, with
w = kernel(q, k) / sum kernel. Swap the kernel and keep the normalizer. As the
query sweeps a loop, each kernel mixes the SAME values differently. The
exp-dot-product, Gaussian, and Yat kernels keep the weights a convex partition
of unity, so the output never leaves the convex hull of the values. The signed
linear kernel produces negative weights and a normalizer that can cross zero, so
its output is flung outside the hull — a mixture in algebra only.
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
OUT_GIF = ROOT / "public" / "attention-kernels-hull.gif"
OUT_PREVIEW = ROOT / "public" / "attention-kernels-hull-preview.png"

W, H = 1100, 560
FPS = 14
END_HOLD = 6

BG = "#fbfaf6"
PANEL = "#ffffff"
INK = "#181818"
MUTED = "#666a70"
BORDER = "#ded9cb"
HULL = "#cdbf9a"

NAMES = ["exp-dot (softmax)", "gaussian", "yat", "linear (signed)"]
COLS = ["#b3661b", "#3a8f5e", "#7a5cc0", "#c0392b"]
N = 6
FRAMES = 72
EPS = 1e-3
TRAIL = 22


def setup():
    # values on an ellipse -> convex position, so the polygon IS the hull
    th = np.linspace(0, 2 * np.pi, N, endpoint=False) + 0.35
    V = np.stack([1.55 * np.cos(th), 1.05 * np.sin(th)], axis=1)
    # keys: spread but nearly centred, so the linear normalizer crosses zero
    k = np.asarray(jax.random.normal(jax.random.key(7), (N, 2)))
    k = k - k.mean(0, keepdims=True) + np.array([0.18, 0.10])
    return jnp.asarray(V), jnp.asarray(k)


V, K = setup()
VNP = np.asarray(V)


def sqdist(q, k):
    return jnp.sum((q[None, :] - k) ** 2, axis=-1)


def kernels(q):
    qd = q @ K.T
    return {
        "exp-dot (softmax)": jnp.exp(qd),
        "gaussian": jnp.exp(-sqdist(q, K) / (2 * 0.8 ** 2)),
        "yat": (qd) ** 2 / (sqdist(q, K) + EPS),
        "linear (signed)": qd,
    }


def outputs(q):
    out = {}
    blow = {}
    for name, kv in kernels(q).items():
        denom = kv.sum()
        w = kv / denom
        out[name] = np.asarray(w @ V)
        blow[name] = bool(abs(float(denom)) < 0.25)
    return out, blow


def query_at(frame):
    a = 2 * np.pi * frame / FRAMES
    return jnp.array([1.5 * np.cos(a), 1.5 * np.sin(a)])


TRAILS = {n: [] for n in NAMES}


def draw_frame(frame: int) -> np.ndarray:
    q = query_at(frame)
    out, blow = outputs(q)
    for n in NAMES:
        TRAILS[n].append(out[n])
        if len(TRAILS[n]) > TRAIL:
            TRAILS[n].pop(0)

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    fig.text(0.5, 0.95, "Four kernels, one normalizer", ha="center",
             color=INK, fontsize=18, weight="bold")
    fig.text(0.5, 0.905,
             "the query sweeps a loop; each kernel mixes the same values  y = Σ wⱼ vⱼ",
             ha="center", color=MUTED, fontsize=11.5)

    # ── left: query space (keys + moving query) ──
    axq = fig.add_axes([0.06, 0.12, 0.36, 0.66])
    axq.set_facecolor(PANEL)
    kk = np.asarray(K)
    axq.scatter(kk[:, 0], kk[:, 1], s=70, color=MUTED, zorder=3)
    for j in range(N):
        axq.annotate(f"k{j}", (kk[j, 0], kk[j, 1]), textcoords="offset points",
                     xytext=(6, 4), color=MUTED, fontsize=8)
    qn = np.asarray(q)
    axq.scatter([qn[0]], [qn[1]], s=170, color=INK, zorder=4, edgecolor="white", lw=1.5)
    axq.add_patch(plt.Circle((0, 0), 1.5, fill=False, color=BORDER, ls="--", lw=1.0))
    axq.set_xlim(-2.4, 2.4); axq.set_ylim(-2.4, 2.4)
    axq.set_aspect("equal"); axq.set_xticks([]); axq.set_yticks([])
    for spine in axq.spines.values():
        spine.set_color(BORDER)
    axq.set_title("query space   (keys kⱼ, query q)", color=INK, fontsize=11, weight="bold", pad=8)

    # ── right: value space (hull + four outputs) ──
    axv = fig.add_axes([0.50, 0.12, 0.36, 0.66])
    axv.set_facecolor(PANEL)
    poly = plt.Polygon(VNP, closed=True, facecolor=HULL, alpha=0.30,
                       edgecolor=HULL, lw=1.5, zorder=1)
    axv.add_patch(poly)
    axv.scatter(VNP[:, 0], VNP[:, 1], s=70, color=MUTED, zorder=3)
    for j in range(N):
        axv.annotate(f"v{j}", (VNP[j, 0], VNP[j, 1]), textcoords="offset points",
                     xytext=(6, 4), color=MUTED, fontsize=8)
    for n, c in zip(NAMES, COLS):
        tr = np.asarray(TRAILS[n])
        if len(tr) > 1:
            axv.plot(tr[:, 0], tr[:, 1], color=c, lw=1.3, alpha=0.45, zorder=2)
        p = np.clip(out[n], -3.2, 3.2)
        marker = "X" if blow[n] else "o"
        axv.scatter([p[0]], [p[1]], s=120, color=c, zorder=5,
                    edgecolor="white", lw=1.2, marker=marker)
    axv.set_xlim(-3.3, 3.3); axv.set_ylim(-3.0, 3.0)
    axv.set_aspect("equal"); axv.set_xticks([]); axv.set_yticks([])
    for spine in axv.spines.values():
        spine.set_color(BORDER)
    axv.set_title("value space   (hull of vⱼ, outputs y)", color=INK, fontsize=11, weight="bold", pad=8)

    # legend
    axl = fig.add_axes([0.875, 0.12, 0.12, 0.66]); axl.axis("off")
    for i, (n, c) in enumerate(zip(NAMES, COLS)):
        y = 0.82 - i * 0.13
        axl.scatter([0.1], [y], s=90, color=c, transform=axl.transAxes,
                    edgecolor="white", lw=1.0)
        axl.text(0.22, y, n, transform=axl.transAxes, va="center",
                 color=INK, fontsize=8.5)

    fig.text(0.5, 0.045,
             "nonnegative kernels keep y inside the hull; the signed linear kernel escapes (✗ = its normalizer near zero)",
             ha="center", color=MUTED, fontsize=10.5)

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba


def main() -> None:
    frames = []
    for frame in range(FRAMES):
        frames.append(draw_frame(frame))
        if (frame + 1) % 18 == 0:
            print(f"rendered {frame + 1}/{FRAMES} frames")
        if frame == FRAMES // 8:
            Image.fromarray(frames[-1]).save(OUT_PREVIEW)
    frames.extend([frames[-1]] * END_HOLD)
    imageio.mimsave(OUT_GIF, frames, duration=1 / FPS, loop=0,
                    palettesize=160, subrectangles=True)
    print(f"wrote {OUT_GIF}")
    print(f"wrote {OUT_PREVIEW}")


if __name__ == "__main__":
    main()
