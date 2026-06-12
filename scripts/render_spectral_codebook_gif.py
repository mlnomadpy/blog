#!/usr/bin/env python3
"""A codebook is the spectral embedding of a label kernel. As a label kernel goes
from flat (structureless) to graded (similarity falls off with class distance),
its top eigenmodes morph the codebook from an even ring (the simplex, flat
spectrum) into the horseshoe of classical MDS (a peaked spectrum). The embedding
and the eigenspectrum are computed live in JAX (jnp.linalg.eigh)."""
from __future__ import annotations
from pathlib import Path
import imageio.v2 as imageio
import jax, jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colormaps  # noqa: E402
from PIL import Image  # noqa: E402

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[1]
OUT_GIF = ROOT / "public" / "spectral-codebook.gif"
OUT_PREVIEW = ROOT / "public" / "spectral-codebook-preview.png"

W, H, FPS, FRAMES, C, SIG = 1100, 520, 18, 96, 9, 2.2
BG, PANEL, INK, MUTED, BORDER, ACCENT = "#fbfaf6", "#ffffff", "#181818", "#666a70", "#ded9cb", "#b3661b"

idx = jnp.arange(C, dtype=jnp.float64)
G = jnp.exp(-(((idx[:, None] - idx[None, :]) / SIG) ** 2))     # graded label kernel
F = jnp.eye(C) - jnp.ones((C, C)) / C                          # flat (centered): the simplex


def mds(S, d=2):
    w, V = jnp.linalg.eigh(S)
    order = jnp.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    coords = V[:, :d] * jnp.sqrt(jnp.clip(w[:d], 0.0, None))    # top-d, sqrt-eigenvalue scaling
    return np.asarray(coords), np.asarray(jnp.clip(w, 0.0, None))


def procrustes(A, B):                                          # orthogonal R minimizing ||A R - B||
    U, _, Vt = np.linalg.svd(A.T @ B)
    return A @ (U @ Vt)


REF = np.stack([np.cos(2 * np.pi * np.arange(C) / C), np.sin(2 * np.pi * np.arange(C) / C)], 1)
COLORS = colormaps["turbo"](np.linspace(0.08, 0.92, C))
prev = None


def draw(frame):
    global prev
    t = frame / (FRAMES - 1)
    S = (1 - t) * F + t * (G - G.mean())                       # flat -> graded (centered)
    E, spec = mds(S)
    E = E / (np.abs(E).max() + 1e-9)
    E = procrustes(E, REF if prev is None else prev)
    prev = E

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    fig.text(0.5, 0.95, "A codebook is the spectral embedding of a label kernel", ha="center", color=INK, fontsize=18, weight="bold")
    fig.text(0.5, 0.905, "top two eigenmodes of the label kernel, scaled by the square roots of their eigenvalues", ha="center", color=MUTED, fontsize=11.5)

    ax = fig.add_axes([0.06, 0.12, 0.5, 0.72]); ax.set_facecolor(PANEL)
    ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color(BORDER)
    ax.plot(E[:, 0], E[:, 1], color=BORDER, lw=1, zorder=1)   # open path in class order (the spectrum)
    ax.scatter(E[:, 0], E[:, 1], s=130, color=COLORS, zorder=3, edgecolors=PANEL, linewidths=1.5)
    for c in range(C):
        ax.text(E[c, 0], E[c, 1], str(c), ha="center", va="center", color=INK, fontsize=8, weight="bold")
    label = "flat spectrum → even ring (the simplex)" if t < 0.2 else ("graded spectrum → the horseshoe (classical MDS)" if t > 0.7 else "structure emerging…")
    ax.set_title(label, color=INK, fontsize=11.5, weight="bold", pad=6)

    axs = fig.add_axes([0.63, 0.18, 0.33, 0.6]); axs.set_facecolor(PANEL)
    bars = spec[:6] / (spec[0] + 1e-9)
    axs.bar(np.arange(len(bars)), bars, color=[ACCENT if i < 2 else BORDER for i in range(len(bars))])
    axs.set_ylim(0, 1.05); axs.set_xticks(range(len(bars))); axs.set_xticklabels([f"λ{i+1}" for i in range(len(bars))], color=MUTED, fontsize=9)
    axs.set_yticks([]); [sp.set_color(BORDER) for sp in axs.spines.values()]
    axs.set_title("eigenspectrum of the label kernel", color=INK, fontsize=11, weight="bold", pad=6)
    fig.text(0.795, 0.12, "structure dial = " + f"{t:.2f}", ha="center", color=ACCENT, fontsize=11, weight="bold")

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba


def main():
    frames = [draw(i) for i in range(FRAMES)]
    Image.fromarray(frames[int(FRAMES * 0.82)]).save(OUT_PREVIEW)
    imageio.mimsave(OUT_GIF, frames, duration=1 / FPS, loop=0, palettesize=128, subrectangles=True)
    print("wrote", OUT_GIF, "and", OUT_PREVIEW)


if __name__ == "__main__":
    main()
