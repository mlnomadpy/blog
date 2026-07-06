#!/usr/bin/env python3
"""A codebook is the spectral embedding of a label kernel. When the label kernel
is flat (structureless) its top eigenmodes give an even ring (the simplex, flat
spectrum); when it is graded (similarity falls off with class distance) the same
two modes give the horseshoe of classical MDS (a peaked spectrum). There is no
temporal process here -- a hand-turned dial is not a training run -- so this is a
static contrast: the two codebooks side by side with their eigenspectra, both
computed in JAX (jnp.linalg.eigh)."""
from __future__ import annotations
from pathlib import Path
import jax, jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colormaps  # noqa: E402
from PIL import Image  # noqa: E402

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[1]
OUT_PNG = ROOT / "public" / "spectral-codebook.png"
OUT_PREVIEW = ROOT / "public" / "spectral-codebook-preview.png"

W, H, C, SIG = 1100, 520, 9, 2.2
BG, PANEL, INK, MUTED, BORDER, ACCENT = "#fbfaf6", "#ffffff", "#181818", "#666a70", "#ded9cb", "#b3661b"

idx = jnp.arange(C, dtype=jnp.float64)
G = jnp.exp(-(((idx[:, None] - idx[None, :]) / SIG) ** 2))     # graded label kernel
G = G - G.mean()
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


def embed(S, ref):
    E, spec = mds(S)
    E = E / (np.abs(E).max() + 1e-9)
    return procrustes(E, ref), spec


def draw():
    # The flat kernel's eigenspace is fully degenerate (C-1 equal eigenvalues),
    # so every orthonormal 2-D slice is an equally valid MDS solution. We take the
    # regular C-gon REF, which is one of them -- the even ring the simplex realizes.
    _, spec_flat = embed(F, REF)
    E_flat = REF / (np.abs(REF).max() + 1e-9)
    E_grad, spec_grad = embed(G, E_flat)

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    fig.text(0.5, 0.95, "A codebook is the spectral embedding of a label kernel", ha="center", color=INK, fontsize=18, weight="bold")
    fig.text(0.5, 0.905, "top two eigenmodes of the label kernel, scaled by the square roots of their eigenvalues", ha="center", color=MUTED, fontsize=11.5)

    panels = [
        (0.03, E_flat, spec_flat, "flat spectrum → even ring (the simplex)"),
        (0.52, E_grad, spec_grad, "graded spectrum → the horseshoe (classical MDS)"),
    ]
    for x0, E, spec, label in panels:
        ax = fig.add_axes([x0 + 0.01, 0.12, 0.28, 0.68]); ax.set_facecolor(PANEL)
        ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color(BORDER)
        ax.plot(E[:, 0], E[:, 1], color=BORDER, lw=1, zorder=1)   # open path in class order (the spectrum)
        ax.scatter(E[:, 0], E[:, 1], s=120, color=COLORS, zorder=3, edgecolors=PANEL, linewidths=1.5)
        for c in range(C):
            ax.text(E[c, 0], E[c, 1], str(c), ha="center", va="center", color=INK, fontsize=8, weight="bold")
        ax.set_title(label, color=INK, fontsize=10.5, weight="bold", pad=6)

        axs = fig.add_axes([x0 + 0.315, 0.32, 0.14, 0.4]); axs.set_facecolor(PANEL)
        bars = spec[:6] / (spec[0] + 1e-9)
        axs.bar(np.arange(len(bars)), bars, color=[ACCENT if i < 2 else BORDER for i in range(len(bars))])
        axs.set_ylim(0, 1.05); axs.set_xticks(range(len(bars))); axs.set_xticklabels([f"λ{i+1}" for i in range(len(bars))], color=MUTED, fontsize=7.5)
        axs.set_yticks([]); [sp.set_color(BORDER) for sp in axs.spines.values()]
        axs.set_title("eigenspectrum", color=INK, fontsize=9.5, weight="bold", pad=5)

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba


def main():
    buf = draw()
    Image.fromarray(buf).save(OUT_PNG)
    Image.fromarray(buf).save(OUT_PREVIEW)
    print("wrote", OUT_PNG, "and", OUT_PREVIEW)


if __name__ == "__main__":
    main()
