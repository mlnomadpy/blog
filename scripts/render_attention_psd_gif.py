#!/usr/bin/env python3
"""Render a GIF of attention's kernel losing positive-definiteness.

A companion figure for "Self-Attention as Kernel Regression in JAX/Flax NNX".
The attention score is a bilinear form s_ij = x_i^T M x_j with M = W_Q W_K^T.
When W_K = W_Q the operator is symmetric PSD and the score matrix is a genuine
(Mercer) kernel: a Gram matrix with nonnegative eigenvalues. As W_K drifts
away from W_Q, M loses its symmetry, and the symmetric part of the score
matrix grows a negative eigenvalue — the point where "attention's kernel" stops
being a kernel in the strict sense. We interpolate W_K from W_Q to an
independent matrix and watch the spectrum cross zero.
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
OUT_GIF = ROOT / "public" / "attention-kernel-psd.gif"
OUT_PREVIEW = ROOT / "public" / "attention-kernel-psd-preview.png"

W, H = 1100, 520
FPS = 12
END_HOLD = 16

BG = "#fbfaf6"
PANEL = "#ffffff"
INK = "#181818"
MUTED = "#666a70"
BORDER = "#ded9cb"
POS = "#3a6ea5"   # nonnegative eigenvalues
NEG = "#c0392b"   # negative eigenvalues — the kernel breaks here

N, D_MODEL, D_HEAD = 9, 10, 10
FRAMES = 46


def setup():
    x = jax.random.normal(jax.random.key(0), (N, D_MODEL))
    wq = jax.random.normal(jax.random.key(1), (D_MODEL, D_HEAD))
    wk_indep = jax.random.normal(jax.random.key(2), (D_MODEL, D_HEAD))
    return x, wq, wk_indep


X, WQ, WK_INDEP = setup()
VMAX = None


def state(t):
    wk = (1.0 - t) * WQ + t * WK_INDEP
    M = WQ @ wk.T
    S = X @ M @ X.T                       # scores s_ij = x_i^T M x_j
    sym = 0.5 * (S + S.T)
    eig = jnp.linalg.eigvalsh(sym)        # ascending
    return np.asarray(S), np.asarray(eig)


# fix a symmetric colour scale for the score heatmap across all frames
VMAX = max(float(np.abs(state(t)[0]).max()) for t in np.linspace(0, 1, FRAMES))


def draw_frame(frame: int) -> np.ndarray:
    t = frame / (FRAMES - 1)
    S, eig = state(t)
    min_eig = float(eig.min())
    psd = min_eig >= -1e-9

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    fig.text(0.5, 0.945, "Where attention's kernel stops being a kernel",
             ha="center", color=INK, fontsize=18, weight="bold")
    fig.text(0.5, 0.90,
             "the key projection drifts off the query projection:  Wₖ = (1−t)·W_q + t·Wₖᶦⁿᵈ",
             ha="center", color=MUTED, fontsize=11.5)

    # ── left: the score matrix, symmetric -> asymmetric ──
    axm = fig.add_axes([0.07, 0.16, 0.40, 0.56])
    axm.imshow(S, cmap="coolwarm", vmin=-VMAX, vmax=VMAX, aspect="equal")
    axm.set_xticks([]); axm.set_yticks([])
    axm.set_xlabel("key j", color=MUTED, fontsize=9.5)
    axm.set_ylabel("query i", color=MUTED, fontsize=9.5)
    for spine in axm.spines.values():
        spine.set_color(BORDER)
    sym_txt = "symmetric" if t < 1e-6 else "non-symmetric"
    axm.set_title(f"scores  s = xᵀMx   ({sym_txt})", color=INK,
                  fontsize=11, weight="bold", pad=8)

    # ── right: eigenvalues of the symmetric part ──
    axe = fig.add_axes([0.575, 0.16, 0.37, 0.56])
    axe.set_facecolor(PANEL)
    xs = np.arange(N)
    cols = [NEG if e < -1e-9 else POS for e in eig]
    axe.bar(xs, eig, color=cols, alpha=0.92)
    axe.axhline(0, color=INK, lw=1.0)
    axe.set_xlim(-0.7, N - 0.3)
    axe.set_ylim(-VMAX * 1.05, VMAX * 1.05)
    axe.set_xticks([])
    axe.tick_params(colors=MUTED, labelsize=9)
    for spine in axe.spines.values():
        spine.set_color(BORDER)
    axe.set_title("eigenvalues of ½(S+Sᵀ)", color=INK, fontsize=11,
                  weight="bold", pad=8)
    verdict = "positive semi-definite — a Mercer kernel" if psd else "indefinite — not a kernel"
    vcol = POS if psd else NEG
    axe.text(0.5, 0.93, verdict, transform=axe.transAxes, ha="center",
             color=vcol, fontsize=11.5, weight="bold")
    axe.text(0.5, 0.84, f"min eigenvalue = {min_eig:+.2f}", transform=axe.transAxes,
             ha="center", color=MUTED, fontsize=10, family="monospace")

    fig.text(0.5, 0.05,
             "softmax will still make the rows nonnegative — but the kernel underneath is no longer positive definite",
             ha="center", color=MUTED, fontsize=10.5)

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba


def main() -> None:
    frames = []
    for frame in range(FRAMES):
        frames.append(draw_frame(frame))
        if (frame + 1) % 12 == 0:
            print(f"rendered {frame + 1}/{FRAMES} frames")
    Image.fromarray(frames[-1]).save(OUT_PREVIEW)
    frames.extend([frames[-1]] * END_HOLD)
    imageio.mimsave(OUT_GIF, frames, duration=1 / FPS, loop=0,
                    palettesize=128, subrectangles=True)
    print(f"wrote {OUT_GIF}")
    print(f"wrote {OUT_PREVIEW}")


if __name__ == "__main__":
    main()
