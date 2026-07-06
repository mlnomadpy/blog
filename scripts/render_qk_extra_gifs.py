#!/usr/bin/env python3
"""Render the gauge-freedom GIF for the QK-projections companion.

For any invertible M, the factor pair (W_Q M, W_K M^{-T}) produces exactly the
same bilinear form B = W_Q W_K^T and therefore the same attention pattern.
The GIF drives M(t) around a smooth loop in GL(d_k): both factor heatmaps
churn continuously while B and the softmax attention pattern do not move, and
the readout prints the real max |B(t) - B(0)| every frame. Every panel is a
real JAX computation.
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
OUT_GIF = ROOT / "public" / "qk-gauge-freedom.gif"
OUT_PREVIEW = ROOT / "public" / "qk-gauge-freedom-preview.png"

W, H = 1100, 480
FPS = 10
FRAMES = 48

BG = "#fbfaf6"
PANEL = "#ffffff"
INK = "#181818"
MUTED = "#666a70"
BORDER = "#ded9cb"
BLUE = "#4a7fb3"
ACCENT = "#b3661b"

D_MODEL, D_K, N_TOK = 12, 4, 10


def gauge(t: float) -> jnp.ndarray:
    """A smooth loop of invertible d_k x d_k matrices: rotation times scaling."""
    a = 2 * jnp.pi * t
    g_rot = jnp.zeros((D_K, D_K)).at[0, 1].set(-1.0).at[1, 0].set(1.0).at[2, 3].set(-0.7).at[3, 2].set(0.7)
    g_scale = jnp.diag(jnp.array([0.55, -0.4, 0.35, -0.5]))
    return jax.scipy.linalg.expm(jnp.sin(a) * g_rot + (1 - jnp.cos(a)) * 0.6 * g_scale)


def heat(ax, mat, vmax, cmap, title, color=INK):
    ax.set_facecolor(PANEL)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.imshow(np.asarray(mat), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.text(0.5, 1.05, title, transform=ax.transAxes, ha="center", va="bottom", color=color, fontsize=11.5)


def main() -> None:
    kq, kk, kx = jax.random.split(jax.random.key(0), 3)
    w_q = jax.random.normal(kq, (D_MODEL, D_K)) / jnp.sqrt(D_MODEL)
    w_k = jax.random.normal(kk, (D_MODEL, D_K)) / jnp.sqrt(D_MODEL)
    x = jax.random.normal(kx, (N_TOK, D_MODEL))

    b0 = w_q @ w_k.T
    scores0 = (x @ b0 @ x.T) / jnp.sqrt(D_K)
    mask = jnp.tril(jnp.ones((N_TOK, N_TOK), dtype=bool))
    attn0 = jax.nn.softmax(jnp.where(mask, scores0, -jnp.inf), axis=-1)

    v_fac = float(max(np.abs(w_q).max(), np.abs(w_k).max())) * 1.4
    v_b = float(np.abs(b0).max())

    frames = []
    for fi in range(FRAMES):
        t = fi / FRAMES
        m = gauge(t)
        w_q_t = w_q @ m
        w_k_t = w_k @ jnp.linalg.inv(m).T
        b_t = w_q_t @ w_k_t.T
        scores_t = (x @ b_t @ x.T) / jnp.sqrt(D_K)
        attn_t = jax.nn.softmax(jnp.where(mask, scores_t, -jnp.inf), axis=-1)
        delta = float(jnp.max(jnp.abs(b_t - b0)))

        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
        fig.text(0.5, 0.94, "Only the product is identified: the factors churn, the relation does not",
                 ha="center", color=INK, fontsize=16, weight="bold")
        fig.text(0.5, 0.885,
                 r"$(W_Q M,\; W_K M^{-\top})\ \Rightarrow\ B = W_Q M M^{-1} W_K^\top$"
                 "   with $M(t)$ looping through invertible matrices",
                 ha="center", color=MUTED, fontsize=11.5)

        a1 = fig.add_axes([0.05, 0.17, 0.13, 0.56])
        a2 = fig.add_axes([0.22, 0.17, 0.13, 0.56])
        a3 = fig.add_axes([0.42, 0.17, 0.245, 0.56])
        a4 = fig.add_axes([0.72, 0.17, 0.245, 0.56])

        heat(a1, w_q_t, v_fac, "PuOr", r"$W_Q M(t)$", ACCENT)
        heat(a2, w_k_t, v_fac, "PuOr", r"$W_K M(t)^{-\top}$", ACCENT)
        heat(a3, b_t, v_b, "BrBG", r"$B = W_Q W_K^\top$  (frozen)")
        heat(a4, attn_t, 1.0, "Blues", "attention pattern  (frozen)", BLUE)
        a1.text(0.5, -0.12, "churning", transform=a1.transAxes, ha="center", color=ACCENT, fontsize=10)
        a2.text(0.5, -0.12, "churning", transform=a2.transAxes, ha="center", color=ACCENT, fontsize=10)
        a3.text(0.5, -0.12, "the only observable object", transform=a3.transAxes, ha="center", color=MUTED, fontsize=10)
        a4.text(0.5, -0.12, "softmax of the causal scores", transform=a4.transAxes, ha="center", color=MUTED, fontsize=10)

        fig.text(0.5, 0.04,
                 rf"max $|B(t) - B(0)|$ = {delta:.2e}      the scores cannot tell the factor pairs apart",
                 ha="center", color=ACCENT, fontsize=12, weight="bold")

        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba()).copy())
        plt.close(fig)
        if fi == FRAMES // 3:
            Image.fromarray(frames[-1]).save(OUT_PREVIEW)
        if (fi + 1) % 16 == 0:
            print(f"rendered {fi + 1}/{FRAMES} frames")

    imageio.mimsave(OUT_GIF, frames, duration=1 / FPS, loop=0, palettesize=128, subrectangles=True)
    print(f"wrote {OUT_GIF} ({OUT_GIF.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
