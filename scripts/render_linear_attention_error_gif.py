#!/usr/bin/env python3
"""Render the error-vs-m GIF for the linear-attention companion.

Left: the exact softmax attention matrix A = softmax(QK^T) for a structured
28-token sequence. Middle: the implied matrix A_hat = D^-1 phi(Q) phi(K)^T
from the first m rows of one growing bank of positive random features.
Right: the relative error of the actual linear-time output
Y_hat = D^-1 phi(Q)(phi(K)^T V) against the exact Y = AV, falling as m grows.
Every frame is a real JAX computation; nothing is precomputed or faked.
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
OUT_GIF = ROOT / "public" / "linear-attention-error.gif"
OUT_PREVIEW = ROOT / "public" / "linear-attention-error-preview.png"

W, H = 1100, 460
FPS = 9
HOLD = 12  # extra frames at the final m

BG = "#fbfaf6"
PANEL = "#ffffff"
INK = "#181818"
MUTED = "#666a70"
BORDER = "#ded9cb"
BLUE = "#4a7fb3"
ACCENT = "#b3661b"

N, D = 28, 8
M_MAX = 512


def build_tokens(key):
    """Structured q/k: tokens on a latent circle plus noise, so the exact
    attention matrix is a recognizable wrapping band, not uniform noise."""
    kq, kk, kv = jax.random.split(key, 3)
    theta = 2 * jnp.pi * jnp.arange(N) / N
    base = jnp.zeros((N, D)).at[:, 0].set(1.5 * jnp.cos(theta)).at[:, 1].set(1.5 * jnp.sin(theta))
    q = (base + 0.2 * jax.random.normal(kq, (N, D))) * D**-0.25
    k = (base + 0.2 * jax.random.normal(kk, (N, D))) * D**-0.25
    v = jax.random.normal(kv, (N, D))
    return q, k, v


def positive_features(x, omega):
    proj = jnp.einsum("td,md->tm", x, omega)
    norm = 0.5 * jnp.sum(x * x, axis=-1, keepdims=True)
    return jnp.exp(proj - norm) / jnp.sqrt(omega.shape[0])


def linear_pieces(q, k, v, omega):
    phi_q = positive_features(q, omega)
    phi_k = positive_features(k, omega)
    a_un = phi_q @ phi_k.T                    # implied unnormalized matrix
    den = a_un.sum(axis=1, keepdims=True)
    a_hat = a_un / den
    y_hat = (phi_q @ (phi_k.T @ v)) / den     # the actual linear-time output
    return a_hat, y_hat


def main() -> None:
    key = jax.random.key(0)
    q, k, v = build_tokens(key)
    a_exact = jax.nn.softmax(q @ k.T, axis=-1)
    y_exact = a_exact @ v

    omega_bank = jax.random.normal(jax.random.key(7), (M_MAX, D))

    # one growing sketch: frame f uses the first m_f rows of the same bank
    ms = np.unique(np.round(np.geomspace(2, M_MAX, 56)).astype(int))
    curve_m, curve_err = [], []

    a_np = np.asarray(a_exact)
    vmax = float(a_np.max())

    frames = []
    for fi, m in enumerate(list(ms) + [ms[-1]] * HOLD):
        held = fi >= len(ms)
        if not held:
            a_hat, y_hat = linear_pieces(q, k, v, omega_bank[:m])
            err = float(jnp.linalg.norm(y_hat - y_exact) / jnp.linalg.norm(y_exact))
            curve_m.append(int(m))
            curve_err.append(err)
            a_hat_np = np.asarray(a_hat)
        err = curve_err[-1]

        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
        fig.text(0.5, 0.945, "The implied attention matrix sharpens as random features accumulate",
                 ha="center", color=INK, fontsize=16, weight="bold")
        fig.text(0.5, 0.895,
                 r"$\hat A = D^{-1}\phi(Q)\phi(K)^\top$ from the first $m$ rows of one bank of positive random features",
                 ha="center", color=MUTED, fontsize=11)

        ax1 = fig.add_axes([0.045, 0.14, 0.26, 0.62])
        ax2 = fig.add_axes([0.355, 0.14, 0.26, 0.62])
        ax3 = fig.add_axes([0.70, 0.17, 0.27, 0.56])

        for ax, title in ((ax1, r"exact  $A=\mathrm{softmax}(QK^\top)$"),
                          (ax2, rf"implied  $\hat A$,  $m={int(m)}$ features")):
            ax.set_facecolor(PANEL)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(BORDER)
            ax.text(0.5, 1.04, title, transform=ax.transAxes, ha="center", va="bottom",
                    color=INK if ax is ax1 else ACCENT, fontsize=11.5)
        ax1.imshow(a_np, cmap="Blues", vmin=0, vmax=vmax, aspect="equal")
        ax2.imshow(a_hat_np, cmap="Oranges", vmin=0, vmax=vmax, aspect="equal")
        ax1.set_xlabel("key j", color=MUTED, fontsize=9)
        ax1.set_ylabel("query i", color=MUTED, fontsize=9)
        ax2.set_xlabel("key j", color=MUTED, fontsize=9)

        ax3.set_facecolor(PANEL)
        for spine in ax3.spines.values():
            spine.set_color(BORDER)
        ax3.set_xscale("log")
        ax3.set_yscale("log")
        ax3.set_xlim(2, M_MAX * 1.15)
        ax3.set_ylim(0.008, 1.2)
        ax3.plot(curve_m, curve_err, color=ACCENT, lw=2.0)
        ax3.scatter([curve_m[-1]], [curve_err[-1]], color=ACCENT, s=45, zorder=3)
        ref = curve_err[0] * np.sqrt(curve_m[0] / np.asarray(ms, float))
        ax3.plot(ms, ref, color=BLUE, lw=1.1, ls="--", alpha=0.8)
        ax3.text(ms[-1], ref[-1] * 1.25, r"$1/\sqrt{m}$", color=BLUE, fontsize=10, ha="right")
        ax3.tick_params(colors=MUTED, labelsize=8.5)
        ax3.set_xlabel("random features m", color=MUTED, fontsize=10)
        ax3.set_title(r"output error  $\|\hat Y - Y\|/\|Y\|$", color=INK, fontsize=11.5)

        fig.text(0.5, 0.045,
                 f"m = {int(m):3d} positive features   relative output error = {err:.3f}",
                 ha="center", color=ACCENT, fontsize=12, weight="bold")

        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba()).copy())
        plt.close(fig)
        if fi == len(ms) - 1:
            Image.fromarray(frames[-1]).save(OUT_PREVIEW)
        if (fi + 1) % 16 == 0:
            print(f"rendered {fi + 1}/{len(ms) + HOLD} frames")

    imageio.mimsave(OUT_GIF, frames, duration=1 / FPS, loop=0, palettesize=128, subrectangles=True)
    print(f"wrote {OUT_GIF} ({OUT_GIF.stat().st_size / 1e6:.2f} MB)")
    print(f"final error at m={int(ms[-1])}: {curve_err[-1]:.4f}")


if __name__ == "__main__":
    main()
