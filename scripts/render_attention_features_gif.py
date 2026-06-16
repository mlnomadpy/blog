#!/usr/bin/env python3
"""Render a GIF of positive random features approximating the softmax kernel.

A companion figure for "Self-Attention as Kernel Regression in JAX/Flax NNX".
The "what this leaves out" payoff: to make attention cheap you replace the
exp-dot-product kernel exp(q.k) with an explicit feature map phi(q).phi(k). With
Performer/FAVOR+ positive random features
    phi(x)_i = exp(w_i . x - ||x||^2 / 2) / sqrt(m),  w_i ~ N(0, I),
we have E[phi(q).phi(k)] = exp(q.k), so the rank-m matrix Phi_Q Phi_K^T is an
unbiased estimate of the full kernel. Grow m and watch the approximation tighten:
the heatmaps converge and the Frobenius error falls like 1/sqrt(m).
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
OUT_GIF = ROOT / "public" / "attention-random-features.gif"
OUT_PREVIEW = ROOT / "public" / "attention-random-features-preview.png"

W, H = 1100, 540
FPS = 4
END_HOLD = 8

BG = "#fbfaf6"
PANEL = "#ffffff"
INK = "#181818"
MUTED = "#666a70"
BORDER = "#ded9cb"
ACCENT = "#b3661b"

N, D = 10, 8
MS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
AVG = 60  # expected relative error over independent feature draws -> a clean curve


def setup():
    # modest input norms keep the exponential estimator low-variance; the 1/sqrt(d)
    # scale on the scores is folded into q, k, so the kernel is exp(q.k).
    Q = jax.random.normal(jax.random.key(21), (N, D)) * 0.25
    K = jax.random.normal(jax.random.key(22), (N, D)) * 0.25
    Kexp = jnp.exp(Q @ K.T)
    return Q, K, Kexp


Q, K, KEXP = setup()
KEXP_NP = np.asarray(KEXP)
VMAX = float(KEXP_NP.max())


def features(x, w):                       # positive random features (FAVOR+)
    m = w.shape[0]
    proj = x @ w.T                        # [n, m]
    return jnp.exp(proj - 0.5 * jnp.sum(x ** 2, axis=-1, keepdims=True)) / jnp.sqrt(m)


def approx(m, seed):
    w = jax.random.normal(jax.random.key(seed), (m, D))
    Khat = features(Q, w) @ features(K, w).T
    err = float(jnp.linalg.norm(Khat - KEXP) / jnp.linalg.norm(KEXP))
    return np.asarray(Khat), err


# precompute: one display approx per m, plus error averaged over AVG draws
DISPLAY, ERRS = [], []
for i, m in enumerate(MS):
    Khat, _ = approx(m, 1000 + i)
    errs = [approx(m, 100 * (i + 1) + s)[1] for s in range(AVG)]
    DISPLAY.append(Khat)
    ERRS.append(float(np.mean(errs)))
ERRS = np.array(ERRS)


def draw_frame(idx: int) -> np.ndarray:
    m = MS[idx]
    Khat = DISPLAY[idx]

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    fig.text(0.5, 0.95, "Making the kernel cheap: positive random features",
             ha="center", color=INK, fontsize=18, weight="bold")
    fig.text(0.5, 0.905,
             "exp(q·k) ≈ φ(q)·φ(k):  a rank-m feature map standing in for the full kernel  (Performer / FAVOR+)",
             ha="center", color=MUTED, fontsize=11.5)

    # true kernel
    ax1 = fig.add_axes([0.05, 0.18, 0.26, 0.56])
    ax1.imshow(KEXP_NP, cmap="magma", vmin=0, vmax=VMAX, aspect="equal")
    ax1.set_xticks([]); ax1.set_yticks([])
    for s in ax1.spines.values():
        s.set_color(BORDER)
    ax1.set_title("true kernel  exp(q·k)", color=INK, fontsize=10.5, weight="bold", pad=6)

    # approximation
    ax2 = fig.add_axes([0.345, 0.18, 0.26, 0.56])
    ax2.imshow(np.clip(Khat, 0, VMAX), cmap="magma", vmin=0, vmax=VMAX, aspect="equal")
    ax2.set_xticks([]); ax2.set_yticks([])
    for s in ax2.spines.values():
        s.set_color(ACCENT); s.set_linewidth(1.6)
    ax2.set_title(f"approx  ΦΦᵀ   (m = {m})", color=INK, fontsize=10.5, weight="bold", pad=6)

    # error curve
    ax3 = fig.add_axes([0.69, 0.18, 0.27, 0.56])
    ax3.set_facecolor(PANEL)
    ax3.plot(MS[: idx + 1], ERRS[: idx + 1], color=ACCENT, lw=2.0, marker="o", ms=4)
    ref = ERRS[0] * np.sqrt(MS[0]) / np.sqrt(np.array(MS))
    ax3.plot(MS, ref, color=MUTED, ls="--", lw=1.0)
    ax3.text(MS[-1], ref[-1] * 1.2, "∝ 1/√m", color=MUTED, fontsize=9, ha="right")
    ax3.set_xscale("log", base=2); ax3.set_yscale("log")
    ax3.set_xlim(0.8, MS[-1] * 1.3)
    ax3.set_ylim(min(ERRS) * 0.6, max(ERRS) * 1.5)
    ax3.set_xlabel("features m", color=MUTED, fontsize=9.5)
    ax3.tick_params(colors=MUTED, labelsize=8.5)
    for s in ax3.spines.values():
        s.set_color(BORDER)
    ax3.set_title("relative Frobenius error", color=INK, fontsize=10.5, weight="bold", pad=6)
    ax3.text(0.95, 0.9, f"err = {ERRS[idx]:.3f}", transform=ax3.transAxes, ha="right",
             color=ACCENT, fontsize=10.5, family="monospace", weight="bold")

    fig.text(0.5, 0.06,
             "the kernel that was never positive definite is approximated by one that is — φ(q)·φ(k) — and that is what buys linear-time attention",
             ha="center", color=MUTED, fontsize=10.5)

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba


def main() -> None:
    frames = []
    for idx in range(len(MS)):
        frames.append(draw_frame(idx))
        print(f"m={MS[idx]:4d}  err={ERRS[idx]:.4f}")
    Image.fromarray(frames[-1]).save(OUT_PREVIEW)
    frames.extend([frames[-1]] * END_HOLD)
    imageio.mimsave(OUT_GIF, frames, duration=1 / FPS, loop=0,
                    palettesize=160, subrectangles=True)
    print(f"wrote {OUT_GIF}")
    print(f"wrote {OUT_PREVIEW}")


if __name__ == "__main__":
    main()
