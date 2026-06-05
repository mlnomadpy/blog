#!/usr/bin/env python3
"""Render a GIF of a bilinear induction head: query reads the current token, key
reads each position's previous token, so i attends to the token after an earlier
copy of token i. The query position sweeps the sequence."""

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
OUT_GIF = ROOT / "public" / "qk-induction.gif"
OUT_PREVIEW = ROOT / "public" / "qk-induction-preview.png"

W, H = 1100, 520
FPS = 9
HOLD = 7

BG = "#fbfaf6"
PANEL = "#ffffff"
INK = "#181818"
MUTED = "#666a70"
BORDER = "#ded9cb"
VCOL = ["#b3661b", "#4a7fb3", "#3a8f5e"]  # one color per vocab symbol

VOCAB = ["a", "b", "c"]
SEQ = ["a", "b", "c", "a", "b", "c", "a", "b"]
IDX = np.array([VOCAB.index(t) for t in SEQ])
N = len(SEQ)
TAU = 0.18
FRAMES = (N - 1) * HOLD


def attention():
    E = jnp.eye(len(VOCAB))
    q = E[jnp.array(IDX)]
    k = jnp.concatenate([jnp.zeros((1, len(VOCAB))), E[jnp.array(IDX[:-1])]])
    s = q @ k.T
    rows, cols = jnp.arange(N)[:, None], jnp.arange(N)[None, :]
    allowed = (cols >= 1) & (cols <= rows)
    s = jnp.where(allowed, s / TAU, -jnp.inf)
    a = jax.nn.softmax(s, axis=-1)
    pred = a @ E[jnp.array(IDX)]
    return np.asarray(a), np.asarray(pred.argmax(-1))


A, PRED = attention()


def draw_frame(frame: int) -> np.ndarray:
    qi = 1 + (frame // HOLD) % (N - 1)

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    fig.text(0.5, 0.945, "A bilinear head that implements induction", ha="center", color=INK, fontsize=18, weight="bold")
    fig.text(0.5, 0.90, "query reads the current token; key reads each position's previous token", ha="center", color=MUTED, fontsize=11.5)

    # sequence strip
    axs = fig.add_axes([0.06, 0.74, 0.88, 0.10])
    axs.set_xlim(0, N)
    axs.set_ylim(0, 1)
    axs.axis("off")
    for i in range(N):
        sel = i == qi
        axs.add_patch(plt.Rectangle((i + 0.06, 0.1), 0.88, 0.8, color=VCOL[IDX[i]], alpha=1.0 if sel else 0.34))
        if sel:
            axs.add_patch(plt.Rectangle((i + 0.06, 0.1), 0.88, 0.8, fill=False, edgecolor=INK, lw=2.2))
        axs.text(i + 0.5, 0.5, SEQ[i], ha="center", va="center", color=INK, fontsize=14, weight="bold")
        axs.text(i + 0.5, -0.18, str(i), ha="center", va="center", color=MUTED, fontsize=9)

    # attention matrix
    axm = fig.add_axes([0.07, 0.14, 0.40, 0.50])
    masked = np.where(np.triu(np.ones((N, N)), 1) > 0, np.nan, A)
    masked = np.where((np.arange(N)[None, :] < 1), np.nan, masked)
    axm.imshow(masked, cmap="Oranges", vmin=0, vmax=1, aspect="equal")
    axm.add_patch(plt.Rectangle((-0.5, qi - 0.5), N, 1, fill=False, edgecolor=INK, lw=2.2))
    axm.set_xticks(range(N))
    axm.set_xticklabels(SEQ, color=MUTED, fontsize=9)
    axm.set_yticks(range(N))
    axm.set_yticklabels(SEQ, color=MUTED, fontsize=9)
    axm.set_xlabel("key j  (its previous token)", color=MUTED, fontsize=9.5)
    axm.set_ylabel("query i  (current token)", color=MUTED, fontsize=9.5)
    for spine in axm.spines.values():
        spine.set_color(BORDER)
    axm.set_title("attention  softmax(QKᵀ, causal)", color=INK, fontsize=11, weight="bold", pad=8)

    # distribution for the selected query
    axd = fig.add_axes([0.57, 0.14, 0.38, 0.50])
    axd.set_facecolor(PANEL)
    row = A[qi].copy()
    xs = np.arange(N)
    axd.bar(xs, row, color=[VCOL[IDX[j]] for j in range(N)], alpha=0.9)
    axd.set_xlim(-0.6, N - 0.4)
    axd.set_ylim(0, 1.05)
    axd.set_xticks(xs)
    axd.set_xticklabels([f"{SEQ[j]}{j}" for j in range(N)], color=MUTED, fontsize=9)
    axd.set_yticks([])
    for spine in axd.spines.values():
        spine.set_color(BORDER)
    axd.set_title(f"row {qi}: where “{SEQ[qi]}” attends", color=INK, fontsize=11, weight="bold", pad=8)
    axd.text(0.5, 0.86, f"predicts “{VOCAB[PRED[qi]]}”", transform=axd.transAxes, ha="center", color=VCOL[PRED[qi]], fontsize=15, weight="bold")

    fig.text(0.5, 0.045, "the bright cells sit one step after earlier copies of the current token — a directed relation no symmetric score could produce", ha="center", color=MUTED, fontsize=10.5)

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba


def main() -> None:
    frames = []
    for frame in range(FRAMES):
        frames.append(draw_frame(frame))
        if frame == (N - 2) * HOLD:  # the last query (clearest induction) for the preview
            Image.fromarray(frames[-1]).save(OUT_PREVIEW)
        if (frame + 1) % 14 == 0:
            print(f"rendered {frame + 1}/{FRAMES} frames")
    imageio.mimsave(OUT_GIF, frames, duration=1 / FPS, loop=0, palettesize=128, subrectangles=True)
    print(f"wrote {OUT_GIF}")
    print(f"wrote {OUT_PREVIEW}")


if __name__ == "__main__":
    main()
