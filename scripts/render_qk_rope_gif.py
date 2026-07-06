#!/usr/bin/env python3
"""Render the RoPE constant-diagonal GIF for the QK-projections companion.

Uses the post's own rope()/rotate_half() math. A fixed content query q and
key k are rotated by their positions with the rotation strength s swept from
0 to 1 (implemented exactly as rope(x, s * positions)). At s = 0 the score
matrix S_ij = rope(q, s*i) . rope(k, s*j) is one constant; as s rises the
matrix organizes into constant diagonals, because the score depends on the
positions only through i - j. One diagonal is highlighted and its values are
plotted flat, with the real max-min spread printed each frame.
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
OUT_GIF = ROOT / "public" / "qk-rope-diagonal.gif"
OUT_PREVIEW = ROOT / "public" / "qk-rope-diagonal-preview.png"

W, H = 1100, 470
FPS = 10
SWEEP = 52

BG = "#fbfaf6"
PANEL = "#ffffff"
INK = "#181818"
MUTED = "#666a70"
BORDER = "#ded9cb"
BLUE = "#4a7fb3"
ACCENT = "#b3661b"

T, DH = 32, 8   # positions, head dimension
DELTA = 8       # the highlighted diagonal: query i, key i - DELTA


# --- the post's RoPE, verbatim ---
def rotate_half(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return jnp.stack([-x2, x1], axis=-1).reshape(x.shape)


def rope(x, positions, base=10000.0):
    """x: [..., t, d_head] (d_head even); positions: [t]."""
    d = x.shape[-1]
    inv_freq = base ** (-jnp.arange(0, d, 2) / d)
    ang = positions[:, None] * inv_freq[None, :]
    cos = jnp.repeat(jnp.cos(ang), 2, axis=-1)
    sin = jnp.repeat(jnp.sin(ang), 2, axis=-1)
    return x * cos + rotate_half(x) * sin


def main() -> None:
    qv = jax.random.normal(jax.random.key(3), (DH,))
    kv = jax.random.normal(jax.random.key(4), (DH,))
    pos = jnp.arange(T, dtype=jnp.float64)

    def score_matrix(strength: float) -> np.ndarray:
        qr = rope(jnp.broadcast_to(qv, (T, DH)), strength * pos)
        kr = rope(jnp.broadcast_to(kv, (T, DH)), strength * pos)
        return np.asarray(jnp.einsum("id,jd->ij", qr, kr))

    s_final = score_matrix(1.0)
    vmax = float(np.abs(s_final).max())

    # eased sweep 0 -> 1; the final frame is held via its duration
    ts = 0.5 - 0.5 * np.cos(np.pi * np.arange(SWEEP) / (SWEEP - 1))
    strengths = list(ts)

    ii = np.arange(DELTA, T)  # query index along the highlighted diagonal

    frames = []
    for fi, s in enumerate(strengths):
        smat = score_matrix(float(s))
        diag = smat[ii, ii - DELTA]
        spread = float(diag.max() - diag.min())

        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
        fig.text(0.5, 0.945, "RoPE organizes the score matrix into constant diagonals",
                 ha="center", color=INK, fontsize=16, weight="bold")
        fig.text(0.5, 0.893,
                 r"$S_{ij}=\mathrm{rope}(q, s\,i)\cdot \mathrm{rope}(k, s\,j)$"
                 "   for one fixed content pair, position strength $s$ swept from 0 to 1",
                 ha="center", color=MUTED, fontsize=11)

        ax1 = fig.add_axes([0.06, 0.13, 0.36, 0.62])
        ax2 = fig.add_axes([0.52, 0.20, 0.44, 0.50])

        ax1.set_facecolor(PANEL)
        ax1.set_xticks([])
        ax1.set_yticks([])
        for spine in ax1.spines.values():
            spine.set_color(BORDER)
        ax1.imshow(smat, cmap="BrBG", vmin=-vmax, vmax=vmax, aspect="equal")
        ax1.plot(ii - DELTA, ii, color=ACCENT, lw=2.2)
        ax1.set_title("score matrix  $S_{ij}$", color=INK, fontsize=12)
        ax1.set_xlabel("key position j", color=MUTED, fontsize=9.5)
        ax1.set_ylabel("query position i", color=MUTED, fontsize=9.5)

        ax2.set_facecolor(PANEL)
        for spine in ax2.spines.values():
            spine.set_color(BORDER)
        ax2.set_xlim(ii[0] - 0.5, ii[-1] + 0.5)
        ax2.set_ylim(-vmax * 1.08, vmax * 1.08)
        ax2.axhline(0, color=BORDER, lw=0.8)
        ax2.plot(ii, diag, color=ACCENT, lw=2.2, marker="o", ms=4)
        ax2.tick_params(colors=MUTED, labelsize=8.5)
        ax2.set_xlabel("query position i along the highlighted diagonal", color=MUTED, fontsize=9.5)
        ax2.set_title(rf"values on the diagonal  $\Delta = i - j = {DELTA}$",
                      color=ACCENT, fontsize=12)
        ax2.text(0.5, 0.06,
                 "a flat line: every (i, j) with the same gap gets the same score",
                 transform=ax2.transAxes, ha="center", color=MUTED, fontsize=10)

        fig.text(0.5, 0.04,
                 f"position strength s = {s:.2f}      "
                 f"spread along the diagonal: max - min = {spread:.4f}",
                 ha="center", color=ACCENT, fontsize=12, weight="bold")

        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba()).copy())
        plt.close(fig)
        if fi == SWEEP - 1:
            Image.fromarray(frames[-1]).save(OUT_PREVIEW)
        if (fi + 1) % 16 == 0:
            print(f"rendered {fi + 1}/{len(strengths)} frames")

    durations = [1 / FPS] * (len(frames) - 1) + [2.5]
    imageio.mimsave(OUT_GIF, frames, duration=durations, loop=0, palettesize=128, subrectangles=True)
    print(f"wrote {OUT_GIF} ({OUT_GIF.stat().st_size / 1e6:.2f} MB)")
    print(f"final diagonal spread at s=1: {float(s_final[ii, ii - DELTA].max() - s_final[ii, ii - DELTA].min()):.2e}")


if __name__ == "__main__":
    main()
