#!/usr/bin/env python3
"""How many modes to spend. The rank-d spectral codebook reconstructs the label
kernel out of its top d eigenmodes; the reconstruction error is exactly the tail
of the spectrum. Live in JAX we sweep the budget d from 1 to C and watch the
rank-d Gram matrix sharpen back into the target kernel while kernel-target
alignment climbs toward 1 and the reconstruction error falls toward 0. Every
number is a jnp.linalg.eigh away."""
from __future__ import annotations
from pathlib import Path
import imageio.v2 as imageio
import jax, jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from PIL import Image  # noqa: E402

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[1]
OUT_GIF = ROOT / "public" / "mds-reconstruction.gif"
OUT_PREVIEW = ROOT / "public" / "mds-reconstruction-preview.png"

W, H, FPS, C, SIG, HOLD = 1100, 520, 14, 10, 2.6, 8
FRAMES = C * HOLD
BG, PANEL, INK, MUTED, BORDER, ACCENT, BLUE, GREEN = "#fbfaf6", "#ffffff", "#181818", "#666a70", "#ded9cb", "#b3661b", "#4a7fb3", "#3a8f5e"
KMAP = LinearSegmentedColormap.from_list("kern", ["#1d3a5f", "#4a7fb3", "#fbf8f1", "#d89a5a", "#b3661b"])

idx = jnp.arange(C, dtype=jnp.float64)
S = jnp.exp(-(((idx[:, None] - idx[None, :]) / SIG) ** 2))      # graded label kernel
S = S - S.mean()                                               # center
w_all, V_all = jnp.linalg.eigh(S)
order = jnp.argsort(w_all)[::-1]
w_all, V_all = np.asarray(w_all[order]), np.asarray(V_all[:, order])
w_pos = np.clip(w_all, 0.0, None)
Snp = np.asarray(S)
VMAX = np.abs(Snp).max()


def rank_d(d):
    Vd, wd = V_all[:, :d], w_pos[:d]
    Shat = (Vd * wd) @ Vd.T                                     # rank-d Gram reconstruction
    err = np.linalg.norm(Shat - Snp) / (np.linalg.norm(Snp) + 1e-12)
    align = float(np.sum(Shat * Snp) / (np.linalg.norm(Shat) * np.linalg.norm(Snp) + 1e-12))
    return Shat, err, align


CURVE = [rank_d(d) for d in range(1, C + 1)]                    # precompute the full sweep


def draw(frame):
    d = 1 + frame // HOLD
    Shat, err, align = CURVE[d - 1]

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    fig.text(0.5, 0.95, "How many modes to spend", ha="center", color=INK, fontsize=18, weight="bold")
    fig.text(0.5, 0.905, "the rank-d codebook rebuilds the label kernel; the error is the tail of the spectrum", ha="center", color=MUTED, fontsize=11.5)

    # target kernel
    axt = fig.add_axes([0.04, 0.16, 0.2, 0.6]); axt.imshow(Snp, cmap=KMAP, vmin=-VMAX, vmax=VMAX)
    axt.set_xticks([]); axt.set_yticks([]); [sp.set_color(BORDER) for sp in axt.spines.values()]
    axt.set_title("target kernel S", color=INK, fontsize=10.5, weight="bold", pad=5)

    # rank-d reconstruction
    axr = fig.add_axes([0.265, 0.16, 0.2, 0.6]); axr.imshow(Shat, cmap=KMAP, vmin=-VMAX, vmax=VMAX)
    axr.set_xticks([]); axr.set_yticks([]); [sp.set_color(ACCENT) for sp in axr.spines.values()]
    [sp.set_linewidth(2) for sp in axr.spines.values()]
    axr.set_title(f"rank-{d} reconstruction", color=ACCENT, fontsize=10.5, weight="bold", pad=5)

    # eigenspectrum, first d highlighted
    axs = fig.add_axes([0.53, 0.55, 0.43, 0.26]); axs.set_facecolor(PANEL)
    bars = w_pos / (w_pos[0] + 1e-12)
    axs.bar(np.arange(C), bars, color=[ACCENT if i < d else BORDER for i in range(C)])
    axs.set_ylim(0, 1.05); axs.set_xticks([]); axs.set_yticks([]); [sp.set_color(BORDER) for sp in axs.spines.values()]
    axs.set_title("spectrum: modes kept (orange)", color=INK, fontsize=10, weight="bold", pad=4)

    # alignment up / error down
    axc = fig.add_axes([0.53, 0.13, 0.43, 0.3]); axc.set_facecolor(PANEL)
    ds = np.arange(1, d + 1)
    aligns = [CURVE[k - 1][2] for k in ds]; errs = [CURVE[k - 1][1] for k in ds]
    axc.plot(ds, aligns, color=GREEN, lw=2.2, marker="o", ms=3, label="kernel-target alignment ↑")
    axc.plot(ds, errs, color=BLUE, lw=2.2, marker="o", ms=3, label="reconstruction error ↓")
    axc.set_xlim(0.8, C + 0.2); axc.set_ylim(-0.03, 1.05); axc.set_xticks(range(1, C + 1))
    axc.tick_params(colors=MUTED, labelsize=8); axc.set_yticks([0, 0.5, 1.0]); axc.set_yticklabels(["0", ".5", "1"], color=MUTED, fontsize=8)
    [sp.set_color(BORDER) for sp in axc.spines.values()]
    axc.set_xlabel("budget d (modes spent)", color=MUTED, fontsize=9.5)
    axc.legend(loc="center right", fontsize=8.5, frameon=False)

    fig.text(0.265, 0.07, f"d = {d}   ·   alignment {align:.2f}   ·   error {err:.2f}", ha="center", color=INK, fontsize=11, weight="bold")
    fig.canvas.draw(); rgba = np.asarray(fig.canvas.buffer_rgba()).copy(); plt.close(fig); return rgba


def main():
    frames = [draw(i) for i in range(FRAMES)]
    Image.fromarray(frames[int(FRAMES * 0.45)]).save(OUT_PREVIEW)
    imageio.mimsave(OUT_GIF, frames, duration=1 / FPS, loop=0, palettesize=128, subrectangles=True)
    print("wrote", OUT_GIF)


if __name__ == "__main__":
    main()
