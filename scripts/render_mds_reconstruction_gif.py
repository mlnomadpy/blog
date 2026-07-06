#!/usr/bin/env python3
"""How many modes to spend. The rank-d spectral codebook reconstructs the label
kernel out of its top d eigenmodes; the reconstruction error is exactly the tail
of the spectrum. This is not a temporal process -- it is a budget you choose --
so it renders as a static figure: the target kernel, three rank-d reconstructions
(d = 1, 3, C), and the full alignment-up / error-down curves over every budget d.
Every number is a jnp.linalg.eigh away."""
from __future__ import annotations
from pathlib import Path
import jax, jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from PIL import Image  # noqa: E402

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[1]
OUT_PNG = ROOT / "public" / "mds-reconstruction.png"
OUT_PREVIEW = ROOT / "public" / "mds-reconstruction-preview.png"

W, H, C, SIG = 1100, 520, 10, 2.6
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


CURVE = [rank_d(d) for d in range(1, C + 1)]                    # the full sweep, all budgets at once


def draw():
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    fig.text(0.5, 0.95, "How many modes to spend", ha="center", color=INK, fontsize=18, weight="bold")
    fig.text(0.5, 0.905, "the rank-d codebook rebuilds the label kernel; the error is the tail of the spectrum", ha="center", color=MUTED, fontsize=11.5)

    # target kernel, then three reconstructions at increasing budget
    show = [("target kernel S", Snp, None, INK)]
    for d in (1, 3, C):
        Shat, err, align = CURVE[d - 1]
        show.append((f"rank-{d}   (err {err:.2f})", Shat, d, ACCENT))
    for k, (title, mat, d, col) in enumerate(show):
        ax = fig.add_axes([0.035 + k * 0.125, 0.5, 0.11, 0.33])
        ax.imshow(mat, cmap=KMAP, vmin=-VMAX, vmax=VMAX)
        ax.set_xticks([]); ax.set_yticks([])
        lw = 2 if d is not None else 1
        [(sp.set_color(col), sp.set_linewidth(lw)) for sp in ax.spines.values()]
        ax.set_title(title, color=col, fontsize=9.5, weight="bold", pad=4)

    # spectrum: every mode, normalized
    axs = fig.add_axes([0.055, 0.13, 0.24, 0.28]); axs.set_facecolor(PANEL)
    bars = w_pos / (w_pos[0] + 1e-12)
    axs.bar(np.arange(C), bars, color=ACCENT)
    axs.set_ylim(0, 1.05); axs.set_xticks(range(C)); axs.set_xticklabels(range(1, C + 1), color=MUTED, fontsize=8)
    axs.set_yticks([]); [sp.set_color(BORDER) for sp in axs.spines.values()]
    axs.set_xlabel("mode", color=MUTED, fontsize=9)
    axs.set_title("spectrum of the label kernel", color=INK, fontsize=10, weight="bold", pad=4)

    # alignment up / error down across the whole budget
    axc = fig.add_axes([0.40, 0.13, 0.55, 0.34]); axc.set_facecolor(PANEL)
    ds = np.arange(1, C + 1)
    aligns = [CURVE[k - 1][2] for k in ds]; errs = [CURVE[k - 1][1] for k in ds]
    axc.plot(ds, aligns, color=GREEN, lw=2.4, marker="o", ms=4, label="kernel-target alignment ↑")
    axc.plot(ds, errs, color=BLUE, lw=2.4, marker="o", ms=4, label="reconstruction error ↓")
    for d, col in ((1, ACCENT), (3, ACCENT), (C, ACCENT)):
        axc.axvline(d, color=col, lw=1, ls=(0, (2, 3)), alpha=0.5)
    axc.set_xlim(0.8, C + 0.2); axc.set_ylim(-0.03, 1.05); axc.set_xticks(range(1, C + 1))
    axc.tick_params(colors=MUTED, labelsize=8.5); axc.set_yticks([0, 0.5, 1.0]); axc.set_yticklabels(["0", ".5", "1"], color=MUTED, fontsize=8.5)
    [sp.set_color(BORDER) for sp in axc.spines.values()]
    axc.set_xlabel("budget d (modes spent)", color=MUTED, fontsize=10)
    axc.set_title("what each budget buys", color=INK, fontsize=10.5, weight="bold", pad=4)
    axc.legend(loc="center right", fontsize=9, frameon=False)

    fig.canvas.draw(); rgba = np.asarray(fig.canvas.buffer_rgba()).copy(); plt.close(fig); return rgba


def main():
    buf = draw()
    Image.fromarray(buf).save(OUT_PNG)
    Image.fromarray(buf).save(OUT_PREVIEW)
    print("wrote", OUT_PNG)


if __name__ == "__main__":
    main()
