"""Render the modality-gap dynamics (gap.py) to a hero GIF and a static figure.

Encoding: COLOUR = modality (A blue, B orange); SHAPE = class (○ / ✕).

  gap_dynamics.gif - the two modality clouds separating over contrastive training
  gap_cones.png    - the final two cones, beside the gap-vs-step curve
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
import glob

BG = "#fbf8f1"; PANEL = "#ffffff"; BORDER = "#e4e1d6"
FG = "#1a1a1a"; MUTED = "#5a5f66"; ACCENT = "#b3661b"; BLUE = "#4a7fb3"
A_COL, B_COL = BLUE, ACCENT

HERE = os.path.dirname(__file__)
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "public", "modality-gap"))


def _serif():
    for p in glob.glob(os.path.join(HERE, "..", "..", "node_modules", "@fontsource", "lora", "files", "*.ttf")):
        try:
            fm.fontManager.addfont(p); return fm.FontProperties(fname=p).get_name()
        except Exception:
            pass
    return "DejaVu Serif"
SERIF = _serif()


def _style(ax):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(BORDER)
    ax.tick_params(colors=MUTED, labelsize=8)


def scat(ax, P, c, color):
    m0, m1 = c == 0, c == 1
    ax.scatter(P[m0, 0], P[m0, 1], s=15, marker="o", c=color, linewidths=0, alpha=0.82, zorder=2)
    ax.scatter(P[m1, 0], P[m1, 1], s=22, marker="x", c=color, linewidths=1.1, alpha=0.9, zorder=2)


def _handles():
    return [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=A_COL, markersize=7, label="modality A"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=B_COL, markersize=7, label="modality B"),
        Line2D([0], [0], marker="o", color=MUTED, markerfacecolor=MUTED, lw=0, markersize=7, label="class 0 ○"),
        Line2D([0], [0], marker="x", color=MUTED, lw=0, markersize=7, label="class 1 ✕"),
    ]


def make_gif(d):
    PA, PB, traj, c = d["PA"], d["PB"], d["traj"], d["c"]
    F = PA.shape[0]; lim = 1.15
    pidx = np.linspace(0, PA.shape[1] - 1, 14).astype(int)
    fig, ax = plt.subplots(figsize=(5.6, 5.3), dpi=96); fig.patch.set_facecolor(BG)
    HOLD = 12

    def frame(k):
        f = min(k, F - 1)
        ax.clear(); _style(ax)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_xticks([]); ax.set_yticks([])
        for i in pidx:
            ax.plot([PA[f, i, 0], PB[f, i, 0]], [PA[f, i, 1], PB[f, i, 1]], color=MUTED, lw=0.5, alpha=0.25, zorder=1)
        scat(ax, PA[f], c, A_COL); scat(ax, PB[f], c, B_COL)
        step, matched, rnd, gap = traj[f]
        ax.set_title(f"contrastive training · step {int(step):>4d}", family=SERIF, fontsize=13, color=FG, pad=10)
        ax.text(0.5, 1.005, f"modality gap = {gap:.2f}    matched-pair cosine = {matched:+.2f}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color=MUTED, family="monospace")
        ax.legend(handles=_handles(), loc="lower left", fontsize=7.5, frameon=False, labelcolor=MUTED, ncol=2)
        ax.plot([-lim, -lim + 2*lim*(f/(F-1))], [-lim + 0.04, -lim + 0.04], color=ACCENT, lw=2.5, zorder=3, solid_capstyle="round")

    anim = FuncAnimation(fig, frame, frames=F + HOLD, interval=90)
    path = os.path.join(OUT, "gap_dynamics.gif")
    anim.save(path, writer=PillowWriter(fps=12)); plt.close(fig)
    print(f"  gap_dynamics.gif: {os.path.getsize(path)/1024:.0f} KB ({F} frames)")


def make_cones(d):
    PA, PB, traj, c = d["PA"], d["PB"], d["traj"], d["c"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2), dpi=120,
                             gridspec_kw={"width_ratios": [1, 1.05]}); fig.patch.set_facecolor(BG)
    ax = axes[0]; _style(ax); lim = 1.15
    for i in np.linspace(0, PA.shape[1] - 1, 16).astype(int):
        ax.plot([PA[-1, i, 0], PB[-1, i, 0]], [PA[-1, i, 1], PB[-1, i, 1]], color=MUTED, lw=0.5, alpha=0.25)
    scat(ax, PA[-1], c, A_COL); scat(ax, PB[-1], c, B_COL)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("After training: two separated cones", family=SERIF, fontsize=11.5, color=FG)
    ax.legend(handles=_handles(), loc="lower left", fontsize=7.5, frameon=False, labelcolor=MUTED, ncol=2)

    ax = axes[1]; _style(ax); step = traj[:, 0]
    ax.plot(step, traj[:, 3], "-o", color=ACCENT, ms=3, lw=1.8, label="modality gap")
    ax.plot(step, traj[:, 1], "-", color=BLUE, lw=1.6, label="matched-pair cosine")
    ax.plot(step, traj[:, 2], "--", color=MUTED, lw=1.4, label="random-pair cosine")
    ax.axhline(0, color=BORDER, lw=1)
    ax.set_xlabel("contrastive training step", color=MUTED, fontsize=9)
    ax.set_ylabel("distance / cosine", color=MUTED, fontsize=9)
    ax.set_title("The objective opens the gap", family=SERIF, fontsize=11.5, color=FG)
    ax.legend(loc="center right", fontsize=8, frameon=False, labelcolor=MUTED)
    fig.tight_layout()
    path = os.path.join(OUT, "gap_cones.png"); fig.savefig(path, facecolor=BG); plt.close(fig)
    print(f"  gap_cones.png: {os.path.getsize(path)/1024:.0f} KB")


def main():
    d = dict(np.load(os.path.join(OUT, "results_gap.npz")))
    make_cones(d); make_gif(d)


if __name__ == "__main__":
    main()
