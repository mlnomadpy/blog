"""Render the separate-vs-represent toy (toy.py) to blog-styled PNGs.

Encoding (consistent across the whole post): marker SHAPE = class (○ / ✕),
COLOUR = modality (A = blue, B = orange).

  toy_data.png     - the four distributions: two modalities × two classes
  toy_collapse.png - the SAME two modalities under a separating space (SigLIP,
                     structure collapsed) and a representing space (autoencoder,
                     structure kept)
  toy_jobs.png     - the two jobs in numbers: both classify; only represent
                     recovers the within-class factor
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import glob

BG = "#fbf8f1"; PANEL = "#ffffff"; BORDER = "#e4e1d6"
FG = "#1a1a1a"; MUTED = "#5a5f66"; ACCENT = "#b3661b"; BLUE = "#4a7fb3"
A_COL, B_COL = BLUE, ACCENT          # modality colours

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


def pca2(X):
    X = X - X.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:2].T


def scatter_mc(ax, P, c, color, s0=16, s1=24):
    """Plot points: class 0 -> filled circle, class 1 -> cross, all one colour."""
    m0, m1 = c == 0, c == 1
    ax.scatter(P[m0, 0], P[m0, 1], s=s0, marker="o", c=color, linewidths=0, alpha=0.82)
    ax.scatter(P[m1, 0], P[m1, 1], s=s1, marker="x", c=color, linewidths=1.1, alpha=0.9)


def _legend(fig):
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=A_COL, markersize=7, label="modality A"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=B_COL, markersize=7, label="modality B"),
        Line2D([0], [0], marker="o", color=MUTED, markerfacecolor=MUTED, markersize=7, lw=0, label="class 0  (○)"),
        Line2D([0], [0], marker="x", color=MUTED, markersize=7, lw=0, label="class 1  (✕)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.5,
               labelcolor=MUTED, bbox_to_anchor=(0.5, -0.02))


def fig_data(d):
    # the four distributions, side by side in one axes: A on the left, B on the right
    PA, PB = pca2(d["xA"]), pca2(d["xB"])
    norm = lambda P: P / (np.abs(P).max() + 1e-9)
    PA, PB = norm(PA), norm(PB)
    PA[:, 0] -= 1.35; PB[:, 0] += 1.35
    fig, ax = plt.subplots(figsize=(8.4, 4.0), dpi=120); fig.patch.set_facecolor(BG); _style(ax)
    scatter_mc(ax, PA, d["c"], A_COL); scatter_mc(ax, PB, d["c"], B_COL)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(-1.35, 1.18, "modality A", color=A_COL, ha="center", fontsize=10, family=SERIF)
    ax.text(1.35, 1.18, "modality B", color=B_COL, ha="center", fontsize=10, family=SERIF)
    ax.set_ylim(-1.3, 1.45)
    ax.set_title("Four distributions: two modalities × two classes",
                 family=SERIF, fontsize=12.5, color=FG)
    _legend(fig)
    fig.savefig(os.path.join(OUT, "toy_data.png"), facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def fig_collapse(d):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2), dpi=120); fig.patch.set_facecolor(BG)

    ax = axes[0]; _style(ax)
    th = np.linspace(0, 2*np.pi, 200); ax.plot(np.cos(th), np.sin(th), color=BORDER, lw=1.1, zorder=0)
    scatter_mc(ax, d["sA"], d["c"], A_COL); scatter_mc(ax, d["sB"], d["c"], B_COL)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_box_aspect(1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Separate · SigLIP", family=SERIF, fontsize=11.5, color=FG)
    ax.text(0.5, -0.06, f"recover t  R² = {max(0,float(d['s_t'])):.2f}   ·   classify {d['s_class']*100:.0f}%",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=MUTED, family="monospace")

    ax = axes[1]; _style(ax)
    scatter_mc(ax, d["aA"], d["c"], A_COL); scatter_mc(ax, d["aB"], d["c"], B_COL)
    ax.set_box_aspect(1); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Represent · autoencoder", family=SERIF, fontsize=11.5, color=FG)
    ax.text(0.5, -0.06, f"recover t  R² = {float(d['a_t']):.2f}   ·   classify {d['a_class']*100:.0f}%",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=MUTED, family="monospace")

    fig.suptitle("Same two modalities, same bottleneck — the objective decides what survives",
                 family=SERIF, fontsize=12.5, color=FG, y=1.0)
    _legend(fig)
    fig.savefig(os.path.join(OUT, "toy_collapse.png"), facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def fig_jobs(d):
    fig, ax = plt.subplots(figsize=(7.0, 3.9), dpi=120); fig.patch.set_facecolor(BG); _style(ax)
    groups = ["classify\n(recover class)", "represent\n(recover t, R²)"]
    sep = [float(d["s_class"]), max(0.0, float(d["s_t"]))]
    rep = [float(d["a_class"]), float(d["a_t"])]
    x = np.arange(2); w = 0.36
    PURPLE = "#9a4f9c"
    b1 = ax.bar(x - w/2, sep, w, color=PURPLE, label="separate (SigLIP)")
    b2 = ax.bar(x + w/2, rep, w, color=ACCENT, label="represent (autoencoder)")
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=9.5, color=FG)
    ax.set_ylim(0, 1.12); ax.set_ylabel("score  (accuracy / R²)", color=MUTED, fontsize=9)
    ax.set_title("Both spaces separate; only the representing space represents",
                 family=SERIF, fontsize=12, color=FG)
    ax.legend(loc="center", bbox_to_anchor=(0.5, 0.5), fontsize=8.5, frameon=False, labelcolor=MUTED)
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8.5, color=FG, family="monospace")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "toy_jobs.png"), facecolor=BG)
    plt.close(fig)


def main():
    d = dict(np.load(os.path.join(OUT, "results_toy.npz")))
    fig_data(d); fig_collapse(d); fig_jobs(d)
    for f in ["toy_data.png", "toy_collapse.png", "toy_jobs.png"]:
        p = os.path.join(OUT, f)
        if os.path.exists(p):
            print(f"  {f}: {os.path.getsize(p)/1024:.0f} KB")


if __name__ == "__main__":
    main()
