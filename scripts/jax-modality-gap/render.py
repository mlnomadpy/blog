"""Render the modality-gap experiment results to blog-styled PNGs.

Reads results.npz (written by demo.py) and produces three figures in
../../public/modality-gap/:

  gap.png         - the modality gap: contrastive (two cones) vs joint (one space)
  scores.png      - what each recovers: denoise MSE and modulation-class accuracy
  reconstruct.png - clean / noisy / MAE-recovered constellations for a few samples
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

BG = "#fbf8f1"; PANEL = "#ffffff"; BORDER = "#e4e1d6"
FG = "#1a1a1a"; MUTED = "#5a5f66"; ACCENT = "#b3661b"
BLUE = "#4a7fb3"; GREEN = "#3a8f5e"; PURPLE = "#9a4f9c"

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
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:2].T


def fig_gap(d):
    # How well did contrastive bring matched modality pairs together? Compare the
    # cosine of matched (waveform_i, constellation_i) pairs against random pairs.
    # If matched is barely above random, there was almost nothing to align: the
    # modalities are complementary, not redundant.
    zA, zB = d["zA"], d["zB"]
    matched = np.sum(zA * zB, axis=1)
    rng = np.random.default_rng(0)
    perm = rng.permutation(zB.shape[0])
    random = np.sum(zA * zB[perm], axis=1)

    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=120)
    fig.patch.set_facecolor(BG); _style(ax)
    bins = np.linspace(-1, 1, 50)
    ax.hist(random, bins=bins, color=MUTED, alpha=0.55, label="random pairs", density=True)
    ax.hist(matched, bins=bins, color=ACCENT, alpha=0.75, label="matched pairs", density=True)
    ax.axvline(1.0, color=GREEN, ls=(0, (3, 3)), lw=1)
    ax.text(0.99, 0.5, "perfect\nalignment", color=GREEN, fontsize=7.5,
            ha="right", va="center", transform=ax.get_xaxis_transform())
    ax.axvline(float(matched.mean()), color=ACCENT, lw=1.5)
    ax.set_title("Contrastive can barely align complementary modalities",
                 family=SERIF, fontsize=12, color=FG)
    ax.set_xlabel("cosine(waveform embedding, constellation embedding)", color=MUTED, fontsize=9)
    ax.set_ylabel("density", color=MUTED, fontsize=9)
    ax.legend(loc="upper left", fontsize=8.5, frameon=False, labelcolor=MUTED)
    ax.text(0.02, 0.74,
            f"matched mean = {matched.mean():.2f}\nrandom mean  = {random.mean():.2f}",
            transform=ax.transAxes, fontsize=8.5, color=MUTED, family="monospace")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "gap.png"), facecolor=BG)
    plt.close(fig)


def fig_scores(d):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), dpi=120)
    fig.patch.set_facecolor(BG)

    GREEN2 = "#3a8f5e"
    chance = 100.0 / 8
    ax = axes[0]; _style(ax)
    floor = float(d["noisy_mse"])
    vals = [floor, float(d["c_denoise"]), float(d["s_denoise"]),
            float(d["m_denoise"]), float(d["t_denoise"])]
    bars = ax.bar(["noisy", "InfoNCE", "SigLIP", "MAE", "multi-\ntask"],
                  vals, color=[MUTED, BLUE, PURPLE, ACCENT, GREEN2], width=0.7)
    ax.axhline(floor, color=MUTED, ls=(0, (3, 3)), lw=1)
    ax.set_title("Recover the clean signal  (MSE, lower = better)",
                 family=SERIF, fontsize=11, color=FG)
    ax.set_ylabel("reconstruction MSE", color=MUTED, fontsize=9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.2f}",
                ha="center", fontsize=8, color=FG, family="monospace")
    ax.text(0.98, 0.93, "below the dashed\nnoise floor = denoising",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=MUTED)

    ax = axes[1]; _style(ax)
    cvals = [float(d["c_class"]) * 100, float(d["s_class"]) * 100,
             float(d["m_class"]) * 100, float(d["t_class"]) * 100]
    bars = ax.bar(["InfoNCE", "SigLIP", "MAE", "multi-task"], cvals,
                  color=[BLUE, PURPLE, ACCENT, GREEN2], width=0.66)
    ax.axhline(chance, color=MUTED, ls=(0, (3, 3)), lw=1)
    ax.set_ylim(0, 100)
    ax.set_title("Modulation class  (linear probe, % acc)",
                 family=SERIF, fontsize=11, color=FG)
    for b, v in zip(bars, cvals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%",
                ha="center", fontsize=8, color=FG, family="monospace")
    ax.text(0.98, 0.10, "chance = 12.5%", transform=ax.transAxes, ha="right",
            fontsize=7.5, color=MUTED)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "scores.png"), facecolor=BG)
    plt.close(fig)


def fig_reconstruct(d):
    Z = d["Z_te"]; W = d["wav_te"]; R = d["rec_te"]
    k = min(4, Z.shape[0])
    fig, axes = plt.subplots(1, k, figsize=(2.1 * k, 2.4), dpi=120)
    fig.patch.set_facecolor(BG)
    if k == 1:
        axes = [axes]
    for j in range(k):
        ax = axes[j]; _style(ax)
        ax.scatter(W[j, 0::2], W[j, 1::2], s=22, c=MUTED, alpha=0.45, label="noisy", linewidths=0)
        ax.scatter(R[j, 0::2], R[j, 1::2], s=22, c=ACCENT, label="MAE", linewidths=0)
        ax.scatter(Z[j, 0::2], Z[j, 1::2], s=42, facecolors="none", edgecolors=GREEN,
                   linewidths=1.3, label="clean")
        ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.8, 1.8); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.legend(loc="upper left", fontsize=6.5, frameon=False, labelcolor=MUTED)
    fig.suptitle("Joint MAE snaps the noisy symbols back to the clean constellation",
                 family=SERIF, fontsize=11, color=FG, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "reconstruct.png"), facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def fig_amc(d):
    snr = d["snr_grid"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=120)
    fig.patch.set_facecolor(BG); _style(ax)
    ax.plot(snr, d["amc_infonce"] * 100, "-o", color=BLUE, ms=4, lw=1.8, label="InfoNCE")
    ax.plot(snr, d["amc_siglip"] * 100, "-s", color=PURPLE, ms=4, lw=1.8, label="SigLIP")
    ax.plot(snr, d["amc_mae"] * 100, "-^", color=ACCENT, ms=4.5, lw=1.8, label="joint MAE")
    if "amc_multitask" in d:
        ax.plot(snr, d["amc_multitask"] * 100, "-D", color="#3a8f5e", ms=3.5, lw=1.8,
                label="multi-task (SigLIP)")
    if "amc_multitask_infonce" in d:
        ax.plot(snr, d["amc_multitask_infonce"] * 100, "-v", color="#7a9a3e", ms=3.5, lw=1.6,
                label="multi-task (InfoNCE)")
    ax.axhline(100.0 / 8, color=MUTED, ls=(0, (3, 3)), lw=1)
    ax.text(snr[-1], 100.0 / 8 + 1.5, "chance 12.5%", color=MUTED, fontsize=7.5, ha="right")
    ax.set_xlabel("test SNR (dB)", color=MUTED, fontsize=9)
    ax.set_ylabel("modulation-class accuracy (%)", color=MUTED, fontsize=9)
    ax.set_title("Linear-probe AMC across SNR", family=SERIF, fontsize=12, color=FG)
    ax.legend(loc="upper left", fontsize=8.5, frameon=False, labelcolor=MUTED)
    ax.set_ylim(8, 90)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "amc_snr.png"), facecolor=BG)
    plt.close(fig)


def fig_gates(d):
    if "gate_mt" not in d:
        return
    gmt, gmae = d["gate_mt"], d["gate_mae"]
    steps = np.arange(1, gmt.shape[0] + 1)
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=120)
    fig.patch.set_facecolor(BG); _style(ax)
    ax.plot(steps, gmt[:, 0], "-o", color=ACCENT, lw=1.8, ms=5, label="multi-task · attention")
    ax.plot(steps, gmt[:, 1], "--o", color=ACCENT, lw=1.8, ms=5, label="multi-task · FFN")
    ax.plot(steps, gmae[:, 0], "-^", color=BLUE, lw=1.6, ms=5, label="MAE · attention")
    ax.plot(steps, gmae[:, 1], "--^", color=BLUE, lw=1.6, ms=5, label="MAE · FFN")
    ax.set_xticks(steps)
    ax.set_xlabel("recursion step (shared block applied $\\ell$ times)", color=MUTED, fontsize=9)
    ax.set_ylabel("CLS-conditioned gate value", color=MUTED, fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.set_title("The gate learns adaptive depth", family=SERIF, fontsize=12, color=FG)
    ax.legend(loc="lower left", fontsize=8, frameon=False, labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "gates.png"), facecolor=BG)
    plt.close(fig)


def fig_align(d):
    # Does reconstruction close the modality gap? Matched-pair cosine for each
    # contrastive head, standalone vs. as a multi-task auxiliary to reconstruction.
    if "ti_matched_cos" not in d:
        return
    fig, ax = plt.subplots(figsize=(7.0, 3.7), dpi=120)
    fig.patch.set_facecolor(BG); _style(ax)
    groups = ["InfoNCE\n(softmax)", "SigLIP\n(sigmoid)"]
    standalone = [float(d["matched_cos"]), float(d["s_matched_cos"])]
    multitask = [float(d["ti_matched_cos"]), float(d["t_matched_cos"])]
    x = np.arange(2); w = 0.36
    b1 = ax.bar(x - w / 2, standalone, w, color=MUTED, label="contrastive only")
    b2 = ax.bar(x + w / 2, multitask, w, color=GREEN, label="+ reconstruction (multi-task)")
    ax.axhline(0, color=BORDER, lw=1)
    ax.axhline(1.0, color=BLUE, ls=(0, (3, 3)), lw=1)
    ax.text(1.45, 1.0, "perfect\nalignment", color=BLUE, fontsize=7, ha="left", va="center")
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=9, color=MUTED)
    ax.set_ylabel("matched-pair cosine", color=MUTED, fontsize=9)
    ax.set_ylim(-0.8, 1.15)
    ax.set_title("Reconstruction closes the gap for InfoNCE, not SigLIP",
                 family=SERIF, fontsize=12, color=FG)
    ax.legend(loc="lower left", fontsize=8, frameon=False, labelcolor=MUTED)
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, v + (0.04 if v >= 0 else -0.10),
                    f"{v:+.2f}", ha="center", fontsize=8, color=FG, family="monospace")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "align.png"), facecolor=BG)
    plt.close(fig)


def main():
    d = dict(np.load(os.path.join(OUT, "results.npz")))
    fig_gap(d); fig_scores(d); fig_reconstruct(d); fig_amc(d); fig_gates(d); fig_align(d)
    for f in ["gap.png", "scores.png", "reconstruct.png", "amc_snr.png", "gates.png", "align.png"]:
        p = os.path.join(OUT, f)
        if os.path.exists(p):
            print(f"  {f}: {os.path.getsize(p) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
