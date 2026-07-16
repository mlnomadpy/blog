#!/usr/bin/env python3
"""Render the JAX-companion figures for "You Don't Have to Solve a Kernel Machine".

Every moving thing is a real number from the Kaggle run of
scripts/kernel_solve_wall.py, read back from the result bundles:

  scripts/results/kgl_blog-solve-wall-v1/kernel_solve_wall.{json,npz}   E1, E2, E3, E4
  scripts/results/kgl_blog-solve-wall-v1b/kernel_solve_wall.json        E3B (K=256, K=1024)
  scripts/results/kgl_blog-solve-wall-v1c/kernel_solve_wall.json        E3B (K=1024, 20 epochs)

Three GIFs (each a real temporal process) + three static PNGs (scoreboards):

  1. solve-wall-agreement   GIF: the descended machine's 400 test predictions
                            scattering against the exact solve's, epoch by epoch,
                            locking onto the diagonal; the per-epoch Pearson r is
                            e1.agree_series.r from the run.
  2. solve-wall-prototypes  GIF: the 32 prototype vectors (e1_proto_traj) migrating
                            through a fixed 2D PCA shadow of the training cloud,
                            from their k-means start to their trained positions.
  3. solve-wall-features    GIF: one fixed garment's conv1 feature maps
                            (e4_fmaps_small) evolving over end-to-end training,
                            with the real validation-accuracy curve as readout.
  4. solve-wall-timing      PNG: measured E2 wall clock (CPU/GPU solve vs one SGD
                            epoch) with Gram-size annotations. A scoreboard.
  5. solve-wall-ladder      PNG: E3 + E3B accuracy vs rows, the memory wall, and
                            the capacity ladder crossing the solve's best line.
  6. solve-wall-compose     PNG: the four E4 accuracies with per-seed dots.

The timing curves, the ladder and the bars are finished results with no temporal
process behind them, so they stay static.

Run: python3 scripts/render_solve_wall_gifs.py   (local; reads bundles, no training)
"""
from __future__ import annotations
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "scripts" / "results"
V1 = json.load(open(RES / "kgl_blog-solve-wall-v1" / "kernel_solve_wall.json"))
V1B = json.load(open(RES / "kgl_blog-solve-wall-v1b" / "kernel_solve_wall.json"))
V1C = json.load(open(RES / "kgl_blog-solve-wall-v1c" / "kernel_solve_wall.json"))
NPZ = np.load(RES / "kgl_blog-solve-wall-v1" / "kernel_solve_wall.npz")
OUT = ROOT / "public"

BG = "#fbfaf6"; PANEL = "#ffffff"; INK = "#181818"; MUTED = "#666a70"
BORDER = "#ded9cb"; ACCENT = "#b3661b"; BLUE = "#4a7fb3"; GREEN = "#3a8f5e"
GREY = "#9aa0a8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})


def save_gif(frames, path, fps, preview_idx=-1):
    imgs = [Image.fromarray(f) for f in frames]
    imageio.mimsave(path, imgs, fps=fps, loop=0)
    imgs[preview_idx].save(str(path).replace(".gif", "-preview.png"))
    kb = Path(path).stat().st_size / 1024
    print(f"  {path.name}: {len(frames)} frames, {kb:.0f} KB")


def save_png(fig, path):
    fig.savefig(path, dpi=110, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  {Path(path).name}: {Path(path).stat().st_size/1024:.0f} KB")


def fig_to_arr(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    return buf.copy()


def style_ax(ax):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(BORDER)
    ax.tick_params(colors=MUTED, labelsize=10)


# ===========================================================================
# 1. AGREEMENT: the descended machine's predictions locking onto the solve's.
#    Frames are real epochs (0..68 of the K=32 telemetry run). The r readout
#    is the Pearson r of exactly the 400 points on screen (the run's held-out
#    prediction snapshots against the solve's), so readout and motion agree;
#    the run's full-test series (e1.agree_series.r) tells the same story but
#    is dragged by off-screen outliers in the chaotic early epochs.
# ===========================================================================
def gif_agreement():
    solve = NPZ["e1_snap_solve"]            # (400,) exact-solve predictions, fixed
    net = NPZ["e1_snap_net"]                # (160, 400) net predictions per epoch
    LAST = 68                               # the run holds r ~ 0.95 here
    lo = float(solve.min()) - 0.35
    hi = float(solve.max()) + 0.35

    # every real epoch 0..LAST is a frame: the finest granularity the run
    # recorded, so the motion is as smooth as the real process allows
    seq = [(net[ep], float(np.corrcoef(net[ep], solve)[0, 1]), ep)
           for ep in range(LAST + 1)]
    frames = []
    hold = 12
    for k, (p, rv, ep) in enumerate(seq + [seq[-1]] * hold):
        fig, ax = plt.subplots(figsize=(5.4, 5.4), dpi=100)
        fig.patch.set_facecolor(BG); style_ax(ax)
        ax.plot([lo, hi], [lo, hi], color=BORDER, lw=1.4, ls="--", zorder=1)
        ax.scatter(solve, p, s=14, c=ACCENT, alpha=0.55, edgecolors="none", zorder=2)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("the solved machine's prediction", color=MUTED, fontsize=11)
        ax.set_ylabel("the descended machine's prediction", color=MUTED, fontsize=11)
        ax.set_title("Two fits of one kernel, one function", color=INK,
                     fontsize=12.5, weight="bold")
        ax.text(0.04, 0.945, f"epoch {ep}", transform=ax.transAxes,
                color=INK, fontsize=12, weight="bold")
        ax.text(0.04, 0.885, f"agreement r = {rv:.3f}", transform=ax.transAxes,
                color=ACCENT, fontsize=12, weight="bold")
        fig.tight_layout()
        frames.append(fig_to_arr(fig)); plt.close(fig)
    save_gif(frames, OUT / "solve-wall-agreement.gif", fps=8)


# ===========================================================================
# 2. PROTOTYPES: the 32 centers migrating through the training cloud, from
#    their k-means start to their trained positions. PCA is fit ONCE on the
#    2000 training rows and never refit; each frame projects the real
#    prototype vectors of that epoch. Trails are the real past positions.
# ===========================================================================
def gif_prototypes():
    X = NPZ["e1_train_X"]                   # (2000, 8)
    traj = NPZ["e1_proto_traj"]             # (160, 32, 8)
    mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    P2 = (X - mu) @ Vt[:2].T                # fixed 2D shadow of the cloud
    T2 = (traj - mu) @ Vt[:2].T             # (160, 32, 2)
    xlo, xhi = P2[:, 0].min() - 0.8, P2[:, 0].max() + 0.8
    ylo, yhi = P2[:, 1].min() - 0.8, P2[:, 1].max() + 0.8

    eps_frames = list(range(0, 160, 3)) + [159]
    frames = []
    hold = 10
    for ep in eps_frames + [eps_frames[-1]] * hold:
        fig, ax = plt.subplots(figsize=(6.1, 4.8), dpi=100)
        fig.patch.set_facecolor(BG); style_ax(ax)
        ax.scatter(P2[:, 0], P2[:, 1], s=7, c=GREY, alpha=0.4, edgecolors="none")
        # trails: each prototype's real path up to this epoch
        for u in range(T2.shape[1]):
            ax.plot(T2[: ep + 1, u, 0], T2[: ep + 1, u, 1], color=ACCENT,
                    lw=0.9, alpha=0.35, zorder=2)
        ax.scatter(T2[0, :, 0], T2[0, :, 1], s=46, facecolors="none",
                   edgecolors=MUTED, linewidths=1.1, zorder=3, label="k-means start")
        ax.scatter(T2[ep, :, 0], T2[ep, :, 1], s=52, c=ACCENT,
                   edgecolors=INK, linewidths=0.6, zorder=4, label="prototype now")
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("training districts, 2D PCA shadow of the 8-dim space",
                      color=MUTED, fontsize=10)
        ax.set_title(f"32 prototypes migrating under gradient descent   epoch {ep}",
                     color=INK, fontsize=12.5, weight="bold")
        ax.legend(loc="upper right", fontsize=9, frameon=False, labelcolor=INK)
        fig.tight_layout()
        frames.append(fig_to_arr(fig)); plt.close(fig)
    save_gif(frames, OUT / "solve-wall-prototypes.gif", fps=10)


# ===========================================================================
# 3. FEATURES: one fixed garment on the left; its 8 conv1 feature maps
#    evolving over the 30 real end-to-end training epochs on the right, with
#    the run's real validation accuracy as the readout. This motion is the
#    thing a solve cannot do: the representation being built.
# ===========================================================================
def gif_features():
    garment = NPZ["e4_garment"]             # (28, 28)
    fmaps = NPZ["e4_fmaps_small"]           # (30, 14, 14, 8)
    acc = NPZ["e4_curve_small"]             # (30,) val accuracy per epoch
    E = fmaps.shape[0]
    vmax = float(fmaps.max())

    # keyframes: every real epoch; one crossfade frame between neighbours
    seq = []
    for ep in range(E):
        seq.append((fmaps[ep], float(acc[ep]), ep))
        if ep < E - 1:
            seq.append((0.5 * (fmaps[ep] + fmaps[ep + 1]),
                        0.5 * float(acc[ep] + acc[ep + 1]), ep + 1))
    frames = []
    hold = 12
    for fm, av, ep in seq + [seq[-1]] * hold:
        fig = plt.figure(figsize=(8.6, 4.4), dpi=100)
        fig.patch.set_facecolor(BG)
        gs = fig.add_gridspec(2, 6, left=0.03, right=0.98, top=0.80, bottom=0.10,
                              wspace=0.08, hspace=0.14)
        axg = fig.add_subplot(gs[:, :2])
        axg.imshow(garment, cmap="gray_r", interpolation="nearest")
        axg.set_xticks([]); axg.set_yticks([])
        for s in axg.spines.values():
            s.set_color(BORDER)
        axg.set_title("the garment (fixed)", color=MUTED, fontsize=10)
        for j in range(8):
            ax = fig.add_subplot(gs[j // 4, 2 + j % 4])
            ax.imshow(fm[:, :, j], cmap="magma", vmin=0, vmax=vmax,
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(BORDER)
        fig.text(0.62, 0.845, "its conv1 feature maps, training end to end",
                 color=MUTED, fontsize=10, ha="center")
        fig.suptitle("Gradient descent building the representation the kernel head compares",
                     color=INK, fontsize=12.5, weight="bold", y=0.985, x=0.5)
        fig.text(0.03, 0.025, f"epoch {ep}", color=INK, fontsize=11, weight="bold")
        fig.text(0.97, 0.025, f"val accuracy {av:.3f}", color=ACCENT, fontsize=11,
                 weight="bold", ha="right")
        frames.append(fig_to_arr(fig)); plt.close(fig)
    save_gif(frames, OUT / "solve-wall-features.gif", fps=9)


# ===========================================================================
# 4. TIMING (static): the measured E2 wall. CPU/GPU exact solve vs one SGD
#    epoch, log-log, with the Gram-matrix footprint annotated.
# ===========================================================================
def fig_timing():
    rows = V1["e2"]["rows"]
    ns = np.array([r["n"] for r in rows])
    cpu = np.array([r["solve_cpu_s"] for r in rows])
    gpu = np.array([r["solve_gpu_s"] for r in rows])
    sgd = np.array([r["sgd_epoch_s"] for r in rows])
    gb = np.array([r["gram_gb_f64"] for r in rows])

    fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=110)
    fig.patch.set_facecolor(BG); style_ax(ax)
    ax.plot(ns, cpu, "-o", color=ACCENT, lw=2.2, ms=6, label="exact solve, CPU float64")
    ax.plot(ns, gpu, "-s", color=BLUE, lw=2.2, ms=6, label="exact solve, GPU float32")
    ax.plot(ns, sgd, "-^", color=GREEN, lw=2.2, ms=6, label="one SGD epoch, K = 32")
    ax.set_xscale("log"); ax.set_yscale("log")
    for n, c, g in zip(ns[2:], cpu[2:], gb[2:]):
        ax.annotate(f"Gram {g:.2f} GB", (n, c), textcoords="offset points",
                    xytext=(6, 9), fontsize=8.5, color=MUTED)
    ax.annotate("at n = 64,000 the Gram alone is 33 GB:\nno solve on a 16 GB box",
                xy=(16000, cpu[-1]), xytext=(2400, 90), fontsize=9.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.1))
    ax.set_xlabel("training rows n", color=MUTED, fontsize=11)
    ax.set_ylabel("seconds (log scale)", color=MUTED, fontsize=11)
    ax.set_title("The measured wall: exact fit vs one gradient epoch",
                 color=INK, fontsize=13, weight="bold")
    ax.legend(loc="lower right", fontsize=9.5, frameon=False, labelcolor=INK)
    save_png(fig, OUT / "solve-wall-timing.png")


# ===========================================================================
# 5. LADDER (static): E3 accuracy vs rows used, the wall, and the E3B
#    capacity ladder crossing the solve's best line. K=1024 at the full set
#    is the v1c 20-epoch run (0.8593 +- 0.0012), which supersedes v1b's
#    8-epoch row at the same K and n.
# ===========================================================================
def fig_ladder():
    e3 = V1["e3"]
    sol_n = [r["n"] for r in e3["solve"]]
    sol_a = [r["acc_mean"] for r in e3["solve"]]
    sgd_n = [r["n"] for r in e3["sgd"]]
    sgd_a = [r["acc_mean"] for r in e3["sgd"]]
    sgd_s = [r["acc_std"] for r in e3["sgd"]]
    b = {(r["K"], r["n"]): r for r in V1B["e3b"]["rows"]}
    c = {(r["K"], r["n"]): r for r in V1C["e3b"]["rows"]}
    k256_n = [64000, 256000, 511012]
    k256_a = [b[(256, n)]["acc_mean"] for n in k256_n]
    k1024_n = [64000, 256000, 511012]
    k1024_a = [b[(1024, 64000)]["acc_mean"], b[(1024, 256000)]["acc_mean"],
               c[(1024, 511012)]["acc_mean"]]
    k1024_s = [b[(1024, 64000)]["acc_std"], b[(1024, 256000)]["acc_std"],
               c[(1024, 511012)]["acc_std"]]
    best_solve = max(sol_a)

    fig, ax = plt.subplots(figsize=(7.8, 5.2), dpi=110)
    fig.patch.set_facecolor(BG); style_ax(ax)
    ax.axvspan(22000, 700000, color=GREY, alpha=0.12, zorder=0)
    ax.text(110000, 0.752, "past the wall:\nno Gram fits in 16 GB", color=MUTED,
            fontsize=9.5, ha="center")
    ax.axhline(best_solve, color=ACCENT, lw=1.3, ls="--", alpha=0.8)
    ax.text(2100, best_solve + 0.003, f"best the solve ever posted  {best_solve:.4f}",
            color=ACCENT, fontsize=9)
    ax.plot(sol_n, sol_a, "-o", color=ACCENT, lw=2.4, ms=7, label="exact solve (all rows as centers)")
    ax.errorbar(sgd_n, sgd_a, yerr=sgd_s, fmt="-^", color=GREEN, lw=2.2, ms=6,
                capsize=2, label="descended, K = 64")
    ax.plot(k256_n, k256_a, "-s", color=BLUE, lw=2.0, ms=6, label="descended, K = 256")
    ax.errorbar(k1024_n, k1024_a, yerr=k1024_s, fmt="-D", color=INK, lw=2.0, ms=6,
                capsize=2, label="descended, K = 1,024")
    ax.annotate(f"{k1024_a[-1]:.4f}", (511012, k1024_a[-1]),
                textcoords="offset points", xytext=(-6, 10), fontsize=10,
                color=INK, weight="bold", ha="right")
    ax.set_xscale("log")
    ax.set_xlim(1600, 700000)
    ax.set_xlabel("training rows used (log scale)", color=MUTED, fontsize=11)
    ax.set_ylabel("held-out accuracy", color=MUTED, fontsize=11)
    ax.set_title("Covertype: the solve stops at 16,000 rows, the ladder keeps climbing",
                 color=INK, fontsize=12.5, weight="bold")
    ax.legend(loc="lower right", fontsize=9, frameon=False, labelcolor=INK)
    save_png(fig, OUT / "solve-wall-ladder.png")


# ===========================================================================
# 6. COMPOSE (static): the four E4 accuracies, per-seed dots on the SGD bars.
# ===========================================================================
def fig_compose():
    e4 = V1["e4"]
    labels = ["exact solve\nraw pixels\nn = 8,000",
              "exact solve\nfrozen random trunk\nn = 8,000",
              "end to end\ntrunk + kernel head\nn = 8,000",
              "end to end\ntrunk + kernel head\nn = 55,000"]
    vals = [e4["acc_solve_raw"], e4["acc_solve_frozen"],
            e4["acc_e2e_small_mean"], e4["acc_e2e_full_mean"]]
    seeds = [None, None, e4["acc_e2e_small_seeds"], e4["acc_e2e_full_seeds"]]
    cols = [ACCENT, GREY, GREEN, GREEN]

    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=110)
    fig.patch.set_facecolor(BG); style_ax(ax)
    xs = np.arange(4)
    ax.bar(xs, vals, width=0.58, color=cols, alpha=0.88, edgecolor="none")
    for x, sd in zip(xs, seeds):
        if sd:
            ax.scatter([x] * len(sd), sd, s=26, c=INK, zorder=3)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.004, f"{v:.4f}", ha="center", color=INK,
                fontsize=10.5, weight="bold")
    ax.set_ylim(0.80, 0.925)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=9, color=INK)
    ax.set_ylabel("Fashion-MNIST test accuracy", color=MUTED, fontsize=11)
    ax.set_title("Composition: what training the representation buys",
                 color=INK, fontsize=13, weight="bold")
    save_png(fig, OUT / "solve-wall-compose.png")


if __name__ == "__main__":
    print("Rendering solve-wall companion figures from the bundles...")
    gif_agreement()     # GIF: real per-epoch predictions locking onto the solve's
    gif_prototypes()    # GIF: real prototype trajectories through the data cloud
    gif_features()      # GIF: real conv1 maps evolving under end-to-end descent
    fig_timing()        # static: measured wall-clock scoreboard
    fig_ladder()        # static: accuracy-vs-n scoreboard with the wall
    fig_compose()       # static: the four E4 accuracies
    print("Done.")
