#!/usr/bin/env python3
"""Render the JAX-companion GIFs for the survival post from the Kaggle bundle.

Every moving thing is a real number produced by scripts/yat_deepsurv.py and read
back from public/yat-deepsurv/*.json. No synthetic trajectories: prototype clouds
are real k-means-seeded centers migrating over real training epochs, KM curves are
real Kaplan-Meier estimates on real risk tertiles, attribution bars are the real
signed contributions a_u phi_u(x), reliability points are real predicted-vs-observed
survival, the OOD walk uses the real kernel evaluated on a real patient.

Six GIFs:
  1. deepsurv-stratify   : both models' risk-tertile KM curves separating over training.
  2. deepsurv-assemble   : one patient's risk assembling bar-by-bar from prototypes.
  3. deepsurv-settle     : the prototype cloud (2D PCA) settling onto patient data.
  4. deepsurv-forget     : cohort deletion, the risk field before/after, cohort re-routes.
  5. deepsurv-abstain    : a patient walked off-distribution, the kernel field going quiet.
  6. deepsurv-reliability: predicted-vs-observed survival drawing itself for both models.

Run: python3 scripts/render_deepsurv_gifs.py   (local, reads the bundle, no training)
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
BUNDLE = ROOT / "public" / "yat-deepsurv"
OUT = ROOT / "public"

BG = "#fbfaf6"; PANEL = "#ffffff"; INK = "#181818"; MUTED = "#666a70"
BORDER = "#ded9cb"; ACCENT = "#b3661b"; BLUE = "#4a7fb3"; GREEN = "#3a8f5e"
GREY = "#9aa0a8"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})


def load(name):
    return json.load(open(BUNDLE / name))


def save_gif(frames, path, fps, preview_idx=0):
    imgs = [Image.fromarray(f) for f in frames]
    imageio.mimsave(path, imgs, fps=fps, loop=0)
    imgs[preview_idx].save(str(path).replace(".gif", "-preview.png"))
    kb = Path(path).stat().st_size / 1024
    print(f"  {path.name}: {len(frames)} frames, {kb:.0f} KB")


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
# 1. RISK STRATIFICATION forming over training (both models).
# ===========================================================================
def gif_stratify():
    tr = load("train_trace.json")
    tg = np.array(tr["tgrid"])
    yat_km = tr["yat_km"]      # [frames][3 tertiles][T]
    mlp_km = tr["mlp_km"]
    epochs = tr["epochs"]
    F = min(len(yat_km), len(mlp_km))
    cols = [BLUE, GREEN, ACCENT]     # low / mid / high risk
    names = ["low-risk third", "mid-risk third", "high-risk third"]
    frames = []
    for fi in list(range(F)) + [F - 1] * 6:
        fi = min(fi, F - 1)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=100)
        fig.patch.set_facecolor(BG)
        for ax, km, title in [(axes[0], yat_km[fi], "Yat DeepSurv"),
                              (axes[1], mlp_km[fi], "standard DeepSurv")]:
            style_ax(ax)
            for g in range(3):
                ax.plot(tg, km[g], color=cols[g], lw=2.4, label=names[g])
            ax.set_ylim(0, 1.02); ax.set_xlim(0, tg[-1])
            ax.set_title(title, color=INK, fontsize=13, weight="bold")
            ax.set_xlabel("months since diagnosis", color=MUTED, fontsize=10)
            ax.set_ylabel("survival", color=MUTED, fontsize=10)
            if title.startswith("Yat"):
                ax.legend(loc="lower left", fontsize=9, frameon=False, labelcolor=INK)
        fig.suptitle(f"Risk stratification forming over training   epoch {epochs[fi]}",
                     color=INK, fontsize=14, weight="bold", y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        frames.append(fig_to_arr(fig)); plt.close(fig)
    save_gif(frames, OUT / "deepsurv-stratify.gif", fps=6)


# ===========================================================================
# 2. A PATIENT'S RISK ASSEMBLING from prototype contributions, bar by bar.
# ===========================================================================
def gif_assemble():
    a = load("attributions.json")
    pt = [p for p in a["patients"] if p["label"] == "high"][0]
    contrib = np.array(pt["contrib"])
    order = np.argsort(-np.abs(contrib))
    K = len(contrib)
    TOP = min(10, K)
    top = order[:TOP]
    logrisk = pt["logrisk"]
    maxabs = np.abs(contrib).max()
    frames = []
    # reveal bars one at a time, running total climbing
    for step in list(range(TOP + 1)) + [TOP] * 8:
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=100,
                                       gridspec_kw={"width_ratios": [1.4, 1]})
        fig.patch.set_facecolor(BG)
        style_ax(axL); style_ax(axR)
        # left: signed bars, revealed
        ys = np.arange(TOP)[::-1]
        for r, u in enumerate(top):
            shown = r < step
            c = contrib[u]
            col = ACCENT if c >= 0 else BLUE
            axL.barh(ys[r], c, color=col, alpha=0.85 if shown else 0.08,
                     edgecolor="none", height=0.7)
            axL.text(-maxabs * 1.28, ys[r], f"#{u}", va="center", ha="left",
                     color=MUTED if shown else BORDER, fontsize=10)
        axL.axvline(0, color=BORDER, lw=1)
        axL.set_xlim(-maxabs * 1.35, maxabs * 1.15)
        axL.set_ylim(-0.6, TOP - 0.4)
        axL.set_yticks([])
        axL.set_xlabel("contribution to log-risk   a_u φ_u(x)", color=MUTED, fontsize=10)
        axL.set_title("prototypes this patient resembles", color=INK, fontsize=12, weight="bold")
        # right: running total vs true log-risk
        running = contrib[top[:step]].sum()
        axR.bar([0], [running], color=ACCENT, alpha=0.85, width=0.5)
        axR.axhline(logrisk, color=INK, lw=1.6, ls="--")
        axR.text(0, logrisk + 0.06, f"true h(x) = {logrisk:.2f}", ha="center",
                 color=INK, fontsize=10)
        axR.set_ylim(min(0, logrisk - 0.4), max(logrisk + 0.5, running + 0.5))
        axR.set_xlim(-0.6, 0.6); axR.set_xticks([])
        axR.set_ylabel("log-risk assembled", color=MUTED, fontsize=10)
        axR.set_title(f"running sum = {running:.2f}", color=INK, fontsize=12, weight="bold")
        fig.suptitle("Your risk is a sum over the patients you resemble",
                     color=INK, fontsize=14, weight="bold", y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        frames.append(fig_to_arr(fig)); plt.close(fig)
    save_gif(frames, OUT / "deepsurv-assemble.gif", fps=3)


# ===========================================================================
# 3. THE PROTOTYPE CLOUD settling onto patient data over training.
# ===========================================================================
def gif_settle():
    tr = load("train_trace.json")
    data2d = np.array(tr["data2d"])
    protos = tr["yat_protos"]     # [frames][K][2]
    readout = np.array(tr["proto_readout"])
    epochs = tr["epochs"]
    F = len(protos)
    # fixed axis limits from data
    xlo, xhi = data2d[:, 0].min() - 0.5, data2d[:, 0].max() + 0.5
    ylo, yhi = data2d[:, 1].min() - 0.5, data2d[:, 1].max() + 0.5
    frames = []
    for fi in list(range(F)) + [F - 1] * 6:
        P = np.array(protos[fi])
        fig, ax = plt.subplots(figsize=(7.4, 5.4), dpi=100)
        fig.patch.set_facecolor(BG); style_ax(ax)
        ax.scatter(data2d[:, 0], data2d[:, 1], s=10, c=GREY, alpha=0.35,
                   edgecolors="none", label="patients")
        for u in range(len(P)):
            col = ACCENT if readout[u] >= 0 else BLUE
            ax.scatter(P[u, 0], P[u, 1], s=140, facecolors="none",
                       edgecolors=col, linewidths=2.2)
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("patient covariate space  (2D PCA)", color=MUTED, fontsize=10)
        ax.set_title(f"Prototypes settling onto patient data   epoch {epochs[fi]}",
                     color=INK, fontsize=13, weight="bold")
        ax.scatter([], [], s=100, facecolors="none", edgecolors=ACCENT, linewidths=2,
                   label="risk-raising prototype")
        ax.scatter([], [], s=100, facecolors="none", edgecolors=BLUE, linewidths=2,
                   label="protective prototype")
        ax.legend(loc="upper right", fontsize=9, frameon=False, labelcolor=INK)
        fig.tight_layout()
        frames.append(fig_to_arr(fig)); plt.close(fig)
    save_gif(frames, OUT / "deepsurv-settle.gif", fps=6)


# ===========================================================================
# 4. COHORT DELETION: the risk field before/after, untouched patients frozen.
# ===========================================================================
def gif_forget():
    ed = load("edit.json")
    emb = load("embedding.json")
    P = np.array(emb["patients"])
    before = np.array(ed["risk_before"]); after = np.array(ed["risk_after"])
    cohort = np.array(ed["cohort_mask"]).astype(bool)
    n = min(len(P), len(before))
    P = P[:n]; before = before[:n]; after = after[:n]; cohort = cohort[:n]
    rlo, rhi = min(before.min(), after.min()), max(before.max(), after.max())
    xlo, xhi = P[:, 0].min() - 1, P[:, 0].max() + 1
    ylo, yhi = P[:, 1].min() - 1, P[:, 1].max() + 1
    frames = []
    steps = 26
    for s in list(range(steps + 1)) + [steps] * 8:
        t = s / steps
        risk = before * (1 - t) + after * t
        fig, ax = plt.subplots(figsize=(7.4, 5.6), dpi=100)
        fig.patch.set_facecolor(BG); style_ax(ax)
        # non-cohort patients (frozen), cohort patients (moving color)
        sc = ax.scatter(P[~cohort, 0], P[~cohort, 1], c=risk[~cohort], cmap="coolwarm",
                        vmin=rlo, vmax=rhi, s=22, alpha=0.6, edgecolors="none")
        ax.scatter(P[cohort, 0], P[cohort, 1], c=risk[cohort], cmap="coolwarm",
                   vmin=rlo, vmax=rhi, s=60, edgecolors=INK, linewidths=0.8)
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        ax.set_xticks([]); ax.set_yticks([])
        state = "before" if t < 0.02 else ("after" if t > 0.98 else "deleting...")
        ax.set_title(f"Forgetting a cohort   ({state})", color=INK, fontsize=13, weight="bold")
        ax.set_xlabel("cohort ringed in black; color = log-risk (blue low, red high)",
                      color=MUTED, fontsize=9)
        cb = fig.colorbar(sc, ax=ax, fraction=0.045)
        cb.ax.tick_params(colors=MUTED, labelsize=9)
        fig.tight_layout()
        frames.append(fig_to_arr(fig)); plt.close(fig)
    save_gif(frames, OUT / "deepsurv-forget.gif", fps=8)


# ===========================================================================
# 5. OOD WALK: a patient pushed off-distribution, the kernel field going quiet.
# ===========================================================================
def gif_abstain():
    proto = load("prototypes.json")
    ood = load("ood.json")
    W = np.array([p["covariates_norm"] for p in proto["prototypes"]])
    b, eps = proto["b"], proto["eps"]

    def kmax(x):
        dot = W @ x
        d2 = (x ** 2).sum() + (W ** 2).sum(1) - 2 * dot
        return float(((dot + b) ** 2 / (d2 + eps)).max())

    # a real in-distribution patient: sit exactly on a prototype (strong resemblance),
    # then push it out along a random direction so its field goes quiet honestly.
    base = W[0].copy()
    rng = np.random.RandomState(3)
    direction = rng.choice([-1, 1], size=base.shape) * (0.6 + rng.rand(*base.shape))

    kin = np.array(ood["kmax_in"]); kout = np.array(ood["kmax_out"])
    CAP = 30
    frames = []
    pushes = np.concatenate([np.linspace(0, 4, 34), np.full(8, 4.0)])
    for push in pushes:
        x = base + push * direction
        km = kmax(x)
        fig, ax = plt.subplots(figsize=(8, 4.4), dpi=100)
        fig.patch.set_facecolor(BG); style_ax(ax)
        bins = np.linspace(0, CAP, 30)
        ax.hist(np.clip(kin, 0, CAP), bins=bins, color=BLUE, alpha=0.55, label="real patients")
        ax.hist(np.clip(kout, 0, CAP), bins=bins, color=GREY, alpha=0.5, label="out-of-distribution")
        ax.axvline(min(km, CAP), color=ACCENT, lw=2.6)
        quiet = km < 4.0
        ax.text(min(km, CAP), ax.get_ylim()[1] * 0.92,
                f" this patient: {km:.1f}" + ("  (abstain)" if quiet else ""),
                color=ACCENT, fontsize=11, weight="bold", ha="left" if km < CAP * 0.6 else "right")
        ax.set_xlim(0, CAP)
        ax.set_xlabel("kernel-max resemblance score", color=MUTED, fontsize=10)
        ax.set_yticks([])
        ax.set_title(f"Walking a patient off the data manifold   push = {push:.1f}σ",
                     color=INK, fontsize=13, weight="bold")
        ax.legend(loc="upper right", fontsize=9, frameon=False, labelcolor=INK)
        fig.tight_layout()
        frames.append(fig_to_arr(fig)); plt.close(fig)
    save_gif(frames, OUT / "deepsurv-abstain.gif", fps=10, preview_idx=len(frames) - 10)


# ===========================================================================
# 6. RELIABILITY drawing itself: predicted vs observed survival, both models.
# ===========================================================================
def gif_reliability():
    ood = load("ood.json")
    horizons = np.array(ood["horizons"])
    ry = ood["reliability_yat"]; rm = ood["reliability_mlp"]
    cols = [BLUE, GREEN, ACCENT]
    Hn = len(horizons)
    frames = []
    for k in list(range(1, Hn + 1)) + [Hn] * 8:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=100)
        fig.patch.set_facecolor(BG)
        for ax, rel, title in [(axes[0], ry, "Yat DeepSurv"), (axes[1], rm, "standard DeepSurv")]:
            style_ax(ax)
            ax.plot([0, 1], [0, 1], color=BORDER, lw=1.2, ls="--")
            for g in range(3):
                pred = np.array(rel[g]["pred"])[:k]
                obs = np.array(rel[g]["obs"])[:k]
                ax.plot(pred, obs, "-o", color=cols[g], lw=2, ms=5,
                        label=["low-risk", "mid-risk", "high-risk"][g] + " third")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xlabel("predicted survival", color=MUTED, fontsize=10)
            ax.set_ylabel("observed survival (Kaplan-Meier)", color=MUTED, fontsize=10)
            ax.set_title(title, color=INK, fontsize=13, weight="bold")
            if title.startswith("Yat"):
                ax.legend(loc="upper left", fontsize=9, frameon=False, labelcolor=INK)
        fig.suptitle("Predicted vs observed survival   (on the diagonal = honest)",
                     color=INK, fontsize=14, weight="bold", y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        frames.append(fig_to_arr(fig)); plt.close(fig)
    save_gif(frames, OUT / "deepsurv-reliability.gif", fps=3, preview_idx=len(frames) - 6)


if __name__ == "__main__":
    print("Rendering DeepSurv companion GIFs from the bundle...")
    gif_stratify()
    gif_assemble()
    gif_settle()
    gif_forget()
    gif_abstain()
    gif_reliability()
    print("Done.")
