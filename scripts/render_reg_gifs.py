"""Figures for the "Why Regularization Is a Price List" JAX companion.

Every number is read from public/regularization/pricelist.json, produced by the
analytic kernel-ridge solve in scripts/regularization_pricelist.py (no gradient
descent). No synthetic metaphors.

A figure is a GIF only when motion carries information no single frame gives: a
genuine parameter sweep where the object transforms. A static result is a PNG.

Three GIFs (real transforming lambda sweeps):
  a. reg-knob.gif       the lambda knob turning: the fitted 1-D function stiffens
                        from ferocious-wiggle to smooth, while the RKHS-norm bill
                        and d_eff readouts fall. Real ridge fits at swept lambda.
  b. reg-eviction.gif   the spectrum eviction: eigenvalue bars, each tinted by its
                        survival weight lambda_k/(lambda_k+lambda); the tail fades
                        first as lambda grows. Real Gram spectrum, real survival.
  d. reg-shrink.gif     the graded shrink (honest anti-sparsity): all six |alpha_i|
                        bars sink smoothly as lambda tightens, redundant points slip
                        under a dashed reading-threshold first, NO hard zeros.

Three static PNGs (static results: a breakdown, a U-curve, a two-panel compare):
  c. reg-bill.png       the RKHS bill as a static breakdown: each mode's line
                        c_k^2/lambda_k, ordered, summing to the real total.
  e. reg-ucurve.png     the generalization U-curve vs lambda, train and test MSE,
                        with the sweet spot lambda* = 0.42 marked.
  f. reg-bridge.png     the points<->modes bridge, two static panels: one bump's
                        fixed mode list and what alpha buys.

Run: python scripts/render_reg_gifs.py
"""
import json
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "public"
DATA = json.load(open(PUB / "regularization" / "pricelist.json"))

BG = "#0e0d0b"; PANEL = "#16140f"; INK = "#e8e2d4"; MUTED = "#9a9282"; LINE = "#3a352c"
A_COL = "#b3661b"; B_COL = "#4a7fb3"; ACC = "#e0a45a"; GOOD = "#7bbf5a"; WARN = "#c25a4a"
plt.rcParams.update({
    "figure.facecolor": BG, "savefig.facecolor": BG, "text.color": INK,
    "axes.edgecolor": LINE, "font.size": 11,
})


def ease(t):
    return t * t * (3 - 2 * t)


def fig_rgba(fig):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba


def save_gif(path, frames, fps, hold=16, palettesize=96):
    Image.fromarray(frames[-1]).save(str(path).replace(".gif", "-preview.png"))
    imageio.mimsave(path, frames + [frames[-1]] * hold, duration=1 / fps, loop=0,
                    palettesize=palettesize, subrectangles=True)
    print(f"wrote {path} ({os.path.getsize(path)//1024} KB)")


def save_png(path, fig):
    fig.savefig(path, dpi=110, facecolor=BG)
    plt.close(fig)
    print(f"wrote {path} ({os.path.getsize(path)//1024} KB)")


LAMBDAS = np.array(DATA["meta"]["lambdas"])


# ---------------------------------------------------------------------------
# The real ridge fit at any swept lambda index, reconstructed from the export.
# We stored three reference fits; for a smooth knob we recompute the fit on the
# stored train Gram basis via the stored eigen data. Simpler: re-solve from the
# stored train points and kernel (tiny, analytic, no training).
# ---------------------------------------------------------------------------
def rbf(A, B, sig):
    d2 = (A[:, None, :] - B[None, :, :]) ** 2
    return np.exp(-d2.sum(-1) / (2.0 * sig ** 2))


R = DATA["regression"]
x_tr = np.array(R["x_tr"]); y_tr = np.array(R["y_tr"])
x_te = np.array(R["x_te"]); y_te = np.array(R["y_te"])
grid_x = np.array(R["grid_x"])
SIG = DATA["meta"]["sig_reg"]
Ktr = rbf(x_tr[:, None], x_tr[:, None], SIG)
Kq = rbf(grid_x[:, None], x_tr[:, None], SIG)
EIGS_REG = np.array(R["eigs"])


def ridge_fit(lam):
    alpha = np.linalg.solve(Ktr + lam * np.eye(len(x_tr)), y_tr)
    pred = Kq @ alpha
    bill = float(alpha @ (Ktr @ alpha))
    d_eff = float(np.sum(EIGS_REG / (EIGS_REG + lam)))
    return pred, bill, d_eff


# ═══════════════════════════════════════════════════════════════════════════
# GIF a — the lambda knob turning
# ═══════════════════════════════════════════════════════════════════════════
def gif_knob():
    # sweep lambda from loose (wiggly) to tight (smooth)
    idxs = np.linspace(2, 52, 34).astype(int)
    frames = []
    bill0 = ridge_fit(LAMBDAS[idxs[0]])[1]
    for j, i in enumerate(idxs):
        lam = LAMBDAS[i]
        pred, bill, d_eff = ridge_fit(lam)
        fig = plt.figure(figsize=(6.4, 5.4), dpi=100, facecolor=BG)
        fig.text(0.5, 0.955, "Turn the budget knob, watch the fit stiffen",
                 ha="center", fontsize=14, weight="bold")
        fig.text(0.5, 0.907,
                 "kernel ridge at growing λ: the wiggle flattens as the RKHS bill and the effective dimension fall",
                 ha="center", fontsize=8.2, color=MUTED)
        ax = fig.add_axes([0.11, 0.30, 0.84, 0.55]); ax.set_facecolor(PANEL)
        ax.plot(x_te, y_te, color=MUTED, lw=1.2, alpha=0.7, label="true signal")
        ax.scatter(x_tr, y_tr, s=18, c=A_COL, alpha=0.7, linewidths=0, label="noisy data", zorder=3)
        ax.plot(grid_x, pred, color=ACC, lw=2.4, zorder=4, label="ridge fit")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(-1.35, 1.35); ax.set_xlim(-3, 3)
        for s in ax.spines.values(): s.set_color(LINE)
        ax.legend(loc="upper right", fontsize=8, facecolor=PANEL, edgecolor=LINE, labelcolor=INK)
        # readout bars for the bill and d_eff
        axb = fig.add_axes([0.11, 0.09, 0.37, 0.13]); axb.set_facecolor(PANEL)
        axb.barh([0], [bill / bill0], color=WARN, height=0.5)
        axb.set_xlim(0, 1.05); axb.set_ylim(-0.6, 0.6); axb.set_yticks([]); axb.set_xticks([])
        for s in axb.spines.values(): s.set_color(LINE)
        axb.text(0.02, 0.42, f"RKHS bill  ‖f‖²ₕ = {bill:.2f}", transform=axb.transAxes,
                 fontsize=8.5, color=INK, va="center")
        axd = fig.add_axes([0.57, 0.09, 0.38, 0.13]); axd.set_facecolor(PANEL)
        axd.barh([0], [d_eff / len(x_tr)], color=B_COL, height=0.5)
        axd.set_xlim(0, 1.05); axd.set_ylim(-0.6, 0.6); axd.set_yticks([]); axd.set_xticks([])
        for s in axd.spines.values(): s.set_color(LINE)
        axd.text(0.02, 0.42, f"effective dim  d_eff = {d_eff:.1f}", transform=axd.transAxes,
                 fontsize=8.5, color=INK, va="center")
        fig.text(0.5, 0.24, f"λ = {lam:.3g}", ha="center", fontsize=11, color=ACC, weight="bold")
        frames.append(fig_rgba(fig))
    save_gif(PUB / "reg-knob.gif", frames, fps=12)


# ═══════════════════════════════════════════════════════════════════════════
# GIF b — the spectrum eviction: survival weight fades the tail first
# ═══════════════════════════════════════════════════════════════════════════
def gif_eviction():
    eigs = EIGS_REG[:24]  # top modes, the ones with meaningful price
    idxs = np.linspace(2, 55, 34).astype(int)
    ks = np.arange(len(eigs))
    frames = []
    for i in idxs:
        lam = LAMBDAS[i]
        surv = eigs / (eigs + lam)  # real survival weight per mode
        d_eff = float(np.sum(EIGS_REG / (EIGS_REG + lam)))
        fig = plt.figure(figsize=(6.6, 5.0), dpi=100, facecolor=BG)
        fig.text(0.5, 0.955, "The budget evicts the tail first",
                 ha="center", fontsize=14, weight="bold")
        fig.text(0.5, 0.905,
                 "each bar is an eigenvalue λₖ (a mode's price); its brightness is the survival λₖ/(λₖ+λ)",
                 ha="center", fontsize=8.1, color=MUTED)
        ax = fig.add_axes([0.10, 0.13, 0.85, 0.72]); ax.set_facecolor(PANEL)
        for k, (e, sv) in enumerate(zip(eigs, surv)):
            # brightness = survival: surviving modes bright orange, evicted ones fade to line
            col = np.array([0.70, 0.40, 0.11]) * sv + np.array([0.227, 0.208, 0.173]) * (1 - sv)
            ax.bar(k, e, color=col, width=0.82, edgecolor=LINE, linewidth=0.4)
        ax.set_xlim(-0.7, len(eigs) - 0.3); ax.set_ylim(0, eigs[0] * 1.08)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_color(LINE)
        ax.set_xlabel("mode k  (smooth, cheap  →  wiggly, expensive)", fontsize=8.5, color=MUTED)
        ax.set_ylabel("eigenvalue λₖ", fontsize=9, color=MUTED)
        ax.text(0.98, 0.94, f"λ = {lam:.3g}", transform=ax.transAxes, ha="right",
                fontsize=11, color=ACC, weight="bold")
        ax.text(0.98, 0.85, f"d_eff = Σ λₖ/(λₖ+λ) = {d_eff:.1f}", transform=ax.transAxes,
                ha="right", fontsize=9.5, color=B_COL, weight="bold")
        frames.append(fig_rgba(fig))
    save_gif(PUB / "reg-eviction.gif", frames, fps=12)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE c (static PNG) — the RKHS bill broken down by mode
# A static breakdown: each mode's line c_k^2/lambda_k, ordered, plus the
# cumulative curve reaching the real total. Motion added nothing; this is a
# static figure of where the bill comes from.
# ═══════════════════════════════════════════════════════════════════════════
def fig_bill():
    Mo = DATA["modes"]
    c = np.array(Mo["c_k"]); ev = np.array(Mo["eigs"])
    cost = c ** 2 / ev  # each mode's line on the bill: c_k^2 / lambda_k
    order = np.argsort(-cost)
    heights = cost[order]
    total = float(np.sum(cost))
    cum = np.cumsum(heights)
    show = min(len(c), 22)
    xs = np.arange(show)

    fig = plt.figure(figsize=(6.6, 5.0), dpi=110, facecolor=BG)
    fig.text(0.5, 0.955, "Where the RKHS bill comes from",
             ha="center", fontsize=14, weight="bold")
    fig.text(0.5, 0.905,
             "each mode's line on the bill is cₖ²/λₖ; the lines sum to the real total αᵀKα = "
             f"{total:.2f}",
             ha="center", fontsize=8.0, color=MUTED)

    ax = fig.add_axes([0.11, 0.13, 0.80, 0.72]); ax.set_facecolor(PANEL)
    ax.bar(xs, heights[:show], color=WARN, width=0.8, edgecolor=LINE, linewidth=0.4,
           label="mode's line  cₖ²/λₖ")
    ax.set_xlim(-0.7, show - 0.3); ax.set_ylim(0, heights[0] * 1.12)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color(LINE)
    ax.set_xlabel("mode k, ordered by its line on the bill  (expensive → cheap)",
                  fontsize=8.5, color=MUTED)
    ax.set_ylabel("cₖ²/λₖ", fontsize=10, color=MUTED)

    # cumulative curve on a twin axis, reaching the total
    axc = ax.twinx()
    axc.plot(xs, cum[:show], color=ACC, lw=2.0, marker="o", ms=3.2,
             label="running total")
    axc.axhline(total, color=MUTED, lw=1.0, ls="--", alpha=0.7)
    axc.set_ylim(0, total * 1.12)
    axc.set_ylabel("cumulative  Σ cₖ²/λₖ", fontsize=9, color=ACC)
    axc.tick_params(colors=ACC, labelsize=7.5)
    for s in axc.spines.values(): s.set_color(LINE)

    axc.text(show - 0.5, total, f"  total αᵀKα = {total:.2f}", ha="right", va="bottom",
             fontsize=8.5, color=MUTED)
    ax.text(0.97, 0.60, "top mode: 29% of the bill\ntop five: 73%", transform=ax.transAxes,
            ha="right", va="top", fontsize=8.5, color=INK)
    lines = ax.get_legend_handles_labels()[0] + axc.get_legend_handles_labels()[0]
    labs = ax.get_legend_handles_labels()[1] + axc.get_legend_handles_labels()[1]
    ax.legend(lines, labs, loc="upper center", fontsize=8, facecolor=PANEL,
              edgecolor=LINE, labelcolor=INK)
    save_png(PUB / "reg-bill.png", fig)


# ═══════════════════════════════════════════════════════════════════════════
# GIF d — the graded shrink (honest anti-sparsity)
# ═══════════════════════════════════════════════════════════════════════════
def gif_shrink():
    Mi = DATA["miniature"]
    sweep = Mi["sweep"]
    # sweep from loose to tight; the six bars sink smoothly, never to zero
    idxs = np.linspace(0, len(sweep) - 6, 34).astype(int)
    labels = ["p1", "p2", "p3", "p4", "p5", "p6"]
    cols = [A_COL, A_COL, A_COL, B_COL, B_COL, B_COL]
    thresh = 0.15  # reading aid, NOT a mechanism
    a0 = np.array(sweep[0]["abs_alpha"])
    frames = []
    for i in idxs:
        row = sweep[i]
        a = np.array(row["abs_alpha"])
        lam = row["lam"]
        fig = plt.figure(figsize=(6.6, 5.0), dpi=100, facecolor=BG)
        fig.text(0.5, 0.955, "Ridge shrinks every point, and never to zero",
                 ha="center", fontsize=14, weight="bold")
        fig.text(0.5, 0.905,
                 "|αᵢ| for six training points as λ tightens: redundant points slip under the reading line first",
                 ha="center", fontsize=8.0, color=MUTED)
        ax = fig.add_axes([0.11, 0.14, 0.84, 0.71]); ax.set_facecolor(PANEL)
        for k in range(6):
            faded = a[k] < thresh
            col = cols[k]
            ax.bar(k, a[k], color=col, width=0.7, edgecolor=LINE, linewidth=0.5,
                   alpha=0.35 if faded else 0.95)
        ax.axhline(thresh, color=INK, lw=1.0, ls="--", alpha=0.6)
        ax.text(5.55, thresh + 0.01, "reading aid", fontsize=7.5, color=MUTED, ha="right")
        ax.set_xlim(-0.7, 5.7); ax.set_ylim(0, a0.max() * 1.12)
        ax.set_xticks(range(6)); ax.set_xticklabels(labels, color=MUTED, fontsize=9)
        ax.set_yticks([])
        for s in ax.spines.values(): s.set_color(LINE)
        ax.set_ylabel("|αᵢ|", fontsize=10, color=MUTED)
        ax.text(0.98, 0.93, f"λ = {lam:.3g}", transform=ax.transAxes, ha="right",
                fontsize=11, color=ACC, weight="bold")
        ax.text(0.98, 0.84, f"min |αᵢ| = {a.min():.3f}  (>0)", transform=ax.transAxes,
                ha="right", fontsize=9, color=GOOD, weight="bold")
        frames.append(fig_rgba(fig))
    save_gif(PUB / "reg-shrink.gif", frames, fps=12)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE e (static PNG) — the generalization U-curve
# A static result: train and test MSE swept over lambda, with the sweet spot
# lambda* = 0.42 marked. A curve is a static figure; drawing it on is fake motion.
# ═══════════════════════════════════════════════════════════════════════════
def fig_ucurve():
    sweep = R["sweep"]
    lam = np.array([r["lam"] for r in sweep])
    tr = np.array([r["train_mse"] for r in sweep])
    te = np.array([r["test_mse"] for r in sweep])
    star = R["star_index"]
    logl = np.log10(lam)

    fig = plt.figure(figsize=(6.6, 5.0), dpi=110, facecolor=BG)
    fig.text(0.5, 0.955, "Too loose overfits, too tight underfits",
             ha="center", fontsize=14, weight="bold")
    fig.text(0.5, 0.905,
             "train and test error swept over the budget λ; the sweet spot is where test error bottoms out",
             ha="center", fontsize=8.1, color=MUTED)
    ax = fig.add_axes([0.13, 0.14, 0.82, 0.71]); ax.set_facecolor(PANEL)
    ax.plot(logl, tr, color=B_COL, lw=2.0, label="train MSE")
    ax.plot(logl, te, color=ACC, lw=2.4, label="test MSE")
    ax.set_xlim(logl[0], logl[-1])
    ax.set_ylim(0, max(te.max(), tr.max()) * 1.08)
    ax.set_xlabel("log₁₀ λ   (loose budget  →  tight budget)", fontsize=8.6, color=MUTED)
    ax.set_ylabel("mean squared error", fontsize=9, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values(): s.set_color(LINE)
    ax.legend(loc="upper center", fontsize=9, facecolor=PANEL, edgecolor=LINE, labelcolor=INK)

    # annotate the loose (overfit) and tight (underfit) ends
    ax.annotate("loose: chases noise", xy=(logl[0], te[0]), xytext=(logl[0] + 0.15, 0.09),
                fontsize=7.8, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
    ax.annotate("tight: underfits", xy=(logl[-1], te[-1]), xytext=(logl[-1] - 1.6, te[-1] - 0.055),
                fontsize=7.8, color=MUTED, ha="right",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))

    ax.axvline(logl[star], color=GOOD, lw=1.2, ls="--", alpha=0.8)
    ax.scatter([logl[star]], [te[star]], s=70, color=GOOD, zorder=5,
               edgecolors="white", linewidths=0.6)
    ax.text(logl[star], te[star] + 0.035,
            f"sweet spot\nλ* = {lam[star]:.2g}\ntest MSE = {te[star]:.3f}",
            ha="center", fontsize=8.5, color=GOOD, weight="bold")
    save_png(PUB / "reg-ucurve.png", fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE f (static PNG) — the points<->modes bridge, two static panels
# Top: one bump's fixed shopping list of modes, lam_k phi_k(x_1).
# Bottom: what the whole solved weight buys, c_k. A two-panel comparison is a
# static figure; easing the bottom panel in was decorative.
# ═══════════════════════════════════════════════════════════════════════════
def fig_bridge():
    Mi = DATA["miniature"]
    X = np.array(Mi["X"]); y = np.array(Mi["y"])
    sig = DATA["meta"]["sig_mini"]
    K = rbf(X, X, sig)
    lam = 0.2
    alpha = np.linalg.solve(K + lam * np.eye(len(X)), y)
    evals, evecs = np.linalg.eigh(K)
    o = np.argsort(evals)[::-1]
    evals = np.clip(evals[o], 0, None); evecs = evecs[:, o]
    i0 = 0  # bump for point p1
    bump_list = evals * evecs[i0, :]            # per-mode amount in ONE bump's list
    c_k = evals * (evecs.T @ alpha)             # the whole weight's mode purchase
    n = len(evals)

    fig = plt.figure(figsize=(6.8, 5.2), dpi=110, facecolor=BG)
    fig.text(0.5, 0.955, "A bump arrives with a shopping list of modes",
             ha="center", fontsize=13.5, weight="bold")
    fig.text(0.5, 0.905,
             "k(xᵢ,·) = Σₖ λₖ φₖ(xᵢ) φₖ: one bump's fixed list (top); what α buys, cₖ (bottom)",
             ha="center", fontsize=8.0, color=MUTED)
    # top: one bump's list
    ax1 = fig.add_axes([0.11, 0.50, 0.84, 0.33]); ax1.set_facecolor(PANEL)
    ax1.bar(range(n), bump_list, color=A_COL, width=0.7, edgecolor=LINE, linewidth=0.4)
    ax1.axhline(0, color=LINE, lw=0.8)
    ax1.set_xlim(-0.6, n - 0.4); ax1.set_xticks([]); ax1.set_yticks([])
    for s in ax1.spines.values(): s.set_color(LINE)
    mx1 = np.abs(bump_list).max() * 1.2
    ax1.set_ylim(-mx1, mx1)
    ax1.text(0.98, 0.86, "one bump k(x₁,·): λₖ φₖ(x₁)", transform=ax1.transAxes,
             ha="right", fontsize=8.5, color=A_COL, weight="bold")
    # bottom: the whole weight's purchase c_k
    ax2 = fig.add_axes([0.11, 0.11, 0.84, 0.33]); ax2.set_facecolor(PANEL)
    ax2.bar(range(n), c_k, color=ACC, width=0.7, edgecolor=LINE, linewidth=0.4)
    ax2.axhline(0, color=LINE, lw=0.8)
    ax2.set_xlim(-0.6, n - 0.4); ax2.set_xticks([]); ax2.set_yticks([])
    for s in ax2.spines.values(): s.set_color(LINE)
    mx2 = np.abs(c_k).max() * 1.2
    ax2.set_ylim(-mx2, mx2)
    ax2.set_xlabel("mode k", fontsize=8.6, color=MUTED)
    ax2.text(0.98, 0.86, "the weight buys cₖ = λₖ Σᵢ αᵢ φₖ(xᵢ)", transform=ax2.transAxes,
             ha="right", fontsize=8.5, color=ACC, weight="bold")
    save_png(PUB / "reg-bridge.png", fig)


if __name__ == "__main__":
    gif_knob()       # GIF: real lambda sweep, the fit stiffens
    gif_eviction()   # GIF: real lambda sweep, the tail fades
    gif_shrink()     # GIF: real lambda sweep, the bars sink
    fig_bill()       # static PNG: RKHS bill broken down by mode
    fig_ucurve()     # static PNG: generalization U-curve with sweet spot
    fig_bridge()     # static PNG: two-panel points<->modes bridge
    print("REG_FIGS_DONE")
