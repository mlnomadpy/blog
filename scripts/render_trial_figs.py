#!/usr/bin/env python3
"""Render the JAX-companion figures for "The White-Box Survival Model on Trial".

Two kinds of artifact, and the split is deliberate:

  GIFs (a real temporal process, every moving thing a real number):
    1. trial-settle    the Yat prototype cloud (real 2D PCA) migrating over real
                       training epochs as the model fits, on a real cohort.
    2. trial-stratify  the three risk-tertile Kaplan-Meier curves pulling apart
                       over real training epochs.
    Both come from an ACTUAL local training run of the real Flax NNX Yat DeepSurv
    on the WHAS500 cohort (sksurv, no download). The motion is the fit happening.

  Static PNGs (a finished result, no process to animate) read straight from the
  trial bundle scripts/results/kgl_blog-deepsurv-trial-v2/trial_results.json:
    3. trial-forest        per-dataset C-index with bootstrap CIs (the viability plot).
    4. trial-ablation      C-index vs prototype count K, kmeans vs random.
    5. trial-calibration   predicted vs observed survival by tertile (METABRIC).
    6. trial-ood           kernel-max OOD AUROCs with the honest below-0.5 bars.
    7. trial-plausibility  prototype in-range flags on METABRIC.

A GIF is earned only where a single frame would lose the information; the forest,
the ablation curve, the calibration panel, the OOD bars and the plausibility table
are all static results, so they are figures, not animations.

Run: python3 scripts/render_trial_figs.py   (local; no Kaggle, ~1 min for the run)
"""
from __future__ import annotations
import json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = json.load(open(ROOT / "scripts" / "results" / "kgl_blog-deepsurv-trial-v2" / "trial_results.json"))
OUT = ROOT / "public"

BG = "#fbfaf6"; PANEL = "#ffffff"; INK = "#181818"; MUTED = "#666a70"
BORDER = "#ded9cb"; ACCENT = "#b3661b"; BLUE = "#4a7fb3"; GREEN = "#3a8f5e"; GREY = "#9aa0a8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})


def save_gif(frames, path, fps, preview_idx=0):
    imgs = [Image.fromarray(f) for f in frames]
    imageio.mimsave(path, imgs, fps=fps, loop=0)
    imgs[preview_idx].save(str(path).replace(".gif", "-preview.png"))
    print(f"  {Path(path).name}: {len(frames)} frames, {Path(path).stat().st_size/1024:.0f} KB")


def save_png(fig, path):
    fig.savefig(path, dpi=110, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  {Path(path).name}: {Path(path).stat().st_size/1024:.0f} KB")


def fig_to_arr(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()


def style_ax(ax):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(BORDER)
    ax.tick_params(colors=MUTED, labelsize=10)


# ===========================================================================
# A REAL local training run of the Yat DeepSurv, capturing per-epoch state for
# the two motion GIFs. This is the same model as the companion code.
# ===========================================================================
def train_capture():
    import jax, jax.numpy as jnp, optax
    from flax import nnx
    from sksurv.datasets import load_whas500

    Xdf, y = load_whas500()
    import pandas as pd
    Xdf = pd.get_dummies(Xdf, drop_first=True)
    ev_f, tm_f = y.dtype.names
    e = y[ev_f].astype("float32"); t = y[tm_f].astype("float32")
    X = Xdf.values.astype("float32")
    good = np.isfinite(X).all(1) & (t > 0)
    X, t, e = X[good], t[good], e[good]
    rng = np.random.RandomState(0); perm = rng.permutation(len(X))
    X, t, e = X[perm], t[perm], e[perm]
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd

    K = 12
    from sklearn.cluster import KMeans
    Winit = KMeans(K, n_init=5, random_state=0).fit(Xn).cluster_centers_.astype("float32")

    class YatLayer(nnx.Module):
        def __init__(self, d, k, *, Winit):
            self.W = nnx.Param(jnp.asarray(Winit))
            self.log_b = nnx.Param(jnp.full((), jnp.log(jnp.expm1(1.0))))
            self.log_eps = nnx.Param(jnp.full((), jnp.log(jnp.expm1(1.0))))
        def __call__(self, x):
            b = jax.nn.softplus(self.log_b.value); eps = jax.nn.softplus(self.log_eps.value)
            dot = x @ self.W.value.T
            d2 = jnp.sum(x**2, -1, keepdims=True) + jnp.sum(self.W.value**2, -1) - 2 * dot
            return (dot + b) ** 2 / (d2 + eps)

    class YatSurv(nnx.Module):
        def __init__(self, d, k, *, rngs, Winit):
            self.yat = YatLayer(d, k, Winit=Winit)
            self.read = nnx.Linear(k, 1, use_bias=False, rngs=rngs)
        def __call__(self, x):
            return self.read(self.yat(x))[:, 0]

    def cox_loss(h, tt, ee):
        order = jnp.argsort(-tt); h = h[order]; ev = ee[order]
        m = jnp.max(h)
        log_cum = m + jnp.log(jnp.cumsum(jnp.exp(h - m)) + 1e-12)
        return -jnp.sum((h - log_cum) * ev) / (jnp.sum(ev) + 1e-8)

    model = YatSurv(Xn.shape[1], K, rngs=nnx.Rngs(0), Winit=Winit)
    opt = nnx.Optimizer(model, optax.adamw(1e-2, weight_decay=1e-4), wrt=nnx.Param)

    @nnx.jit
    def step(model, opt, xb, tb, eb):
        def lf(m): return cox_loss(m(xb), tb, eb)
        loss, grads = nnx.value_and_grad(lf)(model)
        opt.update(model, grads); return loss

    Xj, tj, ej = jnp.asarray(Xn), jnp.asarray(t), jnp.asarray(e)
    # PCA basis on the patient cloud (fixed, so the motion is the prototypes moving)
    Xc = Xn - Xn.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    basis = Vt[:2]                       # [2, d]
    cloud2d = Xc @ basis.T               # patients in the fixed 2D shadow

    snaps, epochs = [], []
    capture_at = [0, 5, 10, 20, 40, 70, 110, 160, 230, 320, 440, 600]
    for ep in range(0, 601):
        if ep in capture_at:
            W = np.asarray(model.yat.W.value)
            W2d = (W - Xn.mean(0)) @ basis.T
            r = np.asarray(model(Xj))
            snaps.append(dict(W2d=W2d.copy(), risk=r.copy(),
                              readout=np.asarray(model.read.kernel.value)[:, 0].copy()))
            epochs.append(ep)
        if ep < 600:
            step(model, opt, Xj, tj, ej)
    return dict(cloud2d=np.asarray(cloud2d), snaps=snaps, epochs=epochs,
                t=t, e=e, evar=float((S[:2] ** 2).sum() / (S ** 2).sum()))


# ---- KM helper for the stratification GIF ----
def km_curve(tt, ee, grid):
    order = np.argsort(tt); ts, es = tt[order], ee[order]
    S, out, n, i = 1.0, [], len(ts), 0
    for g in grid:
        while i < len(ts) and ts[i] <= g:
            if es[i] == 1 and (n - i) > 0:
                S *= (1 - 1.0 / (n - i))
            i += 1
        out.append(S)
    return np.array(out)


def gif_settle(cap):
    cloud, snaps, epochs = cap["cloud2d"], cap["snaps"], cap["epochs"]
    xr = [cloud[:, 0].min(), cloud[:, 0].max()]
    yr = [cloud[:, 1].min(), cloud[:, 1].max()]
    px = (xr[1] - xr[0]) * 0.12; py = (yr[1] - yr[0]) * 0.12
    frames = []
    F = len(snaps)
    for fi in list(range(F)) + [F - 1] * 5:
        fi = min(fi, F - 1)
        s = snaps[fi]
        fig, ax = plt.subplots(figsize=(4.6, 3.4), dpi=100)
        fig.patch.set_facecolor(BG); style_ax(ax)
        ax.scatter(cloud[:, 0], cloud[:, 1], s=14, c=GREY, alpha=0.35, edgecolors="none")
        for u in range(len(s["W2d"])):
            raise_ = s["readout"][u] >= 0
            ax.scatter(s["W2d"][u, 0], s["W2d"][u, 1], s=120,
                       facecolors="none", edgecolors=ACCENT if raise_ else BLUE, linewidths=2.2)
        ax.set_xlim(xr[0] - px, xr[1] + px); ax.set_ylim(yr[0] - py, yr[1] + py)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Prototypes settling onto the cohort   epoch {epochs[fi]}",
                     color=INK, fontsize=11.5, weight="bold")
        ax.set_xlabel(f"2D shadow of patient space ({cap['evar']*100:.0f}% of variance)\n"
                      "orange rings raise risk, blue protect", color=MUTED, fontsize=8.5)
        fig.tight_layout()
        frames.append(fig_to_arr(fig)); plt.close(fig)
    save_gif(frames, OUT / "trial-settle.gif", fps=4, preview_idx=len(snaps) - 1)


def gif_stratify(cap):
    snaps, epochs = cap["snaps"], cap["epochs"]
    t, e = cap["t"], cap["e"]
    grid = np.linspace(0, np.percentile(t, 92), 60)
    cols = [BLUE, GREEN, ACCENT]; names = ["low third", "mid third", "high third"]
    frames = []
    F = len(snaps)
    for fi in list(range(F)) + [F - 1] * 5:
        fi = min(fi, F - 1)
        r = snaps[fi]["risk"]
        cuts = np.percentile(r, [33.33, 66.67])
        grp = np.digitize(r, cuts)
        fig, ax = plt.subplots(figsize=(4.8, 3.4), dpi=100)
        fig.patch.set_facecolor(BG); style_ax(ax)
        for g in range(3):
            m = grp == g
            if m.sum() < 2:
                continue
            ax.plot(grid, km_curve(t[m], e[m], grid), color=cols[g], lw=2.4, label=names[g])
        ax.set_ylim(0, 1.02); ax.set_xlim(0, grid[-1])
        ax.set_title(f"Risk tertiles pulling apart   epoch {epochs[fi]}",
                     color=INK, fontsize=11.5, weight="bold")
        ax.set_xlabel("days since event", color=MUTED, fontsize=9)
        ax.set_ylabel("survival", color=MUTED, fontsize=9)
        ax.legend(loc="lower left", fontsize=8.5, frameon=False, labelcolor=INK)
        fig.tight_layout()
        frames.append(fig_to_arr(fig)); plt.close(fig)
    save_gif(frames, OUT / "trial-stratify.gif", fps=4, preview_idx=len(snaps) - 1)


# ===========================================================================
# STATIC PNGs from the trial bundle.
# ===========================================================================
def png_forest():
    ds_order = BUNDLE["meta"]["datasets"]
    labels = {"coxph": "Cox PH", "coxnet": "pen. Cox", "rsf": "RSF", "mlp": "ReLU DeepSurv", "yat": "Yat DeepSurv"}
    fig, axes = plt.subplots(1, len(ds_order), figsize=(14, 3.6), dpi=100, sharex=False)
    fig.patch.set_facecolor(BG)
    for ax, ds in zip(axes, ds_order):
        style_ax(ax)
        models = list(labels.keys())
        ys = np.arange(len(models))[::-1]
        for yi, m in zip(ys, models):
            s = BUNDLE["summary"][ds][m]
            c = s["cindex"]["mean"]; b = s.get("boot_cindex_seed0", {})
            lo = b.get("lo", c); hi = b.get("hi", c)
            col = ACCENT if m == "yat" else (BLUE if m == "mlp" else GREY)
            ax.plot([lo, hi], [yi, yi], color=col, lw=2.2 if m == "yat" else 1.4, zorder=2)
            ax.scatter([c], [yi], color=col, s=42 if m == "yat" else 26, zorder=3)
        # classical envelope band
        cls = [BUNDLE["summary"][ds][m] for m in ("coxph", "coxnet", "rsf")]
        clo = min(s.get("boot_cindex_seed0", {}).get("lo", s["cindex"]["mean"]) for s in cls)
        chi = max(s.get("boot_cindex_seed0", {}).get("hi", s["cindex"]["mean"]) for s in cls)
        ax.axvspan(clo, chi, color=ACCENT, alpha=0.06, zorder=0)
        ax.set_yticks(ys); ax.set_yticklabels([labels[m] for m in models], fontsize=8.5, color=INK)
        ax.set_title(ds.upper(), color=INK, fontsize=11, weight="bold")
        ax.set_xlabel("C-index", color=MUTED, fontsize=9)
    fig.suptitle("Held-out concordance with bootstrap CIs: does the white-box model land in the pack?",
                 color=INK, fontsize=12.5, weight="bold", y=1.03)
    fig.tight_layout()
    save_png(fig, OUT / "trial-forest.png")


def png_ablation():
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), dpi=100)
    fig.patch.set_facecolor(BG)
    Ks = [6, 12, 24, 48]
    for ax, ds in zip(axes, ["metabric", "gbsg"]):
        style_ax(ax)
        rows = BUNDLE["ablation"][ds]
        for init, col in [("kmeans", ACCENT), ("random", BLUE)]:
            pts = sorted([r for r in rows if r["init"] == init], key=lambda r: r["K"])
            m = [r["cindex_mean"] for r in pts]; sdv = [r["cindex_std"] for r in pts]
            ax.plot(Ks, m, "-o", color=col, lw=2.2, label=init + " seed")
            ax.fill_between(Ks, np.array(m) - sdv, np.array(m) + sdv, color=col, alpha=0.13)
        ax.set_xticks(Ks); ax.set_title(ds.upper(), color=INK, fontsize=11, weight="bold")
        ax.set_xlabel("prototype count K", color=MUTED, fontsize=9)
        ax.set_ylabel("C-index", color=MUTED, fontsize=9)
        ax.legend(fontsize=8.5, frameon=False, labelcolor=INK)
    fig.suptitle("Accuracy is flat in K; k-means seeding is the steadier default",
                 color=INK, fontsize=12.5, weight="bold", y=1.02)
    fig.tight_layout()
    save_png(fig, OUT / "trial-ablation.png")


def png_calibration():
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), dpi=100)
    fig.patch.set_facecolor(BG)
    cols = [BLUE, GREEN, ACCENT]; names = ["low-risk third", "mid-risk third", "high-risk third"]
    for ax, m, title in [(axes[0], "yat", "Yat DeepSurv"), (axes[1], "mlp", "standard DeepSurv")]:
        style_ax(ax)
        rel = BUNDLE["calibration"]["metabric"][m]["reliability"]
        hs = rel["horizons"]
        for gi, g in enumerate(rel["groups"]):
            ax.plot(hs, g["pred"], color=cols[gi], lw=2.2)
            ax.plot(hs, g["obs"], color=cols[gi], lw=1.6, ls="--")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{title}   gap {rel['mean_abs_error']:.3f}", color=INK, fontsize=10.5, weight="bold")
        ax.set_xlabel("months", color=MUTED, fontsize=9)
        if m == "yat":
            ax.set_ylabel("survival", color=MUTED, fontsize=9)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=cols[i], lw=2.2, label=names[i]) for i in range(3)]
    handles += [Line2D([0], [0], color=MUTED, lw=2.2, label="predicted"),
                Line2D([0], [0], color=MUTED, lw=1.6, ls="--", label="observed")]
    axes[1].legend(handles=handles, fontsize=8, frameon=False, labelcolor=INK, loc="upper right")
    fig.suptitle("Predicted vs Kaplan-Meier observed survival by risk tertile (METABRIC)",
                 color=INK, fontsize=12, weight="bold", y=1.02)
    fig.tight_layout()
    save_png(fig, OUT / "trial-calibration.png")


def png_ood():
    ds_order = BUNDLE["meta"]["datasets"]
    short = {"permutation": "shuffled\ncovariates", "extreme_quantile": "far-tail\nsynthetic",
             "real_subgroup": "held-out\nsubgroup"}
    keys = ["permutation", "extreme_quantile", "real_subgroup"]
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=100)
    fig.patch.set_facecolor(BG); style_ax(ax)
    width = 0.26
    xs = np.arange(len(ds_order))
    for ki, k in enumerate(keys):
        vals = [BUNDLE["ood"][ds].get(k, {}).get("auroc", np.nan) for ds in ds_order]
        los = [BUNDLE["ood"][ds].get(k, {}).get("lo", np.nan) for ds in ds_order]
        his = [BUNDLE["ood"][ds].get(k, {}).get("hi", np.nan) for ds in ds_order]
        cols = [ACCENT if v >= 0.55 else (GREY if v >= 0.45 else BLUE) for v in vals]
        pos = xs + (ki - 1) * width
        ax.bar(pos, vals, width, color=cols, edgecolor="none")
        yerr = np.abs(np.vstack([np.array(vals) - los, his - np.array(vals)]))
        ax.errorbar(pos, vals, yerr=yerr, fmt="none", ecolor=INK, elinewidth=1, capsize=2)
        for p, v in zip(pos, vals):
            ax.text(p, 0.02, short[k], ha="center", va="bottom", fontsize=7.5, color=INK, rotation=0)
    ax.axhline(0.5, color=MUTED, ls="--", lw=1)
    ax.text(len(ds_order) - 0.5, 0.51, "0.5: no stranger than a real patient", ha="right", fontsize=8.5, color=MUTED)
    ax.set_xticks(xs); ax.set_xticklabels([d.upper() for d in ds_order], color=INK, fontsize=9.5)
    ax.set_ylim(0, 1); ax.set_ylabel("kernel-max OOD AUROC", color=MUTED, fontsize=9.5)
    ax.set_title("Does the kernel notice a stranger? Honest, and sometimes it does not (bars below 0.5)",
                 color=INK, fontsize=12, weight="bold")
    fig.tight_layout()
    save_png(fig, OUT / "trial-ood.png")


def png_plausibility():
    p = BUNDLE["plausibility"]["metabric"]
    names = p["feature_names"]
    protos = sorted(p["prototypes"], key=lambda r: r["readout"])
    inr = np.array([[c for c in r["per_covar_in_range"]] for r in protos])  # [K, D]
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=100)
    fig.patch.set_facecolor(BG); style_ax(ax)
    # grid of in/out flags: green in-range, orange out
    K, D = inr.shape
    for u in range(K):
        for j in range(D):
            ax.add_patch(plt.Rectangle((j, u), 1, 1, facecolor=GREEN if inr[u, j] else ACCENT,
                                       edgecolor=BG, lw=1, alpha=0.85 if inr[u, j] else 0.9))
    ax.set_xlim(0, D); ax.set_ylim(0, K)
    ax.set_xticks(np.arange(D) + 0.5); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8, color=INK)
    ax.set_yticks(np.arange(K) + 0.5)
    ax.set_yticklabels([f"#{r['id']} ({'+' if r['readout']>=0 else ''}{r['readout']:.2f})" for r in protos],
                       fontsize=7.5, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GREEN, label="in training range"), Patch(color=ACCENT, label="out of range")],
              fontsize=8.5, frameon=False, labelcolor=INK, loc="upper left", bbox_to_anchor=(0, 1.09), ncol=2)
    ax.set_title(f"Are the prototypes real patients?  {p['n_covariates'] if 'n_covariates' in p else ''}"
                 f"{sum(1 for r in protos if r['n_covars_out_of_range']==0)} of {K} land fully in range (METABRIC)",
                 color=INK, fontsize=11.5, weight="bold", pad=26)
    fig.tight_layout()
    save_png(fig, OUT / "trial-plausibility.png")


if __name__ == "__main__":
    print("training the Yat DeepSurv locally (WHAS500) to capture real motion...")
    cap = train_capture()
    print("rendering GIFs (real training process)...")
    gif_settle(cap); gif_stratify(cap)
    print("rendering static PNGs from the trial bundle...")
    png_forest(); png_ablation(); png_calibration(); png_ood(); png_plausibility()
    print(f"done -> {OUT}")
