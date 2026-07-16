"""GIFs + PNGs for the compatibility-kernel companion.

From the real bundles: the curves GIF draws the six validation curves through
training (real per-checkpoint metric); the depth-sweep GIF walks the trained
attention maps through layers (a real transforming sweep through the
network); the checkpoints grid is a static PNG (four snapshots are four
facts); the score-orbit GIF sweeps the query around the plane and recomputes
both score landscapes from their formulas at every frame.
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = os.path.dirname(os.path.abspath(__file__))
PUB = os.path.join(HERE, "..", "public")
V1 = os.path.join(HERE, "results", "kgl_blog-yatattn-v1", "exp", "results",
                  "yat_attention.json")
TELEM_DIR = os.path.join(HERE, "results", "kgl_blog-yatattn-telem")

INK, BLUE, ORANGE = "#222", "#4a7fb3", "#c2553a"
plt.rcParams.update({"figure.facecolor": "#faf8f5", "axes.facecolor": "#faf8f5",
                     "font.size": 11, "axes.edgecolor": "#bbb"})

D = json.load(open(V1))


def gif_curves():
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    curves = [(r["variant"], r["seed"], r["curve"]) for r in D["runs"]]
    steps = sorted({p["step"] for _, _, c in curves for p in c})
    frames = len(steps) + 6

    def draw(f):
        ax.clear()
        smax = steps[min(f, len(steps) - 1)]
        for variant, seed, c in curves:
            pts = [(p["step"], p["val"]) for p in c if p["step"] <= smax]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=ORANGE if variant == "softmax" else BLUE,
                    lw=1.4, alpha=0.85)
        ax.set_xlim(0, 12000); ax.set_ylim(1.4, 2.3)
        ax.set_xlabel("training step"); ax.set_ylabel("validation loss (nats/char)")
        sm, yt = D["summary"]["softmax"], D["summary"]["yat"]
        ax.set_title(f"six runs, two attentions   softmax {sm['best_val_mean']:.4f}  "
                     f"yat {yt['best_val_mean']:.4f}", fontsize=12)
        ax.plot([], [], color=ORANGE, label="softmax")
        ax.plot([], [], color=BLUE, label="yat kernel (no softmax)")
        ax.legend(loc="upper right", fontsize=9)

    anim = FuncAnimation(fig, draw, frames=frames, interval=120)
    out = os.path.join(PUB, "yatattn-curves.gif")
    anim.save(out, writer=PillowWriter(fps=8))
    plt.close(fig)
    print(f"  yatattn-curves.gif: {os.path.getsize(out)//1024} KB")


def load_telem():
    out = {}
    for p in glob.glob(os.path.join(TELEM_DIR, "**", "attn_*_s0.npz"), recursive=True):
        variant = os.path.basename(p).split("_")[1]
        out[variant] = np.load(p)
    return out


def gif_depth_sweep(tel):
    keys = sorted([k for k in tel["softmax"].files if k.startswith("step")],
                  key=lambda k: int(k[4:]))
    last = keys[-1]
    A = {v: tel[v][last].astype(np.float32) for v in ("softmax", "yat")}  # (L,h,T,T)
    L = A["softmax"].shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.8))
    frames = L * 6 + 6

    def draw(f):
        li = min(f // 6, L - 1)
        for ax, v in zip(axes, ("softmax", "yat")):
            ax.clear()
            m = A[v][li].mean(axis=0)          # heads averaged, (T,T)
            ax.imshow(np.sqrt(m), cmap="magma", vmin=0, vmax=np.sqrt(m).max())
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{v}   layer {li + 1}/{L}", fontsize=11)
        fig.suptitle("the trained routing, swept through depth (heads averaged, sqrt scale)",
                     fontsize=12)

    anim = FuncAnimation(fig, draw, frames=frames, interval=140)
    out = os.path.join(PUB, "yatattn-depth-sweep.gif")
    anim.save(out, writer=PillowWriter(fps=7))
    plt.close(fig)
    print(f"  yatattn-depth-sweep.gif: {os.path.getsize(out)//1024} KB")


def png_checkpoints(tel):
    keys = sorted([k for k in tel["softmax"].files if k.startswith("step")],
                  key=lambda k: int(k[4:]))
    li, hi = 3, 0
    fig, axes = plt.subplots(2, len(keys), figsize=(11.5, 5.2))
    for r, v in enumerate(("softmax", "yat")):
        for c, k in enumerate(keys):
            ax = axes[r][c]
            m = tel[v][k].astype(np.float32)[li, hi]
            ax.imshow(np.sqrt(m), cmap="magma")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"step {k[4:]}", fontsize=10)
            if c == 0:
                ax.set_ylabel(v, fontsize=11)
    fig.suptitle(f"layer {li + 1}, head {hi + 1}: four checkpoints, both models (sqrt scale)",
                 fontsize=12)
    out = os.path.join(PUB, "yatattn-checkpoints.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  yatattn-checkpoints.png: {os.path.getsize(out)//1024} KB")


def gif_score_orbit():
    G, R = 52, 2.4
    xs = np.linspace(-R, R, G)
    KX, KY = np.meshgrid(xs, -xs)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7))
    frames = 36

    def draw(f):
        ang = 2 * np.pi * f / frames
        q = np.array([1.3 * np.cos(ang), 1.3 * np.sin(ang)])
        dots = q[0] * KX + q[1] * KY
        bil = dots / np.sqrt(2)
        d2 = (KX - q[0]) ** 2 + (KY - q[1]) ** 2
        yat = dots ** 2 / (d2 + 1.0)
        for ax, (name, V) in zip(axes, [("bilinear q . k", bil),
                                        ("yat kernel", yat)]):
            ax.clear()
            ax.imshow(V, cmap="viridis", extent=[-R, R, -R, R])
            ax.plot([q[0]], [q[1]], "o", color="white", ms=9, mec=INK)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(name, fontsize=11)
        fig.suptitle("the query orbits; one score rotates a ramp, the other carries its peak along",
                     fontsize=12)

    anim = FuncAnimation(fig, draw, frames=frames, interval=110)
    out = os.path.join(PUB, "yatattn-score-orbit.gif")
    anim.save(out, writer=PillowWriter(fps=9))
    plt.close(fig)
    print(f"  yatattn-score-orbit.gif: {os.path.getsize(out)//1024} KB")


if __name__ == "__main__":
    print("rendering yat-attention figures from the real bundles:")
    gif_curves()
    gif_score_orbit()
    tel = load_telem()
    if tel:
        gif_depth_sweep(tel)
        png_checkpoints(tel)
    else:
        print("  (telemetry bundle not present yet; depth sweep + checkpoints skipped)")


def check_entropy():
    """The map reading quoted in the explainer: normalized attention entropy
    per query row (rows with at least 16 visible keys), trained checkpoints."""
    tel = load_telem()
    for v in ("softmax", "yat"):
        keys = sorted([k for k in tel[v].files if k.startswith("step")],
                      key=lambda k: int(k[4:]))
        A = tel[v][keys[-1]].astype(np.float32)
        L, H, T, _ = A.shape
        ents = []
        for l in range(L):
            for h in range(H):
                W = A[l, h]
                for qi in range(16, T):
                    row = W[qi, :qi + 1]
                    row = row / (row.sum() + 1e-9)
                    ents.append(-(row * np.log(row + 1e-12)).sum() / np.log(qi + 1))
        ents = np.array(ents)
        print(f"  {v}: entropy mean {ents.mean():.3f}, "
              f"sharp(<0.3) {(ents < 0.3).mean() * 100:.1f}%, "
              f"diffuse(>0.8) {(ents > 0.8).mean() * 100:.1f}%")
