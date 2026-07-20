"""GIFs + PNGs for the geometry-of-attention companion.

Every moving thing is recomputed from the real math at every frame:
  attngeom-territory.gif  one body travels; both owner maps recomputed from
                          the two score laws per frame (wedges re-aim,
                          pockets re-carve)
  attngeom-hull.gif       the interior body crosses the hull; its territory
                          share under the dot-product law sits at exactly 0
                          until the crossing (the convexity theorem, watched)
  attngeom-ray.gif        a query rides a ray; both weight rows recomputed
                          per frame (softmax sharpens on a fixed winner, the
                          kernel hands off and its mass breathes)
  attngeom-scale.gif      the REAL trained q/k of one head per model, every
                          row's winner recomputed at every scale t (softmax
                          changes zero, the kernel re-elects)
  attngeom-farfield.gif   the kernel owner map as the view zooms out; the
                          antipodal-agreement counter is computed per frame
                          (the unsigned sky arrives on its measured 1/t
                          schedule; the softmax map cannot depend on zoom)
  attngeom-census.png     ownership at the four training checkpoints, both
                          models (four snapshots are four facts: a grid)

Data: bundles kgl_blog-attngeom-qk / kgl_blog-attngeom-goat; toy scalars
match the explainer's panels (b = log 2, eps = 0.25, softening below the
body spacing).
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = os.path.dirname(os.path.abspath(__file__))
PUB = os.path.join(HERE, "..", "public")
QK = os.path.join(HERE, "results", "kgl_blog-attngeom-qk")

PAL = ["#b3661b", "#4a7fb3", "#3a8f5e", "#9a4f9c", "#c2553a", "#5a5f66"]
INK, BLUE, ORANGE = "#222", "#4a7fb3", "#c2553a"
plt.rcParams.update({"figure.facecolor": "#faf8f5", "axes.facecolor": "#faf8f5",
                     "font.size": 11, "axes.edgecolor": "#bbb"})

B, EPS = float(np.log(2)), 0.25          # the explainer panels' toy scalars
KEYS0 = np.array([[0.9, 0.85], [-1.05, 0.6], [0.15, -1.1],
                  [1.35, -0.5], [-0.7, -0.75], [0.35, 0.3]])


def owner_maps(keys, R, grid=110):
    """Top-1 owner index of every query pixel under both laws (real formulas)."""
    xs = np.linspace(-R, R, grid)
    qx, qy = np.meshgrid(xs, xs[::-1])
    q = np.stack([qx, qy], -1).reshape(-1, 2)              # (G*G, 2)
    dots = q @ keys.T                                      # (G*G, K)
    d2 = ((q[:, None, :] - keys[None, :, :]) ** 2).sum(-1)
    kappa = (dots + B) ** 2 / (d2 + EPS)
    return (dots.argmax(-1).reshape(grid, grid),
            kappa.argmax(-1).reshape(grid, grid))


def rgb_map(own, alpha_own=None):
    pal = np.array([[int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)]
                    for c in PAL]) / 255.0
    return pal[own]


def draw_bodies(ax, keys, R, highlight=None):
    for i, k in enumerate(keys):
        if abs(k[0]) > R or abs(k[1]) > R:
            continue
        ec = "#fff"
        ax.scatter([k[0]], [k[1]], s=90 if i == highlight else 60,
                   c=PAL[i % len(PAL)], edgecolors=ec, linewidths=1.6, zorder=5)
    ax.plot([0], [0], marker="+", color=INK, ms=10, mew=1.6, zorder=6)
    ax.set_xlim(-R, R); ax.set_ylim(-R, R)
    ax.set_xticks([]); ax.set_yticks([])


def gif_territory():
    R = 2.0
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.6))
    nF = 56

    def draw(f):
        keys = KEYS0.copy()
        th = 2 * np.pi * f / nF
        keys[3] = [0.6 + 0.9 * np.cos(th), -0.2 + 0.9 * np.sin(th)]  # body 4 orbits
        soft, ker = owner_maps(keys, R)
        for ax, own, name in [(axes[0], soft, "softmax of q·k"),
                              (axes[1], ker, "yat kernel")]:
            ax.clear()
            ax.imshow(rgb_map(own), extent=[-R, R, -R, R], interpolation="bilinear")
            draw_bodies(ax, keys, R, highlight=3)
            ax.set_title(name, fontsize=11)
        axes[0].set_xlabel("every border passes through the origin", fontsize=9)
        axes[1].set_xlabel("pockets re-carve around the moving body", fontsize=9)
        fig.suptitle("one body moves; two laws redraw their maps", fontsize=12)

    anim = FuncAnimation(fig, draw, frames=nF, interval=130)
    out = os.path.join(PUB, "attngeom-territory.gif")
    anim.save(out, writer=PillowWriter(fps=8))
    plt.close(fig)
    print(out, f"{os.path.getsize(out)/1e6:.2f} MB")


def gif_hull():
    R = 2.6
    ring = np.array([[1.15, 0.4], [0.25, 1.2], [-1.0, 0.75],
                     [-0.85, -0.85], [0.65, -1.1]])
    # travel path: hull center -> well outside, and back (pingpong)
    p0, p1 = np.array([0.1, 0.05]), np.array([2.2, 1.0])
    nF = 64
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.6))

    from matplotlib.path import Path as MplPath
    hull_order = ring[np.argsort(np.arctan2(ring[:, 1], ring[:, 0]))]
    hull_path = MplPath(hull_order)

    def draw(f):
        t = f / (nF - 1)
        t = 2 * t if t <= 0.5 else 2 * (1 - t)              # pingpong
        inner = p0 + (p1 - p0) * t
        keys = np.vstack([ring, inner])
        soft, ker = owner_maps(keys, R)
        inside = hull_path.contains_point(inner)
        for ax, own, name in [(axes[0], soft, "softmax of q·k"),
                              (axes[1], ker, "yat kernel")]:
            ax.clear()
            mine = (own == 5)
            img = np.ones(own.shape + (3,)) * 0.93
            img[mine] = np.array([int(PAL[0][1:3], 16), int(PAL[0][3:5], 16),
                                  int(PAL[0][5:7], 16)]) / 255.0
            ax.imshow(img, extent=[-R, R, -R, R], interpolation="bilinear")
            hp = np.vstack([hull_order, hull_order[:1]])
            ax.plot(hp[:, 0], hp[:, 1], "--", color="#999", lw=1.2)
            for k in ring:
                ax.scatter([k[0]], [k[1]], s=55, c="#888", edgecolors="#fff",
                           linewidths=1.4, zorder=5)
            ax.scatter([inner[0]], [inner[1]], s=95, c=PAL[0], edgecolors="#fff",
                       linewidths=1.8, zorder=6)
            share = 100 * mine.mean()
            ax.set_title(f"{name}: it owns {share:.1f}% of view", fontsize=11)
            ax.set_xlim(-R, R); ax.set_ylim(-R, R)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("inside the hull the dot-product law gives it nothing"
                     if inside else "outside the hull it wins an infinite wedge",
                     fontsize=12)

    anim = FuncAnimation(fig, draw, frames=nF, interval=130)
    out = os.path.join(PUB, "attngeom-hull.gif")
    anim.save(out, writer=PillowWriter(fps=8))
    plt.close(fig)
    print(out, f"{os.path.getsize(out)/1e6:.2f} MB")


def gif_ray():
    keys = np.array([[0.55, 0.7], [1.5, 1.4], [2.6, 1.9],
                     [-1.2, 0.9], [0.8, -1.4]])
    R = 3.4
    u = np.array([np.cos(np.arctan2(0.75, 1.0)), np.sin(np.arctan2(0.75, 1.0))])
    nF = 72
    fig = plt.figure(figsize=(8.8, 4.4))
    axP = fig.add_axes([0.05, 0.1, 0.42, 0.78])
    axS = fig.add_axes([0.55, 0.56, 0.4, 0.3])
    axK = fig.add_axes([0.55, 0.12, 0.4, 0.3])

    def rows_at(t):
        q = t * u
        dots = keys @ q
        logits = dots / np.sqrt(2)
        w_soft = np.exp(logits - logits.max()); w_soft /= w_soft.sum()
        d2 = ((q - keys) ** 2).sum(-1)
        kappa = (dots + B) ** 2 / (d2 + EPS)
        mass = kappa.sum()
        return w_soft, kappa / mass, mass, q

    def draw(f):
        p = f / (nF - 1)
        p = 2 * p if p <= 0.5 else 2 * (1 - p)
        t = 0.05 + (3.3 - 0.05) * p
        w_soft, w_ker, mass, q = rows_at(t)
        axP.clear()
        axP.plot([0, u[0] * 2 * R], [0, u[1] * 2 * R], "--", color="#aaa", lw=1)
        for i, k in enumerate(keys):
            axP.scatter([k[0]], [k[1]], s=70, c=PAL[i], edgecolors="#fff",
                        linewidths=1.5, zorder=5)
        axP.scatter([q[0]], [q[1]], s=90, c="#fff", edgecolors=INK,
                    linewidths=1.8, zorder=6)
        axP.plot([0], [0], marker="+", color=INK, ms=9, mew=1.5)
        axP.set_xlim(-R, R); axP.set_ylim(-R, R)
        axP.set_xticks([]); axP.set_yticks([])
        axP.set_title(f"query length t = {t:.2f}", fontsize=11)
        from matplotlib.colors import to_rgba
        for ax, w, name in [(axS, w_soft, "softmax row"), (axK, w_ker, "kernel row")]:
            ax.clear()
            win = int(np.argmax(w))
            cols = [to_rgba(PAL[i], 1.0 if i == win else 0.55) for i in range(5)]
            ax.bar(np.arange(5), w, color=cols)
            ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([0, 1])
        axS.set_title(f"softmax: winner body {int(np.argmax(w_soft)) + 1} "
                      "(fixed on this ray)", fontsize=10)
        axK.set_title(f"kernel: winner body {int(np.argmax(w_ker)) + 1}, "
                      f"mass {mass:.1f}", fontsize=10)

    anim = FuncAnimation(fig, draw, frames=nF, interval=120)
    out = os.path.join(PUB, "attngeom-ray.gif")
    anim.save(out, writer=PillowWriter(fps=8))
    plt.close(fig)
    print(out, f"{os.path.getsize(out)/1e6:.2f} MB")


def load_head(model, li, hi):
    z = np.load(os.path.join(QK, f"geom_{model}_s0.npz"))
    rj = json.load(open(os.path.join(QK, "yat_attention.json")))
    run = next(r for r in rj["runs"] if r["variant"] == model)
    q = z["q"][li, hi].astype(np.float64)
    k = z["k"][li, hi].astype(np.float64)
    b = run.get("b_learned", [[0.0] * 4] * 6)[li][hi]
    eps = run.get("eps_learned", [[1.0] * 4] * 6)[li][hi]
    return q, k, b, eps


def gif_scale():
    # the sharpest head of each model (the export script's pick)
    qS, kS, _, _ = load_head("softmax", 0, 0)
    qY, kY, bY, eY = load_head("yat_b", 0, 0)
    T = qS.shape[0]
    ts = np.exp(np.linspace(np.log(0.25), np.log(4.0), 41))
    i1 = int(np.argmin(np.abs(ts - 1.0)))

    def winners(q, k, t, law, b=0.0, eps=1.0):
        dots = q @ k.T
        win = np.full(T, -1)
        for i in range(1, T):
            ub = i + 1
            if law == "softmax":
                s = t * dots[i, :ub]
            else:
                qq = (q[i] ** 2).sum(); kk = (k[:ub] ** 2).sum(-1)
                d2 = np.maximum(t * t * qq + kk - 2 * t * dots[i, :ub], 0.0)
                s = (t * dots[i, :ub] + b) ** 2 / (d2 + eps)
            win[i] = int(np.argmax(s))
        return win

    WS = np.stack([winners(qS, kS, t, "softmax") for t in ts])
    WY = np.stack([winners(qY, kY, t, "yat", bY, eY) for t in ts])
    chS = (WS != WS[i1]).mean(axis=1) * T / (T - 1)
    chY = (WY != WY[i1]).mean(axis=1) * T / (T - 1)

    order = list(range(len(ts))) + list(range(len(ts) - 2, -1, -1))
    fig = plt.figure(figsize=(8.8, 4.8))
    axA = fig.add_axes([0.07, 0.76, 0.88, 0.12])
    axB = fig.add_axes([0.07, 0.52, 0.88, 0.12])
    axC = fig.add_axes([0.07, 0.1, 0.88, 0.32])

    def draw(f):
        ti = order[f]
        t = ts[ti]
        for ax, Wt, W1, color, name, ch in [
                (axA, WS[ti], WS[i1], ORANGE, "softmax", chS),
                (axB, WY[ti], WY[i1], BLUE, "yat kernel", chY)]:
            ax.clear()
            moved = Wt != W1
            ax.bar(np.arange(T), np.ones(T), width=1.0,
                   color=[color if m else "#ddd" for m in moved])
            ax.set_xlim(0, T); ax.set_ylim(0, 1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{name}: {int(moved.sum())} of {T - 1} queries changed "
                         f"hands at t = {t:.2f}", fontsize=10, loc="left")
        axC.clear()
        axC.plot(ts, 100 * chS, color=ORANGE, lw=2, label="softmax (the theorem: 0)")
        axC.plot(ts, 100 * chY, color=BLUE, lw=2, label="yat kernel")
        axC.axvline(t, color="#999", ls=":", lw=1.2)
        axC.set_xscale("log"); axC.set_xticks([0.25, 0.5, 1, 2, 4])
        axC.set_xticklabels(["0.25x", "0.5x", "1x", "2x", "4x"])
        axC.set_ylabel("% changed winners"); axC.set_ylim(-2, 40)
        axC.legend(fontsize=9, loc="upper left")
        axC.set_title("real trained q/k, layer 1 head 1 of each model; winners "
                      "recomputed from the raw law at every t", fontsize=10)

    anim = FuncAnimation(fig, draw, frames=len(order), interval=110)
    out = os.path.join(PUB, "attngeom-scale.gif")
    anim.save(out, writer=PillowWriter(fps=9))
    plt.close(fig)
    print(out, f"{os.path.getsize(out)/1e6:.2f} MB")


def gif_farfield():
    nF = 60
    Rs = np.exp(np.linspace(np.log(2.0), np.log(120.0), nF))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.6))

    def antipodal_agreement(R):
        th = np.linspace(0, np.pi, 90, endpoint=False)
        u = np.stack([np.cos(th), np.sin(th)], -1) * R
        agree = 0
        for q in u:
            d1 = KEYS0 @ q; d2a = ((q - KEYS0) ** 2).sum(-1)
            k1 = (d1 + B) ** 2 / (d2a + EPS)
            d1m = KEYS0 @ (-q); d2b = ((-q - KEYS0) ** 2).sum(-1)
            k2 = (d1m + B) ** 2 / (d2b + EPS)
            agree += int(np.argmax(k1) == np.argmax(k2))
        return 100 * agree / len(u)

    def draw(f):
        R = Rs[f]
        soft, ker = owner_maps(KEYS0, R, grid=100)
        agree = antipodal_agreement(R)
        for ax, own, name in [(axes[0], soft, "softmax of q·k (cannot depend on zoom)"),
                              (axes[1], ker, "yat kernel")]:
            ax.clear()
            ax.imshow(rgb_map(own), extent=[-R, R, -R, R], interpolation="bilinear")
            draw_bodies(ax, KEYS0, R)
            ax.set_title(name, fontsize=10)
        axes[1].set_xlabel(f"antipodal owner agreement: {agree:.0f}%", fontsize=10)
        axes[0].set_xlabel("same wedges at every radius", fontsize=9)
        fig.suptitle(f"zoom out to radius {R:.0f}: the kernel's sky loses its sign",
                     fontsize=12)

    anim = FuncAnimation(fig, draw, frames=nF, interval=140)
    out = os.path.join(PUB, "attngeom-farfield.gif")
    anim.save(out, writer=PillowWriter(fps=7))
    plt.close(fig)
    print(out, f"{os.path.getsize(out)/1e6:.2f} MB")


def png_census():
    steps = [0, 1200, 6000, 11999]
    li, hi = 3, 0
    fig, axes = plt.subplots(2, 4, figsize=(11.5, 5.6))
    for row, model, name in [(0, "softmax", "softmax"), (1, "yat_b", "yat kernel")]:
        z = np.load(os.path.join(QK, f"attn_{model}_s0.npz"))
        for col, s in enumerate(steps):
            ax = axes[row, col]
            A = z[f"step{s}"][li, hi].astype(np.float32)
            win = A.argmax(-1); win[0] = -1
            T = len(win)
            hold = np.bincount(win[win >= 0], minlength=T)
            big = hold[np.maximum(win, 0)] >= 8
            ii = np.arange(T)[win >= 0]
            ax.scatter(ii, win[win >= 0], s=4,
                       c=[ORANGE if b else "#999" for b in big[win >= 0]])
            ax.plot([0, T], [0, T], "--", color="#bbb", lw=0.8)
            ax.set_xlim(0, T); ax.set_ylim(0, T)
            ax.set_xticks([]); ax.set_yticks([])
            occ = len(set(win[win >= 0].tolist()))
            if row == 0:
                ax.set_title(f"step {s if s < 11999 else 12000}", fontsize=10)
            ax.set_xlabel(f"{name}: {occ} owners", fontsize=9)
    fig.suptitle("layer 4 head 1: every query (x) and the key that owns it (y), "
                 "through training; orange = keys holding 8+", fontsize=12)
    fig.tight_layout()
    out = os.path.join(PUB, "attngeom-census.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(out, f"{os.path.getsize(out)/1e6:.2f} MB")


if __name__ == "__main__":
    gif_territory()
    gif_hull()
    gif_ray()
    gif_scale()
    gif_farfield()
    png_census()
