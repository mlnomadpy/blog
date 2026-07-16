"""GIFs + PNGs for the depth-on-demand companion.

All from the real v2 bundle: the controller GIF replays step doubling on the
exported seed-0 weights (same numpy math as the run); the tolerance-sweep GIF
re-renders the same inputs at each tolerance and morphs the work histogram;
the effort-field GIF shows the plane being swept row by row by the real
renderer. PNGs: the measured tol^(-1/3) power law and the fidelity scoreboard.
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
BUNDLE = os.path.join(HERE, "results", "kgl_blog-adaptive-v2", "exp", "results",
                      "adaptive_depth.json")
D = json.load(open(BUNDLE))

INK, BLUE, ORANGE, GREEN = "#222", "#4a7fb3", "#c2553a", "#2a8f6e"
plt.rcParams.update({"figure.facecolor": "#faf8f5", "axes.facecolor": "#faf8f5",
                     "font.size": 11, "axes.edgecolor": "#bbb"})

DIM, T_TOTAL, L_TRAIN = 4, 6.0, 16


def get_model(ds):
    viz = D[ds][0]["viz"]
    m = {k: (np.array(v[0], np.float64), np.array(v[1], np.float64))
         for k, v in viz["model"].items()}
    return m, np.array(viz["X"]), np.array(viz["y"])


def affine(x, wb):
    return x @ wb[0] + wb[1]


def gradV(m, q):
    h = np.tanh(affine(q, m["b1"]))
    return ((1 - h * h) * m["b2"][0].sum(axis=1)) @ m["b1"][0].T


def leap(m, h, q, p):
    p = p - 0.5 * h * gradV(m, q)
    q = q + h * p
    p = p - 0.5 * h * gradV(m, q)
    return q, p


def encode(m, x):
    z = affine(x[None], m["enc"])[0]
    return z[:DIM].copy(), z[DIM:].copy()


def decode(m, q, p):
    return affine(np.concatenate([q, p])[None], m["dec"])[0]


def controller_events(m, x, tol):
    q, p = encode(m, x)
    t, h, work = 0.0, T_TOTAL / L_TRAIN, 0
    evs = [dict(q=q.copy(), h=h, t=t, err=0.0, rej=False, work=0)]
    h_min = T_TOTAL / 4096
    guard = 0
    while t < T_TOTAL - 1e-9 and guard < 2000:
        guard += 1
        h = min(h, T_TOTAL - t)
        q1, p1 = leap(m, h, q[None].copy().ravel(), p.copy())
        qh, ph = leap(m, h / 2, q.copy(), p.copy())
        q2, p2 = leap(m, h / 2, qh, ph)
        work += 3
        err = max(np.abs(q1 - q2).max(), np.abs(p1 - p2).max())
        if err > tol and h > h_min:
            evs.append(dict(q=q.copy(), h=h, t=t, err=err, rej=True, work=work))
            h /= 2
            continue
        q, p = q2, p2
        t += h
        evs.append(dict(q=q.copy(), h=h, t=t, err=err, rej=False, work=work))
        if err < tol / 4:
            h = min(h * 2, T_TOTAL / 4)
    return evs


def render_one(m, x, tol):
    q, p = encode(m, x)
    t, h, work = 0.0, T_TOTAL / L_TRAIN, 0
    h_min = T_TOTAL / 4096
    guard = 0
    while t < T_TOTAL - 1e-9 and guard < 2000:
        guard += 1
        h = min(h, T_TOTAL - t)
        q1, p1 = leap(m, h, q.copy(), p.copy())
        qh, ph = leap(m, h / 2, q.copy(), p.copy())
        q2, p2 = leap(m, h / 2, qh, ph)
        work += 3
        err = max(np.abs(q1 - q2).max(), np.abs(p1 - p2).max())
        if err > tol and h > h_min:
            h /= 2
            continue
        q, p = q2, p2
        t += h
        if err < tol / 4:
            h = min(h * 2, T_TOTAL / 4)
    return decode(m, q, p), work


def gif_controller():
    m, X, y = get_model("spirals")
    evs = controller_events(m, X[3], 0.03)
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(8.5, 5.6),
                                  gridspec_kw={"height_ratios": [3.2, 1]})
    Q = np.array([e["q"][:2] for e in evs])
    R = np.abs(Q).max() * 1.15

    def draw(i):
        ax.clear(); axh.clear()
        sub = evs[:i + 1]
        acc = np.array([e["q"][:2] for e in sub if not e["rej"]])
        ax.plot(acc[:, 0], acc[:, 1], "-", color=BLUE, lw=1, alpha=0.6)
        ax.plot(acc[:, 0], acc[:, 1], ".", color=BLUE, ms=6)
        cur = sub[-1]
        if cur["rej"]:
            ax.plot([cur["q"][0]], [cur["q"][1]], "x", color=ORANGE, ms=11, mew=2.5)
        ax.set_xlim(-R, R); ax.set_ylim(-R, R)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("the controller stepping one real input through the trained flow", fontsize=12)
        # h bar
        hfrac = min(1.0, cur["h"] / (T_TOTAL / 4))
        axh.barh([0], [hfrac], color=ORANGE if cur["rej"] else BLUE, height=0.55)
        axh.axvline(min(1.0, (T_TOTAL / L_TRAIN) / (T_TOTAL / 4)), color=INK, lw=1.2, ls="--")
        axh.set_xlim(0, 1); axh.set_yticks([])
        axh.set_xticks([])
        state = "REJECTED: halve h and retry" if cur["rej"] else f"accepted, t = {cur['t']:.2f} / {T_TOTAL}"
        axh.set_title(f"step size h = {cur['h']:.4f}   {state}   ({cur['work']} field evaluations)",
                      fontsize=10, color=ORANGE if cur["rej"] else INK)

    anim = FuncAnimation(fig, draw, frames=len(evs), interval=170)
    out = os.path.join(PUB, "adepth-controller.gif")
    anim.save(out, writer=PillowWriter(fps=6))
    plt.close(fig)
    print(f"  adepth-controller.gif: {os.path.getsize(out)//1024} KB")


def png_tol_sweep():
    # five distinct dial settings are five facts, not a process: small multiples,
    # not a slideshow GIF
    m, X, y = get_model("spirals")
    TOLS = [0.3, 0.1, 0.03, 0.01, 0.003]
    N = 100
    fig, axes = plt.subplots(1, 5, figsize=(12.5, 2.9), sharey=True)
    for ax, tol in zip(axes, TOLS):
        works, correct = [], 0
        for i in range(N):
            lg, w = render_one(m, X[i], tol)
            works.append(w)
            correct += int((lg[1] > lg[0]) == bool(y[i]))
        w = np.array(works)
        ax.hist(w, bins=np.arange(0, 300, 14), color=BLUE, alpha=0.85)
        ax.set_title(f"tol {tol}\nwork {w.mean():.0f}  acc {correct}%", fontsize=10)
        ax.set_xlim(0, 300)
    axes[0].set_ylabel("inputs")
    axes[2].set_xlabel("work to render one input (field evaluations)")
    fig.suptitle("the dial moves the bill, not the verdict", fontsize=12, y=1.04)
    out = os.path.join(PUB, "adepth-tol-sweep.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  adepth-tol-sweep.png: {os.path.getsize(out)//1024} KB")


def gif_effort_field():
    m, X, y = get_model("spirals")
    G, R = 30, 2.2
    W = np.zeros((G, G))
    for r in range(G):
        for c in range(G):
            x = np.array([-R + 2 * R * c / (G - 1), R - 2 * R * r / (G - 1)])
            _, w = render_one(m, x, 0.03)
            W[r, c] = w
    fig, ax = plt.subplots(figsize=(6.4, 6.0))

    def draw(rr):
        ax.clear()
        shown = np.full_like(W, np.nan)
        shown[:rr + 1] = W[:rr + 1]
        ax.imshow(shown, extent=[-R, R, -R, R], cmap="inferno", origin="upper",
                  vmin=W.min(), vmax=W.max())
        pts = X[:140]
        ax.plot(pts[y[:140] == 0][:, 0], pts[y[:140] == 0][:, 1], ".", color="#7ee0a3", ms=3)
        ax.plot(pts[y[:140] == 1][:, 0], pts[y[:140] == 1][:, 1], ".", color="#e79ae0", ms=3)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"work to render each point of the plane, tol 0.03   row {min(rr+1, G)}/{G}",
                     fontsize=11)

    anim = FuncAnimation(fig, draw, frames=G + 6, interval=130)
    out = os.path.join(PUB, "adepth-effort-field.gif")
    anim.save(out, writer=PillowWriter(fps=8))
    plt.close(fig)
    print(f"  adepth-effort-field.gif: {os.path.getsize(out)//1024} KB")


def png_power_law():
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for ds, col in [("moons", BLUE), ("rings", GREEN), ("spirals", ORANGE)]:
        tols, acc = [], []
        for ti in range(len(D[ds][0]["sweep"])):
            rows = [s["sweep"][ti] for s in D[ds]]
            tols.append(rows[0]["tol"])
            acc.append(np.mean([r["accepted_mean"] for r in rows]))
        ax.loglog(tols, acc, "o-", color=col, lw=1.8, label=ds)
    t = np.array([0.3, 0.003])
    mid = acc[2]
    ax.loglog(t, mid * (t / tols[2]) ** (-1 / 3), "--", color=INK, lw=1.2,
              label="slope -1/3 (second order)")
    ax.set_xlabel("tolerance"); ax.set_ylabel("accepted steps (mean)")
    ax.legend(fontsize=9)
    ax.set_title("the compute bill follows the integrator's order", fontsize=12)
    out = os.path.join(PUB, "adepth-power-law.png")
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"  adepth-power-law.png: {os.path.getsize(out)//1024} KB")


if __name__ == "__main__":
    print("rendering adaptive-depth figures from the real bundle:")
    gif_controller()
    png_tol_sweep()
    gif_effort_field()
    png_power_law()
