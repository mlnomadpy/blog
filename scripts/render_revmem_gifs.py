"""GIFs + PNGs for the backprop-without-the-memory companion.

Real processes only: the rewind-shear GIF runs genuine float32 momentum
dynamics forward and backward (numpy float32, the same arithmetic as the
run); the ledger GIF animates the two bookkeeping disciplines with meters
labeled by the run's measured megabytes; the error-growth GIF plots
reconstruction error as the rewind proceeds, computed live. PNGs carry the
measured memory wall and the gradient-cliff scoreboard from the bundle.
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
BUNDLE = os.path.join(HERE, "results", "kgl_blog-revmem-v1", "exp", "results",
                      "reversible_memory.json")
D = json.load(open(BUNDLE))

INK, BLUE, ORANGE, PURPLE = "#222", "#4a7fb3", "#c2553a", "#8a6fb3"
plt.rcParams.update({"figure.facecolor": "#faf8f5", "axes.facecolor": "#faf8f5",
                     "font.size": 11, "axes.edgecolor": "#bbb"})


# ── the float32 momentum demo (same math as the in-page panels) ──────────────

def make_field(seed=7, hidden=16, scale=1.4):
    rng = np.random.default_rng(seed)
    W1 = (rng.random((hidden, 2)) * 2 - 1).astype(np.float32) * scale
    b1 = ((rng.random(hidden) * 2 - 1) * 0.4).astype(np.float32)
    W2 = ((rng.random((2, hidden)) * 2 - 1) / np.sqrt(hidden)).astype(np.float32) * scale
    return W1, b1, W2


def field(F, x):
    W1, b1, W2 = F
    h = np.tanh(W1 @ x + b1, dtype=np.float32)
    return (W2 @ h * np.float32(0.15)).astype(np.float32)


def forward(F, x0, v0, mu, L):
    mu = np.float32(mu)
    x, v = x0.astype(np.float32), v0.astype(np.float32)
    S = [(x.copy(), v.copy())]
    for _ in range(L):
        v = (mu * v + field(F, x)).astype(np.float32)
        x = (x + v).astype(np.float32)
        S.append((x.copy(), v.copy()))
    return S


def rewind(F, xL, vL, mu, L):
    mu = np.float32(mu)
    x, v = xL.astype(np.float32), vL.astype(np.float32)
    S = [(x.copy(), v.copy())]
    for _ in range(L):
        x = (x - v).astype(np.float32)
        v = ((v - field(F, x)) / mu).astype(np.float32)
        S.append((x.copy(), v.copy()))
    return S[::-1]


LDEPTH, NP_ = 64, 10
F = make_field()
MUS = [0.9, 0.6, 0.3]
RUNS = {}
for mu in MUS:
    runs = []
    for i in range(NP_):
        a = 2 * np.pi * i / NP_
        x0 = np.array([0.9 * np.cos(a), 0.9 * np.sin(a)], np.float32)
        fw = forward(F, x0, np.zeros(2, np.float32), mu, LDEPTH)
        bw = rewind(F, fw[-1][0], fw[-1][1], mu, LDEPTH)
        runs.append((fw, bw))
    RUNS[mu] = runs


def gif_rewind_shear():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    frames = 2 * LDEPTH + 12

    def draw(t):
        for ax, mu in zip(axes, MUS):
            ax.clear()
            fwN = min(t, LDEPTH)
            bwN = max(0, t - LDEPTH - 6)
            allx = np.array([s[0] for fw, _ in RUNS[mu] for s in fw])
            R = min(max(1.3, np.abs(allx).max() * 1.05), 25)
            for fw, bw in RUNS[mu]:
                P = np.array([s[0] for s in fw[:fwN + 1]])
                ax.plot(P[:, 0], P[:, 1], color=BLUE, lw=0.9, alpha=0.65)
                if bwN:
                    lo = max(0, LDEPTH - bwN)
                    Q = np.array([s[0] for s in bw[lo:]])
                    ax.plot(Q[:, 0], Q[:, 1], ".", color=ORANGE, ms=3.5)
            ax.set_xlim(-R, R); ax.set_ylim(-R, R)
            ax.set_xticks([]); ax.set_yticks([])
            err = max(float(np.linalg.norm(np.array(bw[j][0]) - np.array(fw[j][0])))
                      for fw, bw in RUNS[mu] for j in range(max(0, LDEPTH - bwN), LDEPTH + 1)) if bwN else 0.0
            ax.set_title(f"mu = {mu}" + (f"   err {err:.1e}" if bwN else ""), fontsize=11,
                         color=INK if err < 1e-2 else ORANGE)
        if t <= LDEPTH:
            fig.suptitle(f"forward through {LDEPTH} float32 momentum layers   layer {min(t, LDEPTH)}", fontsize=12)
        else:
            fig.suptitle("rewinding from the endpoint alone: blue history, orange reconstruction", fontsize=12)

    anim = FuncAnimation(fig, draw, frames=frames, interval=110)
    out = os.path.join(PUB, "revmem-rewind-shear.gif")
    anim.save(out, writer=PillowWriter(fps=9))
    plt.close(fig)
    print(f"  revmem-rewind-shear.gif: {os.path.getsize(out)//1024} KB")


def gif_ledger():
    mem = {(r["L"], r["mode"]): r for r in D["memory"]}
    std, rev = mem[(512, "standard")], mem[(512, "reversible")]
    L = 20
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 4.6))
    frames = 2 * L + 10

    def draw(t):
        for ax, (name, col, is_std) in zip(axes, [
                ("standard: store the past", ORANGE, True),
                ("reversible: recompute the past", BLUE, False)]):
            ax.clear()
            fwN = min(t, L); bwN = max(0, t - L - 4)
            for i in range(L):
                filled = False
                if is_std:
                    filled = i < fwN and not (bwN and i >= L - bwN)
                else:
                    head = fwN - 1 if t <= L + 4 else L - 1 - bwN
                    filled = i == head
                ax.add_patch(plt.Rectangle((i, 0), 0.9, 1, facecolor=col if filled else "none",
                                           edgecolor="#999", alpha=0.75 if filled else 1.0))
            stored = (fwN - bwN) if is_std else 1
            frac = max(stored, 0) / L
            ax.add_patch(plt.Rectangle((L + 0.7, 0), 1.1, 1, facecolor="none", edgecolor="#777"))
            ax.add_patch(plt.Rectangle((L + 0.7, 0), 1.1, max(frac, 0.04), facecolor=col, alpha=0.8))
            real = std["temp_mb"] if is_std else rev["temp_mb"]
            ax.text(L + 1.25, -0.32, f"{real:.1f} MB\nat depth 512", ha="center", fontsize=9, color=col)
            ax.set_xlim(-0.3, L + 2.4); ax.set_ylim(-0.75, 1.3)
            ax.axis("off")
            phase = "forward" if t <= L else "backward"
            ax.set_title(f"{name}   ({phase})", loc="left", fontsize=11)
        fig.suptitle("one training step, two bookkeeping disciplines", fontsize=12)

    anim = FuncAnimation(fig, draw, frames=frames, interval=150)
    out = os.path.join(PUB, "revmem-ledger.gif")
    anim.save(out, writer=PillowWriter(fps=7))
    plt.close(fig)
    print(f"  revmem-ledger.gif: {os.path.getsize(out)//1024} KB")


def gif_error_growth():
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    frames = LDEPTH + 8

    def draw(t):
        ax.clear()
        for mu, col in zip(MUS, [BLUE, PURPLE, ORANGE]):
            errs = []
            for j in range(min(t, LDEPTH) + 1):
                idx = LDEPTH - j
                e = max(float(np.linalg.norm(np.array(bw[idx][0]) - np.array(fw[idx][0])))
                        for fw, bw in RUNS[mu])
                errs.append(max(e, 1e-9))
            ax.semilogy(range(len(errs)), errs, color=col, lw=1.8, label=f"mu = {mu}")
            bud = [1.19e-7 * (1 / mu) ** j for j in range(min(t, LDEPTH) + 1)]
            ax.semilogy(range(len(bud)), bud, color=col, ls="--", lw=1.0, alpha=0.6)
        ax.set_xlim(0, LDEPTH); ax.set_ylim(1e-9, 1e3)
        ax.set_xlabel("backward steps taken")
        ax.set_ylabel("max reconstruction error (float32)")
        ax.legend(loc="upper left", fontsize=9)
        ax.set_title("the rewind's error grows as the noise budget (dashed) predicted", fontsize=12)

    anim = FuncAnimation(fig, draw, frames=frames, interval=120)
    out = os.path.join(PUB, "revmem-error-growth.gif")
    anim.save(out, writer=PillowWriter(fps=8))
    plt.close(fig)
    print(f"  revmem-error-growth.gif: {os.path.getsize(out)//1024} KB")


def png_wall():
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    Ls = [8, 32, 128, 512]
    for mode, col in [("standard", ORANGE), ("reversible", BLUE)]:
        ys = [next(r["temp_mb"] for r in D["memory"] if r["L"] == L and r["mode"] == mode) for L in Ls]
        ax.loglog(Ls, ys, "o-", color=col, lw=2, label=f"{mode} backprop")
        for L, y in zip(Ls, ys):
            ax.annotate(f"{y:.1f}", (L, y), textcoords="offset points", xytext=(6, 5), fontsize=9, color=col)
    ax.set_xlabel("depth L"); ax.set_ylabel("activation memory of one step (MB)")
    ax.set_xticks(Ls); ax.set_xticklabels(Ls)
    ax.legend(); ax.set_title("XLA's measured activation memory: O(L) against flat", fontsize=12)
    out = os.path.join(PUB, "revmem-wall.png")
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"  revmem-wall.png: {os.path.getsize(out)//1024} KB")


def png_cliff():
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for mu, col in zip(MUS, [BLUE, PURPLE, ORANGE]):
        rows = [r for r in D["fidelity"] if r["mu"] == mu]
        xs = [r["L"] for r in rows]
        ys = [0 if r["cosine"] is None or r["cosine"] != r["cosine"] else r["cosine"] for r in rows]
        ax.semilogx(xs, ys, "o-", color=col, lw=2, label=f"mu = {mu}")
        Lstar = np.log(1 / 1.19e-7) / np.log(1 / mu)
        if Lstar < 200:
            ax.axvline(Lstar, color=col, ls="--", lw=1, alpha=0.6)
            ax.annotate(f"L* = {Lstar:.0f}", (Lstar, 0.55), fontsize=9, color=col, rotation=90)
    ax.set_xticks([8, 32, 128]); ax.set_xticklabels([8, 32, 128])
    ax.set_xlabel("depth L"); ax.set_ylabel("gradient cosine (rewind vs stored)")
    ax.legend(); ax.set_title("measured gradient deaths against the napkin prediction", fontsize=12)
    out = os.path.join(PUB, "revmem-cliff.png")
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"  revmem-cliff.png: {os.path.getsize(out)//1024} KB")


if __name__ == "__main__":
    print("rendering revmem figures from the real bundle:")
    gif_rewind_shear()
    gif_ledger()
    gif_error_growth()
    png_wall()
    png_cliff()
