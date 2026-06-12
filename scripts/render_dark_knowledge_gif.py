#!/usr/bin/env python3
"""Dark knowledge is the off-one-hot mass. The information that survives a codebook
lives between the prototypes, as the soft mixture a feature makes over them. On a
graded codebook (neighbouring classes are similar) a feature near class k spreads
its assignment onto k's neighbours -- that spread IS the dark knowledge, the class
similarity a teacher distils. Live in JAX we sweep the softmax temperature from
warm to cold and watch the assignment collapse to one-hot, erasing the relation.
"""
from __future__ import annotations
from pathlib import Path
import imageio.v2 as imageio
import jax, jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colormaps  # noqa: E402
from PIL import Image  # noqa: E402

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[1]
OUT_GIF = ROOT / "public" / "dark-knowledge.gif"
OUT_PREVIEW = ROOT / "public" / "dark-knowledge-preview.png"

W, H, FPS, FRAMES, C, SIG = 1100, 520, 16, 84, 9, 2.2
BG, PANEL, INK, MUTED, BORDER, ACCENT, BLUE, GREEN = "#fbfaf6", "#ffffff", "#181818", "#666a70", "#ded9cb", "#b3661b", "#4a7fb3", "#3a8f5e"
COLORS = colormaps["turbo"](np.linspace(0.08, 0.92, C))

# graded label kernel -> horseshoe codebook (classical MDS, top-2 sqrt-scaled)
idx = jnp.arange(C, dtype=jnp.float64)
S = jnp.exp(-(((idx[:, None] - idx[None, :]) / SIG) ** 2)); S = S - S.mean()
w, V = jnp.linalg.eigh(S); o = jnp.argsort(w)[::-1]; w, V = w[o], V[:, o]
proto = V[:, :2] * jnp.sqrt(jnp.clip(w[:2], 0.0, None))
proto = proto / (jnp.linalg.norm(proto, axis=1, keepdims=True) + 1e-9)   # (C,2) unit codes
P = np.asarray(proto)

FOCUS = 6                                                      # the feature sits near class 6
z = proto[FOCUS] + 0.06 * (proto[5] - proto[FOCUS])           # a touch toward its neighbour
z = z / (jnp.linalg.norm(z) + 1e-9)
zn = np.asarray(z)


def soft_assignment(z, prototypes, tau):
    return jax.nn.softmax((z @ prototypes.T) / tau)


# warm -> cold temperature schedule (log space)
TAUS = np.exp(np.linspace(np.log(0.9), np.log(0.02), FRAMES))


def draw(frame):
    tau = float(TAUS[frame])
    a = np.asarray(soft_assignment(z, proto, tau))
    ent = float(-np.sum(a * np.log(a + 1e-12)) / np.log(C))    # normalized entropy in [0,1]
    dark = float(1.0 - a.max())                                # off-top-1 mass

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    fig.text(0.5, 0.95, "Dark knowledge is the off-one-hot mass", ha="center", color=INK, fontsize=18, weight="bold")
    fig.text(0.5, 0.905, "a feature's soft assignment spreads onto similar classes; cool the temperature and it erases", ha="center", color=MUTED, fontsize=11.5)

    # codebook geometry, prototype markers sized by assignment
    ax = fig.add_axes([0.05, 0.1, 0.42, 0.74]); ax.set_facecolor(PANEL); ax.set_aspect("equal")
    th = np.linspace(0, 2 * np.pi, 200); ax.plot(np.cos(th), np.sin(th), color=BORDER, lw=1, zorder=1)
    for c in range(C):
        ax.plot([zn[0], P[c, 0]], [zn[1], P[c, 1]], color=COLORS[c], lw=0.6 + 7 * a[c], alpha=0.35 + 0.5 * a[c], zorder=2)
        ax.scatter([P[c, 0]], [P[c, 1]], s=60 + 900 * a[c], color=COLORS[c], edgecolors=PANEL, linewidths=1.2, zorder=3)
        ax.text(P[c, 0] * 1.14, P[c, 1] * 1.14, str(c), ha="center", va="center", color=MUTED, fontsize=8)
    ax.scatter([zn[0]], [zn[1]], marker="*", s=320, color=INK, edgecolors="white", linewidths=0.8, zorder=5)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_xticks([]); ax.set_yticks([]); [sp.set_color(BORDER) for sp in ax.spines.values()]
    ax.set_title("codebook (★ = feature, marker size = assignment)", color=INK, fontsize=10.5, weight="bold", pad=6)

    # assignment bars, off-top mass shaded
    axb = fig.add_axes([0.57, 0.18, 0.39, 0.56]); axb.set_facecolor(PANEL)
    top = int(a.argmax())
    axb.bar(np.arange(C), a, color=[COLORS[c] if c == top else (0.78, 0.74, 0.66) for c in range(C)])
    axb.bar([top], [a[top]], color=COLORS[top])
    axb.set_ylim(0, 1.02); axb.set_xticks(range(C)); axb.set_xticklabels(range(C), color=MUTED, fontsize=8)
    axb.set_yticks([]); [sp.set_color(BORDER) for sp in axb.spines.values()]
    axb.set_xlabel("class", color=MUTED, fontsize=9.5)
    axb.set_title("soft assignment over the codebook", color=INK, fontsize=10.5, weight="bold", pad=6)
    axb.text(0.02, 0.95, f"τ = {tau:.2f}\ndark-knowledge mass = {dark:.2f}\nnorm. entropy = {ent:.2f}",
             transform=axb.transAxes, va="top", color=INK, fontsize=10, family="monospace")

    state = "WARM · similarity preserved" if dark > 0.25 else ("COOLING" if dark > 0.05 else "COLD · one-hot, relation erased")
    col = GREEN if dark > 0.25 else (BLUE if dark > 0.05 else ACCENT)
    fig.text(0.5, 0.045, state, ha="center", color=col, fontsize=13, weight="bold")
    fig.canvas.draw(); rgba = np.asarray(fig.canvas.buffer_rgba()).copy(); plt.close(fig); return rgba


def main():
    frames = [draw(i) for i in range(FRAMES)]
    Image.fromarray(frames[int(FRAMES * 0.15)]).save(OUT_PREVIEW)
    imageio.mimsave(OUT_GIF, frames, duration=1 / FPS, loop=0, palettesize=128, subrectangles=True)
    print("wrote", OUT_GIF)


if __name__ == "__main__":
    main()
