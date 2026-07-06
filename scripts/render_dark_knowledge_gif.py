#!/usr/bin/env python3
"""Dark knowledge is the off-one-hot mass. The information that survives a codebook
lives between the prototypes, as the soft mixture a feature makes over them. On a
graded codebook (neighbouring classes are similar) a feature near class k spreads
its assignment onto k's neighbours -- that spread IS the dark knowledge, the class
similarity a teacher distils. The softmax temperature is a knob, not a process, so
this is a static contrast: warm (mass spread, similarity preserved) vs cold
(one-hot, relation erased), both computed in JAX.
"""
from __future__ import annotations
from pathlib import Path
import jax, jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colormaps  # noqa: E402
from PIL import Image  # noqa: E402

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[1]
OUT_PNG = ROOT / "public" / "dark-knowledge.png"
OUT_PREVIEW = ROOT / "public" / "dark-knowledge-preview.png"

W, H, C, SIG = 1100, 560, 9, 2.2
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


def stats(tau):
    a = np.asarray(soft_assignment(z, proto, tau))
    ent = float(-np.sum(a * np.log(a + 1e-12)) / np.log(C))    # normalized entropy in [0,1]
    dark = float(1.0 - a.max())                                # off-top-1 mass
    return a, ent, dark


def draw_panel(fig, x0, tau, tag):
    a, ent, dark = stats(tau)

    ax = fig.add_axes([x0, 0.30, 0.24, 0.5]); ax.set_facecolor(PANEL); ax.set_aspect("equal")
    th = np.linspace(0, 2 * np.pi, 200); ax.plot(np.cos(th), np.sin(th), color=BORDER, lw=1, zorder=1)
    for c in range(C):
        ax.plot([zn[0], P[c, 0]], [zn[1], P[c, 1]], color=COLORS[c], lw=0.6 + 7 * a[c], alpha=0.35 + 0.5 * a[c], zorder=2)
        ax.scatter([P[c, 0]], [P[c, 1]], s=50 + 700 * a[c], color=COLORS[c], edgecolors=PANEL, linewidths=1.2, zorder=3)
        ax.text(P[c, 0] * 1.16, P[c, 1] * 1.16, str(c), ha="center", va="center", color=MUTED, fontsize=7.5)
    ax.scatter([zn[0]], [zn[1]], marker="*", s=260, color=INK, edgecolors="white", linewidths=0.8, zorder=5)
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35); ax.set_xticks([]); ax.set_yticks([]); [sp.set_color(BORDER) for sp in ax.spines.values()]
    ax.set_title(f"{tag}   (τ = {tau:.2f})", color=INK, fontsize=10.5, weight="bold", pad=6)

    axb = fig.add_axes([x0 + 0.265, 0.36, 0.20, 0.4]); axb.set_facecolor(PANEL)
    top = int(a.argmax())
    axb.bar(np.arange(C), a, color=[COLORS[c] if c == top else (0.78, 0.74, 0.66) for c in range(C)])
    axb.set_ylim(0, 1.02); axb.set_xticks(range(C)); axb.set_xticklabels(range(C), color=MUTED, fontsize=7)
    axb.set_yticks([]); [sp.set_color(BORDER) for sp in axb.spines.values()]
    axb.set_xlabel("class", color=MUTED, fontsize=8.5)
    axb.text(0.02, 0.96, f"dark mass {dark:.2f}\nentropy {ent:.2f}", transform=axb.transAxes, va="top", color=INK, fontsize=8.5, family="monospace")


def main():
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    fig.text(0.5, 0.95, "Dark knowledge is the off-one-hot mass", ha="center", color=INK, fontsize=18, weight="bold")
    fig.text(0.5, 0.905, "a feature's soft assignment spreads onto similar classes; cool the temperature and it erases", ha="center", color=MUTED, fontsize=11.5)

    draw_panel(fig, 0.035, 0.9, "WARM · similarity preserved")
    draw_panel(fig, 0.525, 0.03, "COLD · one-hot, relation erased")
    fig.text(0.28, 0.075, "warm: mass spreads onto neighbours 5, 7, 8", ha="center", color=GREEN, fontsize=10.5, weight="bold")
    fig.text(0.77, 0.075, "cold: a one-hot spike on class 6", ha="center", color=ACCENT, fontsize=10.5, weight="bold")

    fig.canvas.draw(); buf = np.asarray(fig.canvas.buffer_rgba()).copy(); plt.close(fig)
    Image.fromarray(buf).save(OUT_PNG)
    Image.fromarray(buf).save(OUT_PREVIEW)
    print("wrote", OUT_PNG)


if __name__ == "__main__":
    main()
