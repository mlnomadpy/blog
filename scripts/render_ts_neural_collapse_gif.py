#!/usr/bin/env python3
"""The structured endpoint is a simplex. Live in JAX, a classifier's class means
converge to a simplex equiangular tight frame: every pair of class-mean cosines
collapses onto -1/(C-1), the Welch-bound / neural-collapse optimum. We render the
distribution of off-diagonal class-mean cosines tightening to that line."""
from __future__ import annotations
from pathlib import Path
import imageio.v2 as imageio
import jax, jax.numpy as jnp, optax
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

jax.config.update("jax_enable_x64", False)
ROOT = Path(__file__).resolve().parents[1]
OUT_GIF = ROOT / "public" / "ts-neural-collapse.gif"
OUT_PREVIEW = ROOT / "public" / "ts-neural-collapse-preview.png"

W, H, FPS, FRAMES, EVERY = 1100, 520, 16, 80, 18
C, DIN, DEMB, HID, NPC = 6, 24, 16, 64, 90
BG, PANEL, INK, MUTED, BORDER, ACCENT, BLUE = "#fbfaf6", "#ffffff", "#181818", "#666a70", "#ded9cb", "#b3661b", "#4a7fb3"
import matplotlib.cm as _cm  # noqa: E402
PAL = _cm.tab10(np.arange(C))
TARGET = -1.0 / (C - 1)

rng = np.random.default_rng(3)
means = rng.normal(0, 1.1, (C, DIN))
X = np.concatenate([means[c] + rng.normal(0, 0.9, (NPC, DIN)) for c in range(C)]).astype(np.float32)
y = np.repeat(np.arange(C), NPC)
X = jnp.asarray((X - X.mean(0)) / (X.std(0) + 1e-6)); yj = jnp.asarray(y)


def init(key):
    k = jax.random.split(key, 3)
    he = lambda kk, a, b: jax.random.normal(kk, (a, b)) * np.sqrt(2.0 / a)
    return dict(W1=he(k[0], DIN, HID), b1=jnp.zeros(HID), W2=he(k[1], HID, DEMB), b2=jnp.zeros(DEMB),
                Wc=he(k[2], DEMB, C), bc=jnp.zeros(C))


feats = lambda p, x: jax.nn.relu(x @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]
loss_fn = lambda p: optax.softmax_cross_entropy_with_integer_labels(feats(p, X) @ p["Wc"] + p["bc"], yj).mean()
params = init(jax.random.key(0)); opt = optax.adamw(6e-3, weight_decay=3e-3); state = opt.init(params)


@jax.jit
def step(p, st):
    g = jax.grad(loss_fn)(p); up, st = opt.update(g, st, p); return optax.apply_updates(p, up), st


def draw(frame):
    global params, state
    for _ in range(EVERY):
        params, state = step(params, state)
    Z = np.asarray(feats(params, X))
    mu = np.stack([Z[y == c].mean(0) for c in range(C)]); mu -= mu.mean(0)
    mun = mu / (np.linalg.norm(mu, axis=1, keepdims=True) + 1e-9)
    G = mun @ mun.T
    offs = G[np.triu_indices(C, 1)]
    Zc = Z - Z.mean(0); _, _, Vt = np.linalg.svd(Zc, full_matrices=False); P2 = Zc @ Vt[:2].T
    mu2 = np.stack([P2[y == c].mean(0) for c in range(C)])

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    fig.text(0.5, 0.95, "The structured endpoint is a simplex", ha="center", color=INK, fontsize=18, weight="bold")
    fig.text(0.5, 0.905, "class-mean cosines collapse onto −1/(C−1): the equiangular tight frame / Welch optimum", ha="center", color=MUTED, fontsize=11.5)

    ax = fig.add_axes([0.055, 0.12, 0.42, 0.72]); ax.set_facecolor(PANEL)
    r = np.abs(P2).max() * 1.1 + 1e-6; ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    [sp.set_color(BORDER) for sp in ax.spines.values()]
    for a in range(C):
        for b in range(a + 1, C):
            ax.plot([mu2[a, 0], mu2[b, 0]], [mu2[a, 1], mu2[b, 1]], color=BORDER, lw=0.8, zorder=1)
    for c in range(C):
        ax.scatter(P2[y == c, 0], P2[y == c, 1], s=11, color=PAL[c], alpha=0.35)
        ax.scatter([mu2[c, 0]], [mu2[c, 1]], s=130, color=PAL[c], edgecolors=PANEL, linewidths=1.6, zorder=4)
    ax.set_title("class means (PCA to 2-D)", color=INK, fontsize=11, weight="bold", pad=6)

    axh = fig.add_axes([0.57, 0.18, 0.39, 0.56]); axh.set_facecolor(PANEL)
    axh.hist(offs, bins=np.linspace(-1, 1, 31), color=ACCENT, alpha=0.85)
    axh.axvline(TARGET, color=BLUE, lw=2, ls="--")
    axh.text(TARGET, axh.get_ylim()[1] * 0.92, f"  −1/(C−1) = {TARGET:.2f}", color=BLUE, fontsize=10, weight="bold", va="top")
    axh.set_xlim(-1, 1); axh.set_yticks([]); [sp.set_color(BORDER) for sp in axh.spines.values()]
    axh.set_xlabel("pairwise class-mean cosine", color=MUTED, fontsize=9.5)
    axh.set_title("equiangularity", color=INK, fontsize=11, weight="bold", pad=6)
    axh.text(0.02, 0.92, f"mean {offs.mean():+.2f}\nstd  {offs.std():.3f}", transform=axh.transAxes, va="top", color=INK, fontsize=10, family="monospace")

    fig.canvas.draw(); rgba = np.asarray(fig.canvas.buffer_rgba()).copy(); plt.close(fig); return rgba


def main():
    frames = [draw(i) for i in range(FRAMES)]
    Image.fromarray(frames[-1]).save(OUT_PREVIEW)
    imageio.mimsave(OUT_GIF, frames, duration=1 / FPS, loop=0, palettesize=128, subrectangles=True)
    print("wrote", OUT_GIF)


if __name__ == "__main__":
    main()
