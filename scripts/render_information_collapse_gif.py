#!/usr/bin/env python3
"""Neural collapse grinds the information spectrum to zero. A small encoder is
trained with cross-entropy; live in JAX we split each feature representation into
its between-class covariance Sigma_B (the prototype frame, rank <= C-1, the
separation channel) and within-class covariance Sigma_W (the information: the
gradations the codebook does not carry). As training proceeds, the prototype
frame sharpens into a simplex while the Sigma_W spectrum collapses toward zero."""
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
OUT_GIF = ROOT / "public" / "information-collapse.gif"
OUT_PREVIEW = ROOT / "public" / "information-collapse-preview.png"

W, H, FPS, FRAMES, EVERY = 1100, 520, 16, 80, 9
C, DIN, DEMB, HID, NPC = 4, 20, 6, 64, 120
BG, PANEL, INK, MUTED, BORDER, ACCENT, BLUE = "#fbfaf6", "#ffffff", "#181818", "#666a70", "#ded9cb", "#b3661b", "#4a7fb3"
PAL = ["#b3661b", "#4a7fb3", "#3a8f5e", "#9a4f9c"]

rng = np.random.default_rng(0)
means = rng.normal(0, 1.2, (C, DIN))
X = np.concatenate([means[c] + rng.normal(0, 0.85, (NPC, DIN)) for c in range(C)]).astype(np.float32)
y = np.repeat(np.arange(C), NPC)
X = jnp.asarray((X - X.mean(0)) / (X.std(0) + 1e-6)); yj = jnp.asarray(y)


def init(key):
    k1, k2, k3 = jax.random.split(key, 3)
    he = lambda k, a, b: jax.random.normal(k, (a, b)) * np.sqrt(2.0 / a)
    return dict(W1=he(k1, DIN, HID), b1=jnp.zeros(HID), W2=he(k2, HID, DEMB), b2=jnp.zeros(DEMB),
                Wc=he(k3, DEMB, C), bc=jnp.zeros(C))


def feats(p, x):
    return jax.nn.relu(x @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]


def loss_fn(p):
    logits = feats(p, X) @ p["Wc"] + p["bc"]
    return optax.softmax_cross_entropy_with_integer_labels(logits, yj).mean() + 1e-4 * sum(jnp.sum(v * v) for v in p.values())


params = init(jax.random.key(0))
opt = optax.adam(4e-3); state = opt.init(params)


@jax.jit
def step(p, st):
    g = jax.grad(loss_fn)(p)
    up, st = opt.update(g, st)
    return optax.apply_updates(p, up), st


def spectra(Z):
    mu = jnp.stack([Z[yj == c].mean(0) for c in range(C)])
    gmu = Z.mean(0)
    SB = sum(NPC * jnp.outer(mu[c] - gmu, mu[c] - gmu) for c in range(C)) / Z.shape[0]
    SW = jnp.mean(jax.vmap(lambda z, c: jnp.outer(z - mu[c], z - mu[c]))(Z, yj), 0)
    eb = np.sort(np.asarray(jnp.linalg.eigvalsh(SB)))[::-1]
    ew = np.sort(np.asarray(jnp.linalg.eigvalsh(SW)))[::-1]
    return np.maximum(eb, 0), np.maximum(ew, 0), np.asarray(mu)


def draw(frame):
    global params, state
    for _ in range(EVERY):
        params, state = step(params, state)
    Z = feats(params, X)
    eb, ew, mu = spectra(Z)
    Znp = np.asarray(Z)
    # 2D PCA for the embedding panel
    Zc = Znp - Znp.mean(0)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    P2 = Zc @ Vt[:2].T
    mu2 = (mu - Znp.mean(0)) @ Vt[:2].T
    nc1 = float(ew.sum() / (eb.sum() + 1e-9))     # within / between (collapse -> 0)

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    fig.text(0.5, 0.95, "Neural collapse grinds the information spectrum to zero", ha="center", color=INK, fontsize=18, weight="bold")
    fig.text(0.5, 0.905, "Σ_B (between-class) is the prototype frame; Σ_W (within-class) is the information", ha="center", color=MUTED, fontsize=11.5)

    ax = fig.add_axes([0.055, 0.12, 0.42, 0.72]); ax.set_facecolor(PANEL)
    r = np.abs(P2).max() * 1.1 + 1e-6
    ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    [sp.set_color(BORDER) for sp in ax.spines.values()]
    for c in range(C):
        ax.scatter(P2[y == c, 0], P2[y == c, 1], s=14, color=PAL[c], alpha=0.45)
        ax.scatter([mu2[c, 0]], [mu2[c, 1]], s=140, color=PAL[c], edgecolors=PANEL, linewidths=1.6, zorder=4)
    ax.set_title("representation (PCA to 2-D)", color=INK, fontsize=11.5, weight="bold", pad=6)

    axt = fig.add_axes([0.56, 0.50, 0.40, 0.30]); axb = fig.add_axes([0.56, 0.13, 0.40, 0.30])
    scale = eb[0] + 1e-9
    for axx, vals, col, name in ((axt, eb, ACCENT, "Σ_B  ·  prototype frame (separation)"), (axb, ew, BLUE, "Σ_W  ·  information (within-class)")):
        axx.set_facecolor(PANEL); axx.bar(np.arange(DEMB), vals / scale, color=col)
        axx.set_ylim(0, 1.05); axx.set_xticks([]); axx.set_yticks([]); [sp.set_color(BORDER) for sp in axx.spines.values()]
        axx.set_title(name, color=INK, fontsize=10.5, weight="bold", pad=4)
    axb.text(0.98, 0.86, f"Σ_W/Σ_B = {nc1:.3f}", transform=axb.transAxes, ha="right", color=BLUE, fontsize=10, weight="bold")

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba


def main():
    frames = [draw(i) for i in range(FRAMES)]
    Image.fromarray(frames[-1]).save(OUT_PREVIEW)
    imageio.mimsave(OUT_GIF, frames, duration=1 / FPS, loop=0, palettesize=128, subrectangles=True)
    print("wrote", OUT_GIF, "and", OUT_PREVIEW)


if __name__ == "__main__":
    main()
