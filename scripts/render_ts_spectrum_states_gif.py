#!/usr/bin/env python3
"""Training reshapes the representation's spectrum. Live in JAX we train a small
classifier and, each frame, take the eigenspectrum of the feature covariance. It
starts ~flat and high-rank (RANDOM, isotropic), develops a few dominant modes
(ORGANIZED), and collapses to a low-rank, C-1-mode frame (STRUCTURED) as the
within-class variation is squeezed out. The effective rank (participation ratio)
falls from ~d toward C-1."""
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
OUT_GIF = ROOT / "public" / "ts-spectrum-states.gif"
OUT_PREVIEW = ROOT / "public" / "ts-spectrum-states-preview.png"

W, H, FPS, FRAMES, EVERY = 1100, 520, 16, 80, 16
C, DIN, DEMB, HID, NPC = 5, 24, 12, 64, 90
BG, PANEL, INK, MUTED, BORDER, ACCENT, BLUE, GREEN = "#fbfaf6", "#ffffff", "#181818", "#666a70", "#ded9cb", "#b3661b", "#4a7fb3", "#3a8f5e"
PAL = ["#b3661b", "#4a7fb3", "#3a8f5e", "#9a4f9c", "#c2553a"]

rng = np.random.default_rng(1)
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


def loss_fn(p):
    logits = feats(p, X) @ p["Wc"] + p["bc"]
    return optax.softmax_cross_entropy_with_integer_labels(logits, yj).mean()


params = init(jax.random.key(0)); opt = optax.adamw(5e-3, weight_decay=2e-3); state = opt.init(params)


@jax.jit
def step(p, st):
    g = jax.grad(loss_fn)(p); up, st = opt.update(g, st, p); return optax.apply_updates(p, up), st


def draw(frame):
    global params, state
    for _ in range(EVERY):
        params, state = step(params, state)
    Z = np.asarray(feats(params, X))
    Zc = Z - Z.mean(0)
    cov = Zc.T @ Zc / Z.shape[0]
    ev = np.sort(np.maximum(np.asarray(jnp.linalg.eigvalsh(jnp.asarray(cov))), 0))[::-1]
    eff_rank = (ev.sum() ** 2) / (np.sum(ev ** 2) + 1e-12)        # participation ratio
    # 2D PCA embedding
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    P2 = Zc @ Vt[:2].T
    mu2 = np.stack([P2[y == c].mean(0) for c in range(C)])
    intra = np.mean([np.linalg.norm(P2[y == c] - mu2[c], axis=1).mean() for c in range(C)])
    spread = np.mean([np.linalg.norm(mu2[a] - mu2[b]) for a in range(C) for b in range(a + 1, C)])
    nc = intra / (spread + 1e-9)
    state_lbl, col = ("RANDOM", MUTED) if eff_rank > 0.62 * DEMB else (("STRUCTURED", GREEN) if nc < 0.25 else ("ORGANIZED", BLUE))

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    fig.text(0.5, 0.95, "Training reshapes the representation's spectrum", ha="center", color=INK, fontsize=18, weight="bold")
    fig.text(0.5, 0.905, "eigenspectrum of the feature covariance: flat & high-rank → low-rank, C−1-mode frame", ha="center", color=MUTED, fontsize=11.5)

    ax = fig.add_axes([0.055, 0.12, 0.42, 0.72]); ax.set_facecolor(PANEL)
    r = np.abs(P2).max() * 1.1 + 1e-6; ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    [sp.set_color(BORDER) for sp in ax.spines.values()]
    for c in range(C):
        ax.scatter(P2[y == c, 0], P2[y == c, 1], s=12, color=PAL[c], alpha=0.4)
        ax.scatter([mu2[c, 0]], [mu2[c, 1]], s=120, color=PAL[c], edgecolors=PANEL, linewidths=1.5, zorder=4)
    ax.set_title("representation (PCA to 2-D)", color=INK, fontsize=11, weight="bold", pad=6)

    axs = fig.add_axes([0.56, 0.16, 0.40, 0.6]); axs.set_facecolor(PANEL)
    axs.bar(np.arange(DEMB), ev / (ev[0] + 1e-9), color=[ACCENT if i < C - 1 else BORDER for i in range(DEMB)])
    axs.set_ylim(0, 1.05); axs.set_xticks([]); axs.set_yticks([]); [sp.set_color(BORDER) for sp in axs.spines.values()]
    axs.set_title("feature-covariance spectrum", color=INK, fontsize=11, weight="bold", pad=6)
    axs.text(0.97, 0.86, f"effective rank {eff_rank:.1f} / {DEMB}", transform=axs.transAxes, ha="right", color=ACCENT, fontsize=10.5, weight="bold")

    fig.text(0.5, 0.05, state_lbl, ha="center", color=col, fontsize=15, weight="bold")
    fig.canvas.draw(); rgba = np.asarray(fig.canvas.buffer_rgba()).copy(); plt.close(fig); return rgba


def main():
    frames = [draw(i) for i in range(FRAMES)]
    Image.fromarray(frames[-1]).save(OUT_PREVIEW)
    imageio.mimsave(OUT_GIF, frames, duration=1 / FPS, loop=0, palettesize=128, subrectangles=True)
    print("wrote", OUT_GIF)


if __name__ == "__main__":
    main()
