#!/usr/bin/env python3
"""Distributional simplicity bias: a network fits low-order structure first. On
two moons, the decision boundary starts almost linear (the coarse, low-order
split) and only gradually grows the curvature needed to wrap the moons. We render
the boundary live in JAX along with its length (a complexity proxy) climbing."""
from __future__ import annotations
from pathlib import Path
import imageio.v2 as imageio
import jax, jax.numpy as jnp, optax
import numpy as np
from sklearn.datasets import make_moons
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from PIL import Image  # noqa: E402

jax.config.update("jax_enable_x64", False)
ROOT = Path(__file__).resolve().parents[1]
OUT_GIF = ROOT / "public" / "ts-simplicity-bias.gif"
OUT_PREVIEW = ROOT / "public" / "ts-simplicity-bias-preview.png"

W, H, FPS, FRAMES, EVERY, HID = 1100, 520, 16, 80, 6, 48
BG, PANEL, INK, MUTED, BORDER, ACCENT, BLUE = "#fbfaf6", "#ffffff", "#181818", "#666a70", "#ded9cb", "#b3661b", "#4a7fb3"
REGION = ListedColormap(["#f0d9bf", "#cfe0ef"])

Xn, yn = make_moons(320, noise=0.18, random_state=0)
Xn = (Xn - Xn.mean(0)) / Xn.std(0)
X = jnp.asarray(Xn.astype(np.float32)); yj = jnp.asarray(yn)
gx, gy = np.meshgrid(np.linspace(-2.4, 2.4, 130), np.linspace(-2.4, 2.4, 130))
GRID = jnp.asarray(np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32))


def init(key):
    k = jax.random.split(key, 3)
    he = lambda kk, a, b: jax.random.normal(kk, (a, b)) * np.sqrt(2.0 / a)
    return dict(W1=he(k[0], 2, HID), b1=jnp.zeros(HID), W2=he(k[1], HID, HID), b2=jnp.zeros(HID), W3=he(k[2], HID, 2), b3=jnp.zeros(2))


def net(p, x):
    h = jax.nn.relu(x @ p["W1"] + p["b1"]); h = jax.nn.relu(h @ p["W2"] + p["b2"]); return h @ p["W3"] + p["b3"]


loss_fn = lambda p: optax.softmax_cross_entropy_with_integer_labels(net(p, X), yj).mean()
params = init(jax.random.key(0)); opt = optax.adam(8e-3); state = opt.init(params)


@jax.jit
def step(p, st):
    g = jax.grad(loss_fn)(p); up, st = opt.update(g, st); return optax.apply_updates(p, up), st


hist = []


def draw(frame):
    global params, state
    for _ in range(EVERY):
        params, state = step(params, state)
    pred = np.asarray(net(params, GRID).argmax(-1)).reshape(gx.shape)
    acc = float((np.asarray(net(params, X).argmax(-1)) == yn).mean())
    blen = (np.abs(np.diff(pred, axis=0)).sum() + np.abs(np.diff(pred, axis=1)).sum()) / pred.shape[0]   # boundary length
    hist.append((acc, blen))

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    fig.text(0.5, 0.95, "Networks fit low-order structure first", ha="center", color=INK, fontsize=18, weight="bold")
    fig.text(0.5, 0.905, "the decision boundary starts near-linear and only gradually wraps the moons", ha="center", color=MUTED, fontsize=11.5)

    ax = fig.add_axes([0.055, 0.1, 0.46, 0.76]); ax.set_facecolor(PANEL)
    ax.contourf(gx, gy, pred, levels=[-0.5, 0.5, 1.5], cmap=REGION, alpha=0.85)
    ax.contour(gx, gy, pred, levels=[0.5], colors=[ACCENT], linewidths=2)
    ax.scatter(Xn[yn == 0, 0], Xn[yn == 0, 1], s=12, color="#b3661b", edgecolors="white", linewidths=0.3)
    ax.scatter(Xn[yn == 1, 0], Xn[yn == 1, 1], s=12, color="#4a7fb3", edgecolors="white", linewidths=0.3)
    ax.set_xlim(-2.4, 2.4); ax.set_ylim(-2.4, 2.4); ax.set_xticks([]); ax.set_yticks([]); [sp.set_color(BORDER) for sp in ax.spines.values()]
    ax.set_title("decision boundary", color=INK, fontsize=11, weight="bold", pad=6)

    axc = fig.add_axes([0.6, 0.18, 0.36, 0.56]); axc.set_facecolor(PANEL)
    accs = [h[0] for h in hist]; blens = [h[1] for h in hist]
    n = len(hist); xs = np.arange(n)
    axc.plot(xs, accs, color="#3a8f5e", lw=2, label="train accuracy")
    axc.plot(xs, np.array(blens) / (max(blens) + 1e-9), color=ACCENT, lw=2, label="boundary complexity")
    axc.set_xlim(0, FRAMES); axc.set_ylim(0, 1.05); axc.set_yticks([]); axc.set_xticks([])
    [sp.set_color(BORDER) for sp in axc.spines.values()]
    axc.legend(loc="lower right", fontsize=9, frameon=False)
    axc.set_title("accuracy rises fast; complexity grows slowly", color=INK, fontsize=10.5, weight="bold", pad=6)

    fig.canvas.draw(); rgba = np.asarray(fig.canvas.buffer_rgba()).copy(); plt.close(fig); return rgba


def main():
    frames = [draw(i) for i in range(FRAMES)]
    Image.fromarray(frames[-1]).save(OUT_PREVIEW)
    imageio.mimsave(OUT_GIF, frames, duration=1 / FPS, loop=0, palettesize=128, subrectangles=True)
    print("wrote", OUT_GIF)


if __name__ == "__main__":
    main()
