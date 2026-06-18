#!/usr/bin/env python3
"""Banner GIF for the Yat-MLP JAX companion: ReLU MLP vs Yat-kernel MLP.

Same two-moons data, two units. Left: a standard Linear→ReLU→Linear MLP — a bank
of directions, an opaque piecewise-linear boundary. Right: the Yat-kernel MLP —
a bank of prototypes (drawn as rings), a smooth localized boundary, no activation
function. Both trained live in JAX; snapshots are dense early (the boundary forms
in the first ~25 steps) so the learning is visible rather than a flash.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax
from flax import nnx

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
OUT_GIF = ROOT / "public" / "yat-vs-relu-mlp.gif"
OUT_PREVIEW = ROOT / "public" / "yat-vs-relu-mlp-preview.png"

W, H = 1100, 540
FPS = 10
END_HOLD = 16

BG = "#fbfaf6"
PANEL = "#ffffff"
INK = "#181818"
MUTED = "#666a70"
BORDER = "#ded9cb"
ACCENT = "#b3661b"
BLUE = "#4a7fb3"

N_UNITS = 24
STEPS = 320
GRID_N = 80
DOM = 2.6


def lerp(c0, c1, t):
    return tuple(c0[i] + (c1[i] - c0[i]) * t for i in range(3))


def hexrgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))


BGr, ACr, BLr = hexrgb(BG), hexrgb(ACCENT), hexrgb(BLUE)


class ReluMLP(nnx.Module):
    def __init__(self, d_in, n_units, d_out, *, rngs):
        self.l1 = nnx.Linear(d_in, n_units, rngs=rngs)
        self.l2 = nnx.Linear(n_units, d_out, rngs=rngs)

    def __call__(self, x):
        return self.l2(jax.nn.relu(self.l1(x)))


class YatLayer(nnx.Module):
    def __init__(self, d_in, n_units, *, rngs, b0=0.5, eps0=0.5):
        self.W = nnx.Param(jax.random.normal(rngs.params(), (n_units, d_in)) * 0.7)
        self.log_b = nnx.Param(jnp.full((), jnp.log(jnp.expm1(b0))))
        self.log_eps = nnx.Param(jnp.full((), jnp.log(jnp.expm1(eps0))))

    def __call__(self, x):
        b = jax.nn.softplus(self.log_b.value)
        eps = jax.nn.softplus(self.log_eps.value)
        dot = x @ self.W.value.T
        xn = jnp.sum(x ** 2, -1, keepdims=True)
        wn = jnp.sum(self.W.value ** 2, -1)
        return (dot + b) ** 2 / (xn + wn - 2.0 * dot + eps)


class YatMLP(nnx.Module):
    def __init__(self, d_in, n_units, d_out, *, rngs):
        self.yat = YatLayer(d_in, n_units, rngs=rngs)
        self.readout = nnx.Linear(n_units, d_out, use_bias=True, rngs=rngs)

    def __call__(self, x):
        return self.readout(self.yat(x))


def moons(N, key):
    k1, k2, k3 = jax.random.split(key, 3)
    c = (jax.random.uniform(k1, (N,)) < 0.5).astype(jnp.int32)
    a = jnp.pi * jax.random.uniform(k2, (N,))
    x0 = jnp.where(c == 0, jnp.cos(a), 1 - jnp.cos(a)) - 0.5
    x1 = jnp.where(c == 0, jnp.sin(a), 0.5 - jnp.sin(a)) - 0.25
    X = jnp.stack([x0, x1], 1) + jax.random.normal(k3, (N, 2)) * 0.08
    return X, c


def train_both():
    X, y = moons(400, jax.random.key(0))
    relu = ReluMLP(2, N_UNITS, 2, rngs=nnx.Rngs(1))
    yat = YatMLP(2, N_UNITS, 2, rngs=nnx.Rngs(1))
    o_relu = nnx.Optimizer(relu, optax.adam(3e-2), wrt=nnx.Param)
    o_yat = nnx.Optimizer(yat, optax.adam(3e-2), wrt=nnx.Param)

    gx = jnp.linspace(-DOM, DOM, GRID_N)
    GX, GY = jnp.meshgrid(gx, gx)
    grid = jnp.stack([GX.ravel(), GY.ravel()], 1)

    @nnx.jit
    def step(model, opt, X, y):
        def loss_fn(m):
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(m(X), y))
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        opt.update(model, grads)
        return loss

    @nnx.jit
    def prob(model, grid):
        return jax.nn.softmax(model(grid), -1)[:, 1]

    @nnx.jit
    def acc(model, X, y):
        return jnp.mean(jnp.argmax(model(X), -1) == y)

    snap_steps = set(range(0, 28)) | {
        int(round(v)) for v in np.geomspace(28, STEPS, 16) if v <= STEPS
    }
    snaps = []
    for s in range(STEPS + 1):
        if s > 0:
            step(relu, o_relu, X, y)
            step(yat, o_yat, X, y)
        if s in snap_steps:
            pr = np.asarray(prob(relu, grid)).reshape(GRID_N, GRID_N)
            py = np.asarray(prob(yat, grid)).reshape(GRID_N, GRID_N)
            protos = np.asarray(yat.yat.W.value)
            snaps.append((s, pr, py, float(acc(relu, X, y)), float(acc(yat, X, y)), protos))
    return snaps, np.asarray(X), np.asarray(y)


SNAPS, XDATA, YDATA = train_both()


def field_img(P):
    conf = np.power(np.abs(2 * P - 1), 0.7) * 0.82
    img = np.empty((GRID_N, GRID_N, 3))
    for i in range(GRID_N):
        for j in range(GRID_N):
            hue = ACr if P[i, j] >= 0.5 else BLr
            img[i, j] = lerp(BGr, hue, conf[i, j])
    return img


def panel(ax, title, P, acc, protos=None):
    ax.imshow(field_img(P), extent=[-DOM, DOM, -DOM, DOM], origin="lower", interpolation="bilinear", zorder=0)
    for c, col in ((0, BLUE), (1, ACCENT)):
        m = YDATA == c
        ax.scatter(XDATA[m, 0], XDATA[m, 1], s=9, color=col, edgecolor="white", linewidth=0.3, zorder=3)
    if protos is not None:
        ax.scatter(protos[:, 0], protos[:, 1], s=44, facecolors="none", edgecolors=INK, linewidth=1.3, zorder=4)
    ax.set_xlim(-DOM, DOM); ax.set_ylim(-DOM, DOM)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.set_title(title, color=INK, fontsize=12, weight="bold", pad=6)
    ax.text(0.97, 0.04, f"acc {acc*100:.0f}%", transform=ax.transAxes, ha="right", va="bottom",
            color=MUTED, fontsize=9, family="monospace")


def draw_frame(idx):
    s, pr, py, ar, ay, protos = SNAPS[idx]
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    fig.text(0.5, 0.945, "Same data, two units: ReLU vs the Yat kernel", ha="center", color=INK, fontsize=18, weight="bold")
    fig.text(0.5, 0.895, "left: a Linear→ReLU→Linear MLP.   right: the Yat kernel as the unit — no activation, and the centres (rings) are visible.",
             ha="center", color=MUTED, fontsize=11)
    axl = fig.add_axes([0.055, 0.10, 0.41, 0.72])
    axr = fig.add_axes([0.535, 0.10, 0.41, 0.72])
    panel(axl, "ReLU MLP  (a bank of directions)", pr, ar)
    panel(axr, "Yat-kernel MLP  (a bank of prototypes)", py, ay, protos)
    fig.text(0.5, 0.04, f"step {s} · the ReLU boundary is piecewise-linear and extends to infinity; the Yat boundary wraps the data and its prototypes are readable",
             ha="center", color=MUTED, fontsize=10)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba


def main():
    frames = []
    for i in range(len(SNAPS)):
        frames.append(draw_frame(i))
        if (i + 1) % 12 == 0:
            print(f"rendered {i + 1}/{len(SNAPS)} frames")
    Image.fromarray(frames[-1]).save(OUT_PREVIEW)
    frames.extend([frames[-1]] * END_HOLD)
    imageio.mimsave(OUT_GIF, frames, duration=1 / FPS, loop=0, palettesize=96, subrectangles=True)
    print(f"final acc — ReLU {SNAPS[-1][3]*100:.1f}%  Yat {SNAPS[-1][4]*100:.1f}%")
    print(f"wrote {OUT_GIF}")
    print(f"wrote {OUT_PREVIEW}")


if __name__ == "__main__":
    main()
