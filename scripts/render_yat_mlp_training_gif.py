#!/usr/bin/env python3
"""Render a GIF of a Yat-kernel MLP learning two-moons — no activation function.

A companion figure for the Yat-kernel writeups. We train a tiny Flax NNX MLP
whose only nonlinearity is the Yat kernel itself: each unit computes
(x·w + b)² / (‖x − w‖² + ε), a finite kernel against a learned prototype. The
prototypes (the rows of W) start scattered and visibly migrate onto the data as
training proceeds, and the diverging decision field hardens into the two-moons
boundary. Left: the field + data + migrating prototype rings. Right: the loss
curve (log-y) that drives them.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from flax import nnx  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_GIF = ROOT / "public" / "yat-mlp-training.gif"
OUT_PREVIEW = ROOT / "public" / "yat-mlp-training-preview.png"

W, H = 1100, 520
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
SNAP_EVERY = 8
GRID_N = 70
GRID_LO, GRID_HI = -2.0, 2.0


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
        dist2 = xn + wn - 2.0 * dot
        return (dot + b) ** 2 / (dist2 + eps)


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


def train_snapshots():
    X, y = moons(400, jax.random.key(0))
    model = YatMLP(2, N_UNITS, 2, rngs=nnx.Rngs(1))
    opt = nnx.Optimizer(model, optax.adam(3e-2), wrt=nnx.Param)

    gx = jnp.linspace(GRID_LO, GRID_HI, GRID_N)
    gy = jnp.linspace(GRID_LO, GRID_HI, GRID_N)
    GX, GY = jnp.meshgrid(gx, gy)
    grid = jnp.stack([GX.ravel(), GY.ravel()], 1)

    @nnx.jit
    def step(model, opt, X, y):
        def loss_fn(model):
            logits = model(X)
            return jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(logits, y)
            )

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        opt.update(model, grads)
        return loss

    @nnx.jit
    def evaluate(model, X, y, grid):
        logits = model(X)
        acc = jnp.mean(jnp.argmax(logits, -1) == y)
        glog = model(grid)
        field = glog[:, 1] - glog[:, 0]
        return acc, field

    # Dense-early schedule: the boundary forms in the first ~25 steps, so snapshot
    # every step there, then log-space out to the end. Otherwise most frames sit on
    # an already-converged model and the learning flashes by.
    snap_steps = set(range(0, 28)) | {
        int(round(v)) for v in np.geomspace(28, STEPS, 16) if v <= STEPS
    }

    snaps = []
    loss_hist = []
    cur_loss = float("nan")
    for s in range(STEPS + 1):
        if s > 0:
            cur_loss = float(step(model, opt, X, y))
            loss_hist.append((s, cur_loss))
        if s in snap_steps:
            acc, field = evaluate(model, X, y, grid)
            field = np.asarray(field).reshape(GRID_N, GRID_N)
            protos = np.asarray(model.yat.W.value)
            snaps.append(
                (s, cur_loss, float(acc), field, protos, list(loss_hist))
            )
    return snaps, np.asarray(X), np.asarray(y)


SNAPS, XDATA, YDATA = train_snapshots()
FINAL_LOSS = SNAPS[-1][1]
FINAL_ACC = SNAPS[-1][2]


def draw_frame(idx: int) -> np.ndarray:
    step_i, loss_i, acc_i, field, protos, hist = SNAPS[idx]

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    fig.text(0.5, 0.945, "A Yat-kernel MLP learning — no activation function",
             ha="center", color=INK, fontsize=18, weight="bold")
    fig.text(0.5, 0.90,
             "the prototypes (rings) migrate onto the data; the boundary forms",
             ha="center", color=MUTED, fontsize=11.5)

    # ── left: diverging decision field + data + migrating prototypes ──
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "yat_div", [BLUE, PANEL, ACCENT])
    axf = fig.add_axes([0.055, 0.14, 0.42, 0.62])
    axf.imshow(np.tanh(field), cmap=cmap, vmin=-1, vmax=1, origin="lower",
               extent=[GRID_LO, GRID_HI, GRID_LO, GRID_HI], aspect="equal")

    m0 = YDATA == 0
    m1 = YDATA == 1
    axf.scatter(XDATA[m0, 0], XDATA[m0, 1], s=9, c=BLUE,
                edgecolors="white", linewidths=0.3, zorder=3)
    axf.scatter(XDATA[m1, 0], XDATA[m1, 1], s=9, c=ACCENT,
                edgecolors="white", linewidths=0.3, zorder=3)

    axf.scatter(protos[:, 0], protos[:, 1], s=55, facecolors="none",
                edgecolors=INK, linewidths=1.6, zorder=4)

    axf.set_xlim(GRID_LO, GRID_HI)
    axf.set_ylim(GRID_LO, GRID_HI)
    axf.set_xticks([])
    axf.set_yticks([])
    for spine in axf.spines.values():
        spine.set_color(BORDER)
    axf.set_title("decision field  (logit₁ − logit₀)  +  prototypes",
                  color=INK, fontsize=11, weight="bold", pad=8)

    # ── right: loss curve (log scale), filling in as we train ──
    axl = fig.add_axes([0.575, 0.16, 0.37, 0.56])
    axl.set_facecolor(PANEL)
    if hist:
        steps = [h[0] for h in hist]
        losses = [h[1] for h in hist]
        axl.plot(steps, losses, color=ACCENT, lw=2.0)
        axl.plot(steps[-1], losses[-1], "o", color=ACCENT, ms=7)
    axl.set_yscale("log")
    axl.set_xlim(0, STEPS)
    axl.set_ylim(1e-4, 2)
    axl.set_xlabel("training step", color=MUTED, fontsize=9.5)
    axl.set_title("loss  (softmax cross-entropy)", color=INK,
                  fontsize=11, weight="bold", pad=8)
    axl.tick_params(colors=MUTED, labelsize=9)
    for spine in axl.spines.values():
        spine.set_color(BORDER)
    axl.text(0.96, 0.93, f"step {step_i}", transform=axl.transAxes, ha="right",
             color=MUTED, fontsize=10.5, family="monospace")
    axl.text(0.96, 0.85,
             f"loss {loss_i:.2e}" if loss_i == loss_i else "loss —",
             transform=axl.transAxes, ha="right", color=ACCENT, fontsize=10.5,
             family="monospace", weight="bold")
    axl.text(0.96, 0.77, f"train acc {acc_i * 100:.1f}%",
             transform=axl.transAxes, ha="right", color=INK, fontsize=10.5,
             family="monospace", weight="bold")

    fig.text(0.5, 0.045,
             "each unit is a finite kernel (x·w + b)² / (‖x − w‖² + ε) against a learned prototype — the only nonlinearity in the net",
             ha="center", color=MUTED, fontsize=10.5)

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba


def main() -> None:
    frames = []
    for idx in range(len(SNAPS)):
        frames.append(draw_frame(idx))
        if (idx + 1) % 10 == 0:
            print(f"rendered {idx + 1}/{len(SNAPS)} frames")
    Image.fromarray(frames[-1]).save(OUT_PREVIEW)
    frames.extend([frames[-1]] * END_HOLD)
    imageio.mimsave(OUT_GIF, frames, duration=1 / FPS, loop=0,
                    palettesize=128, subrectangles=True)
    print(f"final loss {FINAL_LOSS:.4f}, final train acc {FINAL_ACC * 100:.2f}%")
    print(f"wrote {OUT_GIF}")
    print(f"wrote {OUT_PREVIEW}")


if __name__ == "__main__":
    main()
