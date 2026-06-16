#!/usr/bin/env python3
"""Render a GIF of a single attention head learning its kernel.

A companion figure for "Self-Attention as Kernel Regression in JAX/Flax NNX".
Each sequence has one flagged token; every position must output that token's
vector. The head can only succeed by learning q/k projections whose kernel
concentrates on the flag. We snapshot the attention matrix for one fixed
example as training proceeds: it starts near-uniform (a wide kernel, every
token averaged) and sharpens to a single bright column on the marker (a narrow
kernel, a hard copy), while the loss falls to ~0. Left: the kernel learning.
Right: the loss it is driven by.
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
OUT_GIF = ROOT / "public" / "attention-kernel-learning.gif"
OUT_PREVIEW = ROOT / "public" / "attention-kernel-learning-preview.png"

W, H = 1100, 520
FPS = 12
END_HOLD = 18

BG = "#fbfaf6"
PANEL = "#ffffff"
INK = "#181818"
MUTED = "#666a70"
BORDER = "#ded9cb"
ACCENT = "#b3661b"

L, D_MODEL = 6, 16
STEPS = 600
SNAP_EVERY = 12  # one frame every SNAP_EVERY steps


class BatchHead(nnx.Module):
    def __init__(self, d_model, d_head, *, rngs):
        self.wq = nnx.Linear(d_model, d_head, use_bias=True, rngs=rngs)
        self.wk = nnx.Linear(d_model, d_head, use_bias=True, rngs=rngs)
        self.wv = nnx.Linear(d_model, d_head, use_bias=True, rngs=rngs)

    def __call__(self, x):
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        s = jnp.einsum("...id,...jd->...ij", q, k) / jnp.sqrt(q.shape[-1])
        a = jax.nn.softmax(s, axis=-1)
        return jnp.einsum("...ij,...jd->...id", a, v), a


def make_example(key):
    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, (L, D_MODEL))
    t = jax.random.randint(k2, (), 0, L)
    x = x.at[t, 0].set(6.0)
    return x, jnp.broadcast_to(x[t], (L, D_MODEL))


def train_snapshots():
    """Train the head, recording (step, loss, alpha_for_display) at intervals."""
    xs, targets = jax.vmap(make_example)(jax.random.split(jax.random.key(3), 512))
    model = BatchHead(D_MODEL, D_MODEL, rngs=nnx.Rngs(0))
    opt = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)

    @nnx.jit
    def step(model, opt, xs, targets):
        def loss_fn(model):
            ys, _ = model(xs)
            return jnp.mean((ys - targets) ** 2)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        opt.update(model, grads)
        return loss

    disp_x = xs[0]
    marker = int(np.asarray(disp_x[:, 0]).argmax())

    snaps = []
    loss_hist = []
    for s in range(STEPS + 1):
        if s > 0:
            loss = float(step(model, opt, xs, targets))
            loss_hist.append((s, loss))
        if s % SNAP_EVERY == 0:
            _, a = model(disp_x[None])
            cur = loss_hist[-1][1] if loss_hist else float("nan")
            snaps.append((s, cur, np.asarray(a[0]), list(loss_hist)))
    return snaps, marker


SNAPS, MARKER = train_snapshots()
FINAL_LOSS = SNAPS[-1][1]


def draw_frame(idx: int) -> np.ndarray:
    step_i, loss_i, alpha, hist = SNAPS[idx]

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    fig.text(0.5, 0.945, "A single head learning its kernel", ha="center",
             color=INK, fontsize=18, weight="bold")
    fig.text(0.5, 0.90,
             "every position must output the flagged token; the kernel learns to route there",
             ha="center", color=MUTED, fontsize=11.5)

    # ── left: attention matrix, sharpening onto the marker column ──
    axm = fig.add_axes([0.07, 0.16, 0.40, 0.56])
    axm.imshow(alpha, cmap="Oranges", vmin=0, vmax=1, aspect="equal")
    axm.add_patch(plt.Rectangle((MARKER - 0.5, -0.5), 1, L, fill=False,
                                edgecolor=ACCENT, lw=2.4))
    axm.set_xticks(range(L))
    axm.set_xticklabels([f"k{j}" if j != MARKER else "flag" for j in range(L)],
                        color=MUTED, fontsize=9)
    axm.set_yticks(range(L))
    axm.set_yticklabels([f"q{i}" for i in range(L)], color=MUTED, fontsize=9)
    axm.set_xlabel("key j", color=MUTED, fontsize=9.5)
    axm.set_ylabel("query i", color=MUTED, fontsize=9.5)
    for spine in axm.spines.values():
        spine.set_color(BORDER)
    axm.set_title("attention  α = softmax(QKᵀ / √d)", color=INK,
                  fontsize=11, weight="bold", pad=8)

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
    axl.set_ylim(1e-5, 20)
    axl.set_xlabel("training step", color=MUTED, fontsize=9.5)
    axl.set_title("loss  (mean squared error)", color=INK,
                  fontsize=11, weight="bold", pad=8)
    axl.tick_params(colors=MUTED, labelsize=9)
    for spine in axl.spines.values():
        spine.set_color(BORDER)
    axl.text(0.96, 0.92, f"step {step_i}", transform=axl.transAxes, ha="right",
             color=MUTED, fontsize=10.5, family="monospace")
    axl.text(0.96, 0.84, f"loss {loss_i:.2e}" if loss_i == loss_i else "loss —",
             transform=axl.transAxes, ha="right", color=ACCENT, fontsize=10.5,
             family="monospace", weight="bold")

    fig.text(0.5, 0.05,
             "the wide, near-uniform kernel collapses to one bright column — a learned nearest-neighbour copy, not a built-in rule",
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
    frames.extend([frames[-1]] * END_HOLD)  # hold on the converged kernel
    imageio.mimsave(OUT_GIF, frames, duration=1 / FPS, loop=0,
                    palettesize=128, subrectangles=True)
    print(f"final loss {FINAL_LOSS:.3e}, marker at column {MARKER}")
    print(f"wrote {OUT_GIF}")
    print(f"wrote {OUT_PREVIEW}")


if __name__ == "__main__":
    main()
