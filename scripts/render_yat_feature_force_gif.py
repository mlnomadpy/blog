#!/usr/bin/env python3
"""Render a GIF of the force field of feature learning for a Yat unit.

A probe prototype P feels a force F(P) = -∇_P L(P), where
L(P) = mean_{x in B} k(P,x) - mean_{x in A} k(P,x) and k is the Yat kernel
k(P,x) = (P·x + b)² / (‖x − P‖² + eps). We draw F as a quiver field over a
grid and animate a probe flowing along it onto class A, leaving a trail. The
point: the gradient is non-zero everywhere — a Yat unit never dies the way a
ReLU unit does once it falls into its flat half-space.
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
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_GIF = ROOT / "public" / "yat-feature-force.gif"
OUT_PREVIEW = ROOT / "public" / "yat-feature-force-preview.png"

W, H = 900, 800
FPS = 12
N_FRAMES = 60
END_HOLD = 12

BG = "#fbfaf6"
PANEL = "#ffffff"
INK = "#181818"
MUTED = "#666a70"
BORDER = "#ded9cb"
ACCENT = "#b3661b"
CLASS_A = "#4a7fb3"
CLASS_B = ACCENT

B_BIAS = 0.5
EPS = 0.4

LIMS = (-2.1, 2.1)
GRID_N = 13
STEP = 0.024
START = jnp.array([1.5, 1.5])


# ── data ──
def make_data():
    key = jax.random.key(7)
    ka, kb = jax.random.split(key)
    a = jax.random.normal(ka, (45, 2)) * 0.32 + jnp.array([-1.0, -0.5])
    b = jax.random.normal(kb, (45, 2)) * 0.32 + jnp.array([1.0, 0.5])
    return a, b


DATA_A, DATA_B = make_data()


# ── Yat kernel and objective ──
def yat_kernel(P, x):
    dot = jnp.dot(P, x) + B_BIAS
    dist2 = jnp.sum((x - P) ** 2) + EPS
    return dot ** 2 / dist2


def loss(P, A, B):
    kB = jax.vmap(lambda x: yat_kernel(P, x))(B)
    kA = jax.vmap(lambda x: yat_kernel(P, x))(A)
    return jnp.mean(kB) - jnp.mean(kA)


grad_loss = jax.jit(jax.grad(loss))


@jax.jit
def force(P):
    return -grad_loss(P, DATA_A, DATA_B)


# ── precompute the quiver field on the grid ──
def compute_field():
    gx = np.linspace(LIMS[0], LIMS[1], GRID_N)
    gy = np.linspace(LIMS[0], LIMS[1], GRID_N)
    GX, GY = np.meshgrid(gx, gy)
    pts = jnp.stack([GX.ravel(), GY.ravel()], axis=-1)
    F = jax.vmap(force)(pts)
    F = np.asarray(F)
    mag = np.linalg.norm(F, axis=-1)
    safe = np.where(mag > 1e-8, mag, 1.0)
    U = F[:, 0] / safe
    V = F[:, 1] / safe
    return GX.ravel(), GY.ravel(), U, V, mag


FX, FY, FU, FV, FMAG = compute_field()
# per-arrow alpha: lower-magnitude arrows more transparent
_norm_mag = (FMAG - FMAG.min()) / (FMAG.max() - FMAG.min() + 1e-9)
F_ALPHA = 0.18 + 0.55 * _norm_mag
ARROW_SCALE = 0.18  # modest arrow length in data units


# ── integrate the probe trajectory ──
# The exact update P += 0.024·F/‖F‖ flows along the (curling) force field onto
# class A, but the arc from (1.5,1.5) is long, so we integrate many small steps
# and then sub-sample N_FRAMES snapshots for the animation.
SUBSTEPS = 5  # integration steps per displayed frame


def integrate():
    P = START
    traj = [np.asarray(P)]
    for _ in range(N_FRAMES - 1):
        for _ in range(SUBSTEPS):
            F = force(P)
            n = jnp.linalg.norm(F) + 1e-9
            P = P + STEP * F / n
        traj.append(np.asarray(P))
    return np.array(traj)


TRAJ = integrate()
FINAL_P = TRAJ[-1]


def draw_frame(idx: int) -> np.ndarray:
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    fig.text(0.5, 0.945, "Why a prototype moves: the force field of feature learning",
             ha="center", color=INK, fontsize=16, weight="bold")
    fig.text(0.5, 0.905,
             "each arrow is a gradient; the prototype is pulled onto the data it should detect",
             ha="center", color=MUTED, fontsize=11)

    ax = fig.add_axes([0.09, 0.10, 0.82, 0.76])
    ax.set_facecolor(PANEL)
    ax.set_xlim(LIMS)
    ax.set_ylim(LIMS)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(BORDER)

    # ── force field: per-arrow alpha by magnitude ──
    for x, y, u, v, a in zip(FX, FY, FU, FV, F_ALPHA):
        ax.arrow(x, y, u * ARROW_SCALE, v * ARROW_SCALE,
                 head_width=0.055, head_length=0.05, length_includes_head=True,
                 color=MUTED, alpha=float(a), lw=0.9, zorder=1)

    # ── data points ──
    ax.scatter(DATA_A[:, 0], DATA_A[:, 1], s=34, color=CLASS_A,
               edgecolors="white", linewidths=0.6, alpha=0.9, zorder=3,
               label="class A")
    ax.scatter(DATA_B[:, 0], DATA_B[:, 1], s=34, color=CLASS_B,
               edgecolors="white", linewidths=0.6, alpha=0.9, zorder=3,
               label="class B")

    # ── probe trail + marker ──
    trail = TRAJ[: idx + 1]
    if len(trail) > 1:
        ax.plot(trail[:, 0], trail[:, 1], color=INK, lw=2.0, alpha=0.85, zorder=4)
    px, py = TRAJ[idx]
    ax.scatter([px], [py], s=130, color=INK, edgecolors="white",
               linewidths=1.6, zorder=5)

    ax.legend(loc="lower right", frameon=True, fontsize=9.5,
              facecolor=PANEL, edgecolor=BORDER)

    fig.text(0.5, 0.045,
             "force everywhere — no half-space where the gradient vanishes — so a Yat unit never dies like a ReLU unit",
             ha="center", color=MUTED, fontsize=10)

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba


def main() -> None:
    frames = []
    for idx in range(N_FRAMES):
        frames.append(draw_frame(idx))
        if (idx + 1) % 10 == 0:
            print(f"rendered {idx + 1}/{N_FRAMES} frames")
    Image.fromarray(frames[-1]).save(OUT_PREVIEW)
    frames.extend([frames[-1]] * END_HOLD)
    imageio.mimsave(OUT_GIF, frames, duration=1 / FPS, loop=0,
                    palettesize=128, subrectangles=True)
    print(f"probe final position ({FINAL_P[0]:.3f}, {FINAL_P[1]:.3f})")
    print(f"wrote {OUT_GIF}")
    print(f"wrote {OUT_PREVIEW}")


if __name__ == "__main__":
    main()
