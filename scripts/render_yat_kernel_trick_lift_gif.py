#!/usr/bin/env python3
"""Render an animated 3D GIF of the kernel trick: lifting two rings.

A companion figure for the finite-feature-map post. Two concentric rings
(inner = class 0, outer = class 1) are not separable by any straight line
in 2-D. The exact degree-2 feature map sends (x1, x2) -> (x1, x2, x1²+x2²);
the squared-radius coordinate z pushes the inner ring low and the outer ring
high, so a single horizontal plane separates them. We animate the lift: a
morph parameter t scales z from a flat z=0 plane up onto the paraboloid, while
the camera slowly orbits. Once the rings have risen enough, a translucent
separating plane appears upstairs and its projected circle is drawn on the
floor — the flat plane upstairs is a circle downstairs.
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
OUT_GIF = ROOT / "public" / "yat-kernel-trick-lift.gif"
OUT_PREVIEW = ROOT / "public" / "yat-kernel-trick-lift-preview.png"

W, H = 1000, 620
FPS = 12
N_FRAMES = 48
END_HOLD = 14

BG = "#fbfaf6"
PANEL = "#ffffff"
INK = "#181818"
MUTED = "#666a70"
BORDER = "#ded9cb"
ACCENT = "#b3661b"
BLUE = "#4a7fb3"

N = 220
R_INNER = 0.55
R_OUTER = 1.25
NOISE = 0.05


def make_rings():
    """Two concentric rings; inner -> class 0, outer -> class 1."""
    key = jax.random.key(7)
    k_ang, k_r0, k_r1, k_n0, k_n1 = jax.random.split(key, 5)
    n0 = N // 2
    n1 = N - n0

    ang0 = jax.random.uniform(k_ang, (n0,), minval=0.0, maxval=2 * jnp.pi)
    ang1 = jax.random.uniform(k_r0, (n1,), minval=0.0, maxval=2 * jnp.pi)
    r0 = R_INNER + NOISE * jax.random.normal(k_r1, (n0,))
    r1 = R_OUTER + NOISE * jax.random.normal(k_n0, (n1,))

    x0 = jnp.stack([r0 * jnp.cos(ang0), r0 * jnp.sin(ang0)], axis=1)
    x1 = jnp.stack([r1 * jnp.cos(ang1), r1 * jnp.sin(ang1)], axis=1)
    x = jnp.concatenate([x0, x1], axis=0)
    y = jnp.concatenate([jnp.zeros(n0), jnp.ones(n1)], axis=0)

    noise2 = NOISE * jax.random.normal(k_n1, x.shape)
    x = x + noise2
    return x, y


X, Y = make_rings()
Z = (X[:, 0] ** 2 + X[:, 1] ** 2)  # squared radius = degree-2 lift height

X_np = np.asarray(X)
Y_np = np.asarray(Y)
Z_np = np.asarray(Z)

# class mean lifted heights -> threshold midpoint at full lift
ZMEAN0 = float(Z_np[Y_np == 0].mean())
ZMEAN1 = float(Z_np[Y_np == 1].mean())
THRESH_FULL = 0.5 * (ZMEAN0 + ZMEAN1)
ZMAX = float(Z_np.max())

# grid for the separating plane upstairs
GRID = 18
gx = np.linspace(-R_OUTER - 0.35, R_OUTER + 0.35, GRID)
PX, PY = np.meshgrid(gx, gx)

# circle on the floor (projected boundary)
theta = np.linspace(0, 2 * np.pi, 200)


def ease(t: float) -> float:
    return float(0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0.0, 1.0)))


def draw_frame(idx: int) -> np.ndarray:
    frac = idx / (N_FRAMES - 1)
    t = ease(frac)                       # morph 0 -> 1
    azim = -60 + 70 * frac               # slow orbit

    z_shown = t * Z_np
    thresh = t * THRESH_FULL

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    fig.text(
        0.5, 0.95,
        "The finite feature map: a flat plane upstairs is a circle downstairs",
        ha="center", color=INK, fontsize=15.5, weight="bold",
    )
    fig.text(
        0.5, 0.905,
        "rings are not separable by any line in 2-D, but a flat plane "
        "separates them after the degree-2 lift",
        ha="center", color=MUTED, fontsize=10.5,
    )

    ax = fig.add_axes([0.02, 0.02, 0.96, 0.84], projection="3d")
    ax.set_facecolor(BG)
    ax.view_init(elev=22, azim=azim)

    # clean, light panes
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((1.0, 1.0, 1.0, 1.0))
        pane.pane.set_edgecolor(BORDER)
        pane._axinfo["grid"]["color"] = (0.87, 0.85, 0.79, 0.6)
        pane._axinfo["grid"]["linewidth"] = 0.6

    # floor circle (projected decision boundary): x1²+x2² = thresh
    if t > 0.4 and thresh > 1e-4:
        rc = np.sqrt(thresh)
        ax.plot(rc * np.cos(theta), rc * np.sin(theta), 0.0,
                color=ACCENT, lw=2.0, ls="--", zorder=1)

    # translucent separating plane upstairs
    if t > 0.4:
        ax.plot_surface(
            PX, PY, np.full_like(PX, thresh),
            color=ACCENT, alpha=0.16, linewidth=0,
            antialiased=True, shade=False, zorder=2,
        )

    # points, colored by class
    m0 = Y_np == 0
    m1 = Y_np == 1
    ax.scatter(X_np[m0, 0], X_np[m0, 1], z_shown[m0],
               color=BLUE, s=16, depthshade=True,
               edgecolors="white", linewidths=0.3, zorder=5)
    ax.scatter(X_np[m1, 0], X_np[m1, 1], z_shown[m1],
               color=ACCENT, s=16, depthshade=True,
               edgecolors="white", linewidths=0.3, zorder=5)

    ax.set_xlim(-R_OUTER - 0.4, R_OUTER + 0.4)
    ax.set_ylim(-R_OUTER - 0.4, R_OUTER + 0.4)
    ax.set_zlim(-0.05, ZMAX * 1.05)
    ax.set_zticks([0, round(ZMAX / 2, 1), round(ZMAX, 1)])

    ax.set_xlabel("x₁", color=MUTED, fontsize=9, labelpad=-6)
    ax.set_ylabel("x₂", color=MUTED, fontsize=9, labelpad=-6)
    ax.set_zlabel("z = x₁² + x₂²", color=MUTED, fontsize=9, labelpad=-4)
    ax.tick_params(colors=MUTED, labelsize=7, pad=-2)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])

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
                    palettesize=160, subrectangles=True)
    print(f"inner z̄={ZMEAN0:.3f}  outer z̄={ZMEAN1:.3f}  thresh={THRESH_FULL:.3f}")
    print(f"wrote {OUT_GIF}")
    print(f"wrote {OUT_PREVIEW}")


if __name__ == "__main__":
    main()
