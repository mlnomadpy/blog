#!/usr/bin/env python3
"""Render a tweet-ready GIF for the gravity/attention bookkeeping analogy.

The math is computed with JAX:
- gravity: direct softened N-body accelerations and a particle-mesh Poisson solve
- attention: exact softmax scores and a finite-feature linear-attention state
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
OUT_GIF = ROOT / "public" / "gravity-attention-bookkeeping.gif"
OUT_PREVIEW = ROOT / "public" / "gravity-attention-bookkeeping-preview.png"

W, H = 1280, 720
FPS = 16
FRAMES = 64
N_GRAV = 44
N_TOK = 48
M_FEAT = 12
GRID = 15
G = 0.018
EPS2 = 0.018

BG = "#fbfaf6"
PANEL = "#ffffff"
INK = "#181818"
MUTED = "#666a70"
BORDER = "#ded9cb"
BLUE = "#4a7fb3"
ACCENT = "#b3661b"
GREEN = "#3a8f5e"


def poisson_matrix(g: int) -> jnp.ndarray:
    h = 2.0 / (g - 1)
    inv_h2 = 1.0 / (h * h)
    a = np.zeros((g * g, g * g), dtype=np.float64)
    for y in range(g):
        for x in range(g):
            k = y * g + x
            if x == 0 or y == 0 or x == g - 1 or y == g - 1:
                a[k, k] = 1.0
                continue
            a[k, k] = -4.0 * inv_h2
            a[k, k - 1] = inv_h2
            a[k, k + 1] = inv_h2
            a[k, k - g] = inv_h2
            a[k, k + g] = inv_h2
    return jnp.asarray(a)


LAPLACE = poisson_matrix(GRID)


def gravity_positions(t: float) -> tuple[np.ndarray, np.ndarray]:
    i = jnp.arange(N_GRAV, dtype=jnp.float64)
    base = i * 2.399963
    r = 0.13 + 0.76 * jnp.sqrt((i + 0.5) / N_GRAV)
    speed = 0.16 + 0.03 * (i % 5)
    x = jnp.cos(base + speed * t) * r
    y = jnp.sin(base * 0.96 + speed * t * 0.82) * r * (0.78 + 0.04 * (i % 3))
    pos = jnp.stack([x, y], axis=1)
    mass = 0.6 + ((i * 29) % 100) / 120.0
    return np.asarray(pos), np.asarray(mass)


def direct_gravity_accel(pos_np: np.ndarray, mass_np: np.ndarray) -> np.ndarray:
    pos = jnp.asarray(pos_np)
    mass = jnp.asarray(mass_np)
    d = pos[None, :, :] - pos[:, None, :]
    r2 = jnp.sum(d * d, axis=-1) + EPS2
    mask = 1.0 - jnp.eye(pos.shape[0], dtype=jnp.float64)
    inv = mask / (r2 * jnp.sqrt(r2))
    accel = jnp.sum(G * mass[None, :, None] * d * inv[:, :, None], axis=1)
    return np.asarray(accel)


def deposit_density(pos: np.ndarray, mass: np.ndarray) -> np.ndarray:
    rho = np.zeros((GRID, GRID), dtype=np.float64)
    scale = GRID - 1.001
    for (x, y), m in zip(pos, mass):
        gx = np.clip((x + 1.0) * 0.5, 0.0, 1.0) * scale
        gy = np.clip((y + 1.0) * 0.5, 0.0, 1.0) * scale
        x0, y0 = int(np.floor(gx)), int(np.floor(gy))
        tx, ty = gx - x0, gy - y0
        for dx, wx in ((0, 1 - tx), (1, tx)):
            for dy, wy in ((0, 1 - ty), (1, ty)):
                xx = min(GRID - 1, max(0, x0 + dx))
                yy = min(GRID - 1, max(0, y0 + dy))
                rho[yy, xx] += m * wx * wy
    cell_area = (2.0 / (GRID - 1)) ** 2
    rho /= cell_area
    rho[0, :] = 0.0
    rho[-1, :] = 0.0
    rho[:, 0] = 0.0
    rho[:, -1] = 0.0
    return rho


def field_potential(pos: np.ndarray, mass: np.ndarray) -> np.ndarray:
    rho = deposit_density(pos, mass)
    rhs = jnp.asarray((4.0 * np.pi * G * rho).reshape(-1))
    phi = jnp.linalg.solve(LAPLACE, rhs).reshape(GRID, GRID)
    return np.asarray(phi)


def attention_state(t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    i = jnp.arange(N_TOK, dtype=jnp.float64)
    angle = 2.0 * jnp.pi * i / N_TOK + 0.23 * jnp.sin(t * 0.7 + i * 0.13)
    pts = jnp.stack([jnp.cos(angle), jnp.sin(angle)], axis=1)
    q = jnp.stack(
        [jnp.cos(angle), jnp.sin(angle), jnp.cos(2 * angle + 0.2), jnp.sin(2 * angle - 0.1)],
        axis=1,
    ) * 1.35
    k = jnp.stack(
        [
            jnp.cos(angle + 0.18),
            jnp.sin(angle - 0.1),
            jnp.cos(2 * angle + 0.35),
            jnp.sin(2 * angle + 0.15),
        ],
        axis=1,
    ) * 1.35
    scores = q @ k.T / jnp.sqrt(q.shape[1])
    scores = scores - jnp.max(scores, axis=1, keepdims=True)
    attn = jnp.exp(scores)
    attn = attn / jnp.sum(attn, axis=1, keepdims=True)

    j = jnp.arange(M_FEAT, dtype=jnp.float64)
    wf = jnp.stack(
        [jnp.cos(j * 1.7), jnp.sin(j * 1.1), jnp.cos(j * 0.8 + 0.4), jnp.sin(j * 1.3 - 0.2)],
        axis=0,
    )
    phi = jnp.exp(k @ wf - 0.5 * jnp.sum(k * k, axis=1, keepdims=True)) / jnp.sqrt(M_FEAT)
    v = jnp.stack([jnp.sin(angle + t), jnp.cos(angle * 0.7 - t * 0.6)], axis=1)
    state = phi.T @ v
    state_strength = jnp.sqrt(jnp.sum(state * state, axis=1))
    return np.asarray(pts), np.asarray(attn), np.asarray(state_strength)


def setup_axis(ax, title: str, subtitle: str) -> None:
    ax.set_facecolor(PANEL)
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.text(0.5, 1.06, title, transform=ax.transAxes, ha="center", va="bottom", color=INK, fontsize=13, weight="bold")
    ax.text(0.5, 1.005, subtitle, transform=ax.transAxes, ha="center", va="bottom", color=MUTED, fontsize=9.5)


def draw_cost_card(ax, left_label: str, left_value: float, left_sub: str, right_label: str, right_value: float, right_sub: str, max_value: float) -> None:
    ax.set_facecolor(PANEL)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.90, "compute at the same N", ha="center", va="center", color=INK, fontsize=12, weight="bold")
    xs = [0.36, 0.64]
    vals = [left_value, right_value]
    labels = [left_label, right_label]
    subs = [left_sub, right_sub]
    cols = [BLUE, ACCENT]
    for x, v, label, sub, col in zip(xs, vals, labels, subs, cols):
        h = max(0.03, min(0.50, 0.50 * v / max_value))
        ax.add_patch(plt.Rectangle((x - 0.055, 0.20), 0.11, 0.50, color=col, alpha=0.12, lw=0))
        ax.add_patch(plt.Rectangle((x - 0.055, 0.20), 0.11, h, color=col, alpha=0.88, lw=0))
        ax.text(x, 0.15, label, ha="center", va="top", color=col, fontsize=9.5, weight="bold")
        ax.text(x, 0.06, sub, ha="center", va="top", color=MUTED, fontsize=8.5)
        ax.text(x, 0.20 + h + 0.025, f"{int(v):,}", ha="center", va="bottom", color=INK, fontsize=9.5)


def draw_frame(frame: int) -> np.ndarray:
    t = frame / FPS
    pos, mass = gravity_positions(t)
    accel = direct_gravity_accel(pos, mass)
    phi = field_potential(pos, mass)
    tok, attn, feat_state = attention_state(t)

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    fig.text(0.5, 0.965, "All-pairs ledgers vs shared fields", ha="center", va="center", color=INK, fontsize=24, weight="bold")
    fig.text(
        0.5,
        0.925,
        "Gravity and attention tell the same compute story: exact pairwise interactions are expensive; a shared state changes the scaling.",
        ha="center",
        va="center",
        color=MUTED,
        fontsize=12,
    )

    ax_g_pair = fig.add_axes([0.055, 0.52, 0.20, 0.31])
    ax_g_field = fig.add_axes([0.285, 0.52, 0.20, 0.31])
    ax_a_pair = fig.add_axes([0.545, 0.52, 0.20, 0.31])
    ax_a_field = fig.add_axes([0.775, 0.52, 0.20, 0.31])
    ax_g_cost = fig.add_axes([0.095, 0.12, 0.34, 0.28])
    ax_a_cost = fig.add_axes([0.585, 0.12, 0.34, 0.28])

    setup_axis(ax_g_pair, "Newtonian ledger", "sum every body-body force")
    edge_count = min(520, N_GRAV * (N_GRAV - 1))
    for e in range(edge_count):
        i = (e * 37 + frame) % N_GRAV
        j = (e * 83 + 11) % N_GRAV
        if i == j:
            continue
        alpha = 0.03 + 0.08 * ((e % 17) / 16)
        ax_g_pair.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], color=BLUE, alpha=alpha, lw=0.45)
    speed = np.linalg.norm(accel, axis=1)
    ax_g_pair.scatter(pos[:, 0], pos[:, 1], s=20 + 45 * mass, c=mass, cmap="copper", edgecolors=INK, linewidths=0.25, zorder=5)
    hot = int(np.argmax(speed))
    ax_g_pair.scatter([pos[hot, 0]], [pos[hot, 1]], s=95, facecolors="none", edgecolors=ACCENT, linewidths=1.4, zorder=6)

    setup_axis(ax_g_field, "Field solve", "deposit rho, solve potential")
    ax_g_field.imshow(phi, extent=(-1, 1, -1, 1), origin="lower", cmap="YlOrBr", alpha=0.74)
    gx, gy = np.gradient(-phi)
    xs = np.linspace(-1, 1, GRID)
    ys = np.linspace(-1, 1, GRID)
    xx, yy = np.meshgrid(xs, ys)
    ax_g_field.quiver(xx[1::2, 1::2], yy[1::2, 1::2], gx[1::2, 1::2], gy[1::2, 1::2], color=ACCENT, alpha=0.45, width=0.006)
    ax_g_field.scatter(pos[:, 0], pos[:, 1], s=18 + 35 * mass, color=GREEN, edgecolors=INK, linewidths=0.25, zorder=5)
    ax_g_field.text(0.5, -0.14, r"$\nabla^2\Phi=4\pi G\rho,\quad a=-\nabla\Phi$", transform=ax_g_field.transAxes, ha="center", color=MUTED, fontsize=9)

    setup_axis(ax_a_pair, "Softmax ledger", "score every query-key pair")
    tok_xy = tok * 0.82
    for e in range(700):
        i = (e * 29 + frame) % N_TOK
        j = (e * 71 + 7) % N_TOK
        alpha = 0.015 + 0.22 * float(attn[i, j])
        ax_a_pair.plot([tok_xy[i, 0], tok_xy[j, 0]], [tok_xy[i, 1], tok_xy[j, 1]], color=BLUE, alpha=alpha, lw=0.45)
    ax_a_pair.scatter(tok_xy[:, 0], tok_xy[:, 1], s=22, color=BLUE, edgecolors=INK, linewidths=0.25, zorder=5)
    ax_a_pair.scatter([tok_xy[-1, 0]], [tok_xy[-1, 1]], s=80, color=ACCENT, edgecolors=INK, linewidths=0.35, zorder=6)
    ax_a_pair.text(0.5, -0.14, r"$A=\mathrm{softmax}(QK^T)$", transform=ax_a_pair.transAxes, ha="center", color=MUTED, fontsize=9)

    setup_axis(ax_a_field, "Feature state", "write keys once, query state")
    ax_a_field.set_xlim(-1.08, 1.08)
    ax_a_field.set_ylim(-1.08, 1.08)
    cols = 4
    rows = int(np.ceil(M_FEAT / cols))
    strengths = feat_state / (feat_state.max() + 1e-9)
    for k in range(M_FEAT):
        c = k % cols
        r = k // cols
        x = -0.48 + c * 0.32
        y = 0.34 - r * 0.32
        ax_a_field.add_patch(plt.Rectangle((x, y), 0.22, 0.22, facecolor=ACCENT, alpha=0.18 + 0.68 * strengths[k], edgecolor=BORDER, lw=0.8))
    ax_a_field.text(0.5, 0.72, r"$\phi(K)^T V$", transform=ax_a_field.transAxes, ha="center", color=ACCENT, fontsize=12, weight="bold")
    stream_x = np.linspace(-0.9, 0.9, 18)
    for k, x in enumerate(stream_x):
        y = -0.78 + 0.045 * np.sin(t * 4 + k)
        ax_a_field.scatter([x], [y], s=18, color=GREEN if k < 17 else BLUE, zorder=4)
        slot = k % M_FEAT
        c = slot % cols
        r = slot // cols
        tx = -0.37 + c * 0.32
        ty = 0.45 - r * 0.32
        pulse = (t * 1.6 + k / len(stream_x)) % 1.0
        ax_a_field.plot([x, tx], [y, ty], color=GREEN, alpha=0.12, lw=0.8)
        ax_a_field.scatter([x + (tx - x) * pulse], [y + (ty - y) * pulse], s=10, color=GREEN, alpha=0.75)
    ax_a_field.annotate("", xy=(0.55, 0.02), xytext=(0.88, -0.58), arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1.8})
    ax_a_field.text(0.5, -0.14, r"$Y\approx\phi(Q)(\phi(K)^T V)$", transform=ax_a_field.transAxes, ha="center", color=MUTED, fontsize=9)

    grav_direct = N_GRAV * (N_GRAV - 1)
    grav_field = 8 * N_GRAV + GRID * GRID
    draw_cost_card(ax_g_cost, "direct", grav_direct, r"$N(N-1)$", "field", grav_field, r"$O(N)+grid$", N_GRAV * (N_GRAV - 1))

    attn_exact = N_TOK * N_TOK
    attn_linear = N_TOK * M_FEAT
    draw_cost_card(ax_a_cost, "softmax", attn_exact, r"$N\times N$", "linear", attn_linear, rf"$N\times m,\ m={M_FEAT}$", N_TOK * N_TOK)

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba


def main() -> None:
    frames = []
    for frame in range(FRAMES):
        frames.append(draw_frame(frame))
        if frame == 0:
            Image.fromarray(frames[-1]).save(OUT_PREVIEW)
        if (frame + 1) % 8 == 0:
            print(f"rendered {frame + 1}/{FRAMES} frames")

    imageio.mimsave(OUT_GIF, frames, duration=1 / FPS, loop=0, palettesize=128, subrectangles=True)
    print(f"wrote {OUT_GIF}")
    print(f"wrote {OUT_PREVIEW}")


if __name__ == "__main__":
    main()
