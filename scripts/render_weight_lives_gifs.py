"""Faithful figures for the "Where Does a Weight Live?" JAX companion.

Two GIFs animate a real computation and one still figure shows a static
comparison; no synthetic metaphors:

  1. weight-lives-representer.gif  a real kernel-ridge weight assembling from the
                                   data, term by term: f = sum_i alpha_i k(x_i, .)
                                   accumulates and its boundary forms; accuracy is
                                   the real partial-sum accuracy.
  2. weight-lives-rings.png        (static figure) nested circles: the best linear
                                   weight (logistic regression, a direction) stays at
                                   chance, while the kernel weight (a placed
                                   combination) separates them. Real accuracies, no
                                   temporal process, so a still figure.
  3. weight-lives-lift.gif         the kernel lift made literal: circles rise by
                                   z = ||x||^2 until a flat plane (a weight you can
                                   place) cuts inner from outer. Real separation.

Run: python scripts/render_weight_lives_gifs.py
"""
import warnings; warnings.filterwarnings('ignore')
import os
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
from sklearn.datasets import make_moons, make_circles
from sklearn.linear_model import LogisticRegression
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import imageio.v2 as imageio
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]; PUB = ROOT / 'public'
BG = '#0e0d0b'; PANEL = '#16140f'; INK = '#e8e2d4'; MUTED = '#9a9282'; LINE = '#3a352c'
A_COL = '#b3661b'; B_COL = '#4a7fb3'; ACC = '#e0a45a'; GOOD = '#7bbf5a'
plt.rcParams.update({'figure.facecolor': BG, 'savefig.facecolor': BG, 'text.color': INK,
                     'axes.edgecolor': LINE, 'font.size': 11})


def ease(t): return t * t * (3 - 2 * t)


def fig_rgba(fig):
    fig.canvas.draw(); rgba = np.asarray(fig.canvas.buffer_rgba()).copy(); plt.close(fig); return rgba


def save_gif(path, frames, fps, hold=14):
    Image.fromarray(frames[-1]).save(str(path).replace('.gif', '-preview.png'))
    imageio.mimsave(path, frames + [frames[-1]] * hold, duration=1 / fps, loop=0,
                    palettesize=128, subrectangles=True)
    print(f'wrote {path} ({os.path.getsize(path)//1024} KB)')


def save_png(path, fig):
    fig.savefig(path, dpi=110, facecolor=BG)
    plt.close(fig)
    print(f'wrote {path} ({os.path.getsize(path)//1024} KB)')


SIG = 0.5
@jax.jit
def gram(X, Y):                                        # RBF kernel matrix, real
    d2 = (X[:, None, :] - Y[None, :, :]) ** 2
    return jnp.exp(-d2.sum(-1) / (2 * SIG ** 2))


def kernel_ridge(X, y, lam=0.4):                       # representer weights alpha, real solve
    G = gram(X, X)
    alpha = jnp.linalg.solve(G + lam * jnp.eye(len(X)), y)
    return np.asarray(alpha)


# ═══════════════════════════════════════════════════════════════════════════
# GIF 1 — the weight assembling from the data, term by term
# ═══════════════════════════════════════════════════════════════════════════
def gif_representer():
    X, y = make_moons(140, noise=0.16, random_state=0); ys = np.where(y == 1, 1.0, -1.0)
    Xj = jnp.asarray(X, jnp.float32)
    alpha = kernel_ridge(Xj, jnp.asarray(ys, jnp.float32))
    gx, gy = np.meshgrid(np.linspace(-1.7, 2.7, 150), np.linspace(-1.3, 1.8, 150))
    GP = jnp.asarray(np.stack([gx.ravel(), gy.ravel()], 1), jnp.float32)
    Kgrid = np.asarray(gram(GP, Xj))                   # [grid, n] kernel of each grid pt to each data pt
    order = np.argsort(-np.abs(alpha))                 # add the heaviest terms first
    steps = list(range(1, len(order) + 1, 5)) + [len(order)]
    Kdata = np.asarray(gram(Xj, Xj))
    frames = []
    for k in steps:
        idx = order[:k]
        field = (Kgrid[:, idx] @ alpha[idx]).reshape(gx.shape)
        acc = 100 * np.mean((np.sign(Kdata[:, idx] @ alpha[idx]) == ys))
        fig = plt.figure(figsize=(6.0, 5.6), dpi=100, facecolor=BG)
        fig.text(0.5, 0.95, 'The weight assembles from the data', ha='center', fontsize=14, weight='bold')
        fig.text(0.5, 0.905, 'f(x) = Σ αᵢ k(xᵢ, x): each data point adds its own kernel bump; the weight lives in their span',
                 ha='center', fontsize=8.6, color=MUTED)
        ax = fig.add_axes([0.06, 0.06, 0.88, 0.80]); ax.set_facecolor(PANEL)
        mx = np.abs(field).max() + 1e-9
        ax.contourf(gx, gy, field / mx, levels=np.linspace(-1, 1, 22), cmap='RdBu_r', alpha=0.85, vmin=-1, vmax=1)
        ax.contour(gx, gy, field, levels=[0], colors=[INK], linewidths=1.8)
        # faint all data, bright the contributing centers
        ax.scatter(X[:, 0], X[:, 1], s=12, c=[A_COL if t > 0 else B_COL for t in ys], alpha=0.25, linewidths=0)
        ax.scatter(X[idx, 0], X[idx, 1], s=30 + 120 * np.abs(alpha[idx]) / (np.abs(alpha).max()),
                   c=[A_COL if ys[i] > 0 else B_COL for i in idx], edgecolors='white', linewidths=0.5, zorder=3)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-1.7, 2.7); ax.set_ylim(-1.3, 1.8)
        for s in ax.spines.values(): s.set_color(LINE)
        ax.set_title(f'{k}/{len(order)} terms · the weight reads the data at {acc:.0f}%', fontsize=10.5, color=INK, pad=5)
        frames.append(fig_rgba(fig))
    save_gif(PUB / 'weight-lives-representer.gif', frames, fps=11)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — rings: a direction (best line) vs a placed kernel weight.
# A static comparison: two fitted readouts on the same nested rings. Nothing here
# is a temporal process — the endpoint accuracies (50% and 100%) are the whole
# story — so this is a still figure, not an animation.
# ═══════════════════════════════════════════════════════════════════════════
def png_rings():
    X, y = make_circles(260, noise=0.07, factor=0.4, random_state=0); ys = np.where(y == 1, 1.0, -1.0)
    Xj = jnp.asarray(X, jnp.float32)
    # the kernel weight: real kernel-ridge, separates the rings
    alpha = kernel_ridge(Xj, jnp.asarray(ys, jnp.float32), lam=0.2)
    gx, gy = np.meshgrid(np.linspace(-1.6, 1.6, 160), np.linspace(-1.6, 1.6, 160))
    GP = jnp.asarray(np.stack([gx.ravel(), gy.ravel()], 1), jnp.float32)
    kfield = np.asarray(gram(GP, Xj) @ alpha).reshape(gx.shape)
    kacc = 100 * np.mean(np.sign(np.asarray(gram(Xj, Xj) @ alpha)) == ys)
    # the best linear weight: logistic regression on raw x,y (a direction)
    lin = LogisticRegression().fit(X, y); lin_acc = 100 * lin.score(X, y)
    w = lin.coef_[0]; b = lin.intercept_[0]
    fig = plt.figure(figsize=(8.6, 4.7), dpi=110, facecolor=BG)
    fig.text(0.5, 0.95, 'A direction cannot, a placed weight can', ha='center', fontsize=14, weight='bold')
    fig.text(0.5, 0.90, 'the same nested rings, read by the best linear weight (a direction) and by the kernel weight (a placed combination)',
             ha='center', fontsize=8.4, color=MUTED)
    for side, kind in enumerate(['line', 'kernel']):
        ax = fig.add_axes([0.04 + side * 0.50, 0.07, 0.44, 0.76]); ax.set_facecolor(PANEL)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.set_aspect('equal')
        for s in ax.spines.values(): s.set_color(LINE)
        if kind == 'line':
            xs = np.array([-1.6, 1.6])
            if abs(w[1]) > 1e-6:
                ys_line = -(w[0] * xs + b) / w[1]
                ax.plot(xs, ys_line, color=INK, lw=2.0)
            zz = (w[0] * gx + w[1] * gy + b)
            ax.contourf(gx, gy, np.sign(zz), levels=[-2, 0, 2], colors=[B_COL, A_COL], alpha=0.10)
            ttl = f'a direction · {lin_acc:.0f}%'
        else:
            ax.contourf(gx, gy, kfield, levels=22, cmap='RdBu_r', alpha=0.80, vmin=-np.abs(kfield).max(), vmax=np.abs(kfield).max())
            ax.contour(gx, gy, kfield, levels=[0], colors=[INK], linewidths=1.8)
            ttl = f'a placed weight · {kacc:.0f}%'
        ax.scatter(X[:, 0], X[:, 1], s=12, c=[A_COL if v > 0 else B_COL for v in ys], alpha=0.85, linewidths=0, zorder=3)
        col = GOOD if (kind == 'kernel') else MUTED
        ax.set_title(ttl, fontsize=11, color=col, weight='bold', pad=5)
    save_png(PUB / 'weight-lives-rings.png', fig)


# ═══════════════════════════════════════════════════════════════════════════
# GIF 3 — the reproducing property: every prediction is a similarity-weighted
#          vote of the data, f(q) = Σ αᵢ k(xᵢ, q). No explicit lift; this is
#          exactly what the kernel-ridge code computes.
# ═══════════════════════════════════════════════════════════════════════════
def gif_vote():
    X, y = make_moons(140, noise=0.16, random_state=0); ys = np.where(y == 1, 1.0, -1.0)
    Xj = jnp.asarray(X, jnp.float32)
    alpha = kernel_ridge(Xj, jnp.asarray(ys, jnp.float32))
    gx, gy = np.meshgrid(np.linspace(-1.7, 2.7, 150), np.linspace(-1.3, 1.8, 150))
    GP = jnp.asarray(np.stack([gx.ravel(), gy.ravel()], 1), jnp.float32)
    field = np.asarray(gram(GP, Xj) @ alpha).reshape(gx.shape)
    fmax = np.abs(field).max() + 1e-9
    # a query path that crosses from the upper moon into the lower moon
    A, B = np.array([-0.7, 0.75]), np.array([1.7, -0.35])
    NF = 40; frames = []
    for fi in range(NF):
        t = ease(min(fi, NF - 1) / (NF - 1)); q = A + (B - A) * t
        kq = np.asarray(jnp.exp(-((Xj - jnp.asarray(q)) ** 2).sum(1) / (2 * SIG ** 2)))  # k(xᵢ, q), real
        contrib = alpha * kq                            # each point's signed vote
        fq = contrib.sum()                              # f(q) = Σ αᵢ k(xᵢ, q)
        top = np.argsort(-kq)[:14]                      # the most-similar points dominate
        fig = plt.figure(figsize=(6.0, 5.7), dpi=104, facecolor=BG)
        fig.text(0.5, 0.95, 'Every prediction is a similarity-weighted vote', ha='center', fontsize=13.5, weight='bold')
        fig.text(0.5, 0.905, 'f(q) = Σ αᵢ k(xᵢ, q): the query is read by its kernel similarity to the data, no coordinates built, no lift',
                 ha='center', fontsize=8.2, color=MUTED)
        ax = fig.add_axes([0.06, 0.06, 0.88, 0.80]); ax.set_facecolor(PANEL)
        ax.contourf(gx, gy, field / fmax, levels=np.linspace(-1, 1, 22), cmap='RdBu_r', alpha=0.35, vmin=-1, vmax=1)
        ax.contour(gx, gy, field, levels=[0], colors=[INK], linewidths=1.2, alpha=0.6)
        ax.scatter(X[:, 0], X[:, 1], s=12, c=[A_COL if v > 0 else B_COL for v in ys], alpha=0.4, linewidths=0)
        # the vote: a link to each dominant point, width ∝ similarity, color ∝ sign of its vote
        kmax = kq[top].max()
        for i in top:
            w = 0.4 + 3.6 * kq[i] / kmax
            col = A_COL if contrib[i] >= 0 else B_COL
            ax.plot([q[0], X[i, 0]], [q[1], X[i, 1]], color=col, lw=w, alpha=0.55, zorder=2)
            ax.scatter([X[i, 0]], [X[i, 1]], s=24, c=col, edgecolors='white', linewidths=0.4, zorder=3)
        cls = A_COL if fq >= 0 else B_COL
        ax.scatter([q[0]], [q[1]], s=150, marker='*', color='white', edgecolors=cls, linewidths=2.2, zorder=5)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-1.7, 2.7); ax.set_ylim(-1.3, 1.8)
        for s in ax.spines.values(): s.set_color(LINE)
        ax.set_title(f'f(q) = {fq:+.2f}  →  {"orange" if fq >= 0 else "blue"} class', fontsize=11, color=cls, weight='bold', pad=5)
        frames.append(fig_rgba(fig))
    save_gif(PUB / 'weight-lives-vote.gif', frames, fps=13, hold=14)


if __name__ == '__main__':
    gif_representer()
    png_rings()
    gif_vote()
    print('WEIGHT_LIVES_GIFS_DONE')
