#!/usr/bin/env python3
"""Contrastive learning splits the structured state into two forces. Live in JAX we
train an encoder onto the 2-sphere (here drawn as the circle, D=2) with the
alignment+uniformity objective of Wang & Isola. The two forces run on different
clocks: UNIFORMITY (everything spread over the sphere) resolves first — the
ORGANIZED state — and while it spreads it actually flings positives apart, so
ALIGNMENT (positives pulled back together) finishes second — the STRUCTURED state.
We render the embedding on the circle plus the two loss curves on their separate
timelines, including alignment's tell-tale early hump."""
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
OUT_GIF = ROOT / "public" / "ts-align-uniform.gif"
OUT_PREVIEW = ROOT / "public" / "ts-align-uniform-preview.png"

W, H, FPS, FRAMES, EVERY = 1100, 520, 16, 80, 3
C, DIN, HID, NPC, AUG = 8, 16, 64, 40, 1.4
BG, PANEL, INK, MUTED, BORDER, ACCENT, BLUE, GREEN = "#fbfaf6", "#ffffff", "#181818", "#666a70", "#ded9cb", "#b3661b", "#4a7fb3", "#3a8f5e"
import matplotlib.cm as _cm  # noqa: E402
PAL = _cm.tab10(np.arange(C))

# Each class is a tight cluster in input space; two augmented views per sample.
rng = np.random.default_rng(5)
means = rng.normal(0, 1.2, (C, DIN))
base = np.concatenate([means[c] + rng.normal(0, 0.55, (NPC, DIN)) for c in range(C)]).astype(np.float32)
y = np.repeat(np.arange(C), NPC)
A = jnp.asarray(base + rng.normal(0, AUG, base.shape).astype(np.float32))   # view 1
B = jnp.asarray(base + rng.normal(0, AUG, base.shape).astype(np.float32))   # view 2
yj = jnp.asarray(y)


def init(key):
    k = jax.random.split(key, 3)
    he = lambda kk, a, b: jax.random.normal(kk, (a, b)) * np.sqrt(2.0 / a)
    return dict(W1=he(k[0], DIN, HID), b1=jnp.zeros(HID), W2=he(k[1], HID, HID), b2=jnp.zeros(HID),
                W3=he(k[2], HID, 2), b3=jnp.zeros(2))


def enc(p, x):
    h = jax.nn.relu(x @ p["W1"] + p["b1"]); h = jax.nn.relu(h @ p["W2"] + p["b2"])
    z = h @ p["W3"] + p["b3"]
    return z / (jnp.linalg.norm(z, axis=1, keepdims=True) + 1e-8)   # project to unit circle


def align_loss(za, zb):
    return (jnp.sum((za - zb) ** 2, axis=1)).mean()                 # ‖f(x)−f(x⁺)‖²


def uniform_loss(z):
    d2 = jnp.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=-1)     # pairwise sq dist
    n = z.shape[0]
    mask = 1.0 - jnp.eye(n)
    mean_off = (jnp.exp(-2.0 * d2) * mask).sum() / (n * (n - 1))
    return jnp.log(mean_off)                                        # Wang-Isola uniformity (t=2)


def loss_fn(p):
    za, zb = enc(p, A), enc(p, B)
    return align_loss(za, zb) + uniform_loss(jnp.concatenate([za, zb], 0))


params = init(jax.random.key(0)); opt = optax.adam(6e-3); state = opt.init(params)


@jax.jit
def step(p, st):
    g = jax.grad(loss_fn)(p); up, st = opt.update(g, st); return optax.apply_updates(p, up), st


hist = []


def draw(frame):
    global params, state
    for _ in range(EVERY):
        params, state = step(params, state)
    za = np.asarray(enc(params, A)); zb = np.asarray(enc(params, B))
    la = float(align_loss(jnp.asarray(za), jnp.asarray(zb)))
    lu = float(uniform_loss(jnp.asarray(np.concatenate([za, zb], 0))))
    hist.append((la, lu))

    # state label: uniformity (spread) resolves first (organized), then alignment (structured)
    if lu > -1.4:
        state_lbl, col = "RANDOM", MUTED
    elif la > 0.02:
        state_lbl, col = "ORGANIZED  ·  spread, positives still loose", BLUE
    else:
        state_lbl, col = "STRUCTURED  ·  uniform + aligned", GREEN

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    fig.text(0.5, 0.95, "Contrastive learning: spread first, then snap together", ha="center", color=INK, fontsize=18, weight="bold")
    fig.text(0.5, 0.905, "the code spreads to fill the sphere (organized) before positives lock back together (structured)", ha="center", color=MUTED, fontsize=11.5)

    ax = fig.add_axes([0.055, 0.1, 0.42, 0.76]); ax.set_facecolor(PANEL); ax.set_aspect("equal")
    th = np.linspace(0, 2 * np.pi, 200); ax.plot(np.cos(th), np.sin(th), color=BORDER, lw=1.2, zorder=1)
    for c in range(C):
        ax.scatter(za[y == c, 0], za[y == c, 1], s=22, color=PAL[c], edgecolors=PANEL, linewidths=0.4, zorder=3)
        ax.scatter(zb[y == c, 0], zb[y == c, 1], s=22, color=PAL[c], edgecolors=PANEL, linewidths=0.4, zorder=3)
    ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25); ax.set_xticks([]); ax.set_yticks([]); [sp.set_color(BORDER) for sp in ax.spines.values()]
    ax.set_title("embedding on the unit circle", color=INK, fontsize=11, weight="bold", pad=6)

    axc = fig.add_axes([0.6, 0.18, 0.36, 0.56]); axc.set_facecolor(PANEL)
    las = np.array([h[0] for h in hist]); lus = np.array([h[1] for h in hist]); xs = np.arange(len(hist))
    las_n = las / (las.max() + 1e-9)
    lus_n = (lus - lus.min()) / (lus.max() - lus.min() + 1e-9)
    axc.plot(xs, lus_n, color=ACCENT, lw=2, label="uniformity  (spread, fast)")
    axc.plot(xs, las_n, color=BLUE, lw=2, label="alignment  ‖f(x)−f(x⁺)‖²  (slow)")
    axc.set_xlim(0, FRAMES); axc.set_ylim(-0.05, 1.05); axc.set_xticks([]); axc.set_yticks([])
    [sp.set_color(BORDER) for sp in axc.spines.values()]
    axc.legend(loc="upper right", fontsize=8.5, frameon=False)
    axc.set_title("two losses, two clocks", color=INK, fontsize=10.5, weight="bold", pad=6)

    fig.text(0.5, 0.045, state_lbl, ha="center", color=col, fontsize=14, weight="bold")
    fig.canvas.draw(); rgba = np.asarray(fig.canvas.buffer_rgba()).copy(); plt.close(fig); return rgba


def main():
    frames = [draw(i) for i in range(FRAMES)]
    Image.fromarray(frames[-1]).save(OUT_PREVIEW)
    imageio.mimsave(OUT_GIF, frames, duration=1 / FPS, loop=0, palettesize=128, subrectangles=True)
    print("wrote", OUT_GIF)


if __name__ == "__main__":
    main()
