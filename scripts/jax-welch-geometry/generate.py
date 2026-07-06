"""Generate GIFs for the Welch/JAX latent-geometry companion post.

Outputs land in public/jax-welch/ and are served by Astro at
/blog/jax-welch/<name>.gif.
"""

from __future__ import annotations

import argparse
import os
import textwrap

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch  # noqa: E402

BG = "#fbfaf6"
PANEL = "#ffffff"
BORDER = "#e4e1d6"
FG = "#1a1a1a"
MUTED = "#5a5f66"
ACCENT = "#b3661b"
BLUE = "#4a7fb3"
GREEN = "#3a8f5e"
RED = "#c2553a"
PALETTE = ["#b3661b", "#4a7fb3", "#3a8f5e", "#9a4f9c", "#c2553a", "#5a5f66"]

HERE = os.path.dirname(__file__)
DEFAULT_OUT = os.path.normpath(os.path.join(HERE, "..", "..", "public", "jax-welch"))


def l2_normalize(x, eps=1e-8):
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)


def gram(x):
    x = l2_normalize(x)
    return x @ x.T


def coherence(x):
    g = gram(x)
    n = g.shape[0]
    return jnp.max(jnp.abs(g - jnp.eye(n, dtype=g.dtype)))


def welch_floor(c, d):
    c = jnp.asarray(c, dtype=jnp.float32)
    d = jnp.asarray(d, dtype=jnp.float32)
    return jnp.sqrt(jnp.maximum(c - d, 0.0) / (d * jnp.maximum(c - 1.0, 1.0)))


def effective_rank(x, eps=1e-12):
    x = x - jnp.mean(x, axis=0, keepdims=True)
    cov = (x.T @ x) / jnp.maximum(x.shape[0] - 1, 1)
    eigs = jnp.clip(jnp.linalg.eigvalsh(cov), 0.0)
    p = eigs / (jnp.sum(eigs) + eps)
    return jnp.exp(-jnp.sum(jnp.where(p > 0, p * jnp.log(p + eps), 0.0)))


def simplex_gram(c, dtype=jnp.float32):
    eye = jnp.eye(c, dtype=dtype)
    return eye + (1.0 - eye) * (-1.0 / (c - 1))


def simplex_error(x):
    g = gram(x)
    target = simplex_gram(g.shape[0], g.dtype)
    return jnp.sqrt(jnp.mean((g - target) ** 2))


def frame_potential(x):
    g = gram(x)
    return jnp.sum(g * g)


def simplex_loss(x):
    return simplex_error(x) ** 2


def smooth_coherence_loss(x, beta=30.0):
    g = gram(x)
    n = g.shape[0]
    eye = jnp.eye(n, dtype=bool)
    vals = jnp.where(eye, -jnp.inf, jnp.abs(g))
    smooth_max = jax.nn.logsumexp(beta * vals) / beta
    cov = (x.T @ x) / n
    tight = jnp.sum((cov - jnp.eye(x.shape[1]) / x.shape[1]) ** 2)
    return smooth_max + 0.15 * tight


# Each mode is a (loss, optimizer) pair. SGD reproduces plain projected gradient
# descent for the simplex; Adam is scale-robust and reliably escapes the
# symmetric saddles of the coherence landscape for the over-complete Welch case.
LOSSES = {"simplex": simplex_loss, "welch": smooth_coherence_loss}
OPTIMS = {"simplex": optax.sgd, "welch": optax.adam}
LRS = {"simplex": 0.04, "welch": 0.05}


def descent_trajectory(x0, loss_fn, opt, steps):
    """Projected-gradient descent on the product of spheres as a single
    jax.lax.scan. Returns the whole trajectory incl. the start: (steps+1, C, d).
    Pure and side-effect free, so it vmaps cleanly over a batch of seeds."""
    opt_state = opt.init(x0)

    def body(carry, _):
        x, state = carry
        _, grad = jax.value_and_grad(loss_fn)(x)
        updates, state = opt.update(grad, state, x)
        x = l2_normalize(optax.apply_updates(x, updates))
        return (x, state), x

    _, xs = jax.lax.scan(body, (x0, opt_state), None, length=steps)
    return jnp.concatenate([x0[None], xs], axis=0)


@jax.jit
def trajectory_metrics(xs):
    """Audit every snapshot of a trajectory in one vmapped, jitted pass."""
    return {
        "coh": jax.vmap(coherence)(xs),
        "rank": jax.vmap(effective_rank)(xs),
        "simplex": jax.vmap(simplex_error)(xs),
        "fp": jax.vmap(frame_potential)(xs),
        "gram": jax.vmap(gram)(xs),
    }


def converged_step(series, eps, tail):
    """Index by which `series` has perceptually settled: the last step whose
    deviation from the final value exceeds eps, plus a short tail margin. Lets the
    animation stop where learning stops instead of idling on a converged frame."""
    series = np.asarray(series)
    beyond = np.where(np.abs(series - series[-1]) > eps)[0]
    t = (int(beyond[-1]) + 1) if len(beyond) else 0
    return min(len(series) - 1, t + tail)


def frontloaded(trim, n_frames, power=2.0):
    """Frame indices over [0, trim], dense early where the descent is fast."""
    u = np.linspace(0.0, 1.0, n_frames) ** power
    return np.unique(np.clip(np.round(u * trim).astype(int), 0, trim))


def run(key, c, d, *, mode, steps=900, n_frames=58, lr=None):
    x0 = l2_normalize(jax.random.normal(key, (c, d)))
    opt = OPTIMS[mode](LRS[mode] if lr is None else lr)
    xs = descent_trajectory(x0, LOSSES[mode], opt, steps)
    # trim to perceptual convergence of the coherence, then sample frames
    # front-loaded so most of them land on the actual descent, not the tail.
    coh_full = np.asarray(jax.vmap(coherence)(xs))
    trim = converged_step(coh_full, eps=0.012, tail=10)
    idx = frontloaded(trim, n_frames)
    xs = xs[idx]
    m = trajectory_metrics(xs)
    xs_np, g_np = np.asarray(xs), np.asarray(m["gram"])
    coh, rank, simx, fp = (np.asarray(m[k]) for k in ("coh", "rank", "simplex", "fp"))
    return [
        {
            "t": int(idx[i]),
            "x": xs_np[i],
            "g": g_np[i],
            "coh": float(coh[i]),
            "rank": float(rank[i]),
            "simplex": float(simx[i]),
            "fp": float(fp[i]),
        }
        for i in range(len(idx))
    ]


def run_many(key, c, d, *, n_seeds=6, steps=600, n_frames=56, lr=0.05):
    """Benign-landscape experiment. Benedetto & Fickus (2003) proved the frame
    potential Σ⟨eᵢ,eⱼ⟩² has no bad local minima, so we descend *it* from n_seeds
    independent random starts at once — a single vmapped scan — and watch every
    seed reach the same floor C²/d − C. Returns the per-seed frame-potential
    curves and the step axis, trimmed to where the curves settle."""
    x0s = l2_normalize(jax.random.normal(key, (n_seeds, c, d)))
    opt = optax.adam(lr)
    batched = jax.vmap(lambda x0: descent_trajectory(x0, frame_potential, opt, steps))
    xs = batched(x0s)  # (n_seeds, steps + 1, C, d)
    # frame_potential sums all i,j; the unit-norm diagonal contributes a constant
    # C, so subtract it for the classic off-diagonal frame potential (floor C²/d − C).
    fp_full = np.asarray(jax.vmap(jax.vmap(frame_potential))(xs)) - c  # (n_seeds, steps+1)
    trim = converged_step(fp_full.max(0), eps=0.05, tail=8)
    idx = frontloaded(trim, n_frames)
    return idx, fp_full[:, idx]


def project(x, phase=0.0):
    """Orthographic projection of unit vectors to the screen plane, also
    returning a per-point depth (+1 toward the viewer) for shading and ordering.
    For d == 2 the points are returned unchanged with zero depth."""
    if x.shape[1] == 2:
        return np.asarray(x[:, :2]), np.zeros(len(x))
    a = -0.42
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(phase), np.sin(phase)
    pts, depth = [], []
    for p in x:
        x1 = p[0] * cb + p[2] * sb
        z1 = -p[0] * sb + p[2] * cb
        y2 = p[1] * ca - z1 * sa
        z2 = p[1] * sa + z1 * ca
        pts.append([x1, y2])
        depth.append(z2)
    return np.asarray(pts), np.asarray(depth)


# ── tiny UI toolkit ─────────────────────────────────────────────────────────
# Everything is laid out in pixels on one full-figure background axis whose data
# range equals the pixel size, so rounded "cards" have circular corners and text
# lands exactly where intended — a real layout instead of subplot soup.
from matplotlib.patches import FancyBboxPatch  # noqa: E402

MONO = "DejaVu Sans Mono"


def new_canvas(W, H, dpi=110):
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    fig.patch.set_facecolor(BG)
    bg = fig.add_axes([0, 0, 1, 1])
    bg.set_xlim(0, W)
    bg.set_ylim(0, H)
    bg.axis("off")
    bg.set_facecolor(BG)
    return fig, bg


def card(bg, x, y, w, h, *, fill=PANEL, ec=BORDER, lw=1.2, r=16, accent=None):
    bg.add_patch(FancyBboxPatch((x + r, y + r), w - 2 * r, h - 2 * r,
                                boxstyle=f"round,pad={r},rounding_size={r}",
                                fc=fill, ec=ec, lw=lw, mutation_aspect=1, zorder=1))
    if accent:
        bg.add_patch(FancyBboxPatch((x + 3, y + 8), 0.1, h - 16,
                                    boxstyle="round,pad=2.5,rounding_size=2.5",
                                    fc=accent, ec="none", zorder=2))


def axes_in(fig, W, H, x, y, w, h):
    ax = fig.add_axes([x / W, y / H, w / W, h / H])
    ax.set_facecolor("none")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def kicker(bg, x, y, text):
    bg.text(x, y, text, fontsize=11, color=ACCENT, family=MONO, fontweight="bold", va="center", ha="left")


def card_label(bg, x, y, text):
    bg.text(x, y, text, fontsize=11.5, color=FG, fontweight="bold", va="center", ha="left")


def stat_card(bg, x, y, w, h, label, value, sub, color, ok):
    card(bg, x, y, w, h, accent=(GREEN if ok else color))
    bg.text(x + 22, y + h - 26, label.upper(), fontsize=10, color=MUTED, family=MONO, va="center", ha="left")
    bg.text(x + 20, y + h * 0.44, value, fontsize=33, color=(GREEN if ok else color),
            fontweight="bold", va="center", ha="left")
    bg.text(x + 22, y + 22, sub, fontsize=10.5, color=(GREEN if ok else MUTED), va="center", ha="left")
    if ok:
        bg.text(x + w - 24, y + h - 26, "✓", fontsize=15, color=GREEN, fontweight="bold", va="center", ha="right")


def panel_card(bg, fig, W, H, x, y, w, h, label, *, corner=None, pad_top=46, pad_bot=22):
    card(bg, x, y, w, h)
    card_label(bg, x + 24, y + h - 26, label)
    if corner:
        bg.text(x + w - 24, y + h - 26, corner, fontsize=10.5, color=MUTED, family=MONO, va="center", ha="right")
    return axes_in(fig, W, H, x + 18, y + pad_bot, w - 36, h - pad_top - pad_bot)


def finish(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def plot_sphere(ax, x, phase, g):
    pts, depth = project(x, phase)
    ax.set_aspect("equal")
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.18, 1.18)
    ax.add_patch(Circle((0, 0), 1.0, fill=False, ec=BORDER, lw=1.1, ls=(0, (2, 3)), zorder=1))
    th = np.linspace(0, 2 * np.pi, 90)
    eq = np.stack([np.cos(th), np.zeros_like(th), np.sin(th)], axis=1)
    eqp, _ = project(eq, phase)
    ax.plot(eqp[:, 0], eqp[:, 1], color=BORDER, lw=1.0, alpha=0.6, zorder=1)
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            s = abs(float(g[i, j]))
            ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]],
                    color=ACCENT, lw=1.0, alpha=0.05 + 0.5 * s, zorder=2)
    for k in np.argsort(depth):
        t = (depth[k] + 1.0) / 2.0
        ax.scatter(pts[k, 0], pts[k, 1], s=80 + 110 * t, color=PALETTE[k % len(PALETTE)],
                   edgecolor="white", linewidth=1.4, alpha=0.5 + 0.5 * t, zorder=3 + t)


def plot_gram(ax, g):
    n = g.shape[0]
    ax.imshow(g, vmin=-1, vmax=1, cmap="coolwarm", extent=[0, n, 0, n], origin="upper", aspect="auto")
    for k in range(n + 1):
        ax.plot([0, n], [k, k], color=PANEL, lw=1.4)
        ax.plot([k, k], [0, n], color=PANEL, lw=1.4)
    if n <= 4:
        for i in range(n):
            for j in range(n):
                v = float(g[i, j])
                ax.text(j + 0.5, n - i - 0.5, f"{v:+.2f}", ha="center", va="center",
                        fontsize=9.5, color="white" if abs(v) > 0.5 else FG)
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)


def plot_spark(ax, series, floor):
    n = len(series)
    ymax = max(max(series) if series else floor, floor) * 1.14 + 1e-6
    ax.set_xlim(0, max(n - 1, 1))
    ax.set_ylim(0, ymax)
    ax.axhspan(0, floor, color=GREEN, alpha=0.06)
    ax.axhline(floor, color=GREEN, lw=1.5, ls=(0, (4, 3)))
    if n >= 2:
        ax.plot(range(n), series, color=ACCENT, lw=2.4, solid_capstyle="round")
        ax.fill_between(range(n), series, ymax, color=ACCENT, alpha=0.04)
    if n:
        ax.scatter(n - 1, series[-1], s=30, color=ACCENT, zorder=4, edgecolor="white", linewidth=1.2)


def draw_frame(hist, idx, *, mode, c, d, title, subtitle, total_steps, dpi=110):
    W, H = 1140, 760
    h = hist[idx]
    fig, bg = new_canvas(W, H, dpi)

    # header
    kicker(bg, 40, H - 32, f"EXPERIMENT · C={c} · d={d} · JAX / optax")
    bg.text(40, H - 62, title, fontsize=23, color=FG, fontweight="bold", va="center", ha="left")
    for i, line in enumerate(textwrap.wrap(subtitle, width=92)[:2]):
        bg.text(40, H - 96 - i * 22, line, fontsize=12.5, color=MUTED, va="center", ha="left")

    top = H - 150          # top of the content cards
    stat_y, stat_h = 40, 170
    body_bot = stat_y + stat_h + 18

    # ── hero: the sphere ──
    sx, sw = 40, 480
    sy, sh = body_bot, top - body_bot
    card(bg, sx, sy, sw, sh)
    card_label(bg, sx + 24, sy + sh - 28, "codes on the sphere")
    bg.text(sx + sw - 24, sy + sh - 28, f"step {h['t']}", fontsize=10.5, color=MUTED, family=MONO, va="center", ha="right")
    ax_geo = axes_in(fig, W, H, sx + 16, sy + 44, sw - 32, sh - 92)
    plot_sphere(ax_geo, h["x"], idx * 0.05, h["g"])
    frac = h["t"] / max(total_steps, 1)
    pbx, pby, pbw = sx + 26, sy + 24, sw - 52
    bg.add_patch(FancyBboxPatch((pbx, pby), pbw, 8, boxstyle="round,pad=0,rounding_size=4", fc=BORDER, ec="none", alpha=0.5, zorder=2))
    bg.add_patch(FancyBboxPatch((pbx, pby), max(pbw * frac, 8), 8, boxstyle="round,pad=0,rounding_size=4", fc=ACCENT, ec="none", zorder=3))

    rx, rw = 556, W - 556 - 40
    # ── descent sparkline card (bottom of right column) ──
    spx, spy, spw, sph = rx, body_bot, rw, 120
    card(bg, spx, spy, spw, sph)
    card_label(bg, spx + 24, spy + sph - 24, "crosstalk descending to its floor")
    floor = float(1.0 / (c - 1)) if mode == "simplex" else float(welch_floor(c, d))
    bg.text(spx + spw - 24, spy + sph - 24, f"floor {floor:.3f}", fontsize=10, color=GREEN, family=MONO, va="center", ha="right")
    ax_spk = axes_in(fig, W, H, spx + 24, spy + 16, spw - 48, sph - 50)
    plot_spark(ax_spk, [hist[k]["coh"] for k in range(idx + 1)], floor)

    # ── gram card (top of right column) ──
    gx, gw = rx, rw
    gy = spy + sph + 18
    gh = top - gy
    card(bg, gx, gy, gw, gh)
    card_label(bg, gx + 24, gy + gh - 26, "cosine table  ⟨eᵢ,eⱼ⟩")
    gsub = "off-diagonal target  −1/(C−1)" if mode == "simplex" else "every |off-diagonal| pressed to the floor"
    bg.text(gx + 24, gy + 22, gsub, fontsize=10, color=MUTED, va="center", ha="left")
    gside = min(gw - 230, gh - 86)
    ax_gram = axes_in(fig, W, H, gx + 26, gy + 50, gside, gside)
    plot_gram(ax_gram, h["g"])
    lx = gx + 26 + gside + 34
    legend_top = gy + 50 + gside - 6
    for k, (cv, lab) in enumerate([(0.85, "+1  aligned"), (0.0, " 0  orthogonal"), (-0.85, "−1  opposite")]):
        yy = legend_top - k * 30
        rgb = matplotlib.colormaps["coolwarm"]((cv + 1) / 2)
        bg.add_patch(FancyBboxPatch((lx, yy - 8), 18, 16, boxstyle="round,pad=0,rounding_size=4", fc=rgb, ec=BORDER, lw=0.8))
        bg.text(lx + 28, yy, lab, fontsize=9.5, color=MUTED, family=MONO, va="center", ha="left")

    # ── stat cards row ──
    target = floor
    gap = max(0.0, h["coh"] - target)
    third_lbl, third_val, third_hint = (("simplex error", h["simplex"], "exact at 0")
                                        if mode == "simplex" else ("Welch gap", gap, "distance to floor"))
    stats = [
        ("crosstalk", f"{h['coh']:.3f}", f"target {target:.3f}", ACCENT, abs(h["coh"] - target) < 0.02),
        ("effective rank", f"{h['rank']:.2f}", f"of {d} dimensions", BLUE, h["rank"] > d - 0.25),
        (third_lbl, f"{third_val:.3f}", third_hint, ACCENT, third_val < 0.02),
    ]
    cw = (W - 80 - 2 * 22) / 3
    for i, (lbl, val, sub, col, ok) in enumerate(stats):
        stat_card(bg, 40 + i * (cw + 22), stat_y, cw, stat_h, lbl, val, sub, col, ok)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def render_case(name, *, c, d, mode, key_seed, title, subtitle, out, fps=15):
    hist = run(jax.random.key(key_seed), c, d, mode=mode)
    total = hist[-1]["t"]
    frames = [draw_frame(hist, i, mode=mode, c=c, d=d, title=title, subtitle=subtitle, total_steps=total) for i in range(len(hist))]
    frames += [frames[-1]] * int(fps * 0.7)
    path = os.path.join(out, f"{name}.gif")
    os.makedirs(out, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, loop=0, subrectangles=True)
    print(f"{name}: {len(frames)} frames -> {path} ({os.path.getsize(path) / 1024:.0f} KB)")
    return hist[-1]


def render_benign_landscape(out, *, c=7, d=3, n_seeds=6, fps=15):
    """Animated proof-by-experiment that the frame-potential landscape is benign:
    every random start dives to the same floor C²/d - C."""
    idx, fp = run_many(jax.random.key(7), c, d, n_seeds=n_seeds)
    floor = float(c * c / d - c)
    ymax = float(max(fp.max(), floor)) * 1.06
    xmax = float(idx[-1])
    frames = []
    for f in range(1, len(idx) + 1):
        frames.append(draw_benign_frame(idx[:f], fp[:, :f], floor, ymax, xmax, c, d, n_seeds))
    frames += [frames[-1]] * int(fps * 0.7)
    path = os.path.join(out, "benign-landscape.gif")
    os.makedirs(out, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, loop=0, subrectangles=True)
    print(f"benign-landscape: {len(frames)} frames -> {path} ({os.path.getsize(path) / 1024:.0f} KB)")
    return float(fp[:, -1].max()), floor


def draw_benign_frame(idx, fp, floor, ymax, xmax, c, d, n_seeds, dpi=110):
    W, H = 1140, 600
    fig, bg = new_canvas(W, H, dpi)
    kicker(bg, 40, H - 30, f"BENIGN LANDSCAPE · C={c} d={d} · JAX / vmap")
    bg.text(40, H - 60, "Every random start reaches the same floor", fontsize=23, color=FG, fontweight="bold", va="center", ha="left")
    bg.text(40, H - 92, "Benedetto & Fickus (2003): the frame potential has no bad local minima.", fontsize=12.5, color=MUTED, va="center", ha="left")
    top = H - 130

    ax = panel_card(bg, fig, W, H, 40, 40, 700, top - 40, "frame potential  Σ⟨eᵢ,eⱼ⟩²  vs. gradient-descent step",
                    corner=f"{n_seeds} seeds", pad_bot=34)
    ylo = floor - 0.1 * (ymax - floor) - 0.5
    ax.set_xlim(0, xmax)
    ax.set_ylim(ylo, ymax)
    ax.axhspan(ylo, floor, color=GREEN, alpha=0.06)
    ax.axhline(floor, color=GREEN, lw=1.6, ls=(0, (5, 4)))
    ax.text(xmax, floor, f" floor  C²/d − C = {floor:.1f}", color=GREEN, fontsize=10, va="bottom", ha="right")
    for s in range(fp.shape[0]):
        ax.plot(idx, fp[s], color=PALETTE[s % len(PALETTE)], lw=2.0, alpha=0.85, solid_capstyle="round")
        ax.scatter(idx[-1], fp[s, -1], s=26, color=PALETTE[s % len(PALETTE)], zorder=3, edgecolor="white", linewidth=1)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(True)
        ax.spines[sp].set_color(BORDER)
    ax.tick_params(left=True, bottom=True, colors=MUTED, labelsize=8.5, length=3)
    ax.set_yticks(np.linspace(np.ceil(ylo), np.floor(ymax), 4))
    ax.set_xticks([0, int(xmax)])

    rx, rw = 764, W - 764 - 40
    ch = (top - 40 - 18) / 2
    spread = float(fp[:, -1].max() - fp[:, -1].min())
    stat_card(bg, rx, 40 + ch + 18, rw, ch, "compute", f"{n_seeds}×", "seeds, one vmapped scan", BLUE, False)
    stat_card(bg, rx, 40, rw, ch, "spread at end", f"{spread:.4f}", "all starts agree", ACCENT, spread < 0.02)
    return finish(fig)


def class_means(z, labels, c):
    oh = jax.nn.one_hot(labels, c, dtype=z.dtype)
    counts = jnp.sum(oh, axis=0)[:, None]
    return (oh.T @ z) / jnp.maximum(counts, 1.0)


def collapse_ratio(z, labels, c):
    means = class_means(z, labels, c)
    global_mean = jnp.mean(z, axis=0, keepdims=True)
    within = jnp.mean(jnp.sum((z - means[labels]) ** 2, axis=-1))
    total = jnp.mean(jnp.sum((z - global_mean) ** 2, axis=-1))
    return within / (total + 1e-8)


# NOTE: metrics 1 (collapse) and 2 (rank) are *not* optimizations — they are
# controlled contrasts that define the two audit numbers before any descent runs.
# Collapse dials the within-class noise scale; rank dials a minor-axis squash.
# There is no learning dynamics to watch, so these ship as static before/after
# figures (a real training run is not being animated), while the descent GIFs
# (simplex/welch/benign) remain animated because motion carries the trajectory.


def _scatter_classes(ax, z, means, labels):
    ax.set_aspect("equal")
    ax.set_xlim(-1.95, 1.95)
    ax.set_ylim(-1.72, 1.72)
    ax.axhline(0, color=BORDER, lw=1, alpha=0.6)
    ax.axvline(0, color=BORDER, lw=1, alpha=0.6)
    for cls in np.unique(labels):
        pts = z[labels == cls]
        ax.scatter(pts[:, 0], pts[:, 1], s=22, color=PALETTE[int(cls)], alpha=0.6, edgecolor="white", linewidth=0.4)
        ax.scatter(means[int(cls), 0], means[int(cls), 1], s=150, color=PALETTE[int(cls)], edgecolor="white", linewidth=1.7, zorder=4)


def render_class_collapse(out):
    """Static: the collapse metric is a contrast, not a process. Loose clouds vs.
    condensed codes around the same fixed class means, with the two numbers the
    audit reports (collapse ratio, within-class variance) at each extreme."""
    key = jax.random.key(12)
    c, m = 3, 34
    angles = jnp.array([0.0, 2 * jnp.pi / 3, 4 * jnp.pi / 3])
    centers = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
    labels = jnp.repeat(jnp.arange(c), m)
    noise = jax.random.normal(key, (c * m, 2))

    def state(sigma):
        z = centers[labels] + sigma * noise
        means = class_means(z, labels, c)
        cr = float(collapse_ratio(z, labels, c))
        wv = float(jnp.mean(jnp.sum((z - means[labels]) ** 2, axis=-1)))
        return np.asarray(z), np.asarray(means), cr, wv

    loose = state(0.58)
    tight = state(0.035)

    W, H = 1140, 600
    fig, bg = new_canvas(W, H)
    kicker(bg, 40, H - 30, "REQUIREMENT 01 · ALIGNMENT · JAX")
    bg.text(40, H - 60, "Are same-class points tightening?", fontsize=23, color=FG, fontweight="bold", va="center", ha="left")
    bg.text(40, H - 92, "Collapse is measurable before we ever mention the simplex or the Welch bound.", fontsize=12.5, color=MUTED, va="center", ha="left")
    top = H - 130
    pw = (700 - 20) / 2
    for k, (data, tag) in enumerate([(loose, "loose"), (tight, "condensed")]):
        z, means, cr, wv = data
        px = 40 + k * (pw + 20)
        ax = panel_card(bg, fig, W, H, px, 40, pw, top - 40, tag, corner=f"cr {data[2]:.3f}")
        _scatter_classes(ax, z, means, labels)

    rx, rw = 724, W - 724 - 40
    ch = (top - 40 - 18) / 2
    stat_card(bg, rx, 40 + ch + 18, rw, ch, "collapse ratio", f"{tight[2]:.3f}", f"{loose[2]:.2f} loose → tight  (within ÷ total)", ACCENT, tight[2] < 0.08)
    stat_card(bg, rx, 40, rw, ch, "within-class variance", f"{tight[3]:.3f}", f"{loose[3]:.2f} → 0 as classes tighten", BLUE, tight[3] < 0.04)
    buf = finish(fig)
    path = os.path.join(out, "class-collapse.png")
    os.makedirs(out, exist_ok=True)
    imageio.imwrite(path, buf)
    print(f"class-collapse: static -> {path} ({os.path.getsize(path) / 1024:.0f} KB)")


def _scatter_rank(ax, z):
    ax.set_aspect("equal")
    ax.set_xlim(-3.3, 3.3)
    ax.set_ylim(-3.3, 3.3)
    ax.axhline(0, color=BORDER, lw=1, alpha=0.6)
    ax.axvline(0, color=BORDER, lw=1, alpha=0.6)
    ax.scatter(z[:, 0], z[:, 1], s=13, color=ACCENT, alpha=0.4, edgecolor="none")
    cov = np.cov(z.T)
    vals, vecs = np.linalg.eigh(cov)
    for k, color in enumerate([BLUE, GREEN]):
        v = vecs[:, k] * np.sqrt(max(vals[k], 0)) * 2.2
        ax.plot([-v[0], v[0]], [-v[1], v[1]], color=color, lw=3.2, alpha=0.9, solid_capstyle="round")


def render_rank_collapse(out):
    """Static: rank is a contrast too. A round cloud (rank ~2) vs. the same cloud
    squashed onto one axis (rank ~1). Effective rank catches the lost dimension."""
    key = jax.random.key(19)
    base = jax.random.normal(key, (420, 2))

    def state(squash):
        z = base * jnp.array([1.0, squash])
        return np.asarray(z), float(effective_rank(z))

    full = state(1.0)
    squashed = state(0.045)

    W, H = 1140, 600
    fig, bg = new_canvas(W, H)
    kicker(bg, 40, H - 30, "REQUIREMENT 02 · RANK · JAX")
    bg.text(40, H - 60, "Is the space using both axes?", fontsize=23, color=FG, fontweight="bold", va="center", ha="left")
    bg.text(40, H - 92, "A separated-looking embedding can still waste a whole dimension.", fontsize=12.5, color=MUTED, va="center", ha="left")
    top = H - 130
    pw = (700 - 20) / 2
    for k, (data, tag) in enumerate([(full, "both axes"), (squashed, "one axis dead")]):
        z, er = data
        px = 40 + k * (pw + 20)
        ax = panel_card(bg, fig, W, H, px, 40, pw, top - 40, tag, corner=f"rank {data[1]:.2f}")
        _scatter_rank(ax, z)

    rx, rw = 724, W - 724 - 40
    ch = (top - 40 - 18) / 2
    stat_card(bg, rx, 40 + ch + 18, rw, ch, "effective rank", f"{squashed[1]:.2f}", f"{full[1]:.2f} round → 1 squashed  (of 2)", RED, False)
    stat_card(bg, rx, 40, rw, ch, "energy in minor axis", "0.05", "1.00 → 0.05: second direction dies", BLUE, False)
    buf = finish(fig)
    path = os.path.join(out, "rank-collapse.png")
    os.makedirs(out, exist_ok=True)
    imageio.imwrite(path, buf)
    print(f"rank-collapse: static -> {path} ({os.path.getsize(path) / 1024:.0f} KB)")


def final_config(mode, c, d, seed, steps=900):
    """Run the full descent (not the trimmed animation) to its true endpoint."""
    x0 = l2_normalize(jax.random.normal(jax.random.key(seed), (c, d)))
    return descent_trajectory(x0, LOSSES[mode], OPTIMS[mode](LRS[mode]), steps)[-1]


def validate():
    """Self-check: confirm each descent actually reaches the geometry the post
    claims. Uses the full descent (the animation is trimmed for pacing) so a
    broken experiment fails loudly instead of shipping a misleading GIF."""
    print("validation (theory vs. JAX descent, full run):")
    s = final_config("simplex", 4, 3, 3)
    s_err, s_coh, s_rank = float(simplex_error(s)), float(coherence(s)), float(effective_rank(s))
    print(f"  simplex  C=4,d=3: coherence {s_coh:.3f}  (target -1/(C-1) = {1/3:.3f})  "
          f"simplex_error {s_err:.4f}  rank {s_rank:.2f}/3")
    assert s_err < 0.03, f"simplex did not converge (error {s_err:.4f})"

    w = final_config("welch", 6, 3, 0)
    w_coh, w_rank = float(coherence(w)), float(effective_rank(w))
    w_floor = float(welch_floor(6, 3))  # 6 equiangular lines (icosahedral) hit this
    print(f"  welch    C=6,d=3: coherence {w_coh:.3f}  (Welch floor {w_floor:.3f}, "
          f"gap {w_coh - w_floor:+.3f})  rank {w_rank:.2f}/3")
    assert w_coh <= w_floor + 0.06, f"welch coherence {w_coh:.3f} far above floor {w_floor:.3f}"
    print("  ok\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of: simplex welch benign class rank")
    ap.add_argument("--skip-validate", action="store_true")
    args = ap.parse_args()
    want = set(args.only) if args.only else {"simplex", "welch", "benign", "class", "rank"}

    if not args.skip_validate:
        validate()

    if "simplex" in want:
        render_case(
            "simplex-descent", c=4, d=3, mode="simplex", key_seed=3,
            title="simplex audit: C=4, d=3",
            subtitle="Can the four class codes share one fair angle? In 3D, yes: the target is a tetrahedron.",
            out=args.out, fps=args.fps,
        )
    if "welch" in want:
        render_case(
            "welch-descent", c=6, d=3, mode="welch", key_seed=0,
            title="Welch audit: C=6, d=3",
            subtitle="Too many codes for orthogonality. The best move is to share crosstalk evenly — down to the Welch floor.",
            out=args.out, fps=args.fps,
        )
    if "benign" in want:
        render_benign_landscape(args.out, fps=args.fps)
    if "class" in want:
        render_class_collapse(args.out)
    if "rank" in want:
        render_rank_collapse(args.out)


if __name__ == "__main__":
    main()
