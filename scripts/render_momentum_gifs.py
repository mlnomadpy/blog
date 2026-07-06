"""Six teaching GIFs for the momentum-ResNet JAX companion.

Every moving thing is a real number from the real computation:

  1. momentum-orbits.gif           the Kepler two-body problem integrated live
                                   (float64, dt = 0.02, 20 orbits) with forward
                                   Euler and leapfrog; Euler pumps in +68.3%
                                   energy and spirals out, leapfrog holds 0.016%.
  2. momentum-depth-lapse.gif      the literal width-2 hidden states of the two
                                   trained rings nets (L = 32), flowing block by
                                   block (public/momentum-resnet/trajectories.json).
  3. momentum-velocity-ledger.gif  the real velocity state v_l of test points
                                   carried through the trained mu = 0.9 net
                                   (forward pass of the exported weights).
  4. momentum-crystallize.gif      the momentum net's decision field, loss and
                                   hidden flow over real training steps
                                   (telemetry from scripts/momentum_gif_data.py,
                                   run on Kaggle; results/kgl_blog-momentum-gifs/).
  5. momentum-rewind.gif           the exact backward pass retracing every point
                                   home in float64 (max err ~3e-13), then the
                                   float32 rewind error rising as (1/mu)^layers.
  6. momentum-ceiling.gif          test accuracy vs depth on rings, 3 seeds each
                                   (public/momentum-resnet/results.json): the
                                   plain flow never touches the 100% line again.

Sources of truth: public/momentum-resnet/{results,trajectories,cliff,inertia}.json
(from scripts/momentum_resnet.py) and the Kaggle telemetry npz.

Run: python3 scripts/render_momentum_gifs.py [orbits|lapse|ledger|crystal|rewind|ceiling]
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import imageio.v2 as imageio
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / 'public'
MR = PUB / 'momentum-resnet'
RESULTS = json.load(open(MR / 'results.json'))
TRAJ = json.load(open(MR / 'trajectories.json'))
INERTIA = json.load(open(MR / 'inertia.json'))
NPZ_PATH = ROOT / 'scripts' / 'results' / 'kgl_blog-momentum-gifs' / 'momentum_gif_data.npz'

BG = '#0e0d0b'; PANEL = '#16140f'; INK = '#e8e2d4'; MUTED = '#9a9282'; LINE = '#3a352c'
C_INNER = '#e0a45a'   # class 0, the enclosed disk
C_OUTER = '#5a9fd0'   # class 1, the surrounding annulus
C_PLAIN = '#c2553a'   # plain residual net / forward Euler
C_MOM = '#36d6c4'     # momentum net / leapfrog
C_GOLD = '#e8c46a'
plt.rcParams.update({'figure.facecolor': BG, 'savefig.facecolor': BG,
                     'text.color': INK, 'axes.edgecolor': LINE, 'font.size': 11})


def ease(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def fig_rgba(fig):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba


def save_gif(path, frames, fps, hold=14, palettesize=128, durations=None):
    """durations in ms per frame (else 1000/fps); the hold extends the last frame."""
    Image.fromarray(frames[-1]).save(str(path).replace('.gif', '-preview.png'))
    dur = list(durations) if durations is not None else [1000.0 / fps] * len(frames)
    dur[-1] += hold * 1000.0 / fps
    imageio.mimsave(path, frames, duration=dur, loop=0,
                    palettesize=palettesize, subrectangles=True)
    im = Image.open(path)
    im.seek(0)
    print(f'wrote {path} ({os.path.getsize(path)//1024} KB, {im.n_frames} frames, '
          f'first dur {im.info.get("duration")} ms)')


def style_ax(ax, panel=PANEL):
    ax.set_facecolor(panel)
    for s in ax.spines.values():
        s.set_color(LINE)


# ═══════════════════════════════════════════════════════════════════════════
# The real dynamics (identical math to scripts/momentum_resnet.py)
# ═══════════════════════════════════════════════════════════════════════════

R0 = np.array([1.0, 0.0]); V0 = np.array([0.0, 0.9])
E0 = 0.5 * 0.9 ** 2 - 1.0                    # -0.595
A_SEMI = -1.0 / (2.0 * E0)                   # 0.8403
ECC = math.sqrt(1.0 + 2.0 * E0 * 0.9 ** 2)   # 0.19
T_ORBIT = 2.0 * math.pi * A_SEMI ** 1.5


def accel(r):
    return -r / (np.linalg.norm(r) ** 3)


def integrate(dt, n_orbits, method):
    n_steps = int(round(n_orbits * T_ORBIT / dt))
    sub = max(1, n_steps // 4000)
    r, v = R0.copy(), V0.copy()
    rs, es = [r.copy()], [E0]
    if method == 'leapfrog':
        a = accel(r)
        for i in range(n_steps):
            v = v + 0.5 * dt * a
            r = r + dt * v
            a = accel(r)
            v = v + 0.5 * dt * a
            if (i + 1) % sub == 0:
                rs.append(r.copy())
                es.append(0.5 * v @ v - 1.0 / np.linalg.norm(r))
    else:
        for i in range(n_steps):
            a = accel(r)
            r = r + dt * v
            v = v + dt * a
            if (i + 1) % sub == 0:
                rs.append(r.copy())
                es.append(0.5 * v @ v - 1.0 / np.linalg.norm(r))
    return np.array(rs), np.array(es), sub * dt


def net_arrays(mu_key, dtype):
    net = INERTIA['nets'][mu_key]
    return (np.array(net['W1'], dtype), np.array(net['b1'], dtype),
            np.array(net['W2'], dtype), np.array(net['b2'], dtype))


def forward_pass(mu, X, dtype=np.float64):
    """The exported trained net, run forward. Returns xs, vs: [L+1, n, 2]."""
    W1, b1, W2, b2 = net_arrays(str(mu), dtype)
    L, h = INERTIA['L'], dtype(INERTIA['h'])
    mu_ = dtype(mu); one = dtype(1)
    x = X.astype(dtype); v = np.zeros_like(x)
    xs, vs = [x.copy()], [v.copy()]
    for l in range(L):
        f = np.tanh(x @ W1[l] + b1[l]) @ W2[l] + b2[l]
        v = mu_ * v + (one - mu_) * f
        x = x + h * v
        xs.append(x.copy()); vs.append(v.copy())
    return np.array(xs), np.array(vs)


def backward_pass(mu, xL, vL, dtype=np.float64):
    """The exact rewind: solve each block for its own past."""
    W1, b1, W2, b2 = net_arrays(str(mu), dtype)
    L, h = INERTIA['L'], dtype(INERTIA['h'])
    mu_ = dtype(mu); one = dtype(1)
    x = xL.astype(dtype); v = vL.astype(dtype)
    xs = [x.copy()]
    for l in range(L - 1, -1, -1):
        x_prev = x - h * v
        f = np.tanh(x_prev @ W1[l] + b1[l]) @ W2[l] + b2[l]
        v = (v - (one - mu_) * f) / mu_
        x = x_prev
        xs.append(x.copy())
    return np.array(xs)          # [L+1, n, 2], index k = k layers rewound


def pick_probes(n_per_class=6):
    X = np.array(INERTIA['test_X'], np.float64)
    y = np.array(INERTIA['test_y'])
    idx = []
    for c in (0, 1):
        cand = np.where(y == c)[0]
        order = np.argsort(np.arctan2(X[cand, 1], X[cand, 0]))
        idx.extend(cand[order[:: max(1, len(cand) // n_per_class)]][:n_per_class])
    idx = np.array(idx)
    return X[idx], y[idx]


# ═══════════════════════════════════════════════════════════════════════════
# GIF 1 -- two integrators, one planet
# ═══════════════════════════════════════════════════════════════════════════
def gif_orbits():
    DT, N_ORB = 0.02, 20
    r_e, e_e, tsamp = integrate(DT, N_ORB, 'euler')
    r_l, e_l, _ = integrate(DT, N_ORB, 'leapfrog')
    S = min(len(r_e), len(r_l))
    drift_e = 100 * (e_e - E0) / abs(E0)              # percent
    drift_l = 100 * (e_l - E0) / abs(E0)
    t_orbits = np.arange(S) * tsamp / T_ORBIT
    apo_run = np.maximum.accumulate(np.linalg.norm(r_e, axis=1))
    # sanity against the published run
    row = [s for s in RESULTS['physics']['sweeps'] if s['dt'] == 0.02][0]
    assert abs(drift_e[-1] / 100 - row['euler_energy_drift']) < 1e-6

    th = np.linspace(0, 2 * np.pi, 400)
    rr = A_SEMI * (1 - ECC ** 2) / (1 + ECC * np.cos(th))
    ell_x, ell_y = rr * np.cos(th), rr * np.sin(th)

    LIM = 3.75
    NF = 150
    frames = []
    for fi in range(NF):
        u = fi / (NF - 1)
        idx = max(1, int((u ** 1.3) * (S - 1)))
        fig = plt.figure(figsize=(8.8, 5.8), dpi=100, facecolor=BG)
        fig.text(0.5, 0.955, 'Two integrators, one planet: where you write the update decides the orbit',
                 ha='center', fontsize=13.5, weight='bold')
        fig.text(0.5, 0.905, 'the same two-body problem (GM = 1), the same launch, the same step dt = 0.02, integrated for 20 orbits',
                 ha='center', fontsize=9, color=MUTED)

        for pi, (rs, name, col) in enumerate(
                [(r_e, 'forward Euler', C_PLAIN), (r_l, 'leapfrog', C_MOM)]):
            ax = fig.add_axes([0.045 + pi * 0.475, 0.30, 0.44, 0.55])
            style_ax(ax)
            ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM * 0.62, LIM * 0.62)
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(name, fontsize=11, color=col, pad=4, weight='bold')
            ax.plot(ell_x, ell_y, color=MUTED, lw=0.9, ls='--', alpha=0.65, zorder=1)
            ax.scatter([0], [0], s=90, marker='*', color=C_GOLD, zorder=3,
                       edgecolors=BG, linewidths=0.5)
            ax.plot(rs[:idx, 0], rs[:idx, 1], color=col, lw=0.7, alpha=0.28, zorder=2)
            tail = max(0, idx - int(T_ORBIT / tsamp))         # last full orbit
            ax.plot(rs[tail:idx, 0], rs[tail:idx, 1], color=col, lw=1.7, alpha=0.95, zorder=4)
            ax.scatter([rs[idx - 1, 0]], [rs[idx - 1, 1]], s=42, color=col,
                       edgecolors=INK, linewidths=0.8, zorder=5)
            if pi == 0:
                ax.text(0.03, 0.05, f'farthest reach {apo_run[idx-1]:.2f}   (true apoapsis 1.00)',
                        transform=ax.transAxes, fontsize=8.5, color=C_PLAIN)
            else:
                ax.text(0.03, 0.05, f'max |dE|/|E| so far {np.abs(drift_l[:idx]).max():.3f}%',
                        transform=ax.transAxes, fontsize=8.5, color=C_MOM)
        fig.text(0.5, 0.865, f'orbit {t_orbits[idx-1]:.1f} of 20', ha='center',
                 fontsize=9.5, color=INK)

        axE = fig.add_axes([0.075, 0.065, 0.87, 0.175])
        style_ax(axE)
        axE.set_xlim(0, 20); axE.set_ylim(-6, 78)
        axE.axhline(0, color=LINE, lw=0.8)
        axE.plot(t_orbits[:idx], drift_e[:idx], color=C_PLAIN, lw=1.6)
        axE.plot(t_orbits[:idx], drift_l[:idx], color=C_MOM, lw=1.6)
        lx = min(t_orbits[idx - 1] + 0.25, 15.6)
        axE.text(lx, min(drift_e[idx - 1], 62), f'Euler {drift_e[idx-1]:+.1f}%',
                 fontsize=8.5, color=C_PLAIN, va='center')
        axE.text(lx, 2.5, f'leapfrog max {np.abs(drift_l[:idx]).max():.3f}%',
                 fontsize=8.5, color=C_MOM, va='bottom')
        axE.set_ylabel('energy drift  (E - E0)/|E0|  %', fontsize=8, color=MUTED)
        axE.set_xlabel('time (orbits)', fontsize=8, color=MUTED)
        axE.tick_params(colors=MUTED, labelsize=7.5)
        frames.append(fig_rgba(fig))
    save_gif(PUB / 'momentum-orbits.gif', frames, fps=12, hold=20, palettesize=96)


# ═══════════════════════════════════════════════════════════════════════════
# GIF 2 -- depth as a time-lapse of the two trained nets
# ═══════════════════════════════════════════════════════════════════════════
def gif_depth_lapse():
    A = {k: np.array(TRAJ[k]['traj']) for k in ('rings_mu0.0', 'rings_mu0.9')}
    Y = {k: np.array(TRAJ[k]['y']) for k in A}
    Lc = A['rings_mu0.0'].shape[1] - 1                # 32
    # camera: the real running extent of the cloud (both nets), never shrinking
    ext = np.maximum.accumulate(
        np.max([np.abs(a).max(axis=(0, 2)) for a in A.values()], axis=0))
    ext = np.maximum(ext, 2.4)

    def stats_at(a, node, frac):
        """Mean cumulative path length and mean turn so far, over drawn points."""
        seg = np.diff(a[:, :node + 1], axis=1)
        pl = np.linalg.norm(seg, axis=2).sum(1).mean() if node >= 1 else 0.0
        if frac > 0 and node < Lc:
            pl += frac * np.linalg.norm(a[:, node + 1] - a[:, node], axis=1).mean()
        if node >= 2:
            d1, d2 = seg[:, :-1], seg[:, 1:]
            dot = (d1 * d2).sum(2)
            nn = np.linalg.norm(d1, axis=2) * np.linalg.norm(d2, axis=2) + 1e-12
            turn = np.degrees(np.arccos(np.clip(dot / nn, -1, 1))).mean()
        else:
            turn = 0.0
        return pl, turn

    TWEEN = 3
    NF = 10 + Lc * TWEEN + 6
    meta = [('rings_mu0.0', 'plain residual  (mu = 0)  =  forward Euler', C_PLAIN, '99.76%'),
            ('rings_mu0.9', 'momentum net  (mu = 0.9)  =  a velocity ledger', C_MOM, '100.0%')]
    frames = []
    for fi in range(NF):
        prog = min(max(0, fi - 10) / TWEEN, Lc)       # in layers
        node, frac = int(prog), prog - int(prog)
        lim = ((1 - frac) * ext[node] + frac * ext[min(node + 1, Lc)]) * 1.12
        fig = plt.figure(figsize=(8.8, 5.3), dpi=105, facecolor=BG)
        fig.text(0.5, 0.955, 'Depth as time: the same test points, carried by two trained nets',
                 ha='center', fontsize=13.5, weight='bold')
        fig.text(0.5, 0.905, 'the literal width-2 hidden state of the rings nets (L = 32, no projection), one block per tick; orange = enclosed class, blue = surrounding class',
                 ha='center', fontsize=8.6, color=MUTED)
        for pi, (key, name, col, accs) in enumerate(meta):
            a, y = A[key], Y[key]
            pos = a[:, node] if node >= Lc else (1 - frac) * a[:, node] + frac * a[:, node + 1]
            ax = fig.add_axes([0.045 + pi * 0.475, 0.115, 0.44, 0.72])
            style_ax(ax)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(name, fontsize=9.8, color=col, pad=4, weight='bold')
            for i in range(a.shape[0]):
                cc = C_INNER if y[i] == 0 else C_OUTER
                xs = list(a[i, :node + 1, 0]) + ([pos[i, 0]] if node < Lc else [])
                ys = list(a[i, :node + 1, 1]) + ([pos[i, 1]] if node < Lc else [])
                ax.plot(xs, ys, color=cc, lw=0.8, alpha=0.45, zorder=2)
            for c, cc in ((0, C_INNER), (1, C_OUTER)):
                m = y == c
                ax.scatter(pos[m, 0], pos[m, 1], s=22, color=cc, zorder=4,
                           edgecolors=BG, linewidths=0.5)
            pl, turn = stats_at(a, node, frac)
            ax.text(0.03, 0.965, f'layer {min(prog, Lc):.0f} / {Lc}', transform=ax.transAxes,
                    fontsize=9, color=INK, va='top', weight='bold')
            ax.text(0.03, 0.03,
                    f'path so far {pl:5.1f}    mean turn {turn:4.1f}' + chr(176) + '/layer',
                    transform=ax.transAxes, fontsize=8.4, color=MUTED)
            if prog >= Lc:
                ax.text(0.97, 0.965, f'test acc {accs}', transform=ax.transAxes,
                        fontsize=9.5, color=col, ha='right', va='top', weight='bold')
        msg = ('layer 0: both nets see the same rings, a disk walled in by an annulus'
               if prog < 1 else
               'the plain net turns a fresh corner at every block; the momentum net coasts through the same journey')
        fig.text(0.5, 0.033, msg, ha='center', fontsize=9.2, color=C_GOLD)
        frames.append(fig_rgba(fig))
    save_gif(PUB / 'momentum-depth-lapse.gif', frames, fps=11, hold=22)


# ═══════════════════════════════════════════════════════════════════════════
# GIF 3 -- the velocity ledger made visible
# ═══════════════════════════════════════════════════════════════════════════
def gif_velocity_ledger():
    Xp, yp = pick_probes(6)
    xs, vs = forward_pass(0.9, Xp)                    # [33, 12, 2] each, real
    Lc = xs.shape[0] - 1
    kin = 0.5 * (vs ** 2).sum(2).mean(1)              # mean kinetic energy, real
    # camera: running extent of the points and their arrow tips, never shrinking
    tips = np.abs(np.concatenate([xs, xs + 0.55 * vs], 1)).max(axis=(1, 2))
    ext = np.maximum(np.maximum.accumulate(tips), 2.4)

    TWEEN = 4
    NF = 10 + Lc * TWEEN + 6
    frames = []
    for fi in range(NF):
        prog = min(max(0, fi - 10) / TWEEN, Lc)
        node, frac = int(prog), prog - int(prog)
        pos = xs[node] if node >= Lc else (1 - frac) * xs[node] + frac * xs[node + 1]
        vel = vs[node] if node >= Lc else (1 - frac) * vs[node] + frac * vs[node + 1]
        lim = ((1 - frac) * ext[node] + frac * ext[min(node + 1, Lc)]) * 1.10
        fig = plt.figure(figsize=(8.6, 5.6), dpi=105, facecolor=BG)
        fig.text(0.5, 0.955, 'The second ledger: v is written by the block, x is moved by v',
                 ha='center', fontsize=13.5, weight='bold')
        fig.text(0.5, 0.907, 'the trained rings net (mu = 0.9, L = 32): each arrow is the real velocity state carried by a test point',
                 ha='center', fontsize=8.8, color=MUTED)
        ax = fig.add_axes([0.09, 0.295, 0.395, 0.60])
        style_ax(ax)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        for i in range(pos.shape[0]):
            cc = C_INNER if yp[i] == 0 else C_OUTER
            xs_i = list(xs[:node + 1, i, 0]) + ([pos[i, 0]] if node < Lc else [])
            ys_i = list(xs[:node + 1, i, 1]) + ([pos[i, 1]] if node < Lc else [])
            ax.plot(xs_i, ys_i, color=cc, lw=0.7, alpha=0.38, zorder=2)
            ax.annotate('', xy=(pos[i, 0] + 0.55 * vel[i, 0], pos[i, 1] + 0.55 * vel[i, 1]),
                        xytext=(pos[i, 0], pos[i, 1]), zorder=5,
                        arrowprops=dict(arrowstyle='-|>', color=C_MOM, lw=1.5,
                                        mutation_scale=11, alpha=0.95))
            ax.scatter([pos[i, 0]], [pos[i, 1]], s=26, color=cc, zorder=4,
                       edgecolors=BG, linewidths=0.5)
        ax.text(0.03, 0.965, f'layer {min(prog, Lc):.0f} / {Lc}', transform=ax.transAxes,
                fontsize=9.5, color=INK, va='top', weight='bold')

        # the update rule, as a legend of what the arrows are
        axT = fig.add_axes([0.57, 0.50, 0.40, 0.34]); axT.axis('off')
        axT.text(0, 0.95, 'each block l:', fontsize=9.5, color=MUTED, va='top')
        axT.text(0.04, 0.74, r'v $\leftarrow$ 0.9 v + 0.1 F$_l$(x)', fontsize=12, color=C_MOM, va='top')
        axT.text(0.04, 0.50, r'x $\leftarrow$ x + h v', fontsize=12, color=INK, va='top')
        axT.text(0, 0.24, 'the block never touches the position;\nit deposits into the velocity ledger,\nand the ledger moves the point',
                 fontsize=8.5, color=MUTED, va='top')

        # kinetic-energy strip: the ledger filling, then funding the escape
        axK = fig.add_axes([0.09, 0.115, 0.87, 0.145])
        style_ax(axK)
        axK.set_xlim(0, Lc + 3.5); axK.set_ylim(0, kin.max() * 1.12)
        kx = np.arange(node + 1).tolist() + ([node + frac] if node < Lc else [])
        ky = kin[:node + 1].tolist() + ([float((1 - frac) * kin[node] + frac * kin[min(node + 1, Lc)])] if node < Lc else [])
        axK.plot(kx, ky, color=C_MOM, lw=1.8)
        axK.fill_between(kx, 0, ky, color=C_MOM, alpha=0.15)
        axK.set_ylabel('mean kinetic\nenergy  ' + r'$\frac{1}{2}|v|^2$', fontsize=7.5, color=MUTED)
        axK.set_xticks([0, 8, 16, 24, 32])
        axK.text(0.995, 0.86, 'layer', transform=axK.transAxes, fontsize=8, color=MUTED, ha='right')
        axK.tick_params(colors=MUTED, labelsize=7.5)
        cur = ky[-1] if ky else 0.0
        axK.text(min(node + frac, Lc) + 0.4, min(cur, kin.max() * 0.82),
                 f'{cur:.2f}', fontsize=8, color=C_MOM, va='bottom')
        p = min(prog, Lc)
        msg = ('the ledger fills slowly: v starts at zero and the first blocks only deposit'
               if p < 9 else
               'inertia carries the enclosed class out across the ring on smooth crossing paths'
               if p < 22 else
               'the final sprint: the accumulated velocity funds the drive to the readout')
        fig.text(0.5, 0.012, msg, ha='center', fontsize=9.2, color=C_GOLD)
        frames.append(fig_rgba(fig))
    save_gif(PUB / 'momentum-velocity-ledger.gif', frames, fps=11, hold=22)


# ═══════════════════════════════════════════════════════════════════════════
# GIF 4 -- training crystallization (Kaggle telemetry)
# ═══════════════════════════════════════════════════════════════════════════
def gif_crystallize():
    d = np.load(NPZ_PATH)
    grids = d['grids'].astype(np.float32)             # [S, 96, 96] P(class 1)
    accs, steps, losses = d['accs'], d['snap_steps'], d['losses']
    trajs = d['trajs']                                # [S, 33, 16, 2]
    py = d['probe_y']; Xtr = d['Xtr']; ytr = d['ytr']
    glim = float(d['grid_lim'])
    S = grids.shape[0]
    cmap = LinearSegmentedColormap.from_list('flow', [C_INNER, '#241f16', C_OUTER])
    tlim = np.abs(trajs).max() * 1.05

    durations = [330 if si < 6 else 220 if si < 16 else 110 for si in range(S)]
    frames = []
    for si in range(S):
        fig = plt.figure(figsize=(9.0, 4.9), dpi=105, facecolor=BG)
        fig.text(0.5, 0.95, 'Training the momentum net: the boundary closes early, the flow keeps crystallizing',
                 ha='center', fontsize=12.8, weight='bold')
        fig.text(0.5, 0.895, 'rings, L = 32, mu = 0.9, real Adam steps; left: P(class) over the input plane; right: the hidden flow and the loss, same moment',
                 ha='center', fontsize=8.4, color=MUTED)

        axF = fig.add_axes([0.045, 0.09, 0.38, 0.71])
        style_ax(axF)
        axF.imshow(grids[si], extent=[-glim, glim, -glim, glim], origin='lower',
                   cmap=cmap, vmin=0, vmax=1, interpolation='bilinear')
        axF.contour(np.linspace(-glim, glim, grids.shape[1]),
                    np.linspace(-glim, glim, grids.shape[2]),
                    grids[si], levels=[0.5], colors=[INK], linewidths=1.1)
        for c, cc in ((0, C_INNER), (1, C_OUTER)):
            m = ytr == c
            axF.scatter(Xtr[m, 0], Xtr[m, 1], s=2.5, color=cc, alpha=0.5, linewidths=0)
        axF.set_xticks([]); axF.set_yticks([])
        axF.set_title('decision field  P(class | input)', fontsize=9, color=MUTED, pad=3)

        axT = fig.add_axes([0.46, 0.09, 0.28, 0.71])
        style_ax(axT)
        axT.set_xlim(-tlim, tlim); axT.set_ylim(-tlim, tlim)
        axT.set_aspect('equal'); axT.set_xticks([]); axT.set_yticks([])
        for i in range(trajs.shape[2]):
            cc = C_INNER if py[i] == 0 else C_OUTER
            axT.plot(trajs[si, :, i, 0], trajs[si, :, i, 1], color=cc, lw=0.9, alpha=0.75)
            axT.scatter([trajs[si, -1, i, 0]], [trajs[si, -1, i, 1]], s=14, color=cc,
                        edgecolors=BG, linewidths=0.4, zorder=4)
        axT.set_title('hidden trajectories (16 probes)', fontsize=9, color=MUTED, pad=3)

        axL = fig.add_axes([0.795, 0.36, 0.185, 0.44])
        style_ax(axL)
        st = steps[si]
        axL.set_xlim(0, len(losses)); axL.set_yscale('log')
        axL.set_ylim(3e-7, 1.2)
        axL.plot(np.arange(st), losses[:st], color=C_GOLD, lw=1.3)
        axL.tick_params(colors=MUTED, labelsize=6.5)
        axL.set_title('train loss', fontsize=8.5, color=MUTED, pad=3)
        axL.set_xticks([0, 2000, 4000]); axL.set_xticklabels(['0', '2k', '4k'], fontsize=6.5)

        fig.text(0.885, 0.245, f'step {st} / {len(losses)}', ha='center', fontsize=9.5,
                 color=INK, weight='bold')
        fig.text(0.885, 0.175, f'test acc {accs[si]*100:.1f}%', ha='center', fontsize=9.5,
                 color=C_MOM if accs[si] >= 1 else MUTED)
        lz = losses[st - 1] if st > 0 else losses[0]
        fig.text(0.885, 0.105, f'loss {lz:.1e}', ha='center', fontsize=8.5, color=C_GOLD)
        msg = ('the readout still sees noise: the field is undecided and the flow barely moves the points'
               if st < 50 else
               'accuracy pins at 100% within 50 steps: the scoreboard is already blind'
               if st < 900 else
               'but the field keeps hardening and the flow keeps organizing, 4000 steps of crystallization')
        fig.text(0.42, 0.012, msg, ha='center', fontsize=9, color=C_GOLD)
        frames.append(fig_rgba(fig))
    save_gif(PUB / 'momentum-crystallize.gif', frames, fps=10, hold=20,
             palettesize=128, durations=durations)


# ═══════════════════════════════════════════════════════════════════════════
# GIF 5 -- the rewind, and the thief of memory
# ═══════════════════════════════════════════════════════════════════════════
def gif_rewind():
    Xp, yp = pick_probes(5)                            # 10 points
    Lc = INERTIA['L']
    xs64, vs64 = forward_pass(0.9, Xp, np.float64)
    bx64 = backward_pass(0.9, xs64[-1], vs64[-1], np.float64)
    err64 = np.array([np.abs(bx64[k] - xs64[Lc - k]).max() for k in range(Lc + 1)])

    err32 = {}
    for mu in (0.9, 0.6, 0.3):
        Xall = np.array(INERTIA['test_X'], np.float64)[:64]
        fx, fv = forward_pass(mu, Xall, np.float32)
        bx = backward_pass(mu, fx[-1], fv[-1], np.float32)
        err32[mu] = np.array([np.abs(bx[k].astype(np.float64) -
                                     fx[Lc - k].astype(np.float64)).max()
                              for k in range(Lc + 1)])
        print(f'  float32 roundtrip mu={mu}: {err32[mu][-1]:.2e}')
    print(f'  float64 roundtrip mu=0.9: {err64[-1]:.2e}')

    lim_full = np.abs(xs64).max() * 1.10
    # camera: grows with the forward cloud, then follows the rewind back home
    extF = np.maximum(np.maximum.accumulate(np.abs(xs64).max(axis=(1, 2))), 2.4)
    extB = np.maximum(np.abs(bx64).max(axis=(1, 2)), 2.4)
    MUCOL = {0.9: C_MOM, 0.6: C_GOLD, 0.3: C_PLAIN}

    N_FWD, N_BWD, N_THIEF = 40, 52, 56
    frames = []

    def base_fig(phase_txt):
        fig = plt.figure(figsize=(9.0, 5.0), dpi=105, facecolor=BG)
        fig.text(0.5, 0.95, 'The rewind: run every block backward, and meet the thief',
                 ha='center', fontsize=13.2, weight='bold')
        fig.text(0.5, 0.897, phase_txt, ha='center', fontsize=8.8, color=MUTED)
        return fig

    def plane_ax(fig, lim):
        ax = fig.add_axes([0.045, 0.09, 0.42, 0.72])
        style_ax(ax)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        return ax

    def err_ax(fig):
        ax = fig.add_axes([0.56, 0.13, 0.415, 0.68])
        style_ax(ax)
        ax.set_xlim(0, Lc); ax.set_yscale('log'); ax.set_ylim(1e-15, 1e12)
        ax.set_xlabel('layers rewound', fontsize=8.5, color=MUTED)
        ax.set_ylabel('max reconstruction error', fontsize=8.5, color=MUTED)
        ax.tick_params(colors=MUTED, labelsize=7)
        return ax

    # phase A: forward
    for fi in range(N_FWD):
        prog = ease(fi / (N_FWD - 1)) * Lc
        node, frac = int(prog), prog - int(prog)
        pos = xs64[node] if node >= Lc else (1 - frac) * xs64[node] + frac * xs64[node + 1]
        fig = base_fig('first, the forward pass of the trained momentum net (mu = 0.9, L = 32), in float64')
        ax = plane_ax(fig, ((1 - frac) * extF[node] + frac * extF[min(node + 1, Lc)]) * 1.10)
        for i in range(pos.shape[0]):
            cc = C_INNER if yp[i] == 0 else C_OUTER
            ax.plot(list(xs64[:node + 1, i, 0]) + [pos[i, 0]],
                    list(xs64[:node + 1, i, 1]) + [pos[i, 1]], color=cc, lw=0.9, alpha=0.55)
            ax.scatter([pos[i, 0]], [pos[i, 1]], s=26, color=cc, zorder=4,
                       edgecolors=BG, linewidths=0.5)
        ax.set_title(f'forward:  layer {prog:.0f} / {Lc}', fontsize=9.5, color=INK, pad=3)
        axE = err_ax(fig)
        axE.text(0.5, 0.5, 'the rewind will be measured here', transform=axE.transAxes,
                 ha='center', fontsize=9, color=MUTED, style='italic')
        frames.append(fig_rgba(fig))

    # phase B: exact rewind
    for fi in range(N_BWD):
        prog = ease(fi / (N_BWD - 1)) * Lc
        k, frac = int(prog), prog - int(prog)
        pos = bx64[k] if k >= Lc else (1 - frac) * bx64[k] + frac * bx64[k + 1]
        fig = base_fig('then hand it ONLY the final (x, v) and run the update rule backward: x = x - h v,  v = (v - 0.1 F(x)) / 0.9')
        ax = plane_ax(fig, ((1 - frac) * extB[k] + frac * extB[min(k + 1, Lc)]) * 1.15)
        for i in range(pos.shape[0]):
            cc = C_INNER if yp[i] == 0 else C_OUTER
            ax.plot(xs64[:, i, 0], xs64[:, i, 1], color=cc, lw=0.9, alpha=0.30)
            ax.plot(bx64[:k + 1, i, 0], bx64[:k + 1, i, 1], color=INK, lw=0.8, alpha=0.5)
            ax.scatter([pos[i, 0]], [pos[i, 1]], s=46, facecolors='none', zorder=5,
                       edgecolors=INK, linewidths=1.3)
            ax.scatter([Xp[i, 0]], [Xp[i, 1]], s=20, color=cc, alpha=0.9, zorder=3,
                       edgecolors=BG, linewidths=0.4)
        ax.set_title(f'rewinding:  {prog:.0f} / {Lc} layers undone', fontsize=9.5,
                     color=INK, pad=3)
        axE = err_ax(fig)
        kk = np.arange(k + 1)
        axE.plot(kk, np.maximum(err64[:k + 1], 1e-15), color=C_MOM, lw=1.8)
        axE.text(0.04, 0.10, f'float64, mu = 0.9\nmax err {err64[k]:.1e}',
                 transform=axE.transAxes, fontsize=9.5, color=C_MOM)
        if fi > N_BWD - 6:
            fig.text(0.5, 0.015, 'every point comes home: no stored activations, the velocity is the receipt',
                     ha='center', fontsize=9.2, color=C_GOLD)
        frames.append(fig_rgba(fig))

    # phase C: the thief (float32, three frictions)
    for fi in range(N_THIEF):
        prog = ease(fi / (N_THIEF - 1)) * Lc
        k = int(prog)
        fig = base_fig('now the same rewind in float32, at three frictions: every backward step divides by mu, so rounding noise grows as (1/mu) per layer')
        ax = plane_ax(fig, lim_full)
        for i in range(Xp.shape[0]):
            cc = C_INNER if yp[i] == 0 else C_OUTER
            ax.plot(xs64[:, i, 0], xs64[:, i, 1], color=cc, lw=0.9, alpha=0.16)
            ax.scatter([Xp[i, 0]], [Xp[i, 1]], s=18, color=cc, alpha=0.5, zorder=3,
                       edgecolors=BG, linewidths=0.4)
        ax.set_title('friction is the thief of memory', fontsize=9.5, color=C_GOLD, pad=3)
        ax.text(0.5, 0.5, 'mu = 0.9  remembers\nmu = 0.3  has burned\nits own history',
                transform=ax.transAxes, ha='center', fontsize=10, color=MUTED)
        axE = err_ax(fig)
        axE.plot(np.arange(Lc + 1), np.maximum(err64, 1e-15), color=C_MOM, lw=1.0,
                 alpha=0.45)
        axE.text(Lc - 0.5, err64[-1] * 3, 'float64 mu=0.9', fontsize=7, color=C_MOM,
                 ha='right', alpha=0.8)
        for mu in (0.9, 0.6, 0.3):
            e = err32[mu]
            kk = np.arange(k + 1)
            guide = e[1] * (1.0 / mu) ** (np.arange(Lc + 1) - 1)
            axE.plot(np.arange(Lc + 1), np.clip(guide, 1e-15, 1e12), color=MUCOL[mu],
                     lw=0.8, ls=':', alpha=0.6)
            axE.plot(kk, np.maximum(e[:k + 1], 1e-15), color=MUCOL[mu], lw=1.8)
            if k >= Lc:
                axE.annotate(f'mu = {mu}:  {e[-1]:.1e}', xy=(Lc, e[-1]),
                             xytext=(Lc - 1, e[-1] * 4), fontsize=8.5, ha='right',
                             color=MUCOL[mu], weight='bold')
        axE.text(0.03, 0.955, 'float32   dotted = the (1/mu)$^k$ prediction',
                 transform=axE.transAxes, fontsize=8, color=MUTED, va='top')
        if fi > N_THIEF - 8:
            fig.text(0.5, 0.015, 'the formula is exact; the arithmetic is not: at mu = 0.3 the noise floor is amplified 5e16-fold',
                     ha='center', fontsize=9.2, color=C_GOLD)
        frames.append(fig_rgba(fig))
    save_gif(PUB / 'momentum-rewind.gif', frames, fps=11, hold=24)


# ═══════════════════════════════════════════════════════════════════════════
# GIF 6 -- the exactness ceiling
# ═══════════════════════════════════════════════════════════════════════════
def gif_ceiling():
    runs = RESULTS['network_runs']
    depths = [8, 32, 128]
    acc = {mu: np.array([[a * 100 for a in runs[f'rings/L{L}/mu{mu}']['accs']]
                         for L in depths]) for mu in ('0.0', '0.9')}
    xpos = np.array([0.0, 1.0, 2.0])
    Y0, Y1 = 99.42, 100.10

    N_INTRO, N_PER, N_RESOLVE = 12, 22, 40
    NF = N_INTRO + 3 * N_PER + N_RESOLVE
    frames = []
    for fi in range(NF):
        fig = plt.figure(figsize=(8.4, 5.2), dpi=105, facecolor=BG)
        fig.text(0.5, 0.95, 'The exactness ceiling: 100% on rings is a line a flow cannot touch',
                 ha='center', fontsize=13.2, weight='bold')
        fig.text(0.5, 0.897, 'test accuracy vs depth, 3 seeds per point, total time fixed (T = 8): deeper means smaller steps, closer to an honest flow',
                 ha='center', fontsize=8.7, color=MUTED)
        ax = fig.add_axes([0.10, 0.165, 0.86, 0.665])
        style_ax(ax)
        ax.set_xlim(-0.35, 2.35); ax.set_ylim(Y0, Y1)
        ax.set_xticks(xpos)
        ax.set_xticklabels(['L = 8\n(h = 1)', 'L = 32\n(h = 0.25)', 'L = 128\n(h = 0.0625)'],
                           fontsize=8.5, color=MUTED)
        ax.set_ylabel('test accuracy (%)', fontsize=9, color=MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.axhline(100, color=INK, lw=1.1, ls='--', alpha=0.85)
        ax.text(-0.30, 100.015, 'exact separation (100%)', fontsize=8.5, color=INK)

        stage = (fi - N_INTRO) / N_PER if fi >= N_INTRO else 0.0
        ndep = min(3, int(stage) + (1 if fi >= N_INTRO else 0))
        frac = min(1.0, stage - int(stage) if stage < 3 else 1.0)
        resolve = ease((fi - (N_INTRO + 3 * N_PER)) / (N_RESOLVE - 1)) if fi >= N_INTRO + 3 * N_PER else 0.0

        for mu, col, name, dx in (('0.0', C_PLAIN, 'plain residual (mu = 0)', -0.05),
                                  ('0.9', C_MOM, 'momentum (mu = 0.9)', 0.05)):
            A = acc[mu]
            for di in range(ndep):
                a = ease(min(1.0, (stage - di) * 1.6))
                for s in range(3):
                    ax.scatter([xpos[di] + dx], [A[di, s]], s=44, color=col, alpha=a,
                               zorder=5, edgecolors=BG, linewidths=0.6)
                if di > 0:
                    am = ease(min(1.0, (stage - di) * 1.6))
                    ax.plot(xpos[di - 1:di + 1] + dx, A[di - 1:di + 1].mean(1), color=col,
                            lw=1.4, alpha=0.6 * am, zorder=3)
                if a >= 1:
                    ax.fill_between([xpos[di] + dx - 0.06, xpos[di] + dx + 0.06],
                                    A[di].min(), A[di].max(), color=col, alpha=0.18, zorder=2)
            ax.scatter([], [], s=44, color=col, label=name)
        ax.legend(loc='lower left', fontsize=8.5, facecolor=PANEL, edgecolor=LINE,
                  labelcolor=INK)

        if resolve > 0:
            # shade the forbidden gap where the plain net is an honest flow
            for di in (1, 2):
                top = 100.0; bot = acc['0.0'][di].max()
                ax.fill_between([xpos[di] - 0.22, xpos[di] + 0.22], bot, top,
                                color=C_PLAIN, alpha=0.22 * resolve, zorder=1)
                ax.text(xpos[di] - 0.25, bot - 0.012, f'best plain seed {bot:.2f}%',
                        ha='right' if di == 1 else 'right', va='top', fontsize=7.6,
                        color=C_PLAIN, alpha=resolve)
            ax.annotate('h = 1 is not yet a flow:\ncoarse steps can jump the wall\n(2 of 3 seeds hit 100%)',
                        xy=(xpos[0] - 0.05, 100.0), xytext=(xpos[0] + 0.02, 99.62),
                        fontsize=7.8, color=MUTED, alpha=resolve,
                        arrowprops=dict(arrowstyle='->', color=MUTED, lw=0.8,
                                        alpha=0.7 * resolve))
            fig.text(0.5, 0.008,
                     'once deep enough to be an honest flow (L >= 32), the plain net never touches 100%\n'
                     'in any of its 6 runs; the momentum net pins the line in every seed at L = 8 and L = 32',
                     ha='center', va='bottom', fontsize=8.8, color=C_GOLD, alpha=resolve)
        frames.append(fig_rgba(fig))
    save_gif(PUB / 'momentum-ceiling.gif', frames, fps=11, hold=26)


GIFS = {'orbits': gif_orbits, 'lapse': gif_depth_lapse, 'ledger': gif_velocity_ledger,
        'crystal': gif_crystallize, 'rewind': gif_rewind, 'ceiling': gif_ceiling}

if __name__ == '__main__':
    which = sys.argv[1:] or list(GIFS)
    for w in which:
        print(f'--- {w} ---')
        GIFS[w]()
    print('MOMENTUM_GIFS_DONE')
