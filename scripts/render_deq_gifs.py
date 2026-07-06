"""Six teaching GIFs for the Yat deep-equilibrium JAX/Flax NNX companion.

Every moving thing is a real number recomputed from the trained operator saved in
public/yat-deq/model.json and public/yat-deq-maze/model.json (the exact params from
scripts/yat_deq.py and scripts/yat_deq_maze.py). Nothing is trained here: each GIF is
a forward iteration of the shared operator, or the implicit-differentiation backward
solve, run live in numpy/JAX. The physical world is the series' one world: the Yat
denominator is a softened inverse-square well, and the fixed-point iteration is the
state rolling down into the single basin the contraction carves.

  1. deq-settle.gif        residual ||z_{k+1}-z_k|| collapsing on a log axis while the
                           PCA-projected state rolls into z* inside the equilibrium cloud.
  2. deq-contraction.gif   four scattered starts collapsing onto one z*; the spread
                           shrinking ~0.66 per step (Banach contraction), log axis.
  3. deq-depth-dial.gif    the decision boundary forming then FREEZING across iterations,
                           accuracy plateauing at turn 6.
  4. deq-adaptive.gif      the test-set settle-time map filling in as the budget sweeps
                           14 -> 65, and the histogram of turns-to-settle building up.
  5. deq-implicit.gif      the CENTERPIECE: the backward fixed-point adjoint residual
                           decaying to zero next to the forward one, "the same
                           contraction runs the backward solve."
  6. deq-maze.gif          a 27x27 flood-fill with 11x11-trained weights climbing
                           76% -> 99.5%, plus the r=0.98 settle-vs-radius scatter.

Run: python scripts/render_deq_gifs.py
"""
import warnings; warnings.filterwarnings('ignore')
import json, os
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import imageio.v2 as imageio
from PIL import Image

jax.config.update('jax_platform_name', 'cpu')
ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / 'public'
DEQ = json.load(open(PUB / 'yat-deq' / 'model.json'))
MAZE = json.load(open(PUB / 'yat-deq-maze' / 'model.json'))

# ── palette (matches the series' warm-dark world) ──
BG = '#0e0d0b'; PANEL = '#16140f'; INK = '#e8e2d4'; MUTED = '#9a9282'; LINE = '#3a352c'
C0 = '#4a7fb3'; C1 = '#c2553a'            # the two moon classes (blue / warm)
ACC = '#c0892a'; GOOD = '#3a8f5e'; HOT = '#d05a2a'
START_COLS = ['#4a7fb3', '#c2553a', '#3a8f5e', '#9a4f9c']
plt.rcParams.update({'figure.facecolor': BG, 'savefig.facecolor': BG, 'text.color': INK,
                     'axes.edgecolor': LINE, 'axes.labelcolor': MUTED,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'font.size': 11})


def ease(t):
    return t * t * (3 - 2 * t)


def fig_rgba(fig):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba


def save_gif(path, frames, fps, hold=16, palettesize=96, scale=None):
    if scale is not None:
        frames = [np.asarray(Image.fromarray(f).resize(
            (int(f.shape[1] * scale), int(f.shape[0] * scale)), Image.LANCZOS)) for f in frames]
    Image.fromarray(frames[-1]).save(str(path).replace('.gif', '-preview.png'))
    frames = frames + [frames[-1]] * hold
    imageio.mimsave(path, frames, duration=1 / fps, loop=0, palettesize=palettesize, subrectangles=True)
    print(f'wrote {path}  ({os.path.getsize(path)//1024} KB, {len(frames)} frames)')


# ═══════════════════════════════════════════════════════════════════════════
# The trained two-moons operator, recomputed from model.json (pure numpy)
# ═══════════════════════════════════════════════════════════════════════════
P = {k: (np.array(v, np.float64) if isinstance(v, list) else v) for k, v in DEQ['params'].items()}
D = DEQ['dims']['d']; BETA = DEQ['solver']['beta']; MP = DEQ['dims']['m']
BVAL, EPS = P['b'], P['eps']
Xte = np.array(DEQ['test']['x']); yte = np.array(DEQ['test']['y'])
PCA_MEAN = np.array(DEQ['pca']['mean']); PCA_BASIS = np.array(DEQ['pca']['basis'])  # (2, D)
PROJ = np.array(DEQ['test']['proj'])                                                # (Nte, 2)
KC = np.array(DEQ['test']['kcount'])
MET = DEQ['metrics']


def yat_feat(z):
    dot = z @ P['W'].T
    d2 = (z ** 2).sum(-1, keepdims=True) + (P['W'] ** 2).sum(-1) - 2 * dot
    return (dot + BVAL) ** 2 / (d2 + EPS)


def Fop(x, z):
    return np.tanh(yat_feat(z) @ P['A'].T + x @ P['Uin'].T + P['z0'])


def picard_step(x, z):
    return (1 - BETA) * z + BETA * Fop(x, z)


def readout(z):
    return z @ P['C'].T + P['cb']


def to_pca(z):                       # project a D-vector (or [N,D]) onto the 2-D equilibrium window
    return (z - PCA_MEAN) @ PCA_BASIS.T


# ═══════════════════════════════════════════════════════════════════════════
# GIF 1 — settling to equilibrium: residual collapses, state rolls to z*
# ═══════════════════════════════════════════════════════════════════════════
def gif_settle():
    # one representative "medium" input (near the median settle time)
    idx = int(np.argsort(np.abs(KC - np.median(KC)))[3])
    x = Xte[idx:idx + 1]
    z = np.zeros((1, D)); path = [z.copy()]; resid = []
    for k in range(80):
        zn = picard_step(x, z)
        resid.append(float(np.linalg.norm(zn - z)))
        z = zn; path.append(z.copy())
    path = np.concatenate(path, 0)            # (K+1, D)
    ppath = to_pca(path)                       # (K+1, 2)
    zstar = ppath[-1]
    cloud = PROJ                               # every input's equilibrium as faint dots
    K = 46                                      # frames of real iteration to show
    xlo, xhi = cloud[:, 0].min() - .3, cloud[:, 0].max() + .3
    ylo, yhi = cloud[:, 1].min() - .3, cloud[:, 1].max() + .3
    frames = []
    for fi in range(K):
        step = fi
        fig = plt.figure(figsize=(9.4, 4.5), dpi=110, facecolor=BG)
        fig.text(0.5, 0.94, 'One input, run until it stops: the state rolls to its fixed point',
                 ha='center', fontsize=14, weight='bold')
        fig.text(0.5, 0.875, 'left: a 2-D window on the 32-D state settling into z*     right: the step size ‖z_{k+1}−z_k‖ collapsing',
                 ha='center', fontsize=9.5, color=MUTED)
        axL = fig.add_axes([0.05, 0.08, 0.42, 0.68]); axL.set_facecolor(PANEL)
        axR = fig.add_axes([0.57, 0.13, 0.39, 0.63]); axR.set_facecolor(PANEL)
        # left: state trajectory over the equilibrium cloud
        axL.scatter(cloud[:, 0], cloud[:, 1], s=6, c=[MUTED], alpha=0.18, linewidths=0, zorder=0)
        axL.plot(ppath[:step + 1, 0], ppath[:step + 1, 1], color=ACC, lw=1.6, alpha=0.9, zorder=2)
        axL.scatter([ppath[0, 0]], [ppath[0, 1]], s=42, facecolor='none', edgecolor=INK, lw=1.2, zorder=3)
        axL.text(ppath[0, 0], ppath[0, 1] - .18, 'start z₀ = 0', ha='center', fontsize=8, color=MUTED)
        axL.scatter([ppath[step, 0]], [ppath[step, 1]], s=70, color=HOT, edgecolor=INK, lw=.8, zorder=5)
        axL.scatter([zstar[0]], [zstar[1]], marker='*', s=210, color=ACC, edgecolor=INK, lw=.6, zorder=4)
        axL.text(zstar[0], zstar[1] + .16, 'z*', ha='center', fontsize=11, color=ACC, weight='bold')
        axL.set_xlim(xlo, xhi); axL.set_ylim(ylo, yhi); axL.set_xticks([]); axL.set_yticks([])
        axL.set_title('state (2-D window on 32 dims)', fontsize=10, color=MUTED, pad=4)
        # right: residual on a log axis, drawing itself
        axR.semilogy(np.arange(1, step + 2), resid[:step + 1], color=HOT, lw=1.8)
        axR.scatter([step + 1], [resid[step]], s=34, color=HOT, zorder=5)
        axR.axhline(DEQ['solver']['tol'], ls='--', color=MUTED, lw=1)
        axR.text(K * 0.98, DEQ['solver']['tol'] * 1.5, 'halt tol', ha='right', fontsize=8, color=MUTED)
        axR.set_xlim(0, K); axR.set_ylim(1e-5, 5)
        axR.set_xlabel('turn  k  (= depth)'); axR.set_ylabel('step size  ‖z_{k+1}−z_k‖')
        axR.set_title('the state stops moving', fontsize=10, color=MUTED, pad=4)
        axR.grid(alpha=.16)
        for ax in (axL, axR):
            for s in ax.spines.values(): s.set_color(LINE)
        fig.text(0.5, 0.015, f'turn {step + 1}   ·   step size {resid[step]:.1e}',
                 ha='center', fontsize=10, color=INK)
        frames.append(fig_rgba(fig))
    save_gif(PUB / 'deq-settle.gif', frames, fps=11, scale=0.82)


# ═══════════════════════════════════════════════════════════════════════════
# GIF 2 — contraction / Banach: scattered starts collapse to one z*
# ═══════════════════════════════════════════════════════════════════════════
def gif_contraction():
    idx = int(np.argsort(np.abs(KC - np.median(KC)))[6])
    x = Xte[idx:idx + 1]
    rng = np.random.RandomState(3)
    starts = np.tanh(rng.randn(4, D) * 1.6)               # four wildly scattered states in (-1,1)^D
    Z = starts.copy(); paths = [Z.copy()]; spread = []
    for k in range(40):
        Zn = np.stack([picard_step(x, Z[i:i + 1])[0] for i in range(4)])
        # spread = mean pairwise distance between the four states
        d = [np.linalg.norm(Zn[i] - Zn[j]) for i in range(4) for j in range(i + 1, 4)]
        spread.append(float(np.mean(d)))
        Z = Zn; paths.append(Z.copy())
    paths = np.stack(paths, 0)                             # (K+1, 4, D)
    pp = np.stack([to_pca(paths[:, i, :]) for i in range(4)], 1)   # (K+1, 4, 2)
    zstar = pp[-1].mean(0)
    # empirical per-step ratio -> the contraction rate
    ratio = np.mean([spread[k + 1] / (spread[k] + 1e-12) for k in range(len(spread) - 1)])
    K = 34
    allp = pp[:K].reshape(-1, 2)
    xlo, xhi = allp[:, 0].min() - .2, allp[:, 0].max() + .2
    ylo, yhi = allp[:, 1].min() - .2, allp[:, 1].max() + .2
    frames = []
    for fi in range(K):
        step = fi
        fig = plt.figure(figsize=(9.4, 4.5), dpi=110, facecolor=BG)
        fig.text(0.5, 0.94, 'Same input, four different starts: they all fall into one z*',
                 ha='center', fontsize=14, weight='bold')
        fig.text(0.5, 0.875, 'a contraction brings any two states closer every turn, so the answer never depends on where you began',
                 ha='center', fontsize=9.5, color=MUTED)
        axL = fig.add_axes([0.05, 0.09, 0.42, 0.72]); axL.set_facecolor(PANEL)
        axR = fig.add_axes([0.57, 0.13, 0.39, 0.66]); axR.set_facecolor(PANEL)
        for i in range(4):
            axL.plot(pp[:step + 1, i, 0], pp[:step + 1, i, 1], color=START_COLS[i], lw=1.5, alpha=0.85, zorder=2)
            axL.scatter([pp[0, i, 0]], [pp[0, i, 1]], s=34, facecolor='none',
                        edgecolor=START_COLS[i], lw=1.3, zorder=3)
            axL.scatter([pp[step, i, 0]], [pp[step, i, 1]], s=48, color=START_COLS[i],
                        edgecolor=INK, lw=.5, zorder=4)
        axL.scatter([zstar[0]], [zstar[1]], marker='*', s=230, color=ACC, edgecolor=INK, lw=.6, zorder=5)
        axL.text(zstar[0], zstar[1] + .12, 'z*', ha='center', fontsize=11, color=ACC, weight='bold')
        axL.set_xlim(xlo, xhi); axL.set_ylim(ylo, yhi); axL.set_xticks([]); axL.set_yticks([])
        axL.set_title('four trajectories (2-D window)', fontsize=10, color=MUTED, pad=4)
        axR.semilogy(np.arange(1, step + 2), spread[:step + 1], color=GOOD, lw=1.9)
        axR.scatter([step + 1], [spread[step]], s=34, color=GOOD, zorder=5)
        axR.set_xlim(0, K); axR.set_ylim(1e-4, 3)
        axR.set_xlabel('turn  k'); axR.set_ylabel('spread between the four states')
        axR.set_title(f'geometric decay  ·  ≈{ratio:.2f} per turn', fontsize=10, color=MUTED, pad=4)
        axR.grid(alpha=.16)
        for ax in (axL, axR):
            for s in ax.spines.values(): s.set_color(LINE)
        fig.text(0.5, 0.015, f'turn {step + 1}   ·   spread {spread[step]:.1e}   ·   ‖J‖ ≈ {MET["jnorm_mean"]:.2f}',
                 ha='center', fontsize=10, color=INK)
        frames.append(fig_rgba(fig))
    save_gif(PUB / 'deq-contraction.gif', frames, fps=11, scale=0.82)


# ═══════════════════════════════════════════════════════════════════════════
# GIF 3 — depth dial: the boundary forms then FREEZES, accuracy plateaus at 6
# ═══════════════════════════════════════════════════════════════════════════
def gif_depth_dial():
    G = 150
    gx, gy = np.meshgrid(np.linspace(-2.4, 2.4, G), np.linspace(-2.4, 2.4, G))
    grid = np.c_[gx.ravel(), gy.ravel()]
    # precompute the class-1 probability at each depth on the whole plane + accuracy
    z = np.zeros((grid.shape[0], D)); zt = np.zeros((Xte.shape[0], D))
    probs, accs = [], []
    KMAX = 22
    for k in range(KMAX):
        z = picard_step(grid, z); zt = picard_step(Xte, zt)
        p1 = 1 / (1 + np.exp(-(readout(z)[:, 1] - readout(z)[:, 0])))
        probs.append(p1.reshape(G, G))
        accs.append(float((np.argmax(readout(zt), 1) == yte).mean()))
    accs = np.array(accs)
    plateau = MET['acc_depth']                     # 6
    frames = []
    for k in range(KMAX):
        fig = plt.figure(figsize=(9.4, 4.6), dpi=110, facecolor=BG)
        fig.text(0.5, 0.94, 'Depth as a dial: the boundary forms, then freezes',
                 ha='center', fontsize=14, weight='bold')
        fig.text(0.5, 0.875, 'apply the shared operator k times to the whole plane; past the plateau, more depth changes nothing',
                 ha='center', fontsize=9.5, color=MUTED)
        axL = fig.add_axes([0.045, 0.05, 0.44, 0.70]); axL.set_facecolor(PANEL)
        axR = fig.add_axes([0.585, 0.14, 0.38, 0.60]); axR.set_facecolor(PANEL)
        axL.contourf(gx, gy, probs[k], levels=18, cmap='RdBu_r', alpha=.72)
        axL.contour(gx, gy, probs[k], levels=[0.5], colors=[INK], linewidths=1.5)
        axL.scatter(Xte[yte == 0, 0], Xte[yte == 0, 1], s=7, c=[C0], edgecolor='w', lw=.15, zorder=3)
        axL.scatter(Xte[yte == 1, 0], Xte[yte == 1, 1], s=7, c=[C1], edgecolor='w', lw=.15, zorder=3)
        axL.set_xlim(-2.4, 2.4); axL.set_ylim(-2.4, 2.4); axL.set_xticks([]); axL.set_yticks([])
        state = 'still forming' if k + 1 < plateau else 'frozen'
        axL.set_title(f'decision boundary at turn {k + 1}  ({state})', fontsize=10, color=MUTED, pad=4)
        axR.plot(np.arange(1, k + 2), accs[:k + 1], color=ACC, lw=1.9)
        axR.scatter([k + 1], [accs[k]], s=34, color=ACC, zorder=5)
        axR.axvline(plateau, ls='--', color=GOOD, lw=1.2)
        axR.text(plateau + .3, 0.55, f'plateau @ {plateau}', fontsize=8.5, color=GOOD)
        axR.set_xlim(0, KMAX); axR.set_ylim(0.45, 1.0)
        axR.set_xlabel('turn  k  (= depth)'); axR.set_ylabel('test accuracy')
        axR.set_title('accuracy vs depth', fontsize=10, color=MUTED, pad=4)
        axR.grid(alpha=.16)
        for ax in (axL, axR):
            for s in ax.spines.values(): s.set_color(LINE)
        fig.text(0.5, 0.015, f'turn {k + 1}   ·   accuracy {accs[k]*100:.1f}%',
                 ha='center', fontsize=10, color=INK)
        frames.append(fig_rgba(fig))
    save_gif(PUB / 'deq-depth-dial.gif', frames, fps=6, scale=0.82)


# ═══════════════════════════════════════════════════════════════════════════
# GIF 4 — adaptive depth: settle-time map fills in, histogram builds up
# ═══════════════════════════════════════════════════════════════════════════
def gif_adaptive():
    kmin, kmax = int(KC.min()), int(KC.max())          # 14 .. 65
    budgets = list(range(kmin, kmax + 1, 2)) + [kmax]
    hist_bins = np.arange(kmin, kmax + 4, 3)
    frames = []
    for bi, budget in enumerate(budgets):
        done = KC <= budget
        fig = plt.figure(figsize=(9.4, 4.6), dpi=110, facecolor=BG)
        fig.text(0.5, 0.94, 'Depth spent, per input: easy points settle first',
                 ha='center', fontsize=14, weight='bold')
        fig.text(0.5, 0.875, 'sweep the depth budget: filled = has settled, hollow = still iterating. The stragglers hug the boundary.',
                 ha='center', fontsize=9.5, color=MUTED)
        axL = fig.add_axes([0.045, 0.06, 0.44, 0.76]); axL.set_facecolor(PANEL)
        axR = fig.add_axes([0.585, 0.15, 0.38, 0.60]); axR.set_facecolor(PANEL)
        for cls, col in ((0, C0), (1, C1)):
            m = (yte == cls) & done
            axL.scatter(Xte[m, 0], Xte[m, 1], s=22, c=[col], edgecolor='w', lw=.2, zorder=3)
            m = (yte == cls) & ~done
            axL.scatter(Xte[m, 0], Xte[m, 1], s=22, facecolor='none', edgecolor=col, lw=1.0, zorder=2)
        axL.set_xlim(-2.4, 2.4); axL.set_ylim(-2.4, 2.4); axL.set_xticks([]); axL.set_yticks([])
        axL.set_title(f'budget = {budget} turns   ·   {done.mean()*100:.0f}% settled',
                      fontsize=10, color=MUTED, pad=4)
        counts, edges = np.histogram(KC[done], bins=hist_bins)
        axR.bar(edges[:-1], counts, width=np.diff(edges) * 0.9, align='edge',
                color=ACC, alpha=0.85, edgecolor=INK, lw=.3)
        axR.axvline(budget, ls='--', color=GOOD, lw=1.2)
        axR.set_xlim(kmin - 1, kmax + 3); axR.set_ylim(0, 130)
        axR.set_xlabel('turns to settle'); axR.set_ylabel('# test inputs')
        axR.set_title(f'easiest {kmin}  ·  median {int(np.median(KC))}  ·  hardest {kmax}',
                      fontsize=10, color=MUTED, pad=4)
        axR.grid(alpha=.16, axis='y')
        for ax in (axL, axR):
            for s in ax.spines.values(): s.set_color(LINE)
        frames.append(fig_rgba(fig))
    save_gif(PUB / 'deq-adaptive.gif', frames, fps=7, scale=0.82)


# ═══════════════════════════════════════════════════════════════════════════
# GIF 5 — implicit differentiation (the centerpiece): forward + backward
#          fixed-point residuals both decay, driven by the SAME contraction
# ═══════════════════════════════════════════════════════════════════════════
def gif_implicit():
    # JAX operator so we can take the real Jacobian-transpose vjp
    Pj = {k: (jnp.array(v) if isinstance(v, (list,)) else v) for k, v in DEQ['params'].items()}
    Xj = jnp.array(Xte); yj = jnp.array(yte)

    def yatj(z):
        dot = z @ Pj['W'].T
        d2 = (z ** 2).sum(-1, keepdims=True) + (Pj['W'] ** 2).sum(-1) - 2 * dot
        return (dot + Pj['b']) ** 2 / (d2 + Pj['eps'])

    def Fj(z):
        return jnp.tanh(yatj(z) @ Pj['A'].T + Xj @ Pj['Uin'].T + Pj['z0'])

    # forward solve, logging residual
    z = jnp.zeros((Xj.shape[0], D)); fwd = []
    for k in range(80):
        zn = (1 - BETA) * z + BETA * Fj(z)
        fwd.append(float(jnp.max(jnp.linalg.norm(zn - z, axis=-1)))); z = zn
    zstar = z

    # loss gradient wrt z* (softmax CE), then adjoint solve (I - J^T) u = g
    def lossz(zz):
        lg = zz @ Pj['C'].T + Pj['cb']
        return -jnp.mean(jax.nn.log_softmax(lg)[jnp.arange(len(yj)), yj])
    g = jax.grad(lossz)(zstar)
    _, vjp_z = jax.vjp(Fj, zstar)                      # J^T applied to a cotangent
    u = jnp.zeros_like(g); bwd = []
    for k in range(80):
        un = (1 - BETA) * u + BETA * (vjp_z(u)[0] + g)
        bwd.append(float(jnp.max(jnp.linalg.norm(un - u, axis=-1)))); u = un

    fwd = np.array(fwd); bwd = np.array(bwd)
    K = 60
    frames = []
    for step in range(K):
        fig = plt.figure(figsize=(9.6, 4.7), dpi=110, facecolor=BG)
        fig.text(0.5, 0.945, 'Training a fixed point: the same contraction runs the backward solve',
                 ha='center', fontsize=13.5, weight='bold')
        fig.text(0.5, 0.885, 'no layers to backprop through. Differentiate the equation z*=F(z*): the adjoint (I − Jᵀ)u = ∂L/∂z* is one more fixed point.',
                 ha='center', fontsize=9.0, color=MUTED)
        axL = fig.add_axes([0.075, 0.17, 0.40, 0.60]); axL.set_facecolor(PANEL)
        axR = fig.add_axes([0.57, 0.17, 0.40, 0.60]); axR.set_facecolor(PANEL)
        axL.semilogy(np.arange(1, step + 2), fwd[:step + 1], color=ACC, lw=1.9)
        axL.scatter([step + 1], [fwd[step]], s=32, color=ACC, zorder=5)
        axL.set_xlim(0, K); axL.set_ylim(1e-8, 5)
        axL.set_xlabel('turn  k'); axL.set_ylabel('residual')
        axL.set_title('FORWARD:  solve  z = F(z)', fontsize=10.5, color=ACC, pad=4)
        axL.grid(alpha=.16)
        axR.semilogy(np.arange(1, step + 2), bwd[:step + 1], color=GOOD, lw=1.9)
        axR.scatter([step + 1], [bwd[step]], s=32, color=GOOD, zorder=5)
        axR.set_xlim(0, K); axR.set_ylim(1e-8, 5)
        axR.set_xlabel('turn  k'); axR.set_ylabel('residual')
        axR.set_title('BACKWARD:  solve  u = Jᵀu + ∂L/∂z*', fontsize=10.5, color=GOOD, pad=4)
        axR.grid(alpha=.16)
        for ax in (axL, axR):
            for s in ax.spines.values(): s.set_color(LINE)
        fig.text(0.5, 0.03, f'turn {step + 1}   ·   forward residual {fwd[step]:.1e}   ·   backward residual {bwd[step]:.1e}',
                 ha='center', fontsize=10, color=INK)
        frames.append(fig_rgba(fig))
    save_gif(PUB / 'deq-implicit.gif', frames, fps=12, scale=0.82)


# ═══════════════════════════════════════════════════════════════════════════
# The trained maze operator (weight-tied recurrent convolution), pure numpy
# ═══════════════════════════════════════════════════════════════════════════
MP_ = {k: (np.array(v, np.float64) if isinstance(v, list) else v) for k, v in MAZE['params'].items()}
DM = MAZE['dims']['d']; MB, MEPS = MP_['b'], MP_['eps']


def m_shift(z, di, dj):
    r = np.roll(z, shift=(di, dj), axis=(0, 1))
    if di > 0: r[:di] = 0
    if di < 0: r[di:] = 0
    if dj > 0: r[:, :dj] = 0
    if dj < 0: r[:, dj:] = 0
    return r


def m_patch(z):
    return np.concatenate([z, m_shift(z, 1, 0), m_shift(z, -1, 0),
                           m_shift(z, 0, 1), m_shift(z, 0, -1)], -1)


def m_F(z, x, free):
    pt = m_patch(z)
    dot = pt @ MP_['W'].T
    d2 = (pt ** 2).sum(-1, keepdims=True) + (MP_['W'] ** 2).sum(-1) - 2 * dot
    phi = (dot + MB) ** 2 / (d2 + MEPS)
    pre = phi @ MP_['A'].T + x @ MP_['Uin'].T + MP_['z0']
    return np.tanh(pre) * free[..., None]


def m_logit(z):
    return (z @ MP_['C'].T + MP_['cb'])[..., 0]


# ═══════════════════════════════════════════════════════════════════════════
# GIF 6 — maze extrapolation: 27x27 flood-fill + settle-vs-radius scatter r=0.98
# ═══════════════════════════════════════════════════════════════════════════
def gif_maze():
    ex = MAZE['examples']['27'][0]                     # radius 52, needs to travel far
    wall = np.array(ex['wall']); free = (1 - wall).astype(np.float64)
    goal = np.zeros_like(free); gi, gj = ex['goal']; goal[gi, gj] = 1.0
    reach = np.array(ex['reach'])
    x = np.stack([free, goal], -1)
    # run the flood, logging belief + accuracy per turn
    z = np.zeros((*free.shape, DM)); beliefs, accs = [], []
    KMAX = 66
    for k in range(KMAX):
        z = m_F(z, x, free)
        b = 1 / (1 + np.exp(-m_logit(z)))
        beliefs.append(b.copy())
        pred = (m_logit(z) > 0).astype(float)
        accs.append(float(((pred == reach) * free).sum() / free.sum()))
    accs = np.array(accs)
    train_len = MAZE['train_size'] * 0 + 30            # T_TRAIN = 30
    # settle-vs-radius scatter (accumulate points)
    radii = np.array(MAZE['adaptive']['radius']); settle = np.array(MAZE['adaptive']['settle'])
    corr = MAZE['adaptive']['corr']
    order = np.argsort(radii)                          # reveal in radius order for a clean sweep
    # display grid: walls dark, goal star, belief warm
    show_ks = list(range(0, KMAX, 2))
    frames = []
    for fi, k in enumerate(show_ks):
        fig = plt.figure(figsize=(9.6, 4.7), dpi=110, facecolor=BG)
        fig.text(0.5, 0.945, 'Trained only on 11×11, flooding a 27×27: the front just keeps going',
                 ha='center', fontsize=13.5, weight='bold')
        fig.text(0.5, 0.885, 'reachable spreads one ring per turn from the goal; a 27×27 needs the front to travel ~40 cells, far past training',
                 ha='center', fontsize=9.0, color=MUTED)
        axL = fig.add_axes([0.05, 0.05, 0.42, 0.74]); axL.set_facecolor(PANEL)
        axR = fig.add_axes([0.58, 0.17, 0.38, 0.60]); axR.set_facecolor(PANEL)
        # left: the maze + current belief
        disp = np.ones((*free.shape, 3)) * np.array([0.086, 0.078, 0.063])   # BG for walls
        bel = beliefs[k]
        warm = np.stack([0.10 + 0.72 * bel, 0.09 + 0.34 * bel, 0.06 + 0.10 * bel], -1)
        m = free[..., None] > 0
        disp = np.where(m, warm, disp)
        axL.imshow(disp, interpolation='nearest')
        axL.scatter([gj], [gi], marker='*', s=150, color='#e8e2d4', edgecolor='#c0892a', lw=.7, zorder=5)
        axL.set_xticks([]); axL.set_yticks([])
        axL.set_title(f'turn {k + 1}   ·   {accs[k]*100:.1f}% of cells correct', fontsize=10, color=MUTED, pad=4)
        # right: accuracy vs turn for THIS maze, with training-length mark + reference dots
        axR.plot(np.arange(1, k + 2), accs[:k + 1] * 100, color=HOT, lw=1.9)
        axR.scatter([k + 1], [accs[k] * 100], s=32, color=HOT, zorder=5)
        axR.axvline(30, ls='--', color=MUTED, lw=1.1)
        axR.text(30.5, 20, 'trained to\n30 turns', fontsize=8, color=MUTED)
        # annotate the milestone numbers as the curve passes them
        if k + 1 >= 22:
            axR.scatter([22], [54.2], s=26, color=MUTED, zorder=4)
            axR.text(22, 60, '54% @22', fontsize=7.5, color=MUTED, ha='center')
        if k + 1 >= 30:
            axR.scatter([30], [76.0], s=26, color=MUTED, zorder=4)
            axR.text(31, 80, '76% @30', fontsize=7.5, color=MUTED)
        axR.set_xlim(0, KMAX); axR.set_ylim(0, 105)
        axR.set_xlabel('turns allowed'); axR.set_ylabel('cell accuracy (%)')
        axR.set_title('every turn past training buys accuracy', fontsize=10, color=MUTED, pad=4)
        axR.grid(alpha=.16)
        for ax in (axL, axR):
            for s in ax.spines.values(): s.set_color(LINE)
        fig.text(0.5, 0.025, f'reaches {accs[-1]*100:.1f}% once the front crosses   ·   a 12-layer stack is stuck at 12 hops forever',
                 ha='center', fontsize=9.5, color=INK)
        frames.append(fig_rgba(fig))
    # tail: the settle-vs-radius scatter accumulating (the r=0.98 payoff)
    NS = 18
    for si in range(NS):
        n = int(np.ceil(len(order) * ease((si + 1) / NS)))
        sel = order[:n]
        fig = plt.figure(figsize=(9.6, 4.7), dpi=110, facecolor=BG)
        fig.text(0.5, 0.945, 'The problem hands the network its depth',
                 ha='center', fontsize=13.5, weight='bold')
        fig.text(0.5, 0.885, 'turns each maze actually needs to settle vs its true propagation distance, across 120 instances of every size',
                 ha='center', fontsize=9.0, color=MUTED)
        ax = fig.add_axes([0.12, 0.16, 0.76, 0.62]); ax.set_facecolor(PANEL)
        lim = max(radii.max(), settle.max()) + 6
        ax.plot([0, lim], [0, lim], ls='--', color=MUTED, lw=1)
        ax.scatter(radii[sel], settle[sel], s=30, color=ACC, edgecolor=INK, lw=.3, alpha=0.85)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel('true propagation distance (BFS radius)'); ax.set_ylabel('turns to settle')
        ax.set_title(f'correlation  r = {corr:.2f}', fontsize=11, color=ACC, pad=4)
        ax.grid(alpha=.16)
        for s in ax.spines.values(): s.set_color(LINE)
        fig.text(0.5, 0.03, 'the network never picks a depth; the instance does',
                 ha='center', fontsize=10, color=INK)
        frames.append(fig_rgba(fig))
    save_gif(PUB / 'deq-maze.gif', frames, fps=9, scale=0.82)


if __name__ == '__main__':
    gif_settle()
    gif_contraction()
    gif_depth_dial()
    gif_adaptive()
    gif_implicit()
    gif_maze()
    print('\nall six DEQ GIFs rendered.')
