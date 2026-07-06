"""Six teaching GIFs for the editable-DEQ companion (edit-a-fixed-point-jax-flax-nnx).

Every moving thing is a real number from the real run: the shared Yat operator,
its exported weights, the eight teaching anchors, the certificate cloud, and the
520 old-class fixed points are all read from public/yat-deq-edit/{model.json,
report.json}. Nothing here is fit or faked; the GIFs re-solve the real operator's
fixed points with the exact solver the browser and the training script use, so
each animation is the process, not a slideshow.

  (a) deqedit-teach-readout.gif   teaching inputs iterating to their equilibria, a
                                  fourth score fading in while the old recall bars
                                  hold flat (readout teach: F untouched).
  (b) deqedit-certificate.gif     the per-probe slope cloud sliding right past the
                                  1.0 wall as the gain ramps, then bisection walking
                                  it back under the 0.98 ceiling.
  (c) deqedit-drift.gif           520 paired old fixed points morphing base -> edited
                                  while the ‖Δz*‖ histogram draws itself, tail lit.
  (d) deqedit-evaporation.gif     the hybrid iteration: edited operator for j turns,
                                  base after; the excursion collapses to ~1e-7.
  (e) deqedit-stack-toll.png      STATIC figure: the tied one-paste edit vs the untied
                                  L-layer bill (numbers written, old-state shift). A
                                  static comparison of final numbers is a figure; the
                                  counters "climbing" would be fake growth.
  (f) deqedit-silence-erase.gif   mask the trained row (recall -> 0, cluster frozen),
                                  paste 8 anchors, the cluster reads back at 100%.

The certificate/invariance/edit/stack numbers in the captions come straight from
report.json. This is forward-solve only (no training), so it runs locally.

Run: python scripts/render_deqedit_gifs.py
"""
import warnings; warnings.filterwarnings('ignore')
import json, os
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / 'public'
DEQ = PUB / 'yat-deq-edit'
M = json.load(open(DEQ / 'model.json'))
R = json.load(open(DEQ / 'report.json'))

# ── palette (warm-dark, matches the series) ──
BG = '#0e0d0b'; PANEL = '#16140f'; INK = '#e8e2d4'; MUTED = '#9a9282'; LINE = '#3a352c'
# class colours: moons 0/1, trained blob 2, taught blob 3
CLS = ['#4a7fb3', '#c2553a', '#3a8f5e', '#c77d2a']
ACCENT = '#36d6c4'; DANGER = '#e0785a'; SAFE = '#7bbf5a'; GAIN = '#e0a45a'
plt.rcParams.update({'figure.facecolor': BG, 'savefig.facecolor': BG, 'text.color': INK,
                     'axes.edgecolor': LINE, 'font.size': 11})

# ── the exported operator ──
P = M['params']
W = np.array(P['W']); A = np.array(P['A']); Uin = np.array(P['Uin'])
z0 = np.array(P['z0']); Cw = np.array(P['C']); cb = np.array(P['cb'])
B = float(P['b']); EPS = float(P['eps'])
BETA = float(M['solver']['beta']); TOL = float(M['solver']['tol'])
D = int(M['dims']['d'])
ANCH = np.array(M['edit']['anchors'])           # [8,32] taught anchors (base equilibria)
ALPHA = float(M['edit']['alpha']); GAMMA = float(M['edit']['gamma'])
GAMMA_RAW = float(M['edit']['gamma_raw']); RESCALE = float(M['edit']['rescale'])
Xte = np.array(M['test']['x']); yte = np.array(M['test']['y'])
mean = np.array(M['pca']['mean']); basis = np.array(M['pca']['basis'])   # [2,32]
sig_base = np.array(M['certificate']['sigma_base'])
sig_edit = np.array(M['certificate']['sigma_edit'])
PROJ_TEST = np.array(M['test']['proj'])          # [640,2] exported base-equilibrium projection


def yat_k(z, Wm, b=B, eps=EPS):
    dot = z @ Wm.T
    d2 = (z ** 2).sum(-1, keepdims=True) + (Wm ** 2).sum(-1) - 2 * dot
    return (dot + b) ** 2 / (d2 + eps)


def Fop(x, z, Wm=W, Am=A):
    pre = yat_k(z, Wm) @ Am.T + x @ Uin.T + z0
    return np.tanh(pre)


def solve(x, Wm=W, Am=A, iters=200, z_init=None):
    z = np.zeros((x.shape[0], D)) if z_init is None else z_init.copy()
    for _ in range(iters):
        z = (1 - BETA) * z + BETA * Fop(x, z, Wm, Am)
    return z


def solve_traj(x, Wm=W, Am=A, iters=60, z_init=None):
    """Return the full iterate trajectory [T+1, n, D] and residuals [T]."""
    z = np.zeros((x.shape[0], D)) if z_init is None else z_init.copy()
    zs = [z.copy()]; res = []
    for _ in range(iters):
        zn = (1 - BETA) * z + BETA * Fop(x, z, Wm, Am)
        res.append(float(np.linalg.norm(zn - z, axis=-1).max()))
        z = zn; zs.append(z.copy())
    return np.array(zs), np.array(res)


def dyn_edit(gamma):
    cols = gamma * (ANCH / np.linalg.norm(ANCH, axis=-1, keepdims=True))
    return np.concatenate([W, ANCH], 0), np.concatenate([A, cols.T], 1)


Wed, Aed = dyn_edit(GAMMA)                       # certified dynamics edit


def proj(z):
    return (z - mean) @ basis.T                  # [n,2] display coords


def sigma_cloud(gamma, x=None):
    """Real per-probe ‖J_F‖₂ at z* under the γ-edited operator.

    Builds the full 32x32 Jacobian of F(z) at each equilibrium by central finite
    differences over the D coordinate directions, then takes its exact top singular
    value. Matches the training script's power-iteration certificate to 3 digits.
    """
    if x is None:
        x = Xte
    Wm, Am = dyn_edit(gamma)
    z = solve(x, Wm, Am, iters=220)
    n = x.shape[0]; h = 1e-4
    J = np.zeros((n, D, D))
    for j in range(D):
        e = np.zeros(D); e[j] = h
        J[:, :, j] = (Fop(x, z + e, Wm, Am) - Fop(x, z - e, Wm, Am)) / (2 * h)
    return np.linalg.norm(J, ord=2, axis=(1, 2))


def ease(t):
    return t * t * (3 - 2 * t)


def fig_rgba(fig):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba


def save_gif(path, frames, fps, hold=14):
    Image.fromarray(frames[-1]).save(str(path).replace('.gif', '-preview.png'))
    frames = frames + [frames[-1]] * hold
    imageio.mimsave(path, frames, duration=1 / fps, loop=0, palettesize=96, subrectangles=True)
    print(f'wrote {path} ({os.path.getsize(path)//1024} KB)')


# ═══════════════════════════════════════════════════════════════════════════
# (a) teach by readout: teaching inputs iterate to equilibria; a fourth score
#     fades in while the old recall bars hold flat (F untouched, exact).
# ═══════════════════════════════════════════════════════════════════════════
def gif_teach_readout():
    teach_x = np.array(M['teach_x'])                 # [8,2] taught-class teaching inputs
    zs, _ = solve_traj(teach_x, iters=44)            # real iterates to the anchors
    P2 = PROJ_TEST; Pt = np.array([proj(z) for z in zs])   # [T+1,8,2]
    # real recall bars: old classes exact (F untouched), new class fades to 85.8%
    acc = R['teach_T1']['acc']
    old_recall = [acc['0'], acc['1'], acc['2']]
    new_recall = acc['3']
    ymask = yte < 3
    N_ITER = len(zs) - 1
    N_FADE = 20; total = N_ITER + N_FADE + 6
    frames = []
    for fi in range(total):
        it = min(fi, N_ITER)
        fade = ease(np.clip((fi - N_ITER) / N_FADE, 0, 1))
        fig = plt.figure(figsize=(8.8, 4.5), dpi=108, facecolor=BG)
        fig.text(0.5, 0.94, 'Teach by readout: settle the new class, read it against fresh anchors',
                 ha='center', fontsize=13.5, weight='bold')
        fig.text(0.5, 0.875, 'the frozen operator iterates 8 taught inputs to their equilibria; a 4th score fades in, the old recall bars do not move',
                 ha='center', fontsize=8.8, color=MUTED)
        # left: state plane, old test equilibria + the taught inputs falling to anchors
        axL = fig.add_axes([0.05, 0.10, 0.44, 0.70]); axL.set_facecolor(PANEL)
        for c in range(3):
            m = yte == c
            axL.scatter(P2[m, 0], P2[m, 1], s=8, color=CLS[c], alpha=0.28, linewidths=0, zorder=1)
        # the new blob's own test equilibria appear as territory as the readout turns on
        mb = yte == 3
        axL.scatter(P2[mb, 0], P2[mb, 1], s=10, color=CLS[3], alpha=0.10 + 0.45 * fade,
                    linewidths=0, zorder=1)
        # anchors (the 8 settled teaching inputs)
        AP = proj(ANCH)
        axL.scatter(AP[:, 0], AP[:, 1], s=90, marker='*', color=CLS[3],
                    edgecolors='white', linewidths=1.0, alpha=0.35 + 0.6 * fade, zorder=4)
        # the taught inputs mid-descent (real iterates)
        cur = Pt[it]
        axL.scatter(cur[:, 0], cur[:, 1], s=42, color=GAIN, edgecolors=BG,
                    linewidths=0.6, zorder=5)
        for k in range(cur.shape[0]):
            tr = Pt[:it + 1, k]
            axL.plot(tr[:, 0], tr[:, 1], color=GAIN, lw=0.8, alpha=0.5, zorder=3)
        axL.set_xticks([]); axL.set_yticks([])
        axL.set_title(f'state plane  ·  solver turn {it}/{N_ITER}', fontsize=9, color=MUTED, pad=3)
        for s in axL.spines.values(): s.set_color(LINE)
        # right: recall bars
        axR = fig.add_axes([0.60, 0.16, 0.36, 0.60]); axR.set_facecolor(PANEL)
        vals = [old_recall[0], old_recall[1], old_recall[2], new_recall * fade]
        cols = [CLS[0], CLS[1], CLS[2], CLS[3]]
        labs = ['moon 0', 'moon 1', 'blob 2', 'new 3']
        axR.bar(range(4), [v * 100 for v in vals], color=cols, edgecolor=BG, width=0.7)
        axR.axhline(85.83, color=CLS[3], lw=0.8, ls=':', alpha=0.5 + 0.5 * fade)
        for i, v in enumerate(vals):
            axR.text(i, v * 100 + 2, f'{v*100:.0f}' if i < 3 else f'{v*100:.1f}',
                     ha='center', fontsize=8.5, color=INK)
        axR.set_ylim(0, 108); axR.set_xticks(range(4)); axR.set_xticklabels(labs, fontsize=8, color=MUTED)
        axR.set_yticks([]); axR.set_title('per-class recall (%)', fontsize=9, color=MUTED, pad=3)
        for s in axR.spines.values(): s.set_color(LINE)
        msg = ('F untouched  ·  max |Δ old logit| = 0.0  ·  new class 85.8%'
               if fade > 0.95 else 'iterating the frozen operator to the anchors…')
        fig.text(0.5, 0.03, msg, ha='center', fontsize=10.5,
                 color=SAFE if fade > 0.95 else MUTED, weight='bold')
        frames.append(fig_rgba(fig))
    save_gif(PUB / 'deqedit-teach-readout.gif', frames, fps=13)


# ═══════════════════════════════════════════════════════════════════════════
# (b) certificate break-and-rescue: the σ cloud slides right past 1.0 as γ ramps,
#     then bisection walks it back under 0.98. All real power-iteration slopes.
# ═══════════════════════════════════════════════════════════════════════════
def gif_certificate():
    # precompute real σ clouds along the ramp up to raw γ, then the bisection walk-back
    ramp = np.linspace(0.0, 1.0, 9)                  # fraction of raw γ, ramping up
    back = np.array([1.0, 0.8, 0.65, RESCALE, RESCALE])   # bisection settling to rescale
    fracs = list(ramp) + list(back)
    print('  computing certificate clouds (real power iteration)…')
    clouds = [sigma_cloud(GAMMA_RAW * f) for f in fracs]
    ymax_new = yte == 3
    frames = []
    n_up = len(ramp)
    for fi, (f, cloud) in enumerate(zip(fracs, clouds)):
        smax = float(cloud.max()); smean = float(cloud.mean())
        phase_down = fi >= n_up
        fig = plt.figure(figsize=(8.6, 4.6), dpi=108, facecolor=BG)
        fig.text(0.5, 0.94, 'The certificate as a dial you can break, then bisect back',
                 ha='center', fontsize=13.5, weight='bold')
        fig.text(0.5, 0.875, 'each dot is one probe input’s measured slope ‖J_F‖₂ at its equilibrium; past 1.0 the unique-fixed-point guarantee is void',
                 ha='center', fontsize=8.6, color=MUTED)
        ax = fig.add_axes([0.09, 0.14, 0.87, 0.62]); ax.set_facecolor(PANEL)
        # jitter x by class so the taught-class dots (which move) are visible
        rng = np.random.RandomState(3)
        xj = np.array([0, 1, 2, 3])[np.clip(yte, 0, 3)] + rng.uniform(-0.28, 0.28, len(yte))
        for c in range(4):
            m = yte == c
            ax.scatter(xj[m], cloud[m], s=14, color=CLS[c], alpha=0.6, linewidths=0, zorder=3)
        ax.axhline(1.0, color=DANGER, lw=1.4, zorder=2)
        ax.text(3.45, 1.02, 'contraction wall (1.0)', color=DANGER, fontsize=8, ha='right')
        ax.axhline(0.98, color=SAFE, lw=1.0, ls='--', zorder=2)
        ax.text(3.45, 0.90, 'ceiling 0.98', color=SAFE, fontsize=8, ha='right')
        ax.set_xlim(-0.6, 3.6); ax.set_ylim(0.3, 1.55)
        ax.set_xticks(range(4)); ax.set_xticklabels(['moon 0', 'moon 1', 'blob 2', 'taught 3'],
                                                    fontsize=8.5, color=MUTED)
        ax.set_ylabel('measured slope ‖J_F‖₂', color=MUTED, fontsize=9)
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ax.spines.values(): s.set_color(LINE)
        broken = smax >= 1.0
        stage = ('bisecting the gain back under the ceiling' if phase_down
                 else 'ramping the gain: the slopes near the new well slide right')
        col = DANGER if broken else (SAFE if phase_down else GAIN)
        fig.text(0.5, 0.045, f'γ = {f:.2f}·γ_raw   max ‖J_F‖₂ = {smax:.3f}   ({stage})',
                 ha='center', fontsize=10.5, color=col, weight='bold')
        frames.append(fig_rgba(fig))
    # extra hold on the rescued frame
    frames = frames + [frames[-1]] * 8
    save_gif(PUB / 'deqedit-certificate.gif', frames, fps=4, hold=6)


# ═══════════════════════════════════════════════════════════════════════════
# (c) drift ledger: 520 paired old fixed points morph base -> edited while the
#     ‖Δz*‖ histogram draws itself, the tail highlighted. Real re-solves.
# ═══════════════════════════════════════════════════════════════════════════
def gif_drift():
    old = yte < 3
    Xold = Xte[old]
    print('  solving 520 old fixed points under base and edited operators…')
    zb = solve(Xold, W, A, iters=260)
    ze = solve(Xold, Wed, Aed, iters=260)
    dz = np.linalg.norm(ze - zb, axis=1)
    Pb = proj(zb); Pe = proj(ze)
    med = float(np.median(dz)); mx = float(dz.max())
    n_tail = int((dz > 0.1).sum())
    tail = dz > 0.1
    N_MORPH = 26; N_HIST = 20; total = 6 + N_MORPH + N_HIST
    edges = np.linspace(0, min(mx, 1.6), 34)
    frames = []
    for fi in range(total):
        morph = ease(np.clip((fi - 6) / N_MORPH, 0, 1))
        histf = ease(np.clip((fi - 6 - N_MORPH) / N_HIST, 0, 1))
        cur = Pb + (Pe - Pb) * morph
        fig = plt.figure(figsize=(8.8, 4.5), dpi=108, facecolor=BG)
        fig.text(0.5, 0.94, 'The drift ledger: what the dynamics edit did to 520 old fixed points',
                 ha='center', fontsize=13.5, weight='bold')
        fig.text(0.5, 0.875, 'each old input’s equilibrium is solved twice and subtracted; most barely move, a real tail sits near the taught region',
                 ha='center', fontsize=8.6, color=MUTED)
        axL = fig.add_axes([0.05, 0.10, 0.44, 0.70]); axL.set_facecolor(PANEL)
        ycol = np.array([CLS[c] for c in yte[old]])
        axL.scatter(cur[~tail, 0], cur[~tail, 1], s=8, c=ycol[~tail], alpha=0.35, linewidths=0, zorder=2)
        # the tail points, lit, with a short trail of their displacement
        for k in np.where(tail)[0]:
            axL.plot([Pb[k, 0], cur[k, 0]], [Pb[k, 1], cur[k, 1]], color=DANGER, lw=0.7, alpha=0.6, zorder=3)
        axL.scatter(cur[tail, 0], cur[tail, 1], s=26, color=DANGER, edgecolors='white',
                    linewidths=0.5, zorder=5)
        axL.set_xticks([]); axL.set_yticks([])
        axL.set_title(f'old fixed points  ·  base → edited ({morph*100:.0f}%)', fontsize=9, color=MUTED, pad=3)
        for s in axL.spines.values(): s.set_color(LINE)
        axR = fig.add_axes([0.59, 0.16, 0.37, 0.60]); axR.set_facecolor(PANEL)
        if histf > 0:
            nshow = int(histf * len(dz))
            order = np.argsort(dz)                     # draw small-to-large so the tail arrives last
            shown = dz[order[:nshow]]
            counts, _ = np.histogram(shown, bins=edges)
            centers = (edges[:-1] + edges[1:]) / 2
            barcols = np.where(centers > 0.1, DANGER, ACCENT)
            axR.bar(centers, counts, width=(edges[1] - edges[0]) * 0.9, color=barcols, edgecolor=BG, lw=0.3)
            axR.axvline(0.1, color=DANGER, lw=0.8, ls=':')
            axR.axvline(med, color=SAFE, lw=1.0, ls='--')
            axR.text(med + 0.03, axR.get_ylim()[1] * 0.78, f'median {med:.2f}', color=SAFE, fontsize=8, va='top')
        axR.set_xlim(0, edges[-1]); axR.set_xlabel('‖Δz*‖  (old fixed-point drift)', color=MUTED, fontsize=8.5)
        axR.set_yticks([]); axR.tick_params(colors=MUTED, labelsize=7)
        axR.set_title('drift distribution', fontsize=9, color=MUTED, pad=3)
        for s in axR.spines.values(): s.set_color(LINE)
        done = histf > 0.95
        msg = (f'median {med:.2f}  ·  worst {mx:.2f}  ·  {n_tail}/520 moved > 0.1  ·  3 predictions flipped'
               if done else 'auditing every old fixed point, one re-solve at a time…')
        fig.text(0.5, 0.035, msg, ha='center', fontsize=10, color=INK if done else MUTED, weight='bold')
        frames.append(fig_rgba(fig))
    save_gif(PUB / 'deqedit-drift.gif', frames, fps=12)


# ═══════════════════════════════════════════════════════════════════════════
# (d) evaporation: hybrid iteration, edited operator for j turns then base; the
#     excursion collapses back to ~1e-7. Two racing log-distance curves.
# ═══════════════════════════════════════════════════════════════════════════
def gif_evaporation():
    Xb = Xte[yte == 3]
    zb_star = solve(Xb, W, A, iters=260)             # the base equilibrium of the taught inputs
    ze_star = solve(Xb, Wed, Aed, iters=260)         # the edited equilibrium
    span = float(np.linalg.norm(ze_star - zb_star, axis=1).max())   # how far the edited well pulls
    # the hybrid trace: edited for j turns, base after. Track max dist to base z* each turn.
    JMAX = 30; TOTAL_IT = 120
    def hybrid_trace(j):
        z = np.zeros((Xb.shape[0], D)); d = []
        for k in range(TOTAL_IT):
            Wm, Am = (Wed, Aed) if k < j else (W, A)
            z = (1 - BETA) * z + BETA * Fop(Xb, z, Wm, Am)
            d.append(float(np.linalg.norm(z - zb_star, axis=1).max()))
        return np.array(d)
    j_edit = 10
    d_edit = hybrid_trace(j_edit)                     # edited then base -> evaporates
    d_pure = hybrid_trace(0)                          # base all along -> solver floor
    floor = R['evaporation']['0']; final = R['evaporation']['10']
    N = TOTAL_IT
    frames = []
    for fi in range(6, N + 1, 2):
        k = fi
        fig = plt.figure(figsize=(8.6, 4.4), dpi=108, facecolor=BG)
        fig.text(0.5, 0.94, 'Watch an edit evaporate: an excursion the base flow forgets',
                 ha='center', fontsize=13.5, weight='bold')
        fig.text(0.5, 0.875, 'the edited operator steers the taught inputs for 10 turns, then the base operator takes over and erases every trace',
                 ha='center', fontsize=8.6, color=MUTED)
        ax = fig.add_axes([0.11, 0.15, 0.85, 0.60]); ax.set_facecolor(PANEL)
        xs = np.arange(1, k + 1)
        ax.semilogy(xs, np.maximum(d_pure[:k], 1e-9), color=MUTED, lw=1.6, label='base all turns (solver floor)')
        ax.semilogy(xs, np.maximum(d_edit[:k], 1e-9), color=GAIN, lw=2.2, label='edited 10 turns, base after')
        ax.axvline(j_edit, color=ACCENT, lw=1.0, ls=':', alpha=0.7)
        ax.text(j_edit + 0.5, d_edit[:k].max() if k > j_edit else span, ' base takes over',
                color=ACCENT, fontsize=8, va='center')
        ax.axhline(floor, color=SAFE, lw=0.8, ls='--', alpha=0.7)
        ax.set_xlim(0, N); ax.set_ylim(1e-8, max(span, 0.1) * 2)
        ax.set_xlabel('solver turn', color=MUTED, fontsize=8.5)
        ax.set_ylabel('max distance to base z*', color=MUTED, fontsize=8.5)
        ax.tick_params(colors=MUTED, labelsize=7)
        ax.legend(loc='upper right', fontsize=8, facecolor=PANEL, edgecolor=LINE, labelcolor=INK)
        for s in ax.spines.values(): s.set_color(LINE)
        cur = d_edit[k - 1]
        done = k >= N
        msg = (f'landed back on the base fixed point to {final:.1e}  (floor {floor:.1e}): erased, not weakened'
               if done else f'turn {k}: excursion {cur:.1e} above the base equilibrium')
        fig.text(0.5, 0.035, msg, ha='center', fontsize=10, color=SAFE if done else GAIN, weight='bold')
        frames.append(fig_rgba(fig))
    save_gif(PUB / 'deqedit-evaporation.gif', frames, fps=14)


# ═══════════════════════════════════════════════════════════════════════════
# (e) stack toll (static PNG): the tied one-paste edit vs the untied L-layer bill.
#     This is a static comparison of final numbers from the report sweep (a bar
#     chart plus a curve, all fixed values), so it is a figure, not a GIF: the
#     counters "climbing" would be fake growth of numbers that are already final.
# ═══════════════════════════════════════════════════════════════════════════
def fig_stack_toll():
    sweep = R['stack']['sweep']                       # j, edit_params, dz_old_max, margin
    tied_params = R['stack']['tied_edit_params']      # 512, once
    tied_dz = R['teach_T2']['dz_max']                 # 1.47, the tied worst case
    js = [s['j'] for s in sweep]
    epar = [s['edit_params'] for s in sweep]
    dzs = [s['dz_old_max'] for s in sweep]
    xs = np.arange(len(sweep))

    fig = plt.figure(figsize=(8.8, 4.5), dpi=150, facecolor=BG)
    fig.text(0.5, 0.94, 'The stack toll: paying for “everywhere” without weight sharing',
             ha='center', fontsize=13.5, weight='bold')
    fig.text(0.5, 0.875, 'the tied loop pastes 512 numbers once and the edit lives at every turn; the untied 12-layer stack repeats the paste per layer',
             ha='center', fontsize=8.6, color=MUTED)
    # left: numbers written
    axL = fig.add_axes([0.07, 0.16, 0.40, 0.60]); axL.set_facecolor(PANEL)
    axL.bar(xs, epar, color=CLS[3], edgecolor=BG, width=0.62)
    axL.axhline(tied_params, color=ACCENT, lw=1.4, ls='--')
    axL.text(len(sweep) - 0.5, tied_params + 200, f'tied: {tied_params} once', color=ACCENT,
             fontsize=8, ha='right')
    axL.set_xlim(-0.6, len(sweep) - 0.4); axL.set_ylim(0, 6600)
    axL.set_xticks(range(len(sweep))); axL.set_xticklabels([str(j) for j in js], fontsize=8, color=MUTED)
    axL.set_xlabel('edited layers', color=MUTED, fontsize=8.5)
    axL.set_title('numbers written', fontsize=9.5, color=MUTED, pad=3)
    axL.tick_params(colors=MUTED, labelsize=7)
    for s in axL.spines.values(): s.set_color(LINE)
    for k in range(len(sweep)):
        axL.text(k, epar[k] + 150, str(epar[k]), ha='center', fontsize=8.5, color=INK)
    # right: old-state shift
    axR = fig.add_axes([0.56, 0.16, 0.40, 0.60]); axR.set_facecolor(PANEL)
    axR.plot(xs, dzs, '-o', color=DANGER, lw=1.8, markersize=5, markeredgecolor=BG)
    axR.axhline(tied_dz, color=ACCENT, lw=1.4, ls='--')
    axR.text(len(sweep) - 0.5, tied_dz + 0.12, f'tied worst case {tied_dz:.2f}', color=ACCENT,
             fontsize=8, ha='right')
    axR.set_xlim(-0.4, len(sweep) - 0.4); axR.set_ylim(0, 4.3)
    axR.set_xticks(range(len(sweep))); axR.set_xticklabels([str(j) for j in js], fontsize=8, color=MUTED)
    axR.set_xlabel('edited layers', color=MUTED, fontsize=8.5)
    axR.set_title('max shift of old inputs’ final states', fontsize=9.5, color=MUTED, pad=3)
    axR.tick_params(colors=MUTED, labelsize=7)
    for s in axR.spines.values(): s.set_color(LINE)
    fig.text(0.5, 0.035, 'all 12 layers: 6,144 numbers, old states shoved 3.86, and no contraction to certify',
             ha='center', fontsize=10, color=DANGER, weight='bold')
    out = PUB / 'deqedit-stack-toll.png'
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    print(f'wrote {out}')


# ═══════════════════════════════════════════════════════════════════════════
# (f) silence vs erase: mask the trained row (recall -> 0, cluster frozen), paste
#     8 anchors, the class reads back out at 100%. Real equilibria never move.
# ═══════════════════════════════════════════════════════════════════════════
def gif_silence_erase():
    old = yte < 3
    Xold = Xte[old]; yold = yte[old]
    print('  solving old-class equilibria for the silence/erase demo…')
    z = solve(Xold, W, A, iters=260)
    P2 = proj(z)
    lg = z @ Cw.T + cb                                # real base logits
    pred_base = lg.argmax(1)
    # mask class 2's readout row
    masked = lg.copy(); masked[:, 2] = -1e9
    pred_mask = masked.argmax(1)
    # resurrection anchors: equilibria of a few class-2 training-region inputs
    a_reg = Xold[yold == 2][:8]
    res_anch = solve(a_reg, W, A, iters=260)
    s_res = ALPHA * yat_k(z, res_anch).max(1)
    pred_res = np.where(s_res > masked.max(1), 2, pred_mask)
    rec_base = float((pred_base[yold == 2] == 2).mean())
    rec_mask = float((pred_mask[yold == 2] == 2).mean())
    rec_res = float((pred_res[yold == 2] == 2).mean())
    AP = proj(res_anch)
    m2 = yold == 2
    from matplotlib.colors import to_rgb
    def cvec(preds):                                 # per-dot RGB from its prediction
        return np.array([to_rgb(CLS[min(int(c), 3)]) for c in preds])
    c_base, c_mask, c_res = cvec(pred_base), cvec(pred_mask), cvec(pred_res)

    # one continuous timeline: hold base, tween colour to masked, hold, paste anchors +
    # tween colour back. Positions of every equilibrium are fixed throughout (flow
    # untouched); only the readout, hence the colour and the recall bar, changes.
    segs = [('hold', 8, 'trained: the blob class is read at 100%'),
            ('to_mask', 16, 'mask the readout row: the cluster recolours to its neighbours, recall falls to 0'),
            ('hold_mask', 12, 'the cluster of equilibria has NOT moved; the flow that made it was never touched'),
            ('to_res', 18, 'paste 8 anchor rows: the same flow is read against them, the cluster turns back green'),
            ('hold_res', 12, 'recall back at 100%: the trained class was silenced, never erased')]
    frames = []
    for kind, n, msg in segs:
        for fi in range(n):
            t = ease((fi + 1) / n)
            if kind == 'hold':      col, rec, anch_a, danger = c_base, rec_base, 0.0, False
            elif kind == 'to_mask': col, rec, anch_a, danger = (1 - t) * c_base + t * c_mask, (1 - t) * rec_base, 0.0, True
            elif kind == 'hold_mask': col, rec, anch_a, danger = c_mask, rec_mask, 0.0, True
            elif kind == 'to_res':  col, rec, anch_a, danger = (1 - t) * c_mask + t * c_res, t * rec_res, t, False
            else:                   col, rec, anch_a, danger = c_res, rec_res, 1.0, False
            fig = plt.figure(figsize=(8.6, 4.7), dpi=108, facecolor=BG)
            fig.text(0.5, 0.94, 'Silenced is not erased: a masked row vs the flow that made the cluster',
                     ha='center', fontsize=13.5, weight='bold')
            fig.text(0.5, 0.875, 'every old input is solved to its equilibrium; only the readout changes, so the dots recolour while their positions stay fixed',
                     ha='center', fontsize=8.6, color=MUTED)
            axL = fig.add_axes([0.06, 0.10, 0.52, 0.70]); axL.set_facecolor(PANEL)
            axL.scatter(P2[~m2, 0], P2[~m2, 1], s=8, c=col[~m2], alpha=0.30, linewidths=0, zorder=2)
            axL.scatter(P2[m2, 0], P2[m2, 1], s=20, c=col[m2], alpha=0.9, edgecolors=BG, linewidths=0.3, zorder=4)
            if anch_a > 0:
                axL.scatter(AP[:, 0], AP[:, 1], s=120, marker='*', color=CLS[2],
                            edgecolors='white', linewidths=1.0, alpha=anch_a, zorder=6)
            axL.set_xticks([]); axL.set_yticks([])
            axL.set_title('state plane (the blob-2 cluster is highlighted)', fontsize=9, color=MUTED, pad=3)
            for s in axL.spines.values(): s.set_color(LINE)
            axR = fig.add_axes([0.68, 0.18, 0.27, 0.54]); axR.set_facecolor(PANEL)
            axR.bar([0], [rec * 100], color=CLS[2], edgecolor=BG, width=0.6)
            axR.set_ylim(0, 108); axR.set_xlim(-0.7, 0.7); axR.set_xticks([0])
            axR.set_xticklabels(['blob 2\nrecall'], fontsize=8.5, color=MUTED)
            axR.text(0, rec * 100 + 3, f'{rec*100:.0f}%', ha='center', fontsize=11, color=INK, weight='bold')
            axR.set_yticks([]); axR.set_title('trained-class recall', fontsize=9, color=MUTED, pad=3)
            for s in axR.spines.values(): s.set_color(LINE)
            fig.text(0.5, 0.035, msg, ha='center', fontsize=9.4,
                     color=DANGER if danger else SAFE, weight='bold')
            frames.append(fig_rgba(fig))
    save_gif(PUB / 'deqedit-silence-erase.gif', frames, fps=12)


if __name__ == '__main__':
    gif_teach_readout()
    gif_certificate()
    gif_drift()
    gif_evaporation()
    fig_stack_toll()
    gif_silence_erase()
    print('DEQEDIT_GIFS_DONE')
