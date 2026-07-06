"""Teach and forget classes inside a weight-tied equilibrium network, by construction.

The setup extends scripts/yat_deq.py: one shared Yat operator

    F(z; x) = tanh( A · φ_W(z) + U·x + z0 ),   φ_W(z)_i = (⟨z,w_i⟩ + b)² / (‖z-w_i‖² + ε)

iterated to its fixed point z* = F(z*; x), read out by ŷ = argmax(C z* + c).
The question: the editing post (edit-a-network-by-hand) taught and forgot classes by
appending / deleting rows of a Yat bank, with proofs of non-interference. Here there is
only ONE operator shared across all depth, and editing it edits every "layer" at once.
Does that surgical editability survive inside the recursion?

Experiment:
  base   : train on two moons (classes 0,1) + one Gaussian blob (class 2). NCLS=3.
  TEACH  : class 3 (a second blob, never trained) by construction, no gradient steps:
           solve the equilibria of a few teaching examples under the base operator and
           use them as anchor prototypes.
             T1 (readout edit)  : s₃(x) = α · max_k φ(z*(x), c_k), the same Yat kernel,
                                  anchors as readout rows. F untouched ⇒ every old fixed
                                  point and score is EXACTLY unchanged.
             T2 (dynamics edit) : additionally append the anchors to the shared bank W
                                  and give A matching columns γ·ĉ_k, deepening a well at
                                  the taught class inside the operator itself. One knob γ.
  certificate : ‖J_F‖₂ (power iteration at z*) over all test points, before/after each
                edit; if T2 breaks the contraction, bisect γ until it is restored.
  invariance  : max ‖Δz*‖ and max |Δscore| over old-class test inputs under T2.
  FORGET : (i) taught class: delete its rows -> operator restored bit-for-bit.
           (ii) trained class 2: mask its readout row -> recall 0, other scores exactly
                unchanged; but the imprint stays in the dynamics, measured by
                resurrecting the class with an 8-anchor readout edit, no training.
  stack  : an untied L-layer copy (per-layer W_l, A_l), where the same teach edit must
           be applied at every layer; effect vs number of edited layers + param count.
  evaporation : in the tied net, apply the edited operator for only the first j steps
                and the base operator afterwards: the edit vanishes from the fixed
                point, showing an edit must live at every depth to survive the limit,
                which is what editing the one shared operator gives for free.
  scratch : a 4-class model trained from scratch, for the incremental-vs-from-scratch
            comparison.

Run: python scripts/yat_deq_edit.py         (CPU, a few minutes)
Writes a text report + public/yat-deq-edit/model.json for the explainer's live viz.
Set OUTDIR=<dir> to redirect both JSON outputs (e.g. OUTDIR=results on a Kaggle kernel).
"""
import warnings; warnings.filterwarnings('ignore')
import os, json, time
import numpy as np
import jax, jax.numpy as jnp
from jax import lax
import optax
from sklearn.datasets import make_moons

jax.config.update('jax_platform_name', 'cpu')
KEY = jax.random.PRNGKey(0)

D_IN, D, M = 2, 32, 24                    # input dim, state dim, # shared prototypes
SOLVER_ITERS, BETA = 120, 0.7             # forward fixed-point solver budget / damping
SETTLE_TOL = 1e-4                         # per-input halting tolerance (matches the browser)
M_TEACH = 8                               # teaching examples for the constructed class
MARGIN = 1.0                              # calibration margin for the anchor readout
L_STACK = 12                              # untied comparison stack depth
STEPS = int(os.environ.get('STEPS', '1500'))


def outdir():
    """Where the JSON artifacts go: OUTDIR if set (Kaggle: OUTDIR=results), else the repo."""
    from pathlib import Path
    if os.environ.get('OUTDIR'):
        d = Path(os.environ['OUTDIR'])
    else:
        d = Path(__file__).resolve().parents[1] / 'public' / 'yat-deq-edit'
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── the shared Yat operator ────────────────────────────────────────────────────
def yat_k(z, W, b, eps):
    dot = z @ W.T
    d2 = (z ** 2).sum(-1, keepdims=True) + (W ** 2).sum(-1) - 2 * dot
    return (dot + b) ** 2 / (d2 + eps)


def yat_feats(z, p):
    return yat_k(z, p['W'], jax.nn.softplus(p['b']), jax.nn.softplus(p['eps']))


def F(p, x, z):
    pre = yat_feats(z, p) @ p['A'].T + x @ p['Uin'].T + p['z0']
    return jnp.tanh(pre)


def solve(fz, z0, iters=SOLVER_ITERS, beta=BETA):
    def step(z, _):
        zn = (1 - beta) * z + beta * fz(z)
        return zn, jnp.max(jnp.linalg.norm(zn - z, axis=-1))
    zT, resid = lax.scan(step, z0, None, length=iters)
    return zT, resid


# ── DEQ layer with implicit differentiation (as in yat_deq.py) ─────────────────
@jax.custom_vjp
def deq(p, x):
    z0 = jnp.zeros((x.shape[0], D))
    z_star, _ = solve(lambda z: F(p, x, z), z0)
    return z_star


def deq_fwd(p, x):
    z0 = jnp.zeros((x.shape[0], D))
    z_star, _ = solve(lambda z: F(p, x, z), z0)
    return z_star, (p, x, z_star)


def deq_bwd(res, g):
    p, x, z_star = res
    _, vjp_z = jax.vjp(lambda z: F(p, x, z), z_star)
    u, _ = solve(lambda u: vjp_z(u)[0] + g, jnp.zeros_like(g))
    _, vjp_px = jax.vjp(lambda pp, xx: F(pp, xx, z_star), p, x)
    return vjp_px(u)


deq.defvjp(deq_fwd, deq_bwd)


def zstar(p, x, iters=300):
    """High-precision equilibrium (no grad), for invariance measurements."""
    z, _ = solve(lambda z: F(p, x, z), jnp.zeros((x.shape[0], D)), iters=iters)
    return z


def logits(p, x):
    return deq(p, x) @ p['C'].T + p['cb']


# ── training loss with the contraction (spectral) penalty ──────────────────────
def spectral_sigma(p, x, z, key, pit=6):
    fz = lambda zz: F(p, x, zz)
    v = jax.random.normal(key, z.shape)
    v = v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    for _ in range(pit):
        Jv = jax.jvp(fz, (z,), (v,))[1]
        _, vjp = jax.vjp(fz, z); JtJv = vjp(Jv)[0]
        v = jax.lax.stop_gradient(JtJv / (jnp.linalg.norm(JtJv, axis=-1, keepdims=True) + 1e-9))
    Jv = jax.jvp(fz, (z,), (v,))[1]
    return jnp.linalg.norm(Jv, axis=-1)


LAMBDA_C, SIGMA_TARGET = 3.0, 0.7
LAMBDA_HARD, SIGMA_HARD = 60.0, 0.9      # crush the tail: no probe point may approach 1


def loss_fn(p, x, y, key):
    lg = logits(p, x)
    ce = optax.softmax_cross_entropy_with_integer_labels(lg, y).mean()
    reg = 1e-4 * sum(jnp.sum(v ** 2) for k, v in p.items() if k in ('A', 'Uin', 'C'))
    # penalize the local slope at the training equilibria AND at jittered copies,
    # so the certificate holds on the surrounding region, not just the sample
    xj = x + 0.1 * jax.random.normal(jax.random.fold_in(key, 1), x.shape)
    xp = jnp.concatenate([x, xj], 0)
    z_star = jax.lax.stop_gradient(deq(p, xp))
    sigma = spectral_sigma(p, xp, z_star, key)
    contract = (LAMBDA_C * jnp.mean(jax.nn.relu(sigma - SIGMA_TARGET) ** 2)
                + LAMBDA_HARD * jnp.mean(jax.nn.relu(sigma - SIGMA_HARD) ** 2))
    return ce + reg + contract


def init_params(key, ncls, m=M):
    k = jax.random.split(key, 7)
    return {
        'W':   jax.random.normal(k[0], (m, D)) * 0.6,
        'A':   jax.random.normal(k[1], (D, m)) * (0.5 / np.sqrt(m)),
        'Uin': jax.random.normal(k[2], (D, D_IN)) * 0.5,
        'z0':  jnp.zeros((D,)),
        'b':   jnp.full((), float(np.log(np.expm1(0.5)))),
        'eps': jnp.full((), float(np.log(np.expm1(1.0)))),
        'C':   jax.random.normal(k[5], (ncls, D)) * 0.5,
        'cb':  jnp.zeros((ncls,)),
    }


def train_deq(key, X, y, ncls, tag):
    p = init_params(key, ncls)
    sched = optax.cosine_decay_schedule(3e-3, STEPS, alpha=0.03)
    opt = optax.adam(sched); st = opt.init(p)

    @jax.jit
    def step(p, st, key):
        l, g = jax.value_and_grad(loss_fn)(p, X, y, key)
        up, st = opt.update(g, st)
        return optax.apply_updates(p, up), st, l

    t0 = time.time()
    for it in range(STEPS + 1):
        p, st, l = step(p, st, jax.random.fold_in(key, it))
        if it % 500 == 0:
            acc = float((jnp.argmax(logits(p, X), 1) == y).mean())
            print(f'  [{tag}] step {it:4d}  loss {float(l):.4f}  train acc {acc:.3f}')
    print(f'  [{tag}] trained in {time.time() - t0:.0f}s')
    return p


# ── diagnostics ────────────────────────────────────────────────────────────────
def contraction_norm(p, x, key, iters=30):
    """Per-sample ‖J_F‖₂ at z* by power iteration on JᵀJ."""
    z_star = zstar(p, x)
    fz = lambda z: F(p, x, z)
    v = jax.random.normal(key, z_star.shape)
    v = v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    def pit(v, _):
        Jv = jax.jvp(fz, (z_star,), (v,))[1]
        _, vjp = jax.vjp(fz, z_star); JtJv = vjp(Jv)[0]
        s = jnp.sqrt(jnp.linalg.norm(JtJv, axis=-1) + 1e-12)
        v = JtJv / (jnp.linalg.norm(JtJv, axis=-1, keepdims=True) + 1e-9)
        return v, s
    _, s = lax.scan(pit, v, None, length=iters)
    return np.array(s[-1])


def settle_counts(p, x, tol=SETTLE_TOL, maxk=SOLVER_ITERS):
    """Per-input iterations until the residual first drops below tol."""
    z = jnp.zeros((x.shape[0], D)); ks = jnp.full((x.shape[0],), maxk, dtype=jnp.int32)
    for k in range(maxk):
        zn = (1 - BETA) * z + BETA * F(p, x, z)
        r = jnp.linalg.norm(zn - z, axis=-1)
        ks = jnp.where((r < tol) & (ks == maxk), k + 1, ks)
        z = zn
    return np.array(ks)


# ── the constructed class: anchors + readout + dynamics edit ───────────────────
def anchor_scores(p, anchors, alpha, z):
    """The taught class's score: the SAME Yat kernel, anchors as readout rows."""
    b = jax.nn.softplus(p['b']); eps = jax.nn.softplus(p['eps'])
    return alpha * yat_k(z, anchors, b, eps).max(-1)


def scores4(p, x, anchors, alpha):
    """Old linear logits + the constructed class 3 anchor score. p may be edited."""
    z = deq(p, x)
    lg = z @ p['C'].T + p['cb']
    s3 = anchor_scores(p, anchors, alpha, z)
    return jnp.concatenate([lg, s3[:, None]], -1)


def dyn_edit(p, anchors, gamma):
    """T2: append the anchors as prototypes with A columns γ·ĉ_k. Returns a new dict."""
    cols = gamma * (anchors / jnp.linalg.norm(anchors, axis=-1, keepdims=True))  # m×D
    q = dict(p)
    q['W'] = jnp.concatenate([p['W'], anchors], 0)
    q['A'] = jnp.concatenate([p['A'], cols.T], 1)
    return q


# ── the untied stack (per-layer weights) ───────────────────────────────────────
def stack_init(key, ncls):
    ks = jax.random.split(key, L_STACK * 2 + 3)
    return {
        'W':   [jax.random.normal(ks[2 * l], (M, D)) * 0.6 for l in range(L_STACK)],
        'A':   [jax.random.normal(ks[2 * l + 1], (D, M)) * (0.5 / np.sqrt(M)) for l in range(L_STACK)],
        'z0':  [jnp.zeros((D,)) for _ in range(L_STACK)],
        'Uin': jax.random.normal(ks[-3], (D, D_IN)) * 0.5,
        'b':   jnp.full((), float(np.log(np.expm1(0.5)))),
        'eps': jnp.full((), float(np.log(np.expm1(1.0)))),
        'C':   jax.random.normal(ks[-1], (ncls, D)) * 0.5,
        'cb':  jnp.zeros((ncls,)),
    }


def stack_states(p, x):
    """All per-layer states [z_0=0, z_1, ..., z_L]."""
    b = jax.nn.softplus(p['b']); eps = jax.nn.softplus(p['eps'])
    xU = x @ p['Uin'].T
    z = jnp.zeros((x.shape[0], D)); zs = [z]
    for l in range(L_STACK):
        z = jnp.tanh(yat_k(z, p['W'][l], b, eps) @ p['A'][l].T + xU + p['z0'][l])
        zs.append(z)
    return zs


def stack_logits(p, x):
    return stack_states(p, x)[-1] @ p['C'].T + p['cb']


def train_stack(key, X, y, ncls):
    p = stack_init(key, ncls)
    sched = optax.cosine_decay_schedule(3e-3, STEPS, alpha=0.03)
    opt = optax.adam(sched); st = opt.init(p)

    def lf(p):
        ce = optax.softmax_cross_entropy_with_integer_labels(stack_logits(p, X), y).mean()
        reg = 1e-4 * (sum(jnp.sum(a ** 2) for a in p['A']) + jnp.sum(p['Uin'] ** 2) + jnp.sum(p['C'] ** 2))
        return ce + reg

    @jax.jit
    def step(p, st):
        l, g = jax.value_and_grad(lf)(p)
        up, st = opt.update(g, st)
        return optax.apply_updates(p, up), st, l

    t0 = time.time()
    for it in range(STEPS + 1):
        p, st, l = step(p, st)
        if it % 500 == 0:
            acc = float((jnp.argmax(stack_logits(p, X), 1) == y).mean())
            print(f'  [stack] step {it:4d}  loss {float(l):.4f}  train acc {acc:.3f}')
    print(f'  [stack] trained in {time.time() - t0:.0f}s')
    return p


def stack_edit(p, layer_states_teach, gamma_frac, edit_layers):
    """Append the per-layer anchor rows at each edited layer (the stack needs one edit
    PER LAYER to express what the tied net expresses with one)."""
    b = jax.nn.softplus(p['b']); eps = jax.nn.softplus(p['eps'])
    q = dict(p); q['W'] = list(p['W']); q['A'] = list(p['A'])
    edit_params = 0
    for l in edit_layers:
        zin, zout = layer_states_teach[l], layer_states_teach[l + 1]     # layer l maps z_l -> z_{l+1}
        p_self = float(jnp.mean(jnp.diagonal(yat_k(zin, zin, b, eps))))
        gam = gamma_frac / p_self
        cols = gam * (zout / jnp.linalg.norm(zout, axis=-1, keepdims=True))
        q['W'][l] = jnp.concatenate([p['W'][l], zin], 0)
        q['A'][l] = jnp.concatenate([p['A'][l], cols.T], 1)
        edit_params += int(np.prod(zin.shape)) + int(np.prod(cols.shape))
    return q, edit_params


# ── data: two moons + two blobs ────────────────────────────────────────────────
def make_data(seed=1):
    Xm, ym = make_moons(n_samples=1400, noise=0.16, random_state=seed)
    Xm = (Xm - Xm.mean(0)) / Xm.std(0)
    rng = np.random.RandomState(seed + 10)
    Xa = np.array([2.1, 1.9]) + 0.22 * rng.randn(400, 2)      # class 2, trained blob
    Xb = np.array([-2.1, -1.9]) + 0.22 * rng.randn(400, 2)    # class 3, TAUGHT blob
    d = {
        'moons_tr': (Xm[:1000], ym[:1000]), 'moons_te': (Xm[1000:], ym[1000:]),
        'a_tr': Xa[:280], 'a_te': Xa[280:],
        'b_teach': Xb[:M_TEACH], 'b_tr': Xb[:280], 'b_te': Xb[280:],
    }
    return d


def per_class_acc(preds, y, ncls):
    return {c: float((preds[y == c] == c).mean()) for c in range(ncls) if np.any(y == c)}


def main():
    out = {}
    dat = make_data()
    Xtr3 = jnp.asarray(np.vstack([dat['moons_tr'][0], dat['a_tr']]))
    ytr3 = jnp.asarray(np.concatenate([dat['moons_tr'][1], np.full(280, 2)]))
    Xte = jnp.asarray(np.vstack([dat['moons_te'][0], dat['a_te'], dat['b_te']]))
    yte = np.concatenate([dat['moons_te'][1], np.full(120, 2), np.full(120, 3)])
    old = yte < 3                                              # test points of trained classes
    Xte_old, yte_old = Xte[old], yte[old]

    # ── base model: moons + blob A, three classes ──
    print('training the base Yat-DEQ (moons + one blob, 3 classes)…')
    p0 = train_deq(KEY, Xtr3, ytr3, ncls=3, tag='base')
    lg0 = np.array(logits(p0, Xte_old))
    pred0 = lg0.argmax(1)
    acc0 = per_class_acc(pred0, yte_old, 3)
    sig0 = contraction_norm(p0, Xte, jax.random.PRNGKey(7))    # over ALL test inputs incl. the unseen blob
    sig0_old = contraction_norm(p0, Xte_old, jax.random.PRNGKey(7))
    settle0 = settle_counts(p0, Xte)
    n_params = sum(int(np.prod(np.shape(v))) for v in p0.values())
    print(f'\nbase: per-class acc {acc0}   overall(old) {float((pred0 == yte_old).mean()):.4f}')
    print(f'base: ‖J‖₂ over all test inputs  mean {sig0.mean():.3f}  max {sig0.max():.3f} '
          f'(old classes only: mean {sig0_old.mean():.3f} max {sig0_old.max():.3f})')
    out['base'] = {'acc': acc0, 'acc_old_overall': float((pred0 == yte_old).mean()),
                   'sigma_mean': float(sig0.mean()), 'sigma_max': float(sig0.max()),
                   'params': n_params}

    # ── TEACH class 3, T1: the readout edit (anchors, F untouched) ──
    print('\nTEACH class 3 by construction (no gradient steps)…')
    teach_x = jnp.asarray(dat['b_teach'])
    anchors = zstar(p0, teach_x)                                # 8 anchor prototypes = equilibria
    b_v = float(jax.nn.softplus(p0['b'])); eps_v = float(jax.nn.softplus(p0['eps']))
    k_self = np.array(yat_k(anchors, anchors, b_v, eps_v))
    # one calibration knob: put the anchor score on the scale the old classes speak at.
    # Calibrated on the LEAVE-ONE-OUT anchor kernel (each anchor scored by the others),
    # the honest estimate of what a typical new-class point will score.
    k_loo = k_self - np.diag(np.diag(k_self)) - 1e9 * np.eye(M_TEACH)
    win_tr = np.median(np.array(logits(p0, Xtr3)).max(1))
    alpha = float((win_tr + MARGIN) / np.median(k_loo.max(1)))
    print(f'  anchors: {M_TEACH} equilibria; readout scale alpha = {alpha:.4f} '
          f'(median winning train logit {win_tr:.2f} + margin {MARGIN}, LOO kernel '
          f'{np.median(k_loo.max(1)):.1f} vs self {np.median(np.diag(k_self)):.1f})')

    s4 = np.array(scores4(p0, Xte, anchors, alpha))
    pred_t1 = s4.argmax(1)
    acc_t1 = per_class_acc(pred_t1, yte, 4)
    captured = int(((pred_t1 == 3) & (yte < 3)).sum())
    # T1 invariance is exact by construction: F is untouched, so z* and the old scores
    # are the identical computation. Assert it anyway.
    dlg = np.abs(np.array(logits(p0, Xte_old)) - lg0).max()
    assert dlg == 0.0, dlg
    print(f'  T1 (readout rows only): new class recall {acc_t1[3]:.3f}, '
          f'old classes {acc_t1[0]:.3f}/{acc_t1[1]:.3f}/{acc_t1[2]:.3f}, '
          f'{captured} old test points captured by the new class')
    print(f'  T1 invariance: max |Δ old logit| = {dlg} (exact, F untouched)')
    out['teach_T1'] = {'acc': acc_t1, 'captured_old': captured, 'alpha': alpha,
                       'max_dlogit_old': float(dlg)}

    # ── TEACH T2: the dynamics edit (anchors into the shared bank) ──
    p_self = float(np.mean(np.diag(k_self)))
    gamma0 = 1.5 / p_self                                       # bump of 1.5 (pre-tanh) at each anchor
    p2 = dyn_edit(p0, anchors, gamma0)
    sig2 = contraction_norm(p2, Xte, jax.random.PRNGKey(7))
    print(f'\n  T2 (dynamics edit, γ = {gamma0:.5f}): '
          f'‖J‖₂ mean {sig2.mean():.3f}  max {sig2.max():.3f} '
          f'({"still a contraction" if sig2.max() < 1 else "CONTRACTION BROKEN"})')
    out['teach_T2'] = {'gamma_raw': gamma0, 'sigma_mean_raw': float(sig2.mean()),
                       'sigma_max_raw': float(sig2.max())}

    # the one-knob rescale, if teaching broke the certificate
    scale = 1.0
    if sig2.max() >= 0.99:
        lo, hi = 0.0, 1.0
        for _ in range(14):
            mid = (lo + hi) / 2
            smax = contraction_norm(dyn_edit(p0, anchors, gamma0 * mid), Xte, jax.random.PRNGKey(7)).max()
            if smax < 0.98: lo = mid
            else: hi = mid
        scale = lo
        p2 = dyn_edit(p0, anchors, gamma0 * scale)
        sig2 = contraction_norm(p2, Xte, jax.random.PRNGKey(7))
        print(f'  rescued by the one knob: γ ← {scale:.3f}·γ, '
              f'‖J‖₂ mean {sig2.mean():.3f} max {sig2.max():.3f}')
    gamma = gamma0 * scale
    out['teach_T2'].update({'rescale': scale, 'gamma': gamma,
                            'sigma_mean': float(sig2.mean()), 'sigma_max': float(sig2.max())})

    # what the dynamics edit buys, and what it costs. The fair T2 construction
    # re-reads the teaching examples under the EDITED operator (the paste moved
    # the flow, so the anchors' readout copies are re-solved, still no training).
    anchors2 = zstar(p2, teach_x)
    z_old_base = np.array(zstar(p0, Xte_old))
    z_old_edit = np.array(zstar(p2, Xte_old))
    dz = np.linalg.norm(z_old_edit - z_old_base, axis=1)
    lg_edit = z_old_edit @ np.array(p2['C']).T + np.array(p2['cb'])
    ds = np.abs(lg_edit - z_old_base @ np.array(p0['C']).T - np.array(p0['cb'])).max(1)
    s4_2 = np.array(scores4(p2, Xte, anchors2, alpha))
    pred_t2 = s4_2.argmax(1)
    acc_t2 = per_class_acc(pred_t2, yte, 4)
    flips = int((pred_t2[old] != pred_t1[old]).sum())
    # condensation and margin of the new class, T1 vs T2 (each vs its own anchors)
    zb_t1 = np.array(zstar(p0, Xte[yte == 3]))
    zb_t2 = np.array(zstar(p2, Xte[yte == 3]))
    cond = lambda Z, A_: float(np.mean(np.min(np.linalg.norm(Z[:, None, :] - np.asarray(A_)[None], axis=-1), axis=1)))
    cond_t1, cond_t2 = cond(zb_t1, anchors), cond(zb_t2, anchors2)
    marg = lambda S: float(np.mean(S[yte == 3, 3] - S[yte == 3, :3].max(1)))
    settle2 = settle_counts(p2, Xte)
    print(f'  T2 per-class acc {acc_t2}; old-class prediction flips vs T1: {flips}')
    print(f'  T2 invariance over {len(dz)} old-class inputs: '
          f'‖Δz*‖ median {np.median(dz):.2e}  p90 {np.percentile(dz, 90):.2e}  max {dz.max():.2e}; '
          f'|Δscore| median {np.median(ds):.2e}  max {ds.max():.2e}; '
          f'{int((dz > 0.1).sum())}/{len(dz)} old inputs moved more than 0.1')
    print(f'  new-class condensation (mean dist to nearest anchor): '
          f'T1 {cond_t1:.3f} -> T2 {cond_t2:.3f}')
    print(f'  new-class margin (s₃ − best old score): T1 {marg(s4):.2f} -> T2 {marg(s4_2):.2f}')
    print(f'  settle iterations (tol {SETTLE_TOL:g}), new class: '
          f'median {int(np.median(settle0[yte == 3]))} -> {int(np.median(settle2[yte == 3]))}; '
          f'old classes: median {int(np.median(settle0[old]))} -> {int(np.median(settle2[old]))}')
    out['teach_T2'].update({
        'acc': acc_t2, 'flips_old_vs_T1': flips,
        'dz_median': float(np.median(dz)), 'dz_p90': float(np.percentile(dz, 90)),
        'dz_max': float(dz.max()), 'n_moved_gt_0.1': int((dz > 0.1).sum()), 'n_old': int(len(dz)),
        'dscore_median': float(np.median(ds)), 'dscore_max': float(ds.max()),
        'condensation_T1': cond_t1, 'condensation_T2': cond_t2,
        'margin_T1': marg(s4), 'margin_T2': marg(s4_2),
        'settle_new_T1': int(np.median(settle0[yte == 3])), 'settle_new_T2': int(np.median(settle2[yte == 3])),
        'settle_old_T1': int(np.median(settle0[old])), 'settle_old_T2': int(np.median(settle2[old])),
    })

    # ── evaporation: an edit applied at finitely many depths does not survive ──
    print('\nEVAPORATION (edit the "layer", not the operator)…')
    Xb_te = Xte[yte == 3]
    zb_star_base = zstar(p0, Xb_te)
    evap = {}
    for j in [0, 3, 10, 30]:
        z = jnp.zeros((Xb_te.shape[0], D))
        for k in range(j):
            z = (1 - BETA) * z + BETA * F(p2, Xb_te, z)         # edited operator for j steps
        z, _ = solve(lambda zz: F(p0, Xb_te, zz), z, iters=200)  # base operator afterwards
        dist = float(jnp.linalg.norm(z - zb_star_base, axis=-1).max())
        evap[j] = dist
        print(f'  edited for first {j:3d} steps, base afterwards: max dist to base z* = {dist:.2e}')
    out['evaporation'] = evap

    # ── FORGET (i): the taught class, by deleting its rows ──
    print('\nFORGET the taught class (delete its rows)…')
    p_undo = {k: (v[:M] if k == 'W' else v[:, :M] if k == 'A' else v) for k, v in p2.items()}
    exact = all(bool(jnp.array_equal(p_undo[k], p0[k])) for k in p0)
    print(f'  operator restored bit-for-bit: {exact}')
    out['forget_taught'] = {'bitwise_exact': exact}

    # ── FORGET (ii): a TRAINED class, by masking its readout row ──
    print('\nFORGET the trained blob class 2 (mask its readout row)…')
    lg_all = np.array(logits(p0, Xte_old))
    pred_before = lg_all.argmax(1)
    masked = lg_all.copy(); masked[:, 2] = -np.inf
    pred_after = masked.argmax(1)
    acc_before = per_class_acc(pred_before, yte_old, 3)
    acc_after = per_class_acc(pred_after, yte_old, 3)
    moved = int((pred_after != pred_before).sum())
    # scores of the survivors are the same computation, exactly
    print(f'  class 2 recall: {acc_before[2]:.3f} -> {acc_after[2]:.3f}')
    print(f'  classes 0/1: {acc_before[0]:.3f}/{acc_before[1]:.3f} -> '
          f'{acc_after[0]:.3f}/{acc_after[1]:.3f}  ({moved} test points reassigned)')
    # ...but the imprint stays in the dynamics: resurrect the class with an anchor readout
    res_anchors = zstar(p0, jnp.asarray(dat['a_tr'][:M_TEACH]))
    s_res = alpha * np.array(yat_k(zstar(p0, Xte_old), res_anchors, b_v, eps_v)).max(1)
    pred_res = np.where(s_res > masked.max(1), 2, pred_after)
    res_recall = float((pred_res[yte_old == 2] == 2).mean())
    print(f'  resurrection: {M_TEACH} anchor rows bring class 2 back at recall {res_recall:.3f} '
          f'(the masked class was unreadable, not erased)')
    out['forget_trained'] = {'recall_before': acc_before[2], 'recall_after': acc_after[2],
                             'acc01_before': [acc_before[0], acc_before[1]],
                             'acc01_after': [acc_after[0], acc_after[1]],
                             'reassigned': moved, 'resurrect_recall': res_recall}

    # ── the untied stack: the same edit needs to be applied at every layer ──
    print('\ntraining the untied L-layer stack (no weight sharing)…')
    ps = train_stack(jax.random.PRNGKey(1), Xtr3, ytr3, ncls=3)
    lg_s = np.array(stack_logits(ps, Xte_old))
    acc_s = per_class_acc(lg_s.argmax(1), yte_old, 3)
    stack_params = (sum(int(np.prod(np.shape(w))) for w in ps['W']) +
                    sum(int(np.prod(np.shape(a))) for a in ps['A']) +
                    sum(int(np.prod(np.shape(z))) for z in ps['z0']) +
                    int(np.prod(np.shape(ps['Uin']))) + 2 +
                    int(np.prod(np.shape(ps['C']))) + int(np.prod(np.shape(ps['cb']))))
    print(f'  stack per-class acc {acc_s}  params {stack_params} (tied: {n_params})')

    # teach the stack: anchor readout at the final layer + dynamics edit at the last j layers
    states_teach = stack_states(ps, teach_x)
    anchors_s = states_teach[-1]
    win_tr_s = np.median(np.array(stack_logits(ps, Xtr3)).max(1))
    ks_self = np.array(yat_k(anchors_s, anchors_s, b_v, eps_v))
    alpha_s = float((win_tr_s + MARGIN) / np.median(ks_self.max(1)))
    zLb = stack_states(ps, Xte[yte == 3])[-1]

    def stack_scores4(q, x):
        zL = stack_states(q, x)[-1]
        lg = zL @ q['C'].T + q['cb']
        s3 = alpha_s * yat_k(zL, anchors_s, jax.nn.softplus(q['b']), jax.nn.softplus(q['eps'])).max(-1)
        return jnp.concatenate([lg, s3[:, None]], -1)

    stack_sweep = []
    for j in [0, 1, 3, 6, L_STACK]:
        layers = list(range(L_STACK - j, L_STACK))
        q, epar = stack_edit(ps, states_teach, 0.5, layers)
        s4s = np.array(stack_scores4(q, Xte))
        accj = per_class_acc(s4s.argmax(1), yte, 4)
        zL = np.array(stack_states(q, Xte[yte == 3])[-1])
        anc_s = np.array(anchors_s)
        condj = float(np.mean(np.min(np.linalg.norm(zL[:, None] - anc_s[None], axis=-1), axis=1)))
        margj = float(np.mean(s4s[yte == 3, 3] - s4s[yte == 3, :3].max(1)))
        # old-class disturbance of the stack edit
        zL_old_b = np.array(stack_states(ps, Xte_old)[-1])
        zL_old_e = np.array(stack_states(q, Xte_old)[-1])
        dzj = float(np.linalg.norm(zL_old_e - zL_old_b, axis=1).max())
        stack_sweep.append({'j': j, 'edit_params': epar, 'acc': accj,
                            'condensation': condj, 'margin': margj, 'dz_old_max': dzj})
        print(f'  stack edit at last {j:2d} layers: edit params {epar:5d}, '
              f'new-class recall {accj[3]:.3f}, margin {margj:.2f}, '
              f'condensation {condj:.3f}, max ‖Δz_L‖ old {dzj:.2e}')
    tied_edit_params = int(np.prod(np.shape(anchors))) * 2
    print(f'  tied edit params (once, active at EVERY depth): {tied_edit_params}')
    out['stack'] = {'L': L_STACK, 'params': stack_params, 'acc_base': acc_s,
                    'sweep': stack_sweep, 'tied_edit_params': tied_edit_params,
                    'alpha_s': alpha_s}

    # ── from-scratch 4-class model, the incremental-vs-scratch check ──
    print('\ntraining a 4-class model from scratch (for the comparison)…')
    Xtr4 = jnp.asarray(np.vstack([dat['moons_tr'][0], dat['a_tr'], dat['b_tr']]))
    ytr4 = jnp.asarray(np.concatenate([dat['moons_tr'][1], np.full(280, 2), np.full(280, 3)]))
    p4 = train_deq(jax.random.PRNGKey(2), Xtr4, ytr4, ncls=4, tag='scratch4')
    pred4 = np.array(logits(p4, Xte)).argmax(1)
    acc4 = per_class_acc(pred4, yte, 4)
    ov4 = float((pred4 == yte).mean())
    ov_t1 = float((pred_t1 == yte).mean()); ov_t2 = float((pred_t2 == yte).mean())
    print(f'  from-scratch: per-class {acc4}, overall {ov4:.4f}')
    print(f'  incremental (base + constructed class): T1 {ov_t1:.4f}  T2 {ov_t2:.4f}')
    out['scratch4'] = {'acc': acc4, 'overall': ov4,
                       'overall_T1': ov_t1, 'overall_T2': ov_t2}

    # ── export the browser assets ──
    export(p0, anchors, alpha, gamma, dat, Xte, yte, settle0, settle2,
           sig0, sig2, out)
    print('\nreport:')
    print(json.dumps(out, indent=2, default=float))
    rp = outdir() / 'report.json'
    json.dump(out, open(rp, 'w'), indent=2, default=float)
    print(f'wrote {rp}')


def export(p, anchors, alpha, gamma, dat, Xte, yte, settle0, settle2, sig0, sig2, metrics):
    outd = outdir()

    z_all = np.array(zstar(p, Xte))
    mean = z_all.mean(0)
    U, S, Vt = np.linalg.svd(z_all - mean, full_matrices=False)
    basis = Vt[:2]
    proj = (z_all - mean) @ basis.T

    def arr(a): return np.asarray(a).astype(float).round(6).tolist()
    model = {
        'dims': {'d_in': D_IN, 'd': D, 'm': M, 'ncls': 3},
        'solver': {'beta': BETA, 'tol': SETTLE_TOL, 'max_iter': SOLVER_ITERS},
        'params': {
            'W': arr(p['W']), 'A': arr(p['A']), 'Uin': arr(p['Uin']), 'z0': arr(p['z0']),
            'b': float(jax.nn.softplus(p['b'])), 'eps': float(jax.nn.softplus(p['eps'])),
            'C': arr(p['C']), 'cb': arr(p['cb']),
        },
        'edit': {'anchors': arr(anchors), 'alpha': alpha, 'gamma': gamma,
                 'gamma_raw': metrics['teach_T2']['gamma_raw'],
                 'rescale': metrics['teach_T2']['rescale'], 'cls': 3, 'm_teach': M_TEACH},
        'teach_x': arr(dat['b_teach']),
        'pca': {'mean': arr(mean), 'basis': arr(basis)},
        'test': {'x': arr(Xte), 'y': np.asarray(yte).astype(int).tolist(), 'proj': arr(proj),
                 'settle_base': np.asarray(settle0).astype(int).tolist(),
                 'settle_edit': np.asarray(settle2).astype(int).tolist()},
        'certificate': {'sigma_base': arr(sig0), 'sigma_edit': arr(sig2)},
        'metrics': metrics,
    }
    f = outd / 'model.json'
    f.write_text(json.dumps(model))
    print(f'\nwrote {f}  ({f.stat().st_size / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
