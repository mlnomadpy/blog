"""The Yat Deep Equilibrium Model: one shared Yat operator, iterated to its fixed point.

A normal deep net stacks L *different* layers. Here we share the prototypes W (and the
mixing A) across depth, so the whole network is a single map

    F(z; x) = tanh( A · φ_W(z) + U · x + z0 ),   φ_W(z)_i = (⟨z,w_i⟩ + b)² / (‖z-w_i‖² + ε)

applied over and over. "Depth" becomes *iteration count*, and the function the network
computes is defined not by a stack but by ONE fixed-point equation

    z* = F(z*; x) ,   ŷ = C z* + c .

Because it is the fixed point of a *shared* Yat operator, the model is:
  • compact         — the entire (arbitrary-depth) net is two matrices (W, A) + injection U.
  • white-box-ish   — the same M prototypes describe the computation at every depth.
  • depth-free      — we solve the equation directly (Anderson/Picard), not by unrolling,
                      and differentiate through z* with the implicit function theorem
                      (constant memory, no backprop-through-iterations).
  • adaptive-depth  — if F is a contraction (we measure ‖J‖₂ at z*), halt when the residual
                      ‖z_{k+1}-z_k‖ falls below a tol; early iterates already classify.

Run: python scripts/yat_deq.py
Outputs a text report + two PNGs (/tmp/yat_deq_convergence.png, /tmp/yat_deq_boundary.png).
"""
import warnings; warnings.filterwarnings('ignore')
import os, json
import numpy as np
import jax, jax.numpy as jnp
from jax import lax
import optax
from sklearn.datasets import make_moons

jax.config.update('jax_platform_name', 'cpu')
KEY = jax.random.PRNGKey(0)

D_IN, D, M, NCLS = 2, 32, 24, 2          # input dim, equilibrium-state dim, # shared prototypes, classes
SOLVER_ITERS, BETA, TOL = 120, 0.7, 1e-6  # forward fixed-point solver budget / damping / tolerance


# ── the shared Yat operator F(z; x) ────────────────────────────────────────────
def yat_feats(z, p):                      # φ_W(z): similarities of state z to the M shared prototypes -> [bsz, M]
    W = p['W']; b = jax.nn.softplus(p['b']); eps = jax.nn.softplus(p['eps'])
    dot = z @ W.T
    d2 = (z ** 2).sum(-1, keepdims=True) + (W ** 2).sum(-1) - 2 * dot
    return (dot + b) ** 2 / (d2 + eps)


def F(p, x, z):                           # one application of the shared operator (input x injected every step)
    # tanh is 1-Lipschitz and bounds the state to (-1,1)^D, so ‖J_F‖ ≤ ‖A‖·Lip(φ_W): controllable -> contraction.
    pre = yat_feats(z, p) @ p['A'].T + x @ p['Uin'].T + p['z0']
    return jnp.tanh(pre)


# ── fixed-point solver: damped Picard z_{k+1} = (1-β)z_k + β F(z_k) ─────────────
def solve(fz, z0, iters=SOLVER_ITERS, beta=BETA):
    def step(z, _):
        zn = (1 - beta) * z + beta * fz(z)
        return zn, jnp.max(jnp.linalg.norm(zn - z, axis=-1))
    zT, resid = lax.scan(step, z0, None, length=iters)
    return zT, resid                      # zT ≈ z*, resid = max per-sample residual at each iter


# ── DEQ layer: forward solves the equilibrium, backward uses the IMPLICIT function theorem ──
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
    # z* = F(z*), so dz* = J dz* + ∂F.  Cotangent obeys (I - Jᵀ) u = g  ->  u = Jᵀu + g (a fixed point).
    p, x, z_star = res
    _, vjp_z = jax.vjp(lambda z: F(p, x, z), z_star)          # Jᵀ · (·)
    u, _ = solve(lambda u: vjp_z(u)[0] + g, jnp.zeros_like(g))  # solve adjoint by the SAME iteration
    _, vjp_px = jax.vjp(lambda pp, xx: F(pp, xx, z_star), p, x)
    gp, gx = vjp_px(u)                                          # param / input grads without unrolling the solver
    return gp, gx


deq.defvjp(deq_fwd, deq_bwd)


# ── full model + loss ──────────────────────────────────────────────────────────
def logits(p, x):
    z_star = deq(p, x)
    return z_star @ p['C'].T + p['cb']


def spectral_sigma(p, x, z, key, pit=6):
    """Differentiable estimate of ‖J‖₂ of F at state z via power iteration on JᵀJ (direction detached)."""
    fz = lambda zz: F(p, x, zz)
    v = jax.random.normal(key, z.shape)
    v = v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    for _ in range(pit):
        Jv = jax.jvp(fz, (z,), (v,))[1]
        _, vjp = jax.vjp(fz, z); JtJv = vjp(Jv)[0]
        v = jax.lax.stop_gradient(JtJv / (jnp.linalg.norm(JtJv, axis=-1, keepdims=True) + 1e-9))
    Jv = jax.jvp(fz, (z,), (v,))[1]                 # Rayleigh quotient: ‖J v‖ for the dominant v -> ≈ ‖J‖₂
    return jnp.linalg.norm(Jv, axis=-1)             # per-sample, differentiable in params


LAMBDA_C, SIGMA_TARGET = 3.0, 0.7                   # push ‖J‖₂ below target (with margin) so F is a genuine contraction


def loss_fn(p, x, y, key):
    lg = logits(p, x)
    ce = optax.softmax_cross_entropy_with_integer_labels(lg, y).mean()
    reg = 1e-4 * sum(jnp.sum(v ** 2) for k, v in p.items() if k in ('A', 'Uin', 'C'))
    z_star = jax.lax.stop_gradient(deq(p, x))
    sigma = spectral_sigma(p, x, z_star, key)
    contract = LAMBDA_C * jnp.mean(jax.nn.relu(sigma - SIGMA_TARGET) ** 2)   # 0 once ‖J‖₂ < target
    return ce + reg + contract


def init_params(key):
    k = jax.random.split(key, 7)
    return {
        'W':   jax.random.normal(k[0], (M, D)) * 0.6,          # the M shared prototypes (in state space)
        'A':   jax.random.normal(k[1], (D, M)) * (0.5 / np.sqrt(M)),  # small -> encourages contraction
        'Uin': jax.random.normal(k[2], (D, D_IN)) * 0.5,       # input injection
        'z0':  jnp.zeros((D,)),
        'b':   jnp.full((), float(np.log(np.expm1(0.5)))),     # softplus -> ~0.5
        'eps': jnp.full((), float(np.log(np.expm1(1.0)))),     # softplus -> ~1.0
        'C':   jax.random.normal(k[5], (NCLS, D)) * 0.5,       # linear readout on z*
        'cb':  jnp.zeros((NCLS,)),
    }


# ── diagnostics ─────────────────────────────────────────────────────────────────
def contraction_norm(p, x, key, iters=30):
    """Estimate ‖J‖₂ of F at z* by power iteration on JᵀJ (‖J‖₂ < 1  ⇒  F is a contraction)."""
    z0 = jnp.zeros((x.shape[0], D)); z_star, _ = solve(lambda z: F(p, x, z), z0)
    fz = lambda z: F(p, x, z)
    v = jax.random.normal(key, z_star.shape)
    v = v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    def pit(v, _):
        Jv = jax.jvp(fz, (z_star,), (v,))[1]                   # J v
        _, vjp = jax.vjp(fz, z_star); JtJv = vjp(Jv)[0]        # Jᵀ (J v)
        s = jnp.sqrt(jnp.linalg.norm(JtJv, axis=-1) + 1e-12)   # ≈ ‖J‖₂ per sample
        v = JtJv / (jnp.linalg.norm(JtJv, axis=-1, keepdims=True) + 1e-9)
        return v, s
    _, s = lax.scan(pit, v, None, length=iters)
    return s[-1]                                               # per-sample ‖J‖₂ estimate


def depth_profile(p, x, y, max_k=SOLVER_ITERS):
    """Accuracy and residual as a function of unroll depth k -> shows adaptive-depth halting is safe."""
    z = jnp.zeros((x.shape[0], D)); accs, resids = [], []
    for _ in range(max_k):
        zn = (1 - BETA) * z + BETA * F(p, x, z)
        resids.append(float(jnp.max(jnp.linalg.norm(zn - z, axis=-1))))
        z = zn
        pred = jnp.argmax(z @ p['C'].T + p['cb'], axis=-1)
        accs.append(float((pred == y).mean()))
    return np.array(accs), np.array(resids)


# ── data ─────────────────────────────────────────────────────────────────────────
Xall, yall = make_moons(n_samples=1400, noise=0.16, random_state=1)
Xall = (Xall - Xall.mean(0)) / Xall.std(0)
ntr = 1000
Xtr, ytr = jnp.asarray(Xall[:ntr]), jnp.asarray(yall[:ntr])
Xte, yte = jnp.asarray(Xall[ntr:]), jnp.asarray(yall[ntr:])


# ── train ─────────────────────────────────────────────────────────────────────────
def main():
    params = init_params(KEY)
    STEPS = int(os.environ.get('STEPS', '1500'))
    sched = optax.cosine_decay_schedule(3e-3, STEPS, alpha=0.03)   # decay LR so the contraction constraint settles
    opt = optax.adam(sched); opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, x, y, key):
        l, g = jax.value_and_grad(loss_fn)(params, x, y, key)
        updates, opt_state = opt.update(g, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, l

    print(f'training Yat-DEQ  (state d={D}, {M} shared prototypes, fixed-point solver {SOLVER_ITERS} iters)…')
    for it in range(STEPS + 1):
        params, opt_state, l = step(params, opt_state, Xtr, ytr, jax.random.fold_in(KEY, it))
        if it % 250 == 0:
            tr = float((jnp.argmax(logits(params, Xtr), 1) == ytr).mean())
            te = float((jnp.argmax(logits(params, Xte), 1) == yte).mean())
            jn = contraction_norm(params, Xte, jax.random.PRNGKey(it))
            print(f'  step {it:4d}  loss {l:.4f}  train {tr:.3f}  test {te:.3f}  '
                  f'‖J‖₂≈{float(jn.mean()):.3f} (max {float(jn.max()):.3f})')

    # final report
    tr = float((jnp.argmax(logits(params, Xtr), 1) == ytr).mean())
    te = float((jnp.argmax(logits(params, Xte), 1) == yte).mean())
    jn = contraction_norm(params, Xte, jax.random.PRNGKey(7))
    _, resid = solve(lambda z: F(params, Xte, z), jnp.zeros((Xte.shape[0], D)))
    accs, resids = depth_profile(params, Xte, yte)
    # effective depth = first iter where the residual is below tol
    eff = int(np.argmax(resids < TOL)) if np.any(resids < TOL) else SOLVER_ITERS
    # depth at which accuracy first reaches within 0.5% of the equilibrium accuracy
    acc_star = accs[-1]; acc_depth = int(np.argmax(accs >= acc_star - 0.005)) + 1

    print('\n── Yat Deep Equilibrium Model ─────────────────────────────────')
    print(f'  train acc                : {tr:.3f}')
    print(f'  test  acc                : {te:.3f}')
    print(f'  contraction ‖J‖₂ at z*   : mean {float(jn.mean()):.3f}, max {float(jn.max()):.3f}'
          f'   {"→ CONTRACTION (unique z*, any-depth safe)" if float(jn.max()) < 1 else "→ NOT contractive"}')
    print(f'  fixed-point residual     : {float(resid[-1]):.2e} after {SOLVER_ITERS} iters')
    print(f'  effective depth (res<tol): {eff} iterations to reach tol={TOL:g}')
    print(f'  depth to hit final acc   : {acc_depth} iterations (adaptive halt is safe past here)')
    print(f'  parameter count          : {sum(int(np.prod(v.shape)) for v in params.values())} '
          f'(shared across ALL depth)')

    # ── plots ──
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].semilogy(np.array(resid), color='#c0392b')
    ax[0].axhline(TOL, ls='--', c='gray', lw=1); ax[0].set_title('fixed-point residual  ‖z_{k+1}−z_k‖')
    ax[0].set_xlabel('iteration (= depth)'); ax[0].set_ylabel('max residual'); ax[0].grid(alpha=.3)
    ax[1].plot(np.arange(1, len(accs) + 1), accs, color='#2471a3')
    ax[1].axvline(acc_depth, ls='--', c='green', lw=1, label=f'plateau @ {acc_depth}')
    ax[1].set_ylim(0.4, 1.0); ax[1].set_title('test accuracy vs unroll depth')
    ax[1].set_xlabel('iteration (= depth)'); ax[1].set_ylabel('accuracy'); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.suptitle('Yat-DEQ: one shared Yat operator iterated to equilibrium', fontweight='bold')
    fig.tight_layout(); fig.savefig('/tmp/yat_deq_convergence.png', dpi=130); plt.close(fig)

    # decision boundary from the equilibrium readout
    gx, gy = np.meshgrid(np.linspace(-2.4, 2.4, 220), np.linspace(-2.4, 2.4, 220))
    grid = jnp.asarray(np.c_[gx.ravel(), gy.ravel()])
    probs = jax.nn.softmax(logits(params, grid), -1)[:, 1].reshape(gx.shape)
    fig, ax = plt.subplots(figsize=(5.4, 5))
    ax.contourf(gx, gy, np.array(probs), levels=20, cmap='RdBu_r', alpha=.75)
    ax.contour(gx, gy, np.array(probs), levels=[0.5], colors='k', linewidths=1.4)
    Xnp = np.array(Xte); ynp = np.array(yte)
    ax.scatter(Xnp[ynp == 0, 0], Xnp[ynp == 0, 1], s=8, c='#2166ac', edgecolor='w', lw=.2)
    ax.scatter(Xnp[ynp == 1, 0], Xnp[ynp == 1, 1], s=8, c='#b2182b', edgecolor='w', lw=.2)
    ax.set_title(f'Yat-DEQ decision surface  (test acc {te:.3f})'); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig('/tmp/yat_deq_boundary.png', dpi=130); plt.close(fig)

    json.dump({'train': tr, 'test': te, 'Jnorm_mean': float(jn.mean()), 'Jnorm_max': float(jn.max()),
               'final_residual': float(resid[-1]), 'effective_depth': eff, 'acc_depth': acc_depth},
              open('/tmp/yat_deq_report.json', 'w'), indent=2)
    print('\n  wrote /tmp/yat_deq_convergence.png, /tmp/yat_deq_boundary.png, /tmp/yat_deq_report.json')

    export_browser_assets(params, tr, te, float(jn.mean()), float(jn.max()),
                          float(resid[-1]), eff, acc_depth)


def export_browser_assets(p, tr, te, jn_mean, jn_max, residual, eff, acc_depth):
    """Dump public/yat-deq/model.json so the explainer's viz can run the exact operator live in-browser."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    out = root / 'public' / 'yat-deq'; out.mkdir(parents=True, exist_ok=True)

    # equilibria over the test set, then a 2-D PCA window on the 32-D state for the trajectory viz
    z_star = np.array(deq(p, Xte))
    mean = z_star.mean(0)
    U, S, Vt = np.linalg.svd(z_star - mean, full_matrices=False)
    basis = Vt[:2]                                   # (2, D): top-2 principal directions of the equilibrium cloud
    proj = (z_star - mean) @ basis.T                 # (Nte, 2)

    # per-input iterations-to-converge (adaptive depth), replicating the browser iteration exactly
    def iterate_count(x, tol=1e-4, maxk=SOLVER_ITERS):
        z = jnp.zeros((x.shape[0], D)); ks = jnp.full((x.shape[0],), maxk, dtype=jnp.int32)
        for k in range(maxk):
            zn = (1 - BETA) * z + BETA * F(p, x, z)
            r = jnp.linalg.norm(zn - z, axis=-1)
            ks = jnp.where((r < tol) & (ks == maxk), k + 1, ks)
            z = zn
        return np.array(ks)
    kcount = iterate_count(Xte)

    def arr(a): return np.asarray(a).astype(float).round(6).tolist()
    model = {
        'dims': {'d_in': D_IN, 'd': D, 'm': M, 'ncls': NCLS},
        'solver': {'beta': BETA, 'tol': 1e-4, 'max_iter': SOLVER_ITERS},
        'params': {
            'W': arr(p['W']), 'A': arr(p['A']), 'Uin': arr(p['Uin']), 'z0': arr(p['z0']),
            'b': float(jax.nn.softplus(p['b'])), 'eps': float(jax.nn.softplus(p['eps'])),
            'C': arr(p['C']), 'cb': arr(p['cb']),
        },
        'pca': {'mean': arr(mean), 'basis': arr(basis)},
        'test': {'x': arr(Xte), 'y': np.array(yte).astype(int).tolist(),
                 'proj': arr(proj), 'kcount': kcount.astype(int).tolist()},
        'metrics': {'train_acc': tr, 'test_acc': te, 'jnorm_mean': jn_mean, 'jnorm_max': jn_max,
                    'residual': residual, 'effective_depth': eff, 'acc_depth': acc_depth},
    }
    (out / 'model.json').write_text(json.dumps(model))
    kb = (out / 'model.json').stat().st_size / 1024
    print(f'  wrote {out / "model.json"}  ({kb:.0f} KB)  '
          f'[{M} prototypes, {len(model["test"]["y"])} test points]')


if __name__ == '__main__':
    main()
