"""Why go deep? So you can think longer. A recursive Yat operator that solves grid
reachability by propagation, and solves BIGGER grids at test time just by iterating more.

Two moons never needed depth: the answer was settled in a handful of turns. This task
does. Reachability on a random-wall grid is a propagation problem: "reachable" spreads
one cell per step out from the goal, so an instance whose farthest reachable cell is 30
hops away genuinely needs ~30 turns. We share ONE Yat operator across all of them
(the same prototypes W and the same mixing A at every step), applied to a 5-cell
neighborhood and masked by the walls, so it is a weight-tied recurrent convolution:

    z_{k+1}[i,j] = tanh( A · φ_W( patch(z_k)[i,j] ) + U·x[i,j] + z0 ) · free[i,j]

Trained by unrolling on small grids (11×11), then at test time we run it on 15/21/27
grids with more iterations. If it learned the LOCAL rule (a cell is reachable if a free
neighbor is), it extrapolates: the same operator solves mazes far larger than any it
trained on, purely by thinking longer. And each instance needs exactly as many turns as
its own propagation distance, so depth is spent adaptively.

Run: python scripts/yat_deq_maze.py
"""
import warnings; warnings.filterwarnings('ignore')
import os, json
from pathlib import Path
from collections import deque
import numpy as np
import jax, jax.numpy as jnp
from jax import lax
import optax

jax.config.update('jax_platform_name', 'cpu')
KEY = jax.random.PRNGKey(0)

D, M = 12, 16                 # state channels, shared prototypes
KN = 5                        # neighborhood: self + up/down/left/right (4-connected)
PWALL = 0.30                  # wall density of the random grids
TRAIN_N = 11                  # training grid size
T_TRAIN = 30                  # unrolled iterations during training (enough for 11×11 to converge)


# ── maze generation + ground-truth reachability (numpy) ─────────────────────────
def make_maze(n, key):
    r = np.random.RandomState(int(key) % (2 ** 32))
    wall = (r.rand(n, n) < PWALL)
    free = np.argwhere(~wall)
    if len(free) == 0: return make_maze(n, key + 1)
    gi, gj = free[r.randint(len(free))]
    wall[gi, gj] = False
    reach = np.zeros((n, n), bool); reach[gi, gj] = True
    dq = deque([(gi, gj)]); dist = {(gi, gj): 0}; rad = 0
    while dq:
        i, j = dq.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < n and 0 <= b < n and not wall[a, b] and not reach[a, b]:
                reach[a, b] = True; dist[(a, b)] = dist[(i, j)] + 1; rad = max(rad, dist[(a, b)])
                dq.append((a, b))
    goal = np.zeros((n, n), np.float32); goal[gi, gj] = 1.0
    return (~wall).astype(np.float32), goal, reach.astype(np.float32), rad


def batch(n, bs, seed):
    fr, go, re = [], [], []
    for k in range(bs):
        f, g, r, _ = make_maze(n, seed * 100003 + k)
        fr.append(f); go.append(g); re.append(r)
    free = np.stack(fr)[..., None]; goal = np.stack(go)[..., None]
    x = np.concatenate([free, goal], -1)                    # [B,H,W,2]  input = (free, goal)
    return jnp.asarray(free), jnp.asarray(x), jnp.asarray(np.stack(re))


# ── the shared Yat operator on a grid ───────────────────────────────────────────
def shift(z, di, dj):                                        # z[i-di, j-dj], zero outside the grid
    r = jnp.roll(z, shift=(di, dj), axis=(1, 2))
    if di > 0: r = r.at[:, :di].set(0)
    if di < 0: r = r.at[:, di:].set(0)
    if dj > 0: r = r.at[:, :, :dj].set(0)
    if dj < 0: r = r.at[:, :, dj:].set(0)
    return r


def patch(z):                                                # [B,H,W,5D] the 5-cell neighborhood
    return jnp.concatenate([z, shift(z, 1, 0), shift(z, -1, 0), shift(z, 0, 1), shift(z, 0, -1)], -1)


def F(p, z, x, free):                                         # one application of the shared operator
    b = jax.nn.softplus(p['b']); eps = jax.nn.softplus(p['eps'])
    pt = patch(z)                                            # [B,H,W,5D]
    dot = pt @ p['W'].T                                      # [B,H,W,M]
    d2 = (pt ** 2).sum(-1, keepdims=True) + (p['W'] ** 2).sum(-1) - 2 * dot
    phi = (dot + b) ** 2 / (d2 + eps)                        # Yat feature
    pre = phi @ p['A'].T + x @ p['Uin'].T + p['z0']
    return jnp.tanh(pre) * free                              # walls carry nothing


def solve(p, x, free, iters):                                # iterate from z=0, return states? just final + logits path
    z = jnp.zeros((*x.shape[:3], D))
    def step(z, _): zn = F(p, z, x, free); return zn, None
    z, _ = lax.scan(step, z, None, length=iters)
    return z


def logits(p, z):                                            # per-cell reachability logit
    return (z @ p['C'].T + p['cb'])[..., 0]                  # [B,H,W]


def init_params(key):
    k = jax.random.split(key, 6)
    return {
        'W':   jax.random.normal(k[0], (M, KN * D)) * 0.3,
        'A':   jax.random.normal(k[1], (D, M)) * (0.6 / np.sqrt(M)),
        'Uin': jax.random.normal(k[2], (D, 2)) * 0.6,
        'z0':  jnp.zeros((D,)),
        'b':   jnp.full((), float(np.log(np.expm1(0.5)))),
        'eps': jnp.full((), float(np.log(np.expm1(1.0)))),
        'C':   jax.random.normal(k[4], (1, D)) * 0.5,
        'cb':  jnp.zeros((1,)),
    }


def loss_fn(p, x, free, reach):
    z = solve(p, x, free, T_TRAIN)
    lg = logits(p, z)
    fm = free[..., 0]
    bce = optax.sigmoid_binary_cross_entropy(lg, reach)
    return (bce * fm).sum() / fm.sum()


def cell_acc(p, x, free, reach, iters):
    z = solve(p, x, free, iters); pred = (logits(p, z) > 0).astype(jnp.float32)
    fm = free[..., 0]
    return ((pred == reach) * fm).sum() / fm.sum()


def main():
    params = init_params(KEY)
    STEPS = int(os.environ.get('STEPS', '1500'))
    sched = optax.cosine_decay_schedule(3e-3, STEPS, alpha=0.05)
    opt = optax.adam(sched); opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, x, free, reach):
        l, g = jax.value_and_grad(loss_fn)(params, x, free, reach)
        updates, opt_state = opt.update(g, opt_state)
        return optax.apply_updates(params, updates), opt_state, l

    print(f'training recursive Yat maze-solver on {TRAIN_N}×{TRAIN_N} random-wall grids '
          f'(unrolled {T_TRAIN} iters, state d={D}, {M} shared prototypes)…')
    for it in range(STEPS + 1):
        free, x, reach = batch(TRAIN_N, 48, it + 1)
        params, opt_state, l = step(params, opt_state, x, free, reach)
        if it % 200 == 0:
            fv, xv, rv = batch(TRAIN_N, 64, 90000 + it)
            a = float(cell_acc(params, xv, fv, rv, T_TRAIN))
            print(f'  step {it:4d}  loss {l:.4f}  train-size cell acc {a:.3f}')

    # ── extrapolation: bigger grids, more iterations ──
    print('\n── extrapolation (train size = 11) ──')
    sizes = [11, 15, 21, 27]
    extrap = {}
    for n in sizes:
        fv, xv, rv = batch(n, 80, 500000 + n)
        # accuracy as a function of how long we let it think
        iters_grid = list(range(2, 2 * n + 12, 2))
        accs = [float(cell_acc(params, xv, fv, rv, T)) for T in iters_grid]
        extrap[n] = {'iters': iters_grid, 'acc': accs, 'best': max(accs)}
        print(f'  {n:2d}×{n:<2d}: best cell acc {max(accs):.3f}  '
              f'(at {iters_grid[int(np.argmax(accs))]} iters; acc@{2*TRAIN_N} iters '
              f'= {accs[min(len(accs)-1, TRAIN_N-1)]:.3f})')

    # ── adaptive depth: iterations-to-converge vs BFS radius, per instance ──
    def iters_to_settle(p, x, free, tol=0.02, maxk=120):
        z = jnp.zeros((*x.shape[:3], D)); prev = jnp.zeros_like(logits(p, z))
        ks = maxk
        for k in range(maxk):
            z = F(p, z, x, free); lg = jax.nn.sigmoid(logits(p, z))
            ch = float(jnp.abs(lg - prev).max()); prev = lg
            if ch < tol: ks = k + 1; break
        return ks
    radii, settle = [], []
    for k in range(120):
        n = [11, 15, 21, 27][k % 4]
        f, g, r, rad = make_maze(n, 700000 + k)
        free1 = jnp.asarray(f)[None, ..., None]
        x1 = jnp.asarray(np.concatenate([f[..., None], g[..., None]], -1))[None]
        radii.append(int(rad)); settle.append(iters_to_settle(params, x1, free1))
    corr = float(np.corrcoef(radii, settle)[0, 1])
    print(f'\n  adaptive depth: iterations-to-settle vs BFS radius correlation r = {corr:.2f}')

    export_assets(params, extrap, radii, settle, corr)


def export_assets(p, extrap, radii, settle, corr):
    root = Path(__file__).resolve().parents[1]
    out = root / 'public' / 'yat-deq-maze'; out.mkdir(parents=True, exist_ok=True)

    def arr(a): return np.asarray(a).astype(float).round(6).tolist()
    # a few showcase mazes per size for the live viz (walls, goal, ground-truth reachable)
    examples = {}
    for n in [11, 15, 21, 27]:
        ex = []
        for s in range(6):
            f, g, r, rad = make_maze(n, 900000 + n * 10 + s)
            gi, gj = np.argwhere(g > 0)[0]
            ex.append({'wall': (1 - f).astype(int).tolist(), 'goal': [int(gi), int(gj)],
                       'reach': r.astype(int).tolist(), 'radius': int(rad)})
        examples[str(n)] = ex

    model = {
        'dims': {'d': D, 'm': M, 'kn': KN},
        'params': {'W': arr(p['W']), 'A': arr(p['A']), 'Uin': arr(p['Uin']), 'z0': arr(p['z0']),
                   'b': float(jax.nn.softplus(p['b'])), 'eps': float(jax.nn.softplus(p['eps'])),
                   'C': arr(p['C']), 'cb': arr(p['cb'])},
        'train_size': TRAIN_N,
        'extrap': {str(k): v for k, v in extrap.items()},
        'adaptive': {'radius': radii, 'settle': settle, 'corr': corr},
        'examples': examples,
    }
    (out / 'model.json').write_text(json.dumps(model))
    kb = (out / 'model.json').stat().st_size / 1024
    print(f'  wrote {out / "model.json"}  ({kb:.0f} KB)')


if __name__ == '__main__':
    main()
