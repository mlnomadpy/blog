"""D3: A network that conserves energy (Hamiltonian nets).

Extends the Arc D program from `momentum_resnet.py` (D1) and
`velocity_ledger.py` (D2). D1 established depth=time, skip=forward-Euler,
and that leapfrog holds energy where Euler leaks it. This experiment takes
the next integrator off the shelf: make the residual field the *symplectic
gradient of a learned scalar energy*, and the network conserves that energy
by construction (Greydanus et al., 2019, Hamiltonian Neural Networks).

Two parts, same shape as D1:

  Part A (the physics fact). An ideal pendulum, H(q,p) = p^2/2 + (1 - cos q),
  whose true energy is exactly conserved. A plain MLP trained to predict the
  vector field (q_dot, p_dot) rolls out with energy drift; an HNN that
  predicts a scalar H_theta and takes (dH/dp, -dH/dq) conserves energy over
  the same long rollout. This is the existence proof at the level of physics.

  Part B (the network fact). Hidden state z = (q, p), each in R^d. A plain
  residual block z += h * F(z) versus a Hamiltonian/leapfrog block that steps
  a learned potential V_theta(q). On moons / rings / spirals, the two match on
  accuracy (existence proof, not a benchmark win), but only the Hamiltonian
  net keeps a flat energy trace through depth, and it stays stable when depth
  is pushed far past training (a fixed total time T, h = T / L, more steps).

Everything is a real run. Dumps JSON to public/hamiltonian-net/ for the
explainer viz; the companion re-renders GIFs from the same numbers.

Run on Kaggle GPU (never locally). Self-contained: JAX + optax + numpy.
"""

import json
import math
import os
import time
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
import optax

# On Kaggle the launcher captures a `results/` dir next to this script; a local
# scripts/export_hamiltonian_viz.py later reshapes it into public/hamiltonian-net/.
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

SMOKE = os.environ.get("SMOKE", "0") == "1"


# ─────────────────────────────────────────────────────────────────────────────
# Part A: the physics fact. Ideal pendulum, plain-MLP field vs HNN field.
# H(q, p) = p^2 / 2 + (1 - cos q)  (unit mass, unit gravity/length).
# ─────────────────────────────────────────────────────────────────────────────

def true_energy(q, p):
    return 0.5 * p ** 2 + (1.0 - np.cos(q))


def true_field(q, p):
    # q_dot = dH/dp = p ;  p_dot = -dH/dq = -sin q
    return p, -np.sin(q)


def sample_pendulum(rng, n):
    """Random (q, p) states over a band of energies below the separatrix (2.0)."""
    q = rng.uniform(-2.0, 2.0, size=n)
    p = rng.uniform(-2.0, 2.0, size=n)
    return np.stack([q, p], axis=1).astype(np.float32)


def params_to_lists(p):
    """Nested (w, b) pairs / dicts -> plain lists for JSON export."""
    if isinstance(p, dict):
        return {k: params_to_lists(v) for k, v in p.items()}
    if isinstance(p, (list, tuple)):
        return [params_to_lists(v) for v in p]
    return np.asarray(p).tolist()


def mlp_params(key, sizes):
    params = []
    keys = jax.random.split(key, len(sizes) - 1)
    for k, (a, b) in zip(keys, zip(sizes[:-1], sizes[1:])):
        w = jax.random.normal(k, (a, b)) * math.sqrt(2.0 / a)
        params.append((w, jnp.zeros((b,))))
    return params


def mlp_apply(params, x):
    h = x
    for w, b in params[:-1]:
        h = jnp.tanh(h @ w + b)
    w, b = params[-1]
    return h @ w + b


def hnn_energy(params, qp):
    # scalar energy per state; qp: (..., 2) -> (...,)
    return mlp_apply(params, qp)[..., 0]


def hnn_field(params, qp):
    # symplectic gradient of the learned scalar: (dH/dp, -dH/dq)
    g = jax.vmap(jax.grad(lambda s: hnn_energy(params, s)))(qp)  # (n, 2)
    dHdq, dHdp = g[:, 0], g[:, 1]
    return jnp.stack([dHdp, -dHdq], axis=1)


def baseline_field(params, qp):
    return mlp_apply(params, qp)  # (n, 2), predicts (q_dot, p_dot) directly


def train_field(kind, key, states, targets, steps=4000, lr=1e-3):
    sizes = [2, 64, 64, 1] if kind == "hnn" else [2, 64, 64, 2]
    params = mlp_params(key, sizes)
    opt = optax.adam(lr)
    st = opt.init(params)
    S = jnp.asarray(states)
    T = jnp.asarray(targets)

    def loss_fn(params):
        pred = hnn_field(params, S) if kind == "hnn" else baseline_field(params, S)
        return jnp.mean((pred - T) ** 2)

    grad_fn = jax.value_and_grad(loss_fn)

    def upd(carry, _):
        params, st = carry
        loss, g = grad_fn(params)
        updates, st = opt.update(g, st)
        params = optax.apply_updates(params, updates)
        return (params, st), loss

    # whole training loop runs on-device as one scan (no per-step host sync)
    run = jax.jit(lambda p, s: jax.lax.scan(upd, (p, s), None, length=steps))
    (params, st), losses = run(params, st)
    return params, float(losses[-1])


def rollout(field_fn, q0, p0, dt, n_steps):
    """RK4 rollout of a learned field from a single (q0, p0); returns energy trace."""
    s0 = jnp.asarray([[q0, p0]], dtype=jnp.float32)

    def step(s, _):
        k1 = field_fn(s)
        k2 = field_fn(s + 0.5 * dt * k1)
        k3 = field_fn(s + 0.5 * dt * k2)
        k4 = field_fn(s + dt * k3)
        s2 = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return s2, s2[0]                       # record (q, p) each step

    _, traj = jax.jit(lambda s: jax.lax.scan(step, s, None, length=n_steps))(s0)
    traj = np.asarray(traj)                    # (n_steps, 2)
    q = np.concatenate([[q0], traj[:, 0]])
    p = np.concatenate([[p0], traj[:, 1]])
    return q, p, true_energy(q, p)


def run_physics(seed=0):
    print("=" * 72)
    print("PART A: pendulum, plain-MLP field vs HNN field")
    rng = np.random.default_rng(seed)
    fit_steps = 300 if SMOKE else 4000
    n = 200 if SMOKE else 1200
    states = sample_pendulum(rng, 400 if SMOKE else 2000)
    qd, pd = true_field(states[:, 0], states[:, 1])
    targets = np.stack([qd, pd], axis=1).astype(np.float32)

    key = jax.random.PRNGKey(seed)
    kb, kh = jax.random.split(key)
    base_p, base_loss = train_field("base", kb, states, targets, steps=fit_steps)
    hnn_p, hnn_loss = train_field("hnn", kh, states, targets, steps=fit_steps)
    print(f"  field-fit MSE: baseline {base_loss:.2e}, HNN {hnn_loss:.2e}")

    q0, p0, dt = 1.6, 0.0, 0.05   # a wide swing, well below separatrix
    E0 = float(true_energy(np.array(q0), np.array(p0)))

    bq, bp, be = rollout(lambda s: baseline_field(base_p, s), q0, p0, dt, n)
    hq, hp, he = rollout(lambda s: hnn_field(hnn_p, s), q0, p0, dt, n)

    base_drift = float(np.max(np.abs(be - E0)) / abs(E0))
    hnn_drift = float(np.max(np.abs(he - E0)) / abs(E0))
    print(f"  rollout {n} steps, E0 = {E0:.4f}: "
          f"baseline max|dE|/E0 = {base_drift:.2%}, HNN = {hnn_drift:.2%}")

    sub = max(1, n // 400)
    phys = dict(
        E0=E0, dt=dt, n_steps=n, q0=q0, p0=p0,
        baseline_field_mse=base_loss, hnn_field_mse=hnn_loss,
        baseline_energy_drift=base_drift, hnn_energy_drift=hnn_drift,
        trace=dict(
            t=[float(i * dt) for i in range(0, n + 1, sub)],
            baseline_q=bq[::sub].tolist(), baseline_p=bp[::sub].tolist(),
            baseline_E=be[::sub].tolist(),
            hnn_q=hq[::sub].tolist(), hnn_p=hp[::sub].tolist(),
            hnn_E=he[::sub].tolist(),
        ),
        # trained field weights, exported so the explainer can integrate the
        # real learned fields live in the browser
        models=dict(baseline=params_to_lists(base_p),
                    hnn=params_to_lists(hnn_p)),
    )
    return phys


# ─────────────────────────────────────────────────────────────────────────────
# Part B: the network fact. Plain ResNet vs Hamiltonian (leapfrog) ResNet.
# Hidden state z = (q, p), q,p in R^d. Plain: z += h F(z). Hamiltonian:
# leapfrog on a learned potential V_theta(q), kinetic p^2/2, so the learned
# energy E = ||p||^2/2 + V(q) is conserved (bounded drift) along depth.
# ─────────────────────────────────────────────────────────────────────────────

DIM = 4          # q in R^DIM, p in R^DIM ; residual stream is 2*DIM
HIDDEN = 32      # width inside each block's potential/field MLP
T_TOTAL = 6.0    # fixed total integration time; h = T / L
DEPTHS = (16,) if SMOKE else (16, 64)
SEEDS = (0,) if SMOKE else (0, 1, 2)
N_TRAIN, N_TEST = (256, 512) if SMOKE else (1024, 2048)
STEPS, LR = (400 if SMOKE else 5000), 3e-3
OPT = optax.adam(LR)   # module-level so the jitted training scan is reused across seeds


def make_moons(n, rng, noise=0.08):
    t = rng.uniform(0, math.pi, n // 2)
    x0 = np.stack([np.cos(t), np.sin(t)], 1)
    x1 = np.stack([1 - np.cos(t), 1 - np.sin(t) - 0.5], 1)
    X = np.concatenate([x0, x1], 0) + rng.normal(0, noise, (n, 2))
    y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(np.int32)
    return X.astype(np.float32), y


def make_rings(n, rng, noise=0.04):
    r = np.where(rng.random(n) < 0.5, 0.5, 1.0)
    th = rng.uniform(0, 2 * math.pi, n)
    X = np.stack([r * np.cos(th), r * np.sin(th)], 1) + rng.normal(0, noise, (n, 2))
    y = (r > 0.75).astype(np.int32)
    return X.astype(np.float32), y


def make_spirals(n, rng, noise=0.05, turns=1.75):
    k = n // 2
    t = np.sqrt(rng.random(k)) * turns * 2 * math.pi
    a = np.stack([t * np.cos(t), t * np.sin(t)], 1) / (turns * 2 * math.pi)
    b = -a
    X = np.concatenate([a, b], 0) + rng.normal(0, noise, (n, 2))
    y = np.concatenate([np.zeros(k), np.ones(k)]).astype(np.int32)
    return X.astype(np.float32), y


DATASETS = {"moons": make_moons, "rings": make_rings, "spirals": make_spirals}


def dense(key, a, b):
    w = jax.random.normal(key, (a, b)) * math.sqrt(2.0 / a)
    return (w, jnp.zeros((b,)))


def init_params(key, kind):
    ks = jax.random.split(key, 8)
    p = {}
    p["enc"] = dense(ks[0], 2, 2 * DIM)             # x -> (q0, p0)
    # one shared block (weight-tied across depth, like D1's tied field)
    p["b1"] = dense(ks[1], DIM if kind == "ham" else 2 * DIM, HIDDEN)
    p["b2"] = dense(ks[2], HIDDEN, DIM if kind == "ham" else 2 * DIM)
    p["dec"] = dense(ks[3], 2 * DIM, 2)             # z_final -> logits
    return p


def potential(p, q):
    # scalar V_theta(q) per row; q: (n, DIM) -> (n,)
    h = jnp.tanh(q @ p["b1"][0] + p["b1"][1])
    return (h @ p["b2"][0] + p["b2"][1]).sum(axis=1)  # sum -> scalar field per row


def grad_V(p, q):
    return jax.vmap(jax.grad(lambda qi: potential(p, qi[None])[0]))(q)


def plain_field(p, z):
    h = jnp.tanh(z @ p["b1"][0] + p["b1"][1])
    return h @ p["b2"][0] + p["b2"][1]


def _depth_step(params, kind, h, q, pmom):
    if kind == "ham":
        pmom = pmom - 0.5 * h * grad_V(params, q)     # half kick
        q = q + h * pmom                               # drift
        pmom = pmom - 0.5 * h * grad_V(params, q)      # half kick
    else:
        z = jnp.concatenate([q, pmom], axis=1)
        z = z + h * plain_field(params, z)
        q, pmom = z[:, :DIM], z[:, DIM:]
    return q, pmom


def forward(params, X, kind, L):
    """Logits only. Depth is a lax.scan over the (weight-tied) block."""
    h = T_TOTAL / L
    z = X @ params["enc"][0] + params["enc"][1]
    q, pmom = z[:, :DIM], z[:, DIM:]

    def body(carry, _):
        return _depth_step(params, kind, h, *carry), None

    (q, pmom), _ = jax.lax.scan(body, (q, pmom), None, length=L)
    z = jnp.concatenate([q, pmom], axis=1)
    return z @ params["dec"][0] + params["dec"][1]


def trace_forward(params, X, kind, L):
    """Per-depth telemetry: (energy_trace [ham only, else zeros], rms_trace), length L."""
    h = T_TOTAL / L
    z = X @ params["enc"][0] + params["enc"][1]
    q, pmom = z[:, :DIM], z[:, DIM:]

    def body(carry, _):
        q, pmom = _depth_step(params, kind, h, *carry)
        zc = jnp.concatenate([q, pmom], axis=1)
        rms = jnp.sqrt((zc ** 2).mean())
        E = (0.5 * (pmom ** 2).sum(1) + potential(params, q)).mean() \
            if kind == "ham" else jnp.float32(0.0)
        return (q, pmom), (E, rms)

    _, (Es, rms) = jax.lax.scan(body, (q, pmom), None, length=L)
    return Es, rms


_forward_jit = jax.jit(forward, static_argnums=(2, 3))
_trace_jit = jax.jit(trace_forward, static_argnums=(2, 3))


@partial(jax.jit, static_argnames=("kind", "L", "steps"))
def _train_run(params, opt_state, X, y, kind, L, steps):
    """Whole training loop as one on-device scan; reused across seeds/datasets
    for a given (kind, L) (data + init are arguments, not baked constants)."""
    def loss_fn(p):
        logits = forward(p, X, kind, L)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    grad_fn = jax.value_and_grad(loss_fn)

    def upd(carry, _):
        p, s = carry
        loss, g = grad_fn(p)
        updates, s = OPT.update(g, s)
        p = optax.apply_updates(p, updates)
        return (p, s), loss

    (params, opt_state), losses = jax.lax.scan(upd, (params, opt_state),
                                               None, length=steps)
    return params, losses


def train_one(ds_name, kind, L, seed):
    rng = np.random.default_rng(seed)
    Xtr, ytr = DATASETS[ds_name](N_TRAIN, rng)
    Xte, yte = DATASETS[ds_name](N_TEST, rng)
    params = init_params(jax.random.PRNGKey(seed), kind)
    st = OPT.init(params)

    params, _ = _train_run(params, st, jnp.asarray(Xtr), jnp.asarray(ytr),
                           kind, L, STEPS)

    Xte_j = jnp.asarray(Xte)
    def acc_at(Ldepth):
        logits = _forward_jit(params, Xte_j, kind, Ldepth)
        return float((np.asarray(logits.argmax(1)) == yte).mean())

    acc = acc_at(L)
    # extrapolation: same fixed total time, more steps (finer integration)
    acc_deep = acc_at(L * 4)

    Es, rms = _trace_jit(params, jnp.asarray(Xte[:256]), kind, L)
    rms = np.asarray(rms)
    if kind == "ham":
        E = np.asarray(Es)
        e_drift = float(np.max(np.abs(E - E[0])) / (abs(E[0]) + 1e-6))
        e_trace = E.tolist()
    else:                                        # plain net has no learned energy
        e_drift, e_trace = None, []
    rms_ratio = float(rms[-1] / (rms[0] + 1e-9))
    out = dict(dataset=ds_name, kind=kind, L=L, seed=seed,
               acc=acc, acc_deep=acc_deep, energy_drift=e_drift,
               energy_trace=e_trace, rms_ratio=rms_ratio,
               rms_trace=rms.tolist())
    if seed == 0:   # export the trained net + data sample for live in-browser viz
        out["model"] = params_to_lists(params)
        # the datasets are generated class-0-then-class-1: shuffle before
        # slicing or the viz sample is single-class
        perm = np.random.default_rng(123).permutation(len(Xte))
        k = min(400, len(Xte))
        out["viz_X"] = Xte[perm[:k]].tolist()
        out["viz_y"] = yte[perm[:k]].tolist()
    return out


def run_networks():
    print("=" * 72)
    print("PART B: plain ResNet vs Hamiltonian (leapfrog) ResNet")
    rows = []
    for ds in DATASETS:
        for L in DEPTHS:
            for kind in ("plain", "ham"):
                accs, deeps, drifts, rms_ratios = [], [], [], []
                e_trace, r_trace, model, viz = None, None, None, None
                for s in SEEDS:
                    r = train_one(ds, kind, L, s)
                    accs.append(r["acc"]); deeps.append(r["acc_deep"])
                    if r["energy_drift"] is not None:
                        drifts.append(r["energy_drift"])
                    if r["rms_ratio"] is not None:
                        rms_ratios.append(r["rms_ratio"])
                    if s == 0:
                        e_trace, r_trace = r["energy_trace"], r["rms_trace"]
                        model = r.get("model")
                        viz = dict(X=r.get("viz_X"), y=r.get("viz_y"))
                edrift = float(np.mean(drifts)) if drifts else None
                rratio = float(np.mean(rms_ratios)) if rms_ratios else None
                rows.append(dict(
                    dataset=ds, kind=kind, L=L,
                    acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)),
                    acc_deep_mean=float(np.mean(deeps)),
                    energy_drift_mean=edrift, rms_ratio_mean=rratio,
                    energy_trace_seed0=e_trace, rms_trace_seed0=r_trace,
                    model_seed0=model, viz_data=viz,
                ))
                edstr = f"{edrift:.2%}" if edrift is not None else "  n/a"
                print(f"  {ds:<8} L={L:<4} {kind:<6} "
                      f"acc {np.mean(accs):.3f}+-{np.std(accs):.3f}, "
                      f"acc@4L {np.mean(deeps):.3f}, "
                      f"energy drift {edstr}, rms x{rratio:.2f}")
    return rows


def main():
    t0 = time.time()
    phys = run_physics()
    nets = run_networks()
    with open(os.path.join(OUT, "physics.json"), "w") as f:
        json.dump(phys, f)
    with open(os.path.join(OUT, "networks.json"), "w") as f:
        json.dump(nets, f)
    summary = dict(
        physics=dict(baseline_energy_drift=phys["baseline_energy_drift"],
                     hnn_energy_drift=phys["hnn_energy_drift"]),
        networks=[{k: r[k] for k in
                   ("dataset", "kind", "L", "acc_mean", "acc_std",
                    "acc_deep_mean", "energy_drift_mean", "rms_ratio_mean")}
                  for r in nets],
    )
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"done in {time.time()-t0:.0f}s -> {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
