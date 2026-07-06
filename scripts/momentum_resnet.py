"""momentum_resnet.py, the experiment behind "skip connections are half of Newton".

Part A (the physics fact, computed, not metaphor):
    integrate a real two-body central-force problem (Kepler, GM=1) with forward
    Euler and with leapfrog (kick-drift-kick) at the same step size; measure
    energy drift, apoapsis growth, and the divergence between the two orbits.

Part B (the network fact):
    one architecture with one knob. A residual block is
        v_{l+1} = mu * v_l + (1 - mu) * F_l(x_l)
        x_{l+1} = x_l + h * v_{l+1}
    with v_0 = 0. At mu = 0 this is EXACTLY the plain residual network
    x_{l+1} = x_l + h * F_l(x_l) (forward Euler of the field F); at mu > 0 it is
    a momentum residual network (Sander et al. 2021), the symplectic-Euler
    discretization of a damped second-order system. Parameter counts are matched
    by construction (mu is a fixed scalar hyperparameter, not trained).

    The residual stream is kept at width 2, so every hidden trajectory drawn in
    the post is the literal hidden state, not a projection. Blocks are per-layer
    (untied): F_l(x) = W2_l tanh(W1_l x + b1_l) + b2_l with hidden width 16.
    Total integration time is fixed at T = L * h = 8, so depth L in {8, 32, 128}
    is a refinement of the same flow (h = 1, 0.25, 0.0625).

    Tasks: two moons, two interleaved spirals, concentric rings (disk inside an
    annulus). The rings task is the topological probe: a first-order planar flow
    is a homeomorphism and cannot pull an enclosed class out of the class that
    surrounds it, while a momentum net's doubled (x, v) state can.

    Measured: test accuracy vs depth (3 seeds), hidden-trajectory statistics
    (path length, norm drift, turning angle), per-block gradient norms at init
    and after training, kinetic energy through depth for the momentum net, and
    the exact-inversion error of the momentum net (run the dynamics backward).

Exports JSON to public/momentum-resnet/ and prints every number used in the post.
Run:  python3 scripts/momentum_resnet.py          (CPU, a few minutes)
"""

import json
import math
import os
import time

import numpy as np

import jax
import jax.numpy as jnp
import optax

OUT = os.path.join(os.path.dirname(__file__), "..", "public", "momentum-resnet")
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Part A: the physics fact. Kepler two-body, forward Euler vs leapfrog.
# ─────────────────────────────────────────────────────────────────────────────

R0 = np.array([1.0, 0.0])
V0 = np.array([0.0, 0.9])          # elliptical orbit, e = 0.19
E0 = 0.5 * (0.9 ** 2) - 1.0        # = -0.595
A_SEMI = -1.0 / (2.0 * E0)         # semi-major axis 0.8403
T_ORBIT = 2.0 * math.pi * A_SEMI ** 1.5  # 4.8399


def accel(r):
    return -r / (np.linalg.norm(r) ** 3)


def integrate(dt, n_orbits, method):
    """Returns (r_hist, energy_hist) sampled every `sub` steps."""
    n_steps = int(round(n_orbits * T_ORBIT / dt))
    sub = max(1, n_steps // 4000)
    r, v = R0.copy(), V0.copy()
    rs, es = [r.copy()], [E0]
    if method == "leapfrog":
        a = accel(r)
        for i in range(n_steps):
            v = v + 0.5 * dt * a          # kick
            r = r + dt * v                # drift
            a = accel(r)
            v = v + 0.5 * dt * a          # kick
            if (i + 1) % sub == 0:
                rs.append(r.copy())
                es.append(0.5 * v @ v - 1.0 / np.linalg.norm(r))
    else:  # forward Euler: both updates from the OLD state
        for i in range(n_steps):
            a = accel(r)
            r = r + dt * v
            v = v + dt * a
            if (i + 1) % sub == 0:
                rs.append(r.copy())
                es.append(0.5 * v @ v - 1.0 / np.linalg.norm(r))
    return np.array(rs), np.array(es)


def run_physics():
    print("=" * 72)
    print("PART A: Kepler two-body, forward Euler vs leapfrog (GM = 1)")
    print(f"  r0 = {R0.tolist()}, v0 = {V0.tolist()}, E0 = {E0:.4f}, "
          f"period = {T_ORBIT:.4f}")
    phys = {"r0": R0.tolist(), "v0": V0.tolist(), "E0": E0,
            "period": T_ORBIT, "sweeps": []}
    N_ORB = 20
    for dt in (0.05, 0.02, 0.01, 0.005):
        r_e, e_e = integrate(dt, N_ORB, "euler")
        r_l, e_l = integrate(dt, N_ORB, "leapfrog")
        drift_e = (e_e[-1] - E0) / abs(E0)
        drift_l_max = np.max(np.abs(e_l - E0)) / abs(E0)
        apo_e = np.max(np.linalg.norm(r_e, axis=1))
        apo_l = np.max(np.linalg.norm(r_l, axis=1))
        apo_true = A_SEMI * (1 + math.sqrt(1 + 2 * E0 * (0.9 ** 2)))  # a(1+e)
        div = np.linalg.norm(r_e[-1] - r_l[-1])
        row = dict(dt=dt, n_orbits=N_ORB,
                   euler_energy_drift=float(drift_e),
                   leapfrog_energy_max_dev=float(drift_l_max),
                   euler_apoapsis=float(apo_e), leapfrog_apoapsis=float(apo_l),
                   true_apoapsis=float(apo_true),
                   euler_vs_leapfrog_final_sep=float(div))
        phys["sweeps"].append(row)
        print(f"  dt={dt:<6} after {N_ORB} orbits: "
              f"Euler dE/|E0| = {drift_e:+.3%}, "
              f"leapfrog max|dE|/|E0| = {drift_l_max:.2e}, "
              f"apoapsis {apo_true:.3f} -> Euler {apo_e:.3f} / leap {apo_l:.3f}, "
              f"final separation {div:.3f}")
    return phys


# ─────────────────────────────────────────────────────────────────────────────
# Part B: the network fact.
# ─────────────────────────────────────────────────────────────────────────────

WIDTH = 2       # residual-stream width: the hidden state IS 2-D (no projection)
HIDDEN = 16     # hidden width inside each block F_l
T_TOTAL = 8.0   # fixed total integration time, h = T / L
DEPTHS = (8, 32, 128)
MUS_MAIN = (0.0, 0.9)
SEEDS = (0, 1, 2)
N_TRAIN, N_TEST = 1024, 2048
STEPS, LR = 4000, 3e-3


def make_moons(n, rng, noise=0.08):
    n2 = n // 2
    t = rng.uniform(0, math.pi, n2)
    a = np.stack([np.cos(t), np.sin(t)], 1)
    b = np.stack([1 - np.cos(t), 0.5 - np.sin(t)], 1)
    X = np.concatenate([a, b]) + rng.normal(0, noise, (n2 * 2, 2))
    y = np.concatenate([np.zeros(n2), np.ones(n2)]).astype(np.int32)
    X = (X - X.mean(0)) / X.std(0).mean()
    return X.astype(np.float32), y


def make_spirals(n, rng, noise=0.05, turns=1.75):
    n2 = n // 2
    t = np.sqrt(rng.uniform(0.05, 1.0, n2)) * turns * 2 * math.pi
    r = 0.15 + 0.85 * t / (turns * 2 * math.pi)
    a = np.stack([r * np.cos(t), r * np.sin(t)], 1)
    b = np.stack([r * np.cos(t + math.pi), r * np.sin(t + math.pi)], 1)
    X = np.concatenate([a, b]) + rng.normal(0, noise, (n2 * 2, 2))
    y = np.concatenate([np.zeros(n2), np.ones(n2)]).astype(np.int32)
    X = X / X.std(0).mean() * 0.9
    return X.astype(np.float32), y


def make_rings(n, rng, noise=0.04):
    """Class 0: a disk. Class 1: an annulus that fully encircles it."""
    n2 = n // 2
    th0 = rng.uniform(0, 2 * math.pi, n2)
    r_in = 0.6 * np.sqrt(rng.uniform(0, 1, n2))
    th1 = rng.uniform(0, 2 * math.pi, n2)
    r_out = rng.uniform(1.3, 1.8, n2)
    a = np.stack([r_in * np.cos(th0), r_in * np.sin(th0)], 1)
    b = np.stack([r_out * np.cos(th1), r_out * np.sin(th1)], 1)
    X = np.concatenate([a, b]) + rng.normal(0, noise, (n2 * 2, 2))
    y = np.concatenate([np.zeros(n2), np.ones(n2)]).astype(np.int32)
    return X.astype(np.float32), y


DATASETS = {"moons": make_moons, "spirals": make_spirals, "rings": make_rings}


def init_params(key, L):
    k1, k2, k3 = jax.random.split(key, 3)
    s1 = 1.0 / math.sqrt(WIDTH)
    s2 = 1.0 / math.sqrt(HIDDEN)
    return {
        "W1": jax.random.normal(k1, (L, WIDTH, HIDDEN)) * s1,
        "b1": jnp.zeros((L, HIDDEN)),
        "W2": jax.random.normal(k2, (L, HIDDEN, WIDTH)) * s2,
        "b2": jnp.zeros((L, WIDTH)),
        "Wr": jax.random.normal(k3, (WIDTH, 2)) * s1,
        "br": jnp.zeros((2,)),
    }


def forward(params, X, mu, h, collect=False):
    """X: [n, 2]. Returns logits [n, 2] (and trajectories if collect)."""
    v0 = jnp.zeros_like(X)

    def block(carry, p):
        x, v = carry
        f = jnp.tanh(x @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]
        v = mu * v + (1.0 - mu) * f
        x = x + h * v
        return (x, v), ((x, v) if collect else None)

    blocks = {k: params[k] for k in ("W1", "b1", "W2", "b2")}
    (xL, vL), traj = jax.lax.scan(block, (X, v0), blocks)
    logits = xL @ params["Wr"] + params["br"]
    if collect:
        xs = jnp.concatenate([X[None], traj[0]], 0)   # [L+1, n, 2]
        vs = jnp.concatenate([v0[None], traj[1]], 0)  # [L+1, n, 2]
        return logits, xs, vs
    return logits


def loss_fn(params, X, y, mu, h):
    logits = forward(params, X, mu, h)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


def per_block_grad_norms(params, X, y, mu, h):
    g = jax.grad(loss_fn)(params, X, y, mu, h)
    sq = sum(jnp.sum(g[k] ** 2, axis=tuple(range(1, g[k].ndim)))
             for k in ("W1", "b1", "W2", "b2"))
    return jnp.sqrt(sq)  # [L]


def train_one(ds_name, L, mu, seed, collect_final=False):
    rng = np.random.default_rng(1000 + seed)
    Xtr, ytr = DATASETS[ds_name](N_TRAIN, rng)
    Xte, yte = DATASETS[ds_name](N_TEST, rng)
    h = T_TOTAL / L
    params = init_params(jax.random.PRNGKey(seed), L)

    gn_init = np.asarray(per_block_grad_norms(params, jnp.array(Xtr),
                                              jnp.array(ytr), mu, h))

    opt = optax.adam(LR)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state):
        l, g = jax.value_and_grad(loss_fn)(params, jnp.array(Xtr),
                                           jnp.array(ytr), mu, h)
        updates, opt_state = opt.update(g, opt_state)
        return optax.apply_updates(params, updates), opt_state, l

    losses = []
    for i in range(STEPS):
        params, opt_state, l = step(params, opt_state)
        if i % 200 == 0 or i == STEPS - 1:
            losses.append(float(l))

    logits, xs, vs = forward(params, jnp.array(Xte), mu, h, collect=True)
    acc = float((jnp.argmax(logits, 1) == jnp.array(yte)).mean())

    # trajectory statistics on the test set (xs: [L+1, n, 2])
    xs_np = np.asarray(xs)
    seg = np.diff(xs_np, axis=0)                      # [L, n, 2]
    seglen = np.linalg.norm(seg, axis=2)              # [L, n]
    path_len = float(seglen.sum(0).mean())
    norms = np.linalg.norm(xs_np, axis=2)             # [L+1, n]
    norm_max = float(norms.max(0).mean())
    norm_end = float(norms[-1].mean())
    # mean turning angle between consecutive segments (smoothness)
    d1, d2 = seg[:-1], seg[1:]
    dot = (d1 * d2).sum(2)
    nn_ = np.linalg.norm(d1, axis=2) * np.linalg.norm(d2, axis=2) + 1e-12
    turn = float(np.degrees(np.arccos(np.clip(dot / nn_, -1, 1))).mean())
    kinetic = (0.5 * (np.asarray(vs) ** 2).sum(2).mean(1)).tolist()  # [L+1]

    gn_tr = np.asarray(per_block_grad_norms(params, jnp.array(Xtr),
                                            jnp.array(ytr), mu, h))

    out = dict(acc=acc, loss_final=losses[-1], path_len=path_len,
               norm_max=norm_max, norm_end=norm_end, turn_deg=turn,
               gn_init=gn_init.tolist(), gn_trained=gn_tr.tolist(),
               kinetic=kinetic, losses=losses, h=h)
    if collect_final:
        out["_params"] = params
        out["_test"] = (Xte, yte)
        out["_traj"] = xs_np
        out["_logits"] = np.asarray(logits)
    return out


def invert_momentum(params, xL, vL, mu, h, L):
    """Run the momentum dynamics exactly backward (float64)."""
    p64 = {k: np.asarray(v, dtype=np.float64) for k, v in params.items()}
    x = np.asarray(xL, dtype=np.float64)
    v = np.asarray(vL, dtype=np.float64)
    for l in range(L - 1, -1, -1):
        x_prev = x - h * v
        f = np.tanh(x_prev @ p64["W1"][l] + p64["b1"][l]) @ p64["W2"][l] + p64["b2"][l]
        v = (v - (1.0 - mu) * f) / mu
        x = x_prev
    return x, v


def run_networks():
    print("=" * 72)
    print(f"PART B: residual nets, stream width {WIDTH}, per-layer blocks "
          f"(hidden {HIDDEN}), T = L*h = {T_TOTAL}, {STEPS} Adam steps @ lr {LR}")
    results = {}
    t0 = time.time()
    for ds in DATASETS:
        for L in DEPTHS:
            for mu in MUS_MAIN:
                accs, rows = [], []
                for seed in SEEDS:
                    r = train_one(ds, L, mu, seed)
                    accs.append(r["acc"])
                    rows.append(r)
                key = f"{ds}/L{L}/mu{mu}"
                best = int(np.argmax(accs))
                results[key] = dict(
                    acc_mean=float(np.mean(accs)), acc_min=float(np.min(accs)),
                    acc_max=float(np.max(accs)), accs=[float(a) for a in accs],
                    path_len=rows[best]["path_len"], turn_deg=rows[best]["turn_deg"],
                    norm_max=rows[best]["norm_max"], norm_end=rows[best]["norm_end"],
                    loss_final=rows[best]["loss_final"], h=rows[best]["h"],
                    gn_init=rows[best]["gn_init"], gn_trained=rows[best]["gn_trained"],
                    kinetic=rows[best]["kinetic"])
                print(f"  {key:<22} acc {np.mean(accs)*100:5.1f}% "
                      f"(min {np.min(accs)*100:5.1f}, max {np.max(accs)*100:5.1f}) "
                      f"path {rows[best]['path_len']:5.2f} "
                      f"turn {rows[best]['turn_deg']:5.1f} deg "
                      f"[{time.time()-t0:5.0f}s]")
    return results


def run_exports(results):
    """Best-seed detailed runs for the viz exports."""
    print("=" * 72)
    print("EXPORTS")

    # 1) trajectories: rings and spirals at L=32, mu 0 vs 0.9, best seed each
    traj_out = {}
    for ds in ("rings", "spirals"):
        for mu in MUS_MAIN:
            key = f"{ds}/L32/mu{mu}"
            best_seed = SEEDS[int(np.argmax(results[key]["accs"]))]
            r = train_one(ds, 32, mu, best_seed, collect_final=True)
            Xte, yte = r["_test"]
            # a balanced, spatially spread subsample of 32 test points
            idx = []
            for c in (0, 1):
                cand = np.where(yte == c)[0]
                order = np.argsort(np.arctan2(Xte[cand, 1], Xte[cand, 0]))
                idx.extend(cand[order[:: max(1, len(cand) // 16)]][:16])
            idx = np.array(idx)
            traj_out[f"{ds}_mu{mu}"] = dict(
                mu=mu, L=32, h=r["h"], acc=r["acc"],
                y=yte[idx].tolist(),
                pred=np.argmax(r["_logits"][idx], 1).tolist(),
                traj=np.round(r["_traj"][:, idx, :].transpose(1, 0, 2), 4).tolist(),
                path_len=r["path_len"], turn_deg=r["turn_deg"])
            print(f"  traj {ds} mu={mu}: seed {best_seed}, acc {r['acc']*100:.1f}%, "
                  f"32 real trajectories of {33} states")
    with open(os.path.join(OUT, "trajectories.json"), "w") as f:
        json.dump(traj_out, f)

    # 2) inertia dial: rings, L=32, nets trained at mu in {0, .3, .6, .9},
    #    weights exported so the browser can run the real forward pass live
    inertia = {"L": 32, "h": T_TOTAL / 32, "width": WIDTH, "hidden": HIDDEN,
               "nets": {}}
    rng = np.random.default_rng(7)
    Xd, yd = make_rings(256, rng)
    inertia["test_X"] = np.round(Xd, 4).tolist()
    inertia["test_y"] = yd.tolist()
    inv_report = {}
    for mu in (0.0, 0.3, 0.6, 0.9):
        # reuse main-seed sweep when available, else train fresh (single seed)
        key = f"rings/L32/mu{mu}"
        if key in results:
            best_seed = SEEDS[int(np.argmax(results[key]["accs"]))]
        else:
            best_seed = 0
        r = train_one("rings", 32, mu, best_seed, collect_final=True)
        p = r["_params"]
        inertia["nets"][str(mu)] = dict(
            mu=mu, acc=r["acc"],
            W1=np.round(np.asarray(p["W1"]), 5).tolist(),
            b1=np.round(np.asarray(p["b1"]), 5).tolist(),
            W2=np.round(np.asarray(p["W2"]), 5).tolist(),
            b2=np.round(np.asarray(p["b2"]), 5).tolist(),
            Wr=np.round(np.asarray(p["Wr"]), 5).tolist(),
            br=np.round(np.asarray(p["br"]), 5).tolist())
        print(f"  inertia net mu={mu}: acc {r['acc']*100:.1f}%")

        # 3) exact inversion (the time-reversibility payoff), mu > 0 only
        if mu > 0:
            X8 = jnp.array(Xd[:64])
            _, xs, vs = forward(p, X8, mu, T_TOTAL / 32, collect=True)
            x0r, _ = invert_momentum(p, np.asarray(xs[-1]), np.asarray(vs[-1]),
                                     mu, T_TOTAL / 32, 32)
            err = float(np.abs(x0r - Xd[:64]).max())
            inv_report[str(mu)] = err
            print(f"    inversion: max|x0_reconstructed - x0| = {err:.2e} "
                  f"(64 points, dynamics run exactly backward)")
    with open(os.path.join(OUT, "inertia.json"), "w") as f:
        json.dump(inertia, f)
    return inv_report


def main():
    t0 = time.time()
    phys = run_physics()
    results = run_networks()
    inv = run_exports(results)

    # 4) cliff.json: accuracy vs depth for every dataset and both mus,
    #    plus the per-block gradient-norm profiles at L=128
    cliff = {"depths": list(DEPTHS), "datasets": {}}
    for ds in DATASETS:
        cliff["datasets"][ds] = {}
        for mu in MUS_MAIN:
            cliff["datasets"][ds][str(mu)] = dict(
                acc_mean=[results[f"{ds}/L{L}/mu{mu}"]["acc_mean"] for L in DEPTHS],
                acc_min=[results[f"{ds}/L{L}/mu{mu}"]["acc_min"] for L in DEPTHS],
                acc_max=[results[f"{ds}/L{L}/mu{mu}"]["acc_max"] for L in DEPTHS])
    cliff["gradnorms_L128"] = {
        ds: {str(mu): dict(init=results[f"{ds}/L128/mu{mu}"]["gn_init"],
                           trained=results[f"{ds}/L128/mu{mu}"]["gn_trained"])
             for mu in MUS_MAIN}
        for ds in DATASETS}
    with open(os.path.join(OUT, "cliff.json"), "w") as f:
        json.dump(cliff, f)

    summary = {"physics": phys,
               "network_runs": results,
               "inversion_max_err": inv,
               "config": dict(width=WIDTH, hidden=HIDDEN, T=T_TOTAL,
                              depths=list(DEPTHS), mus=list(MUS_MAIN),
                              seeds=list(SEEDS), n_train=N_TRAIN,
                              n_test=N_TEST, steps=STEPS, lr=LR)}
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("=" * 72)
    print(f"done in {time.time()-t0:.0f}s -> {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
