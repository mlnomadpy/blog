"""D5: Depth on demand (adaptive-step integration of a trained flow).

D3 (hamiltonian_net.py) showed a leapfrog classifier treats depth as
resolution: fixed total time T, layers are steps, more layers just render the
same flow finer. This experiment takes the next integrator off the numerical
analysis shelf: ADAPTIVE step size. If depth is a rendering of a flow, the
renderer does not need a fixed step. It can take big steps where the flow is
gentle and small steps where it is sharp, controlled by a local error estimate
(step doubling: one h-step vs two h/2-steps; if they disagree beyond tol,
halve; if they agree comfortably, double).

The claim: computation time becomes an input-dependent, controllable quantity
AT INFERENCE, with no training changes. Trained once at fixed depth, the same
weights then run with an error controller, and:
  E1  fidelity: adaptive predictions agree with a fine fixed-grid reference,
      with agreement rising as tol tightens.
  E2  the effort histogram: steps-per-input varies across the dataset; easy
      inputs finish in a few steps, hard ones spend many.
  E3  effort correlates with difficulty: steps vs (distance to the decision
      boundary / final margin), Pearson r reported.
  E4  the accuracy-vs-mean-steps curve as tol sweeps: accuracy holds while the
      MEAN depth drops well below the fixed training depth.

Model: the D3 leapfrog net (hidden state (q, p), one shared block = one
leapfrog step of a learned potential V(q)), trained at L=16, T=6 per D3.
Adaptive rendering never changes the weights, only the step schedule.

Writes results/adaptive_depth.json. Run on Kaggle GPU (or --cpu, tiny nets).
"""

import json
import math
import os
import time

import numpy as np

import jax
import jax.numpy as jnp
import optax

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
SMOKE = os.environ.get("SMOKE", "0") == "1"

DIM = 4
HIDDEN = 32
T_TOTAL = 6.0
L_TRAIN = 16
SEEDS = (0,) if SMOKE else (0, 1, 2)
N_TRAIN, N_TEST = (256, 512) if SMOKE else (1024, 2048)
STEPS, LR = (400 if SMOKE else 5000), 3e-3
OPT = optax.adam(LR)
TOLS = (0.3, 0.1, 0.03, 0.01, 0.003) if not SMOKE else (0.1, 0.01)
L_REF = 256          # the fine fixed-grid reference rendering


# ── data (same generators as D3) ─────────────────────────────────────────────

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


# ── the D3 leapfrog net ──────────────────────────────────────────────────────

def dense(key, a, b):
    w = jax.random.normal(key, (a, b)) * math.sqrt(2.0 / a)
    return (w, jnp.zeros((b,)))


def init_params(key):
    ks = jax.random.split(key, 4)
    return dict(enc=dense(ks[0], 2, 2 * DIM), b1=dense(ks[1], DIM, HIDDEN),
                b2=dense(ks[2], HIDDEN, DIM), dec=dense(ks[3], 2 * DIM, 2))


def potential(p, q):
    h = jnp.tanh(q @ p["b1"][0] + p["b1"][1])
    return (h @ p["b2"][0] + p["b2"][1]).sum(axis=-1)


def grad_V(p, q):
    return jax.vmap(jax.grad(lambda qi: potential(p, qi[None])[0]))(q)


def leap(p, h, q, pm):
    pm = pm - 0.5 * h * grad_V(p, q)
    q = q + h * pm
    pm = pm - 0.5 * h * grad_V(p, q)
    return q, pm


def forward_fixed(p, X, L):
    h = T_TOTAL / L
    z = X @ p["enc"][0] + p["enc"][1]
    q, pm = z[:, :DIM], z[:, DIM:]

    def body(c, _):
        return leap(p, h, *c), None

    (q, pm), _ = jax.lax.scan(body, (q, pm), None, length=L)
    return jnp.concatenate([q, pm], 1) @ p["dec"][0] + p["dec"][1]


@jax.jit
def train_run(params, st, X, y):
    def loss_fn(pp):
        lg = forward_fixed(pp, X, L_TRAIN)
        return optax.softmax_cross_entropy_with_integer_labels(lg, y).mean()

    def upd(c, _):
        pp, s = c
        loss, g = jax.value_and_grad(loss_fn)(pp)
        u, s = OPT.update(g, s)
        return (optax.apply_updates(pp, u), s), loss

    (params, st), losses = jax.lax.scan(upd, (params, st), None, length=STEPS)
    return params, losses


# ── the adaptive renderer (step doubling, per input, numpy loop) ─────────────

def np_affine(x, wb):
    return x @ np.asarray(wb[0]) + np.asarray(wb[1])


def np_gradV(p, q):
    h = np.tanh(np_affine(q, p["b1"]))
    w1, w2 = np.asarray(p["b1"][0]), np.asarray(p["b2"][0])
    return ((1 - h * h) * w2.sum(axis=1)) @ w1.T


def np_leap(p, h, q, pm):
    pm = pm - 0.5 * h * np_gradV(p, q)
    q = q + h * pm
    pm = pm - 0.5 * h * np_gradV(p, q)
    return q, pm


def render_adaptive(p, x, tol, h0=T_TOTAL / L_TRAIN, h_min=T_TOTAL / 4096):
    """Integrate ONE input to time T with step doubling.
    Returns (logits, work, accepted): `work` counts every leapfrog evaluation
    including the error probes (the controller's true cost, 3 per accepted
    step), `accepted` counts committed steps (the rendered resolution)."""
    z = np_affine(x[None], p["enc"])
    q, pm = z[:, :DIM], z[:, DIM:]
    t, h, work, accepted = 0.0, h0, 0, 0
    while t < T_TOTAL - 1e-9:
        h = min(h, T_TOTAL - t)
        q1, p1 = np_leap(p, h, q, pm)                    # one h step
        qh, ph = np_leap(p, h / 2, q, pm)                # two h/2 steps
        q2, p2 = np_leap(p, h / 2, qh, ph)
        work += 3
        err = float(np.max(np.abs(np.concatenate([q1 - q2, p1 - p2], 1))))
        if err > tol and h > h_min:
            h = h / 2                                     # too sharp: refine, retry
            continue
        q, pm = q2, p2                                    # accept the finer render
        accepted += 2                                     # two committed h/2 steps
        t += h
        if err < tol / 4:
            h = min(h * 2, T_TOTAL / 4)                   # gentle: stride out
    lg = np_affine(np.concatenate([q, pm], 1), p["dec"])[0]
    return lg, work, accepted


def np_forward_fixed(p, X, L):
    h = T_TOTAL / L
    z = np_affine(X, p["enc"])
    q, pm = z[:, :DIM], z[:, DIM:]
    for _ in range(L):
        q, pm = np_leap(p, h, q, pm)
    return np_affine(np.concatenate([q, pm], 1), p["dec"])


def main():
    t0 = time.time()
    results = {}
    for ds_name, gen in DATASETS.items():
        per_seed = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            Xtr, ytr = gen(N_TRAIN, rng)
            Xte, yte = gen(N_TEST, rng)
            perm = np.random.default_rng(123).permutation(len(Xte))
            Xte, yte = Xte[perm], yte[perm]
            params = init_params(jax.random.PRNGKey(seed))
            params, _ = train_run(params, OPT.init(params),
                                  jnp.asarray(Xtr), jnp.asarray(ytr))
            params = jax.tree.map(np.asarray, params)

            # the fine fixed reference and the fixed-depth baseline
            ref = np_forward_fixed(params, Xte, L_REF).argmax(1)
            base = np_forward_fixed(params, Xte, L_TRAIN)
            acc_fixed = float((base.argmax(1) == yte).mean())
            margin = np.abs(base[:, 1] - base[:, 0])       # difficulty proxy

            n_eval = 300 if not SMOKE else 80
            sweep = []
            for tol in TOLS:
                preds, works, accs_n = [], [], []
                for i in range(n_eval):
                    lg, w, a = render_adaptive(params, Xte[i], tol)
                    preds.append(int(lg.argmax()))
                    works.append(w); accs_n.append(a)
                preds = np.array(preds); works = np.array(works); accepted = np.array(accs_n)
                agree = float((preds == ref[:n_eval]).mean())
                acc = float((preds == yte[:n_eval]).mean())
                r = float(np.corrcoef(works, -np.log(margin[:n_eval] + 1e-9))[0, 1])
                keep = seed == 0 and tol == TOLS[-3 if len(TOLS) > 2 else -1]
                sweep.append(dict(tol=tol, agree_ref=agree, acc=acc,
                                  work_mean=float(works.mean()),
                                  work_min=int(works.min()), work_max=int(works.max()),
                                  accepted_mean=float(accepted.mean()),
                                  corr_work_difficulty=r,
                                  work_hist=np.histogram(works, bins=12)[0].tolist(),
                                  works=works.tolist() if keep else None,
                                  accepted=accepted.tolist() if keep else None,
                                  margins=margin[:n_eval].tolist() if keep else None))
                print(f"  {ds_name} seed{seed} tol={tol:<6} agree {agree:.3f} "
                      f"acc {acc:.3f} (fixed {acc_fixed:.3f}) "
                      f"work {works.mean():.1f} [{works.min()},{works.max()}] "
                      f"accepted {accepted.mean():.1f} (ref {L_REF}) "
                      f"r(work,difficulty) {r:+.2f}")
            entry = dict(seed=seed, acc_fixed=acc_fixed, sweep=sweep)
            if seed == 0:
                # viz export: the trained weights + a balanced test sample, so the
                # in-page panels can run the same net and the same controller live
                entry["viz"] = dict(
                    model={k: [np.asarray(params[k][0]).tolist(),
                               np.asarray(params[k][1]).tolist()] for k in params},
                    X=Xte[:300].tolist(), y=yte[:300].tolist(),
                )
            per_seed.append(entry)
        results[ds_name] = per_seed
    with open(os.path.join(RESULTS_DIR, "adaptive_depth.json"), "w") as f:
        json.dump(results, f)
    print(f"done in {time.time()-t0:.0f}s -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
