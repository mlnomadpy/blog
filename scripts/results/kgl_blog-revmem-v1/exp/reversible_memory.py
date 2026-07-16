"""D4: Backprop without the memory (reversible nets, O(1)-memory training).

D1 (momentum_resnet.py) proved the momentum residual step is algebraically
invertible: given the output state you can reconstruct every intermediate state
by running the ledger backwards, with float error amplified by (1/mu)^L. This
experiment makes that invertibility load-bearing: a custom-VJP training step
that stores NO intermediate activations. The backward pass reconstructs each
layer's input by inverting the forward step, then computes that layer's
gradient locally. Peak activation memory is O(1) in depth instead of O(L).

The momentum step (velocity ledger, mu in (0, 1)):
    v_{l+1} = mu * v_l + f_theta(x_l)
    x_{l+1} = x_l + h * v_{l+1}
Inverse, given (x_{l+1}, v_{l+1}):
    x_l = x_{l+1} - h * v_{l+1}
    v_l = (v_{l+1} - f_theta(x_l)) / mu

Three measurements, all real:
  E1  memory: peak device-memory of one training step vs depth L, standard
      backprop (stores L activations) vs reversible (stores 1), at a width
      where the difference is visible. Read from the JAX device allocator.
  E2  gradient fidelity: cosine similarity + relative error between reversible
      gradients and standard gradients, vs depth and vs mu. The (1/mu)^L noise
      budget from D1, now priced where it matters.
  E3  training: the reversible step trains end to end (FMNIST subset,
      pixel-flat) to the same accuracy as standard backprop at matched config.

Writes results/reversible_memory.json (+ .npz curves).
Run on Kaggle GPU (P100): memory numbers need the real device allocator.
"""

import json
import os
import time

import numpy as np

import jax
import jax.numpy as jnp
import optax

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
SMOKE = os.environ.get("SMOKE", "0") == "1"

RESULTS, ARRAYS = {}, {}

WIDTH = 256          # residual stream width (state x and velocity v)
HIDDEN = 512         # block hidden width
H_STEP = 0.5         # integration step in the ledger
MU = 0.9
BATCH = 256
LR = 1e-3
DEPTHS_MEM = (8, 32, 128, 512) if not SMOKE else (8, 32)
DEPTHS_FID = (8, 32, 128) if not SMOKE else (8,)
MUS_FID = (0.9, 0.6, 0.3)
TRAIN_STEPS = 1500 if not SMOKE else 60
TRAIN_L = 64 if not SMOKE else 8


# ── the block and the two training steps ─────────────────────────────────────

def init_params(key, width=WIDTH, hidden=HIDDEN):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    s1, s2 = 1.0 / np.sqrt(width), 1.0 / np.sqrt(hidden)
    return dict(
        w1=jax.random.normal(k1, (width, hidden)) * s1, b1=jnp.zeros((hidden,)),
        w2=jax.random.normal(k2, (hidden, width)) * s2, b2=jnp.zeros((width,)),
        enc=jax.random.normal(k3, (784, width)) * (1.0 / 28.0),
        dec=jax.random.normal(k4, (width, 10)) * s1,
    )


def block(p, x):
    return jnp.tanh(x @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"]


def fwd_scan(p, x0, v0, L):
    """Standard forward: autodiff will store per-layer residuals (O(L) memory)."""
    def step(carry, _):
        x, v = carry
        v = MU * v + block(p, x)
        x = x + H_STEP * v
        return (x, v), None
    (x, v), _ = jax.lax.scan(step, (x0, v0), None, length=L)
    return x, v


# reversible: forward stores nothing; backward reconstructs by inversion
@jax.custom_vjp
def rev_forward(p, x0, v0, L_arr):
    L = L_arr.shape[0]
    return fwd_scan(p, x0, v0, L)


def rev_fwd(p, x0, v0, L_arr):
    out = rev_forward(p, x0, v0, L_arr)
    # residuals: ONLY the output state (constant in depth), never the trajectory
    return out, (p, out[0], out[1], L_arr)


def rev_bwd(res, g):
    p, xL, vL, L_arr = res
    gx, gv = g
    L = L_arr.shape[0]

    def back(carry, _):
        x_next, v_next, gx_n, gv_n, gp = carry
        # invert one layer to recover its input
        x_prev = x_next - H_STEP * v_next
        f_val, f_vjp = jax.vjp(lambda pp, xx: block(pp, xx), p, x_prev)
        v_prev = (v_next - f_val) / MU
        # backprop the same layer given recovered inputs:
        #   x_next = x_prev + H*(mu*v_prev + f(x_prev))
        #   v_next = mu*v_prev + f(x_prev)
        gv_local = gv_n + H_STEP * gx_n          # d/dv_next of both outputs
        gp_f, gx_f = f_vjp(gv_local)
        gx_p = gx_n + gx_f
        gv_p = MU * gv_local
        gp = jax.tree.map(jnp.add, gp, gp_f)
        return (x_prev, v_prev, gx_p, gv_p, gp), None

    gp0 = jax.tree.map(jnp.zeros_like, p)
    (x0r, v0r, gx0, gv0, gp), _ = jax.lax.scan(
        back, (xL, vL, gx, gv, gp0), None, length=L)
    return gp, gx0, gv0, None


rev_forward.defvjp(rev_fwd, rev_bwd)


def loss_std(p, X, y, L):
    x0 = X @ p["enc"]
    v0 = jnp.zeros_like(x0)
    xL, _ = fwd_scan(p, x0, v0, L)
    logits = xL @ p["dec"]
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


def loss_rev(p, X, y, L_arr):
    x0 = X @ p["enc"]
    v0 = jnp.zeros_like(x0)
    xL, _ = rev_forward(p, x0, v0, L_arr)
    logits = xL @ p["dec"]
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


# ── E1: peak memory of one training step vs depth ────────────────────────────

def peak_mem_bytes():
    stats = jax.local_devices()[0].memory_stats()
    return stats.get("peak_bytes_in_use", 0) if stats else 0


def measure_one(L, mode):
    """One (L, mode) peak-memory point. Run in a FRESH process: the allocator's
    peak_bytes_in_use is a process-lifetime high-water mark, so points must not
    share a process or later readings inherit earlier peaks. Reports both the
    runtime allocator peak (BFC, prealloc off) and XLA's static memory analysis
    of the compiled step (temp bytes = the activation storage)."""
    rngX = np.random.default_rng(0)
    X = jnp.asarray(rngX.normal(size=(BATCH, 784)).astype(np.float32))
    y = jnp.asarray(rngX.integers(0, 10, BATCH).astype(np.int32))
    p = init_params(jax.random.PRNGKey(0))
    L_arr = jnp.zeros((L,))
    if mode == "standard":
        g = jax.jit(jax.grad(lambda pp: loss_std(pp, X, y, L)))
    else:
        g = jax.jit(jax.grad(lambda pp: loss_rev(pp, X, y, L_arr)))
    compiled = g.lower(p).compile()
    ma = compiled.memory_analysis()
    temp_mb = getattr(ma, "temp_size_in_bytes", 0) / 1e6 if ma else 0.0
    out = g(p); jax.block_until_ready(out)
    t0 = time.time()
    out = g(p); jax.block_until_ready(out)
    dt = time.time() - t0
    print(json.dumps(dict(L=L, mode=mode, peak_mb=peak_mem_bytes() / 1e6,
                          temp_mb=temp_mb, step_s=dt)))


def run_memory():
    print("=" * 72)
    print("E1: peak device memory of one gradient step vs depth (fresh process per point)")
    import subprocess
    import sys
    rows = []
    for L in DEPTHS_MEM:
        for mode in ("standard", "reversible"):
            env = dict(os.environ, MEASURE=f"{L},{mode}",
                       XLA_PYTHON_CLIENT_PREALLOCATE="false")   # BFC, no 90% grab
            r = subprocess.run([sys.executable, os.path.abspath(__file__)],
                               capture_output=True, text=True, env=env)
            line = [ln for ln in r.stdout.strip().splitlines() if ln.startswith("{")]
            if not line:
                print(f"  L={L} {mode}: FAILED\n{r.stdout[-500:]}\n{r.stderr[-500:]}")
                continue
            row = json.loads(line[-1])
            rows.append(row)
            print(f"  L={row['L']:<5} {row['mode']:<11} peak {row['peak_mb']:8.1f} MB"
                  f"   xla temp {row['temp_mb']:8.1f} MB   step {row['step_s']*1e3:7.1f} ms")
    RESULTS["memory"] = rows


# ── E2: gradient fidelity vs depth and mu ────────────────────────────────────

def run_fidelity(X, y):
    print("=" * 72)
    print("E2: reversible-vs-standard gradient agreement ((1/mu)^L is the budget)")
    global MU
    rows = []
    mu0 = MU
    for mu in MUS_FID:
        MU = mu
        for L in DEPTHS_FID:
            p = init_params(jax.random.PRNGKey(1))
            L_arr = jnp.zeros((L,))
            g_std = jax.jit(jax.grad(lambda pp: loss_std(pp, X, y, L)))(p)
            g_rev = jax.jit(jax.grad(lambda pp: loss_rev(pp, X, y, L_arr)))(p)
            flat_s = jnp.concatenate([v.ravel() for v in jax.tree.leaves(g_std)])
            flat_r = jnp.concatenate([v.ravel() for v in jax.tree.leaves(g_rev)])
            cos = float(jnp.dot(flat_s, flat_r) /
                        (jnp.linalg.norm(flat_s) * jnp.linalg.norm(flat_r) + 1e-30))
            rel = float(jnp.linalg.norm(flat_r - flat_s) /
                        (jnp.linalg.norm(flat_s) + 1e-30))
            rows.append(dict(mu=mu, L=L, cosine=cos, rel_err=rel,
                             budget=float((1.0 / mu) ** L)))
            print(f"  mu={mu} L={L:<5} cos {cos:.6f}  rel {rel:.2e}  (1/mu)^L {(1/mu)**L:.2e}")
    MU = mu0
    RESULTS["fidelity"] = rows


# ── E3: it trains. Same config, both backprops, real data ────────────────────

def load_fmnist(n_train=20000, n_test=4000):
    """FMNIST via keras npz mirror baked into Kaggle images, else sklearn fetch."""
    try:
        from tensorflow.keras.datasets import fashion_mnist
        (Xtr, ytr), (Xte, yte) = fashion_mnist.load_data()
    except Exception:
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml("Fashion-MNIST", version=1, return_X_y=True, as_frame=False)
        X = X.reshape(-1, 28, 28).astype(np.uint8); y = y.astype(int)
        Xtr, ytr, Xte, yte = X[:60000], y[:60000], X[60000:], y[60000:]
    rng = np.random.default_rng(0)
    i = rng.permutation(len(Xtr))[:n_train]
    j = rng.permutation(len(Xte))[:n_test]
    f = lambda A: (A.reshape(len(A), -1).astype(np.float32) / 255.0 - 0.2860) / 0.3530
    return f(Xtr[i]), ytr[i].astype(np.int32), f(Xte[j]), yte[j].astype(np.int32)


def run_training():
    print("=" * 72)
    print(f"E3: training at L={TRAIN_L}, standard vs reversible backprop")
    Xtr, ytr, Xte, yte = load_fmnist(4000 if SMOKE else 20000, 1000 if SMOKE else 4000)
    Xtr, ytr = jnp.asarray(Xtr), jnp.asarray(ytr)
    n = len(Xtr)
    L_arr = jnp.zeros((TRAIN_L,))
    opt = optax.adam(LR)
    curves = {}
    for mode in ("standard", "reversible"):
        p = init_params(jax.random.PRNGKey(0))
        st = opt.init(p)
        if mode == "standard":
            vg = jax.jit(jax.value_and_grad(lambda pp, XB, yB: loss_std(pp, XB, yB, TRAIN_L)))
        else:
            vg = jax.jit(jax.value_and_grad(lambda pp, XB, yB: loss_rev(pp, XB, yB, L_arr)))

        @jax.jit
        def upd(p, st, XB, yB):
            loss, g = vg(p, XB, yB)
            u, st = opt.update(g, st)
            return optax.apply_updates(p, u), st, loss

        losses = []
        rng = np.random.default_rng(1)
        t0 = time.time()
        for s in range(TRAIN_STEPS):
            idx = rng.integers(0, n, BATCH)
            p, st, loss = upd(p, st, Xtr[idx], ytr[idx])
            if s % 50 == 0:
                losses.append(float(loss))
        dt = time.time() - t0

        def acc(X, y):
            x0 = jnp.asarray(X) @ p["enc"]
            xL, _ = fwd_scan(p, x0, jnp.zeros_like(x0), TRAIN_L)
            return float((np.asarray((xL @ p["dec"]).argmax(1)) == y).mean())

        a = acc(Xte, yte)
        curves[mode] = dict(losses=losses, test_acc=a, wall_s=dt)
        print(f"  {mode:<11} test acc {a:.4f}   wall {dt:.0f}s")
    RESULTS["training"] = curves


def main():
    t0 = time.time()
    print(f"device: {jax.devices()}")
    rngX = np.random.default_rng(0)
    X = jnp.asarray(rngX.normal(size=(BATCH, 784)).astype(np.float32))
    y = jnp.asarray(rngX.integers(0, 10, BATCH).astype(np.int32))
    run_memory()
    run_fidelity(X, y)
    run_training()
    with open(os.path.join(RESULTS_DIR, "reversible_memory.json"), "w") as f:
        json.dump(RESULTS, f, indent=1)
    print(f"done in {time.time()-t0:.0f}s -> {RESULTS_DIR}")


if __name__ == "__main__":
    if os.environ.get("MEASURE"):
        L_s, mode_s = os.environ["MEASURE"].split(",")
        measure_one(int(L_s), mode_s)
    else:
        main()
