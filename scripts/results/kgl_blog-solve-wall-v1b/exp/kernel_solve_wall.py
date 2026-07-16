"""You don't have to solve a kernel machine: the solve-vs-descend battery.

Four experiments, one Mercer kernel (the Yat kernel
k(x, x') = (x.x')^2 / (||x - x'||^2 + eps)):

E1 agreement   California Housing. Exact Yat-kernel ridge solve vs an SGD-trained
               K-prototype Yat net vs Gaussian RFF ridge at matched parameter
               budget. Prediction agreement (Pearson r), test RMSE, K ladder,
               per-epoch agreement telemetry for the convergence GIF.
E2 wall        Measured Gram-build + Cholesky-solve wall clock and memory as n
               grows (CPU float64 and GPU float32), against one SGD epoch and
               SGD time-to-parity at the same n.
E3 scale       Covertype (581k x 54, class 2 vs rest). Exact solve on the biggest
               subsample the machine can hold vs the SGD net minibatching through
               the full training set. Accuracy vs n-used.
E4 compose     Fashion-MNIST. Exact solve on raw pixels (8k subsample) vs frozen
               random conv trunk + solve vs conv trunk + Yat head trained
               end-to-end by SGD, at matched 8k and at the full 60k.

LR fairness: every gradient-trained model picks its own learning rate by inner
validation. Every SGD result is best-val-epoch, not last-epoch. Headline numbers
are means over SEEDS seeds.

Env:
  MODE=smoke|full   (smoke: tiny subsets, 1 seed, plumbing check)
  SEEDS=3
Outputs: results/kernel_solve_wall.json (scalars), results/kernel_solve_wall.npz
(per-epoch series, prediction snapshots, feature-map stacks for GIFs).
"""

import json
import os
import time

import numpy as np

MODE = os.environ.get("MODE", "full")
SMOKE = MODE == "smoke"
SEEDS = int(os.environ.get("SEEDS", "1" if SMOKE else "3"))
EPS = 1e-2  # Yat kernel softening, matches the series convention
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

import jax
import jax.numpy as jnp
import optax
from flax import nnx

print(f"devices: {jax.devices()}  mode={MODE} seeds={SEEDS}", flush=True)

try:
    import wandb

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "neoyat-goatv"),
        name=f"blog-solve-wall-{MODE}",
        config={"mode": MODE, "seeds": SEEDS},
    )
    WB = True
except Exception as e:  # noqa: BLE001
    print(f"wandb off: {e}", flush=True)
    WB = False

RESULTS: dict = {"mode": MODE, "seeds": SEEDS, "eps": EPS}
ARRAYS: dict = {}


# ---------------------------------------------------------------- kernel math
def yat_gram(A, B, eps=EPS):
    """k(a,b) = (a.b)^2 / (||a-b||^2 + eps), float64 on CPU by default."""
    dot = A @ B.T
    sq = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * dot
    return (dot * dot) / (np.maximum(sq, 0.0) + eps)


def solve_krr(Xtr, ytr, lam, eps=EPS, dtype=np.float64):
    """Exact Yat-kernel ridge: alpha = (K + lam I)^-1 y. Returns predict fn."""
    from scipy.linalg import cho_factor, cho_solve

    K = yat_gram(Xtr.astype(dtype), Xtr.astype(dtype), eps).astype(dtype)
    K[np.diag_indices_from(K)] += lam
    cf = cho_factor(K, lower=True)
    alpha = cho_solve(cf, ytr.astype(dtype))

    def predict(X, chunk=4096):
        outs = []
        for i in range(0, len(X), chunk):
            outs.append(yat_gram(X[i : i + chunk].astype(dtype), Xtr.astype(dtype), eps) @ alpha)
        return np.concatenate(outs)

    return predict, alpha


def median_sq_dist(X, m=1000, seed=0):
    idx = np.random.default_rng(seed).choice(len(X), min(m, len(X)), replace=False)
    d2 = ((X[idx, None] - X[None, idx]) ** 2).sum(-1)
    return float(np.median(d2[d2 > 0]))


def pick_lam_eps(Xtr, ytr, Xva, yva, lams=(1e-3, 1e-2, 1e-1, 1.0), epss=None):
    """eps is the kernel's bandwidth; select it on val jointly with the ridge,
    exactly as one would a Gaussian bandwidth. The descended net then uses the
    SAME eps, so both fits carry the same kernel."""
    if epss is None:
        med = median_sq_dist(Xtr)
        epss = (1e-2, 1.0, med / 4, med)
    best = (None, None, np.inf)
    for eps in epss:
        for lam in lams:
            pred, _ = solve_krr(Xtr, ytr, lam, eps=eps)
            rmse = float(np.sqrt(np.mean((pred(Xva) - yva) ** 2)))
            if rmse < best[2]:
                best = (lam, eps, rmse)
    return best


# ---------------------------------------------------------------- the Yat net
class YatNet(nnx.Module):
    """h(x) = sum_u a_u * (w_u.x + b_u)^2 / (||x - w_u||^2 + eps).

    out_dim=1 for regression/binary, 10 for FMNIST (a is K x out)."""

    def __init__(self, protos, out_dim, rngs, eps=EPS):
        K, d = protos.shape
        self.eps = eps
        self.w = nnx.Param(jnp.asarray(protos, jnp.float32))
        self.b = nnx.Param(jnp.zeros((K,), jnp.float32))
        self.a = nnx.Param(
            0.01 * jax.random.normal(rngs.params(), (K, out_dim), jnp.float32)
        )

    def phi(self, x):
        dot = x @ self.w.value.T + self.b.value[None, :]
        sq = ((x[:, None, :] - self.w.value[None, :, :]) ** 2).sum(-1)
        return (dot * dot) / (sq + self.eps)

    def __call__(self, x):
        return self.phi(x) @ self.a.value


def kmeans_protos(X, K, seed):
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=K, n_init=4, random_state=seed).fit(
        X[: min(len(X), 20000)]
    )
    return km.cluster_centers_.astype(np.float32)


def train_yat_net(
    Xtr, ytr, Xva, yva, K, lr, seed, epochs, batch, loss_kind, out_dim=1,
    epoch_hook=None, eps=EPS,
):
    """Adam, best-val-epoch selection. Returns (best_val, best_params, curves)."""
    rngs = nnx.Rngs(seed)
    model = YatNet(kmeans_protos(Xtr, K, seed), out_dim, rngs, eps=eps)
    opt = nnx.Optimizer(model, optax.adamw(lr, weight_decay=1e-4), wrt=nnx.Param)

    def loss_fn(m, xb, yb):
        out = m(xb)
        if loss_kind == "mse":
            return jnp.mean((out[:, 0] - yb) ** 2)
        if loss_kind == "bce":
            return jnp.mean(optax.sigmoid_binary_cross_entropy(out[:, 0], yb))
        return jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(out, yb.astype(jnp.int32))
        )

    @nnx.jit
    def step(m, o, xb, yb):
        l, grads = nnx.value_and_grad(loss_fn)(m, xb, yb)
        o.update(m, grads)
        return l

    @nnx.jit
    def fwd(m, xb):
        return m(xb)

    def val_metric(m):
        outs = [np.asarray(fwd(m, jnp.asarray(Xva[i : i + 8192])))
                for i in range(0, len(Xva), 8192)]
        out = np.concatenate(outs)
        if loss_kind == "mse":
            return float(np.sqrt(np.mean((out[:, 0] - yva) ** 2)))  # lower better
        if loss_kind == "bce":
            return 1.0 - float(np.mean((out[:, 0] > 0) == (yva > 0.5)))  # error
        return 1.0 - float(np.mean(out.argmax(1) == yva))  # error

    rng = np.random.default_rng(seed)
    n = len(Xtr)
    best = (np.inf, None, -1)
    curves = []
    for ep in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n - batch + 1, batch):
            idx = perm[i : i + batch]
            step(model, opt, jnp.asarray(Xtr[idx]), jnp.asarray(ytr[idx]))
        vm = val_metric(model)
        curves.append(vm)
        if vm < best[0]:
            best = (vm, _snapshot(model), ep)
        if epoch_hook is not None:
            epoch_hook(ep, model)
    _restore(model, best[1])
    return best[0], model, {"val_curve": curves, "best_epoch": best[2]}


def sweep_lr(Xtr, ytr, Xva, yva, K, lrs, seed, epochs, batch, loss_kind, out_dim=1,
             eps=EPS):
    best = (None, np.inf)
    for lr in lrs:
        vm, _, _ = train_yat_net(
            Xtr, ytr, Xva, yva, K, lr, seed, epochs, batch, loss_kind, out_dim, eps=eps
        )
        print(f"    lr sweep K={K} lr={lr}: val={vm:.4f}", flush=True)
        if vm < best[1]:
            best = (lr, vm)
    return best[0]


@nnx.jit
def _fwd(m, xb):
    return m(xb)


def _snapshot(model):
    return jax.tree.map(lambda a: np.asarray(a).copy(), nnx.state(model, nnx.Param))


def _restore(model, snap):
    nnx.update(model, jax.tree.map(jnp.asarray, snap))


def predict_np(model, X, chunk=8192):
    return np.concatenate(
        [np.asarray(_fwd(model, jnp.asarray(X[i : i + chunk]))) for i in range(0, len(X), chunk)]
    )


# ================================================================ E1 agreement
def e1_agreement():
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split

    print("== E1 agreement (California Housing) ==", flush=True)
    D = fetch_california_housing()
    X, y = D.data.astype(np.float64), D.target.astype(np.float64)
    Xtr, Xrest, ytr, yrest = train_test_split(X, y, train_size=4000, random_state=0)
    Xva, Xrest, yva, yrest = train_test_split(Xrest, yrest, train_size=1000, random_state=0)
    Xte, yte = Xrest[:4000], yrest[:4000]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    ymu, ysd = ytr.mean(), ytr.std()
    Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    ytr, yva, yte = (ytr - ymu) / ysd, (yva - ymu) / ysd, (yte - ymu) / ysd
    if SMOKE:
        Xtr, ytr = Xtr[:800], ytr[:800]

    # -- exact solve; (lam, eps) picked on val, eps shared with the net below
    lam, eps, _ = pick_lam_eps(Xtr, ytr, Xva, yva)
    t0 = time.time()
    pred_solve, alpha = solve_krr(Xtr, ytr, lam, eps=eps)
    solve_s = time.time() - t0
    ps_te = pred_solve(Xte)
    rmse_solve = float(np.sqrt(np.mean((ps_te - yte) ** 2)))
    print(f"  solve: lam={lam} eps={eps:.3g} rmse={rmse_solve:.4f} "
          f"({solve_s:.1f}s, n={len(Xtr)})", flush=True)

    # -- RFF ridge at matched parameter budgets
    def rff_rmse(Dfeat, seed=0, lams=(1e-3, 1e-2, 1e-1, 1.0)):
        rng = np.random.default_rng(seed)
        # median heuristic bandwidth
        idx = rng.choice(len(Xtr), min(1000, len(Xtr)), replace=False)
        d2 = ((Xtr[idx, None] - Xtr[None, idx]) ** 2).sum(-1)
        gamma = 1.0 / np.median(d2[d2 > 0])
        W = rng.normal(0, np.sqrt(2 * gamma), (Xtr.shape[1], Dfeat))
        bph = rng.uniform(0, 2 * np.pi, Dfeat)
        f = lambda A: np.sqrt(2.0 / Dfeat) * np.cos(A @ W + bph)
        Ztr, Zva, Zte = f(Xtr), f(Xva), f(Xte)
        best = (np.inf, None)
        for lm in lams:
            wgt = np.linalg.solve(Ztr.T @ Ztr + lm * np.eye(Dfeat), Ztr.T @ ytr)
            vr = np.sqrt(np.mean((Zva @ wgt - yva) ** 2))
            if vr < best[0]:
                best = (vr, wgt)
        return float(np.sqrt(np.mean((Zte @ best[1] - yte) ** 2)))

    # -- SGD Yat nets: K ladder
    Ks = [8, 32] if SMOKE else [8, 16, 32, 64, 128]
    epochs = 3 if SMOKE else 160
    ladder = []
    agree_series = None
    snap_pred = None
    K_HEAD = 32
    for K in Ks:
        lr = sweep_lr(Xtr, ytr, Xva, yva, K, (1e-2, 3e-3, 1e-3, 3e-4), 0,
                      max(3, epochs // 3), 256, "mse", eps=eps)
        rmses, agrees = [], []
        for seed in range(SEEDS):
            hooks = {}
            if K == K_HEAD and seed == 0:
                # per-epoch telemetry for the convergence GIF
                sub = np.arange(0, len(Xte), max(1, len(Xte) // 400))[:400]
                snaps, rs, rmts, wsnaps = [], [], [], []

                def hook(ep, m, sub=sub, snaps=snaps, rs=rs, rmts=rmts, wsnaps=wsnaps):
                    p = predict_np(m, Xte)[:, 0]
                    snaps.append(p[sub].astype(np.float32))
                    rs.append(float(np.corrcoef(p, ps_te)[0, 1]))
                    rmts.append(float(np.sqrt(np.mean((p - yte) ** 2))))
                    wsnaps.append(np.asarray(m.w.value, np.float32).copy())

                hooks["epoch_hook"] = hook
            _, model, cur = train_yat_net(
                Xtr, ytr, Xva, yva, K, lr, seed, epochs, 256, "mse", eps=eps, **hooks
            )
            p = predict_np(model, Xte)[:, 0]
            rmses.append(float(np.sqrt(np.mean((p - yte) ** 2))))
            agrees.append(float(np.corrcoef(p, ps_te)[0, 1]))
            if K == K_HEAD and seed == 0:
                agree_series = {"r": rs, "rmse": rmts}
                snap_pred = np.stack(snaps)
                ARRAYS["e1_snap_solve"] = ps_te[sub].astype(np.float32)
                ARRAYS["e1_snap_true"] = yte[sub].astype(np.float32)
                ARRAYS["e1_snap_net"] = snap_pred
                ARRAYS["e1_proto_traj"] = np.stack(wsnaps)
                ARRAYS["e1_train_X"] = Xtr[:2000].astype(np.float32)
        ladder.append(
            {"K": K, "params": int(K * (Xtr.shape[1] + 2)), "lr": lr,
             "rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)),
             "agree_mean": float(np.mean(agrees)), "agree_std": float(np.std(agrees)),
             "rmse_seeds": rmses, "agree_seeds": agrees}
        )
        print(f"  net K={K}: rmse={np.mean(rmses):.4f}±{np.std(rmses):.4f} "
              f"agree r={np.mean(agrees):.4f}", flush=True)

    head = next(l for l in ladder if l["K"] == K_HEAD)
    rff_small = rff_rmse(K_HEAD)                      # same feature count
    rff_matched = rff_rmse(head["params"])            # same parameter count
    RESULTS["e1"] = {
        "n_train": len(Xtr), "lam": lam, "eps": eps, "solve_rmse": rmse_solve,
        "solve_seconds": solve_s, "ladder": ladder, "K_head": K_HEAD,
        "rff_rmse_same_features": rff_small, "rff_rmse_same_params": rff_matched,
        "agree_series": agree_series,
        "ysd": float(ysd),
    }
    print(f"  rff D={K_HEAD}: {rff_small:.4f}  D={head['params']}: {rff_matched:.4f}", flush=True)


# ================================================================ E2 the wall
def e2_wall():
    from sklearn.datasets import fetch_california_housing

    print("== E2 the wall (measured) ==", flush=True)
    D = fetch_california_housing()
    X = ((D.data - D.data.mean(0)) / (D.data.std(0) + 1e-8)).astype(np.float64)
    y = ((D.target - D.target.mean()) / D.target.std()).astype(np.float64)
    ns = [500, 1000, 2000] if SMOKE else [500, 1000, 2000, 4000, 8000, 16000]
    rows = []
    for n in ns:
        Xn, yn = X[:n], y[:n]
        # CPU float64 exact solve
        t0 = time.time()
        _, _ = solve_krr(Xn, yn, 1e-2)
        cpu_s = time.time() - t0
        # GPU float32 exact solve (build + cholesky on device)
        t0 = time.time()
        Xj = jnp.asarray(Xn, jnp.float32)
        dot = Xj @ Xj.T
        sq = (Xj * Xj).sum(1)[:, None] + (Xj * Xj).sum(1)[None, :] - 2 * dot
        Kg = (dot * dot) / (jnp.maximum(sq, 0.0) + EPS) + 1e-2 * jnp.eye(n)
        L = jnp.linalg.cholesky(Kg)
        al = jax.scipy.linalg.cho_solve((L, True), jnp.asarray(yn, jnp.float32))
        al.block_until_ready()
        gpu_s = time.time() - t0
        del Kg, L, al
        # one SGD epoch (K=32, batch 256) at the same n
        rngs = nnx.Rngs(0)
        model = YatNet(Xn[:32].astype(np.float32), 1, rngs)
        opt = nnx.Optimizer(model, optax.adamw(3e-3, weight_decay=1e-4), wrt=nnx.Param)

        @nnx.jit
        def step(m, o, xb, yb):
            l, g = nnx.value_and_grad(
                lambda mm, xx, yy: jnp.mean((mm(xx)[:, 0] - yy) ** 2)
            )(m, xb, yb)
            o.update(m, g)
            return l

        step(model, opt, jnp.asarray(Xn[:256], jnp.float32), jnp.asarray(yn[:256], jnp.float32))
        t0 = time.time()
        for i in range(0, n - 256 + 1, 256):
            step(model, opt, jnp.asarray(Xn[i : i + 256], jnp.float32),
                 jnp.asarray(yn[i : i + 256], jnp.float32))
        jax.block_until_ready(model.a.value)
        sgd_epoch_s = time.time() - t0
        rows.append({"n": n, "solve_cpu_s": cpu_s, "solve_gpu_s": gpu_s,
                     "gram_gb_f64": n * n * 8 / 1e9, "gram_gb_f32": n * n * 4 / 1e9,
                     "sgd_epoch_s": sgd_epoch_s})
        print(f"  n={n}: cpu {cpu_s:.2f}s gpu {gpu_s:.2f}s sgd-epoch {sgd_epoch_s:.3f}s "
              f"gram {n*n*8/1e9:.2f}GB", flush=True)
    RESULTS["e2"] = {"rows": rows, "gpu_gb": 16.0}


# ================================================================ E3 scale
def e3_scale():
    from sklearn.datasets import fetch_covtype
    from sklearn.model_selection import train_test_split

    print("== E3 past the wall (covtype) ==", flush=True)
    D = fetch_covtype()
    X = D.data.astype(np.float32)
    y = (D.target == 2).astype(np.float32)  # most common class vs rest
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=50000, random_state=0)
    Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=20000, random_state=0)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    print(f"  train pool {len(Xtr)}, val {len(Xva)}, test {len(Xte)}", flush=True)

    solve_ns = [1000, 2000] if SMOKE else [2000, 8000, 16000]
    sgd_ns = [1000, 4000] if SMOKE else [2000, 8000, 16000, 64000, 256000, len(Xtr)]
    rows_solve, rows_sgd = [], []

    # (lam, eps) picked once on val at n=2000; eps is then THE kernel for E3,
    # shared by every solve size and by the descended net.
    _, eps3, _ = pick_lam_eps(
        Xtr[:2000].astype(np.float64), (2.0 * ytr[:2000] - 1.0).astype(np.float64),
        Xva[:5000].astype(np.float64), (2.0 * yva[:5000] - 1.0).astype(np.float64),
    )
    print(f"  kernel eps={eps3:.3g}", flush=True)

    for n in solve_ns:
        accs = []
        for seed in range(min(SEEDS, 3)):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(Xtr), n, replace=False)
            ylab = 2.0 * ytr[idx] - 1.0
            best = (0.0, None)
            for lam in (1e-2, 1e-1, 1.0):
                pred, _ = solve_krr(Xtr[idx].astype(np.float64), ylab.astype(np.float64),
                                    lam, eps=eps3)
                acc_va = float(np.mean((pred(Xva[:10000].astype(np.float64)) > 0)
                                       == (yva[:10000] > 0.5)))
                if acc_va > best[0]:
                    best = (acc_va, pred)
            acc = float(np.mean((best[1](Xte.astype(np.float64)) > 0) == (yte > 0.5)))
            accs.append(acc)
        rows_solve.append({"n": n, "acc_mean": float(np.mean(accs)),
                           "acc_std": float(np.std(accs)), "acc_seeds": accs})
        print(f"  solve n={n}: acc={np.mean(accs):.4f}±{np.std(accs):.4f}", flush=True)

    K = 16 if SMOKE else 64
    lr = sweep_lr(Xtr[:16000], ytr[:16000], Xva, yva, K, (1e-2, 3e-3, 1e-3, 3e-4), 0,
                  3 if SMOKE else 20, 512, "bce", eps=eps3)
    for n in sgd_ns:
        epochs = 2 if SMOKE else max(6, min(40, int(40 * 16000 / n)))
        accs = []
        nseeds = SEEDS if n in (sgd_ns[0], sgd_ns[-1]) else max(1, SEEDS - 1)
        for seed in range(nseeds):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(Xtr), n, replace=False)
            _, model, _ = train_yat_net(Xtr[idx], ytr[idx], Xva, yva, K, lr, seed,
                                        epochs, 512, "bce", eps=eps3)
            p = predict_np(model, Xte)[:, 0]
            accs.append(float(np.mean((p > 0) == (yte > 0.5))))
        rows_sgd.append({"n": n, "acc_mean": float(np.mean(accs)),
                         "acc_std": float(np.std(accs)), "acc_seeds": accs,
                         "epochs": epochs})
        print(f"  sgd n={n}: acc={np.mean(accs):.4f}±{np.std(accs):.4f}", flush=True)

    RESULTS["e3"] = {"solve": rows_solve, "sgd": rows_sgd, "K": K, "lr": lr,
                     "eps": eps3,
                     "wall_note": "float64 Gram at n=16k is 2.0 GB; 64k would be 33 GB",
                     "n_pool": len(Xtr)}


# ---------------------------------------------------------------- E3B capacity
def e3_big():
    """E3 extension: past the wall, gradient descent scales CAPACITY with data.
    K in {256, 1024} prototypes at the large-n points, where the solve cannot go."""
    from sklearn.datasets import fetch_covtype
    from sklearn.model_selection import train_test_split

    print("== E3B capacity past the wall (covtype) ==", flush=True)
    D = fetch_covtype()
    X = D.data.astype(np.float32)
    y = (D.target == 2).astype(np.float32)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=50000, random_state=0)
    Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=20000, random_state=0)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    eps3 = float(os.environ.get("EPS3", "1.0"))  # E3's val-picked kernel bandwidth

    Ks = [64, 256] if SMOKE else [256, 1024]
    ns = [4000] if SMOKE else [64000, 256000, len(Xtr)]
    rows = []
    for K in Ks:
        lr = sweep_lr(Xtr[:64000], ytr[:64000], Xva, yva, K, (3e-3, 1e-3, 3e-4), 0,
                      2 if SMOKE else 6, 512, "bce", eps=eps3)
        for n in ns:
            epochs = 2 if SMOKE else (16 if n <= 64000 else 8)
            accs = []
            for seed in range(min(SEEDS, 2)):
                rng = np.random.default_rng(seed)
                idx = rng.choice(len(Xtr), n, replace=False)
                _, model, _ = train_yat_net(Xtr[idx], ytr[idx], Xva, yva, K, lr, seed,
                                            epochs, 512, "bce", eps=eps3)
                p = predict_np(model, Xte)[:, 0]
                accs.append(float(np.mean((p > 0) == (yte > 0.5))))
            rows.append({"K": K, "n": n, "lr": lr, "epochs": epochs,
                         "acc_mean": float(np.mean(accs)),
                         "acc_std": float(np.std(accs)), "acc_seeds": accs})
            print(f"  sgd K={K} n={n}: acc={np.mean(accs):.4f}±{np.std(accs):.4f}",
                  flush=True)
    RESULTS["e3b"] = {"rows": rows, "eps": eps3}


# ================================================================ E4 compose
class ConvYat(nnx.Module):
    """Two conv layers -> GAP-ish flatten -> Yat prototype head (K x 10)."""

    def __init__(self, K, rngs, feat_dim=64):
        self.c1 = nnx.Conv(1, 16, (3, 3), strides=2, rngs=rngs)
        self.c2 = nnx.Conv(16, 32, (3, 3), strides=2, rngs=rngs)
        self.proj = nnx.Linear(32 * 7 * 7, feat_dim, rngs=rngs)
        self.w = nnx.Param(0.5 * jax.random.normal(rngs.params(), (K, feat_dim)))
        self.b = nnx.Param(jnp.zeros((K,)))
        self.a = nnx.Param(0.01 * jax.random.normal(rngs.params(), (K, 10)))

    def features(self, x):
        h = jax.nn.relu(self.c1(x))
        h = jax.nn.relu(self.c2(h))
        h = h.reshape((h.shape[0], -1))
        return self.proj(h)

    def __call__(self, x):
        f = self.features(x)
        dot = f @ self.w.value.T + self.b.value[None, :]
        sq = ((f[:, None, :] - self.w.value[None, :, :]) ** 2).sum(-1)
        return ((dot * dot) / (sq + EPS)) @ self.a.value


def _load_fmnist():
    import torchvision

    tr = torchvision.datasets.FashionMNIST("/tmp/fmnist", train=True, download=True)
    te = torchvision.datasets.FashionMNIST("/tmp/fmnist", train=False, download=True)
    Xtr = tr.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    Xte = te.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    return Xtr, tr.targets.numpy().astype(np.int32), Xte, te.targets.numpy().astype(np.int32)


def _solve_ovr(Xtr, ytr, Xva, yva, Xte, yte, tag):
    """Exact Yat-kernel ridge, one-vs-rest over 10 classes; (lam, eps) on val.
    Returns (test acc, predictions on the whole test set)."""
    from scipy.linalg import cho_factor, cho_solve

    Y = np.eye(10)[ytr] * 2.0 - 1.0
    med = median_sq_dist(Xtr)
    best = (0.0, None)
    for eps in (1e-2, med / 4, med):
        K = yat_gram(Xtr.astype(np.float64), Xtr.astype(np.float64), eps)
        for lam in (1e-1, 1.0, 10.0):
            Kl = K.copy()
            Kl[np.diag_indices_from(Kl)] += lam
            cf = cho_factor(Kl, lower=True)
            A = cho_solve(cf, Y)

            def pred(Xq, A=A, eps=eps, chunk=2048):
                outs = []
                for i in range(0, len(Xq), chunk):
                    outs.append(yat_gram(Xq[i : i + chunk].astype(np.float64),
                                         Xtr.astype(np.float64), eps) @ A)
                return np.concatenate(outs).argmax(1)

            acc_va = float(np.mean(pred(Xva) == yva))
            if acc_va > best[0]:
                best = (acc_va, pred)
    pte = best[1](Xte)
    acc = float(np.mean(pte == yte))
    print(f"  {tag}: acc={acc:.4f}", flush=True)
    return acc, pte


def e4_compose():
    print("== E4 composition (Fashion-MNIST) ==", flush=True)
    Xtr_all, ytr_all, Xte, yte = _load_fmnist()
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(Xtr_all))
    va_idx, tr_idx = perm[:5000], perm[5000:]
    Xva, yva = Xtr_all[va_idx], ytr_all[va_idx]
    Xtr_pool, ytr_pool = Xtr_all[tr_idx], ytr_all[tr_idx]
    n_small = 1000 if SMOKE else 8000
    small = rng.choice(len(Xtr_pool), n_small, replace=False)
    Xs, ys = Xtr_pool[small], ytr_pool[small]
    if SMOKE:
        Xte, yte = Xte[:2000], yte[:2000]

    # (a) exact solve on raw pixels
    acc_raw, pred_raw = _solve_ovr(Xs, ys, Xva[:5000], yva[:5000], Xte, yte,
                                   f"solve raw n={n_small}")

    # (b) frozen random conv trunk -> exact solve on its features
    frozen = ConvYat(64, nnx.Rngs(0))

    @nnx.jit
    def feats(m, xb):
        return m.features(xb)

    def featurize(X):
        out = []
        for i in range(0, len(X), 4096):
            xb = jnp.asarray(X[i : i + 4096].reshape(-1, 28, 28, 1))
            out.append(np.asarray(feats(frozen, xb)))
        Z = np.concatenate(out)
        return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8) * np.sqrt(Z.shape[1])

    acc_frozen, pred_frozen = _solve_ovr(featurize(Xs), ys, featurize(Xva[:5000]),
                                         yva[:5000], featurize(Xte), yte,
                                         f"solve frozen-trunk n={n_small}")

    # (c) end-to-end: conv trunk + Yat head by SGD, matched-data and full-data
    def train_e2e(Xd, yd, seed, lr, epochs, hook=None):
        rngs = nnx.Rngs(seed)
        model = ConvYat(64, rngs)
        opt = nnx.Optimizer(model, optax.adamw(lr, weight_decay=1e-4), wrt=nnx.Param)

        def loss_fn(m, xb, yb):
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(m(xb), yb))

        @nnx.jit
        def step(m, o, xb, yb):
            l, g = nnx.value_and_grad(loss_fn)(m, xb, yb)
            o.update(m, g)
            return l

        @nnx.jit
        def fwd(m, xb):
            return m(xb)

        def acc_of(X_, y_):
            preds = []
            for i in range(0, len(X_), 4096):
                xb = jnp.asarray(X_[i : i + 4096].reshape(-1, 28, 28, 1))
                preds.append(np.asarray(fwd(model, xb)).argmax(1))
            return float(np.mean(np.concatenate(preds) == y_))

        rng = np.random.default_rng(seed)
        best = (0.0, None, -1)
        curve = []
        for ep in range(epochs):
            perm = rng.permutation(len(Xd))
            for i in range(0, len(Xd) - 256 + 1, 256):
                idx = perm[i : i + 256]
                step(model, opt, jnp.asarray(Xd[idx].reshape(-1, 28, 28, 1)),
                     jnp.asarray(yd[idx]))
            av = acc_of(Xva, yva)
            curve.append(av)
            if av > best[0]:
                best = (av, _snapshot(model), ep)
            if hook is not None:
                hook(ep, model)
        _restore(model, best[1])
        return model, acc_of(Xte, yte), curve

    epochs_small = 2 if SMOKE else 30
    epochs_full = 2 if SMOKE else 12
    # LR sweep on the small set, one seed
    best_lr, best_acc = None, 0.0
    for lr in (3e-3, 1e-3, 3e-4):
        _, acc, _ = train_e2e(Xs, ys, 0, lr, max(2, epochs_small // 3))
        print(f"    e2e lr sweep lr={lr}: te={acc:.4f}", flush=True)
        if acc > best_acc:
            best_lr, best_acc = lr, acc

    accs_small, accs_full = [], []
    for seed in range(SEEDS):
        hook = None
        if seed == 0:
            # telemetry: feature maps of one fixed garment per epoch (for the GIF)
            fixed = jnp.asarray(Xte[:1].reshape(1, 28, 28, 1))
            fmaps, teaccs = [], []

            def hook(ep, m, fixed=fixed, fmaps=fmaps):
                h1 = np.asarray(jax.nn.relu(m.c1(fixed)))[0]  # 14x14x16
                fmaps.append(h1[..., :8].astype(np.float32))

        _, acc, curve = train_e2e(Xs, ys, seed, best_lr, epochs_small, hook)
        accs_small.append(acc)
        if seed == 0:
            ARRAYS["e4_fmaps_small"] = np.stack(fmaps)
            ARRAYS["e4_curve_small"] = np.asarray(curve, np.float32)
            ARRAYS["e4_garment"] = Xte[0].reshape(28, 28).astype(np.float32)
    model_full0 = None
    for seed in range(SEEDS):
        model, acc, curve = train_e2e(Xtr_pool, ytr_pool, seed, best_lr, epochs_full)
        accs_full.append(acc)
        if seed == 0:
            model_full0 = model
            ARRAYS["e4_curve_full"] = np.asarray(curve, np.float32)

    # fixed 12-garment panel: images, every fit's verdict, conv1 maps frozen vs trained
    panel = []
    for c in range(10):
        panel.extend(np.where(yte == c)[0][:2 if c < 2 else 1].tolist())
    panel = np.asarray(panel[:12])
    fixed = jnp.asarray(Xte[panel].reshape(-1, 28, 28, 1))

    @nnx.jit
    def conv1(m, xb):
        return jax.nn.relu(m.c1(xb))

    @nnx.jit
    def logits(m, xb):
        return m(xb)

    ARRAYS["e4_panel_images"] = Xte[panel].reshape(-1, 28, 28).astype(np.float32)
    ARRAYS["e4_panel_true"] = yte[panel].astype(np.int32)
    ARRAYS["e4_panel_pred_raw"] = pred_raw[panel].astype(np.int32)
    ARRAYS["e4_panel_pred_frozen"] = pred_frozen[panel].astype(np.int32)
    ARRAYS["e4_panel_pred_e2e"] = np.asarray(logits(model_full0, fixed)).argmax(1).astype(np.int32)
    ARRAYS["e4_panel_fmaps_frozen"] = np.asarray(conv1(frozen, fixed))[..., :8].astype(np.float32)
    ARRAYS["e4_panel_fmaps_trained"] = np.asarray(conv1(model_full0, fixed))[..., :8].astype(np.float32)

    RESULTS["e4"] = {
        "n_small": n_small, "acc_solve_raw": acc_raw, "acc_solve_frozen": acc_frozen,
        "lr": best_lr,
        "acc_e2e_small_mean": float(np.mean(accs_small)),
        "acc_e2e_small_std": float(np.std(accs_small)), "acc_e2e_small_seeds": accs_small,
        "acc_e2e_full_mean": float(np.mean(accs_full)),
        "acc_e2e_full_std": float(np.std(accs_full)), "acc_e2e_full_seeds": accs_full,
        "n_full": len(Xtr_pool),
    }
    print(f"  e2e n={n_small}: {np.mean(accs_small):.4f}±{np.std(accs_small):.4f}  "
          f"e2e n=full: {np.mean(accs_full):.4f}±{np.std(accs_full):.4f}", flush=True)


# ================================================================ main
def main():
    t0 = time.time()
    only = os.environ.get("ONLY", "")
    if only == "e3b":
        e3_big()
    else:
        e1_agreement()
        e2_wall()
        e3_scale()
        e4_compose()
    RESULTS["total_seconds"] = time.time() - t0
    with open(os.path.join(RESULTS_DIR, "kernel_solve_wall.json"), "w") as f:
        json.dump(RESULTS, f, indent=1)
    np.savez_compressed(os.path.join(RESULTS_DIR, "kernel_solve_wall.npz"), **ARRAYS)
    if WB:
        wandb.summary.update({"total_seconds": RESULTS["total_seconds"]})
        wandb.finish()
    print(f"DONE in {RESULTS['total_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
