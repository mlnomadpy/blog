"""Export the lazy-training post's assets from the yat_lazy runs.

Reads results/kgl_blog-yatlazy-*/yat_lazy.json (+ prototypes npz) and writes
public/lazy-training/:

  curves.json    per arm: mean/std best-acc per width (the width ladder),
                 plus the trained-vs-frozen gap per width and the movement
                 statistics per width for the trained arms
  rate.json      the Monte Carlo rate check: fit err(m) = a + c * m^(-alpha)
                 on the frozen arms, report alpha with residuals
  protos.json    24 frozen-random and 24 data-seeded prototypes (16x16
                 downsampled f16) for the gallery panel

Local, seconds-scale.
"""

import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "public", "lazy-training")
os.makedirs(OUT, exist_ok=True)


def export_subset():
    """A small balanced Fashion-MNIST subset at 14x14 for the in-browser
    panels (the live readout training and the Monte Carlo kernel estimate).
    Independent of the Kaggle bundle; images quantized to uint8."""
    import torchvision
    tr = torchvision.datasets.FashionMNIST("/tmp/fmnist", train=True, download=True)
    te = torchvision.datasets.FashionMNIST("/tmp/fmnist", train=False, download=True)

    def prep(ds, per_class, seed):
        X = ds.data.numpy().astype(np.float32) / 255.0
        y = ds.targets.numpy()
        rng = np.random.default_rng(seed)
        idx = np.concatenate([rng.choice(np.where(y == c)[0], per_class, replace=False)
                              for c in range(10)])
        rng.shuffle(idx)
        X14 = X[idx].reshape(-1, 14, 2, 14, 2).mean((2, 4))
        return (np.round(X14 * 255).astype(int).reshape(len(idx), -1).tolist(),
                y[idx].astype(int).tolist())

    Xtr, ytr = prep(tr, 200, 0)     # 2,000 train
    Xte, yte = prep(te, 80, 1)      # 800 test
    obj = {"d": 196, "side": 14, "classes": ["T-shirt", "Trouser", "Pullover",
           "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"],
           "Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte}
    p = os.path.join(OUT, "fmnist14.json")
    with open(p, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    print(f"fmnist14.json  {os.path.getsize(p)/1e6:.2f} MB")


def find_bundle():
    cands = sorted(glob.glob(os.path.join(HERE, "results", "kgl_blog-yatlazy-*",
                                          "yat_lazy.json")))
    cands = [c for c in cands if "smoke" not in c]
    assert cands, "no full yat_lazy bundle downloaded yet"
    return cands[-1]


def main():
    path = find_bundle()
    print("bundle:", path)
    D = json.load(open(path))
    rows = [r for r in D["rows"] if not r.get("sweep")]
    widths = D["widths"]

    arms = sorted({r["arm"] for r in rows})
    curves = {"widths": widths, "arms": {}}
    for arm in arms:
        mean, std, move = [], [], []
        for m in widths:
            accs = [r["best_acc"] for r in rows if r["arm"] == arm and r["m"] == m]
            mean.append(round(float(np.mean(accs)), 2) if accs else None)
            std.append(round(float(np.std(accs)), 2) if accs else None)
            mv = [r["w_move_rel"] for r in rows
                  if r["arm"] == arm and r["m"] == m and "w_move_rel" in r]
            move.append(round(float(np.mean(mv)), 4) if mv else None)
        curves["arms"][arm] = {"mean": mean, "std": std}
        if any(v is not None for v in move):
            curves["arms"][arm]["move"] = move

    # gaps: what training the features buys, per width
    def gap(trained, frozen):
        a, b = curves["arms"].get(trained), curves["arms"].get(frozen)
        if not a or not b:
            return None
        return [round(x - y, 2) if x is not None and y is not None else None
                for x, y in zip(a["mean"], b["mean"])]
    curves["gaps"] = {
        "yat_trained_minus_frozen": gap("yat_trained", "yat_frozen"),
        "yat_data_trained_minus_frozen": gap("yat_data_trained", "yat_data"),
        "rbf_data_trained_minus_frozen": gap("rbf_data_trained", "rbf_data"),
    }

    # the Monte Carlo rate check: err(m) = a + c m^(-alpha), fitted on each
    # frozen arm's ascending region (yat_frozen stalls past m = 512 at the
    # 12-epoch budget; the fit describes the climb, the stall is reported
    # separately in the post)
    ASCENDING_CAP = {"yat_frozen": 512}
    rate = {}
    for arm in ("yat_frozen", "yat_data", "relu_frozen"):
        if arm not in curves["arms"]:
            continue
        mean = curves["arms"][arm]["mean"]
        cap = ASCENDING_CAP.get(arm)
        pts = [(m, 100 - acc) for m, acc in zip(widths, mean)
               if acc is not None and (cap is None or m <= cap)]
        if len(pts) < 4:
            continue
        ms = np.array([p[0] for p in pts], float)
        er = np.array([p[1] for p in pts], float)
        best = None
        for a0 in np.linspace(0, er.min() * 0.98, 60):
            y = np.log(er - a0)
            A = np.vstack([np.ones_like(ms), -np.log(ms)]).T
            coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
            r = float(res[0]) if len(res) else 0.0
            if best is None or r < best[0]:
                best = (r, a0, float(np.exp(coef[0])), float(coef[1]))
        _, a0, c, alpha = best
        rate[arm] = {"err_floor": round(a0, 2), "coef": round(c, 2),
                     "alpha": round(alpha, 3),
                     "pts": [[int(m), round(float(e), 2)] for m, e in pts]}
    print(json.dumps(rate, indent=1))

    # prototype galleries, downsampled to 16x16 for the panel
    npzs = sorted(glob.glob(os.path.join(os.path.dirname(path),
                                         "yat_lazy_prototypes.npz")))
    protos = {}
    if npzs:
        z = np.load(npzs[-1])
        for k in z.files:
            W = z[k].astype(np.float32).reshape(-1, 28, 28)
            W16 = W.reshape(-1, 14, 2, 14, 2).mean((2, 4))
            lo, hi = W16.min(), W16.max()
            protos[k] = np.round((W16 - lo) / (hi - lo + 1e-9), 3).tolist()
            protos[k + "_range"] = [round(float(lo), 3), round(float(hi), 3)]

    # numbers quoted in prose: the movement power law, the bank-redundancy
    # measurement, and the big-m budget patches
    numbers = {}
    mv = curves["arms"].get("yat_trained", {}).get("move")
    if mv and all(v is not None for v in mv):
        ww = np.log(np.array(widths, float))
        A = np.vstack([np.ones_like(ww), ww]).T
        coef, res, *_ = np.linalg.lstsq(A, np.log(np.array(mv)), rcond=None)
        pred = A @ coef
        yv = np.log(np.array(mv))
        numbers["movement_slope"] = round(float(coef[1]), 3)
        numbers["movement_r2"] = round(float(1 - np.sum((yv - pred) ** 2)
                                             / np.sum((yv - yv.mean()) ** 2)), 4)

    def bank_redundancy(m=256, n=4000, seed=0):
        import torchvision
        tr = torchvision.datasets.FashionMNIST("/tmp/fmnist", train=True, download=True)
        X = (tr.data.numpy().astype(np.float32) / 255.0).reshape(-1, 784)[:n]
        rng = np.random.default_rng(seed)
        banks = {"random": rng.normal(0, 1 / np.sqrt(784), (m, 784)).astype(np.float32),
                 "data": X[rng.choice(n, m, replace=False)]}
        out = {}
        for name, W in banks.items():
            dot = X @ W.T
            d2 = (X ** 2).sum(1, keepdims=True) + (W ** 2).sum(1) - 2 * dot
            F = (dot + 0.5) ** 2 / (np.maximum(d2, 0) + 0.5)
            Fz = (F - F.mean(0)) / (F.std(0) + 1e-9)
            C = (Fz.T @ Fz) / n
            ev = np.maximum(np.linalg.eigvalsh(C)[::-1], 0)
            out[name] = {"eff_rank": round(float(ev.sum() ** 2 / (ev ** 2).sum()), 1),
                         "mean_abs_corr": round(float(np.abs(C - np.eye(m)).sum()
                                                      / (m * m - m)), 3)}
        return out
    numbers["redundancy_m256"] = bank_redundancy()

    for tag, slug in [("bigm_lr01_12ep", "kgl_blog-yatlazy-bigm"),
                      ("bigm_lr01_36ep", "kgl_blog-yatlazy-bigm2")]:
        p = os.path.join(HERE, "results", slug, "yat_lazy.json")
        if os.path.exists(p):
            rows2 = [r for r in json.load(open(p))["rows"] if not r.get("sweep")]
            numbers[tag] = {str(m): round(float(np.mean(
                [r["best_acc"] for r in rows2 if r["m"] == m])), 2)
                for m in sorted({r["m"] for r in rows2})}
    curves["numbers"] = numbers

    for name, obj in [("curves", curves), ("rate", rate), ("protos", protos)]:
        p = os.path.join(OUT, f"{name}.json")
        with open(p, "w") as f:
            json.dump(obj, f, separators=(",", ":"))
        print(f"{name}.json  {os.path.getsize(p)/1e3:.0f} KB")


if __name__ == "__main__":
    import sys
    if "--subset-only" in sys.argv:
        export_subset()
    else:
        export_subset()
        main()
