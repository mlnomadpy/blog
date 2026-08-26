"""Eigen-analysis of the trained Yat network's kernel + asset export.

Reads the per-epoch model snapshots from kgl_blog-mercer-v1 and performs the
empirical Mercer decomposition on the real test set, per epoch:

  features   Phi_e = phi(W_e, X_test)                     (n x m)
  centered   C_e = (Phi_e - mean)^T (Phi_e - mean) / n    (m x m)
  eigh       C_e = U diag(lam) U^T ; mode values psi_k = Phi_c u_k / sqrt(n lam_k)

Writes public/mercer-microscope/:
  spectrum.json   per-epoch eigenvalues, participation ratio, kernel-target
                  alignment (cumulative label energy by mode + the Cristianini
                  scalar), head energy by mode, test accuracy
  modes.json      top modes of the final trained kernel: eigenvalue share,
                  most-positive / most-negative test thumbs (14x14 uint8),
                  per-class mean mode value (the class x mode map)
  trunc.json      the browser payload for the live truncation panel: an
                  800-image test subset's features under BOTH final kernels
                  (trained + frozen twin), their eigenbases, heads, labels,
                  thumbs, so rank-k truncation is recomputed live on the page
  numbers.json    every statistic quoted in prose

Local, seconds-scale: a replay of exported weights.
"""

import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "..", "public", "mercer-microscope")
os.makedirs(OUT, exist_ok=True)

BUNDLE = sorted(g for g in glob.glob(os.path.join(RES, "kgl_blog-mercer-*"))
                if "smoke" not in g)[-1]
print("bundle:", BUNDLE)


def load_test():
    import torchvision
    te = torchvision.datasets.FashionMNIST("/tmp/fmnist", train=False, download=True)
    X = (te.data.numpy().astype(np.float32) / 255.0).reshape(-1, 784)
    y = te.targets.numpy().astype(np.int64)
    return X, y


X, y = load_test()
n = len(X)
Y = np.eye(10)[y]
Yc = Y - Y.mean(0)
CLS = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
       "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]


def feats(W, b, eps):
    dot = X @ W.T
    d2 = (X ** 2).sum(1, keepdims=True) + (W ** 2).sum(1) - 2 * dot
    return (dot + b) ** 2 / (np.maximum(d2, 0) + eps)


def decompose(Phi):
    mu = Phi.mean(0)
    Pc = Phi - mu
    C = (Pc.T @ Pc) / n
    lam, U = np.linalg.eigh(C)
    lam, U = lam[::-1].copy(), U[:, ::-1].copy()
    lam = np.maximum(lam, 0)
    psi = Pc @ U / (np.sqrt(n * np.maximum(lam, 1e-30)))    # (n, m), unit norm cols
    return mu, lam, U, psi


def alignment_curve(lam, psi):
    """Cumulative share of centered-label energy captured by the top-k modes."""
    proj = (psi.T @ Yc) ** 2                                # (m, 10)
    per_mode = proj.sum(1)
    return np.cumsum(per_mode) / (Yc ** 2).sum()


def cristianini(Phi):
    """<K, yy^T> / (||K|| ||yy^T||) via the feature trick (finite rank)."""
    Pc = Phi - Phi.mean(0)
    G = Pc.T @ Yc                                           # (m, 10)
    num = (G ** 2).sum()
    C = Pc.T @ Pc
    kf = np.sqrt((C ** 2).sum())
    yf = np.sqrt(((Yc.T @ Yc) ** 2).sum())
    return float(num / (kf * yf + 1e-30))


def main():
    D = json.load(open(os.path.join(BUNDLE, "yat_mercer.json")))
    rows = D["rows"]
    m = D["m"]
    z = {arm: np.load(os.path.join(BUNDLE, f"mercer_{arm}_s0.npz"))
         for arm in ("trained", "frozen")}
    nE = D["epochs"] + 1

    spectrum = {"epochs": nE, "arms": {}}
    numbers = {"m": m}
    per_arm_final = {}
    for arm in ("trained", "frozen"):
        za = z[arm]
        eig_by_ep, pr, align10, align40, crist, ac_by_ep = [], [], [], [], [], []
        for e in range(nE):
            W = za[f"W{e}"].astype(np.float64)
            Phi = feats(W, float(za["b"][e]), float(za["eps"][e]))
            mu, lam, U, psi = decompose(Phi)
            ac = alignment_curve(lam, psi)
            ac_by_ep.append(np.round(ac, 4).tolist())
            eig_by_ep.append(np.round(np.log10(np.maximum(lam, 1e-12)), 3).tolist())
            pr.append(round(float(lam.sum() ** 2 / (lam ** 2).sum()), 2))
            align10.append(round(float(ac[9]), 4))
            align40.append(round(float(ac[39]), 4))
            crist.append(round(cristianini(Phi), 4))
            if e == nE - 1:
                per_arm_final[arm] = (Phi, mu, lam, U, psi, ac,
                                      za[f"A{e}"].astype(np.float64),
                                      za[f"bias{e}"].astype(np.float64))
        curve = next(r["curve"] for r in rows if r["arm"] == arm and r["seed"] == 0)
        spectrum["arms"][arm] = dict(
            eigvals_log10=eig_by_ep, pr=pr, align10=align10, align40=align40,
            cristianini=crist, acc=[None] + curve, align_curve=ac_by_ep)
        numbers[arm] = dict(pr_init=pr[0], pr_final=pr[-1],
                            align10_init=align10[0], align10_final=align10[-1],
                            cristianini_init=crist[0], cristianini_final=crist[-1])

    # ── truncation: keep top-k modes of the final kernel, no retraining ──
    ks = sorted(set([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]))
    trunc_acc = {}
    for arm, (Phi, mu, lam, U, psi, ac, A, bias) in per_arm_final.items():
        accs = []
        Pc = Phi - mu
        base = mu @ A + bias
        for k in ks:
            Pk = Pc @ U[:, :k] @ U[:, :k].T
            pred = (Pk @ A + base).argmax(1)
            accs.append(round(100 * float((pred == y).mean()), 2))
        full = (Phi @ A + bias).argmax(1)
        trunc_acc[arm] = dict(ks=ks, acc=accs,
                              full=round(100 * float((full == y).mean()), 2))
    numbers["trunc"] = {a: t for a, t in trunc_acc.items()}
    for arm, t in trunc_acc.items():
        need = next((k for k, a in zip(t["ks"], t["acc"])
                     if a >= t["full"] - 1.0), None)
        numbers[arm]["k_within_1pt"] = need

    # ── the modes of the trained kernel, as pictures ──
    Phi, mu, lam, U, psi, ac, A, bias = per_arm_final["trained"]
    thumbs, thumb_ids = {}, {}
    X14 = (X.reshape(-1, 28, 28).reshape(-1, 14, 2, 14, 2).mean((2, 4)) * 255).astype(np.uint8)

    def tid(i):
        i = int(i)
        thumbs[str(i)] = X14[i].reshape(-1).tolist()
        return i

    NMODE = 16
    modes = []
    lam_share = lam / lam.sum()
    head_energy = (U.T @ A) ** 2                             # (m, 10)
    head_share = head_energy.sum(1) * lam / (head_energy.sum(1) * lam).sum()
    proj = (psi.T @ Yc) ** 2
    label_share = proj.sum(1) / (Yc ** 2).sum()
    for k2 in range(NMODE):
        v = psi[:, k2]
        top = np.argsort(v)[::-1][:5]
        bot = np.argsort(v)[:5]
        cls_mean = [round(float(v[y == c].mean()), 3) for c in range(10)]
        modes.append(dict(
            k=k2, lam_share=round(float(lam_share[k2]), 4),
            label_share=round(float(label_share[k2]), 4),
            head_share=round(float(head_share[k2]), 4),
            top=[tid(i) for i in top], bot=[tid(i) for i in bot],
            cls=cls_mean))
    modes_obj = dict(modes=modes, thumbs=thumbs, classes=CLS)

    # ── the browser payload for live truncation ──
    sub = np.random.default_rng(0).choice(n, 800, replace=False)
    sub.sort()
    KMAX = 128
    payload = dict(y=y[sub].tolist(), classes=CLS, kmax=KMAX,
                   thumbs=[X14[i].reshape(-1).tolist() for i in sub[:24]])
    for arm, (Phi, mu, lam, U, psi_, ac, A, bias) in per_arm_final.items():
        # eigen-coordinates: truncation in the browser is just zeroing columns
        E = (Phi - mu)[sub] @ U[:, :KMAX]                    # (800, KMAX)
        Ahat = U[:, :KMAX].T @ A                             # (KMAX, 10)
        payload[arm] = dict(
            E=np.round(E, 2).tolist(),
            Ahat=np.round(Ahat, 4).tolist(),
            base=np.round(mu @ A + bias, 4).tolist(),
            full_acc=trunc_acc[arm]["full"],
            lam_log10=np.round(np.log10(np.maximum(lam, 1e-12)), 3).tolist())

    for name, obj in [("spectrum", spectrum), ("modes", modes_obj),
                      ("trunc", payload), ("numbers", numbers)]:
        p = os.path.join(OUT, f"{name}.json")
        with open(p, "w") as f:
            json.dump(obj, f, separators=(",", ":"))
        print(f"{name}.json  {os.path.getsize(p)/1e6:.2f} MB")
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
