"""Spectral surgery: delete a CONCEPT from a trained Yat network.

The microscope post decomposed the trained network's kernel into ranked
orthogonal Mercer modes. Because the modes are orthogonal, the readout splits
exactly into per-mode terms, so a single mode can be removed by projection,
with no retraining and no approximation:

    readout logits   f(x) = base + sum_k  <Phi_c(x), u_k> (u_k^T A)
    delete mode k    A <- A - u_k (u_k^T A)          (silence: readout only)
    erase mode k     Phi <- Phi (I - u_k u_k^T)      (erase: the features too)

Experiments (all a local replay of the exported weights of
kgl_blog-mercer-v1; no training of the network anywhere):

  E1 selectivity   per-mode surgery -> per-class and per-class-pair damage
  E2 prediction    predict each pair's damage from the class-mode profile and
                   the readout coefficients BEFORE cutting, then measure it
  E3 controls      the same amount of readout energy removed along a random
                   direction, and a prototype-row deletion (the old edit), to
                   show what non-selective damage looks like
  E4 silence vs    after silencing, refit a fresh readout on the untouched
     erase         features (the concept should come back); after erasing,
                   refit on the projected features (it should not)
  E5 order         cumulative deletion from the top of the spectrum vs the
                   bottom

Writes public/spectral-surgery/{surgery,predict,recovery,live,numbers}.json.
Run: python scripts/spectral_surgery.py
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.join(HERE, "results", "kgl_blog-mercer-v1")
OUT = os.path.join(HERE, "..", "public", "spectral-surgery")
os.makedirs(OUT, exist_ok=True)

CLS = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
       "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]
NMODE = 16                      # modes given the full treatment
EPOCH = 12                      # the final snapshot


def load_data():
    import torchvision
    tr = torchvision.datasets.FashionMNIST("/tmp/fmnist", train=True, download=True)
    te = torchvision.datasets.FashionMNIST("/tmp/fmnist", train=False, download=True)
    Xtr = (tr.data.numpy().astype(np.float32) / 255.0).reshape(-1, 784)
    Xte = (te.data.numpy().astype(np.float32) / 255.0).reshape(-1, 784)
    return Xtr, tr.targets.numpy().astype(np.int64), Xte, te.targets.numpy().astype(np.int64)


Xtr, ytr, Xte, yte = load_data()
z = np.load(os.path.join(BUNDLE, f"mercer_trained_s0.npz"))
W = z[f"W{EPOCH}"].astype(np.float64)
A = z[f"A{EPOCH}"].astype(np.float64)
bias = z[f"bias{EPOCH}"].astype(np.float64)
B_, EPS_ = float(z["b"][EPOCH]), float(z["eps"][EPOCH])
print(f"[model] m={W.shape[0]} b={B_:.3f} eps={EPS_:.4f}")


def feats(X):
    dot = X @ W.T
    d2 = (X ** 2).sum(1, keepdims=True) + (W ** 2).sum(1) - 2 * dot
    return (dot + B_) ** 2 / (np.maximum(d2, 0) + EPS_)


Ftr, Fte = feats(Xtr), feats(Xte)
mu = Ftr.mean(0)                                   # the modes are defined on the
Pc_tr, Pc_te = Ftr - mu, Fte - mu                  # TRAIN measure, then applied
C = (Pc_tr.T @ Pc_tr) / len(Pc_tr)
lam, U = np.linalg.eigh(C)
lam, U = lam[::-1].copy(), U[:, ::-1].copy()
lam = np.maximum(lam, 0)
print(f"[modes] top-5 eigenvalue share: "
      f"{np.round(lam[:5] / lam.sum(), 3)}")

base = mu @ A + bias
Ahat = U.T @ A                                     # readout in mode coordinates
Ete = Pc_te @ U                                    # test eigen-coordinates
Etr = Pc_tr @ U


def acc_from(E, Ahat_, y):
    return 100.0 * float(((E @ Ahat_ + base).argmax(1) == y).mean())


def confusion(E, Ahat_, y):
    pred = (E @ Ahat_ + base).argmax(1)
    M = np.zeros((10, 10))
    for t, p in zip(y, pred):
        M[t, p] += 1
    return M


FULL = acc_from(Ete, Ahat, yte)
CM0 = confusion(Ete, Ahat, yte)
print(f"[base] full-model test accuracy {FULL:.2f}%")


def pair_damage(CM):
    """symmetric off-diagonal confusion count per class pair"""
    P = np.zeros((10, 10))
    for a in range(10):
        for b in range(a + 1, 10):
            P[a, b] = P[b, a] = CM[a, b] + CM[b, a]
    return P


P0 = pair_damage(CM0)


def silence(k):
    """delete mode k from the readout: zero its row in mode coordinates"""
    Ah = Ahat.copy()
    Ah[k] = 0
    return Ah


LIVE_ONLY = os.environ.get("LIVE_ONLY", "0") == "1"

# ── E1 + E2: per-mode selectivity, and the prediction made beforehand ──
mode_rows, scatter = [], []
cls_mean_full = np.array([[Etr[ytr == c, k].mean() / np.sqrt(max(lam[k], 1e-30))
                           for c in range(10)] for k in range(len(lam))])  # (m,10)
cls_mean = cls_mean_full[:NMODE]
for k in range(NMODE):
    Ah = silence(k)
    acc = acc_from(Ete, Ah, yte)
    CM = confusion(Ete, Ah, yte)
    P = pair_damage(CM)
    dpair = P - P0
    per_class = np.array([100.0 * (CM[c, c] / CM[c].sum()) -
                          100.0 * (CM0[c, c] / CM0[c].sum()) for c in range(10)])
    # the prediction, from quantities the microscope already exported:
    # a pair is at risk when the mode separates the two classes AND the readout
    # uses the mode differently for them
    pred = np.zeros((10, 10))
    for a in range(10):
        for b in range(a + 1, 10):
            sep = abs(cls_mean[k, a] - cls_mean[k, b])
            use = abs(Ahat[k, a] - Ahat[k, b])
            pred[a, b] = pred[b, a] = sep * use * np.sqrt(max(lam[k], 0))
    if k < 8:
        for a in range(10):
            for b in range(a + 1, 10):
                scatter.append([k, a, b, round(float(pred[a, b]), 4),
                                round(float(dpair[a, b]), 1)])
    mode_rows.append(dict(
        k=k, acc=round(acc, 2), drop=round(FULL - acc, 2),
        per_class=np.round(per_class, 2).tolist(),
        dpair=np.round(dpair, 1).tolist(),
        pred=np.round(pred / (pred.max() + 1e-12), 3).tolist(),
        lam_share=round(float(lam[k] / lam.sum()), 4),
        worst_pair=[int(i) for i in np.unravel_index(np.argmax(dpair), dpair.shape)]))
    print(f"  mode {k+1:2d}: acc {acc:5.2f} (drop {FULL-acc:5.2f})  "
          f"worst pair {CLS[mode_rows[-1]['worst_pair'][0]]}/{CLS[mode_rows[-1]['worst_pair'][1]]}")

sc = np.array([[r[3], r[4]] for r in scatter], dtype=float)
rho = float(np.corrcoef(sc[:, 0], sc[:, 1])[0, 1])
# rank correlation too (the prediction is ordinal by construction)
def rank(v):
    o = np.argsort(v); r = np.empty_like(o, dtype=float); r[o] = np.arange(len(v))
    return r
srho = float(np.corrcoef(rank(sc[:, 0]), rank(sc[:, 1]))[0, 1])
print(f"[E2] predicted vs measured pair damage: r = {rho:.3f}, rank r = {srho:.3f}")

# ── E3: controls, matched by the readout energy removed ──
rng = np.random.default_rng(0)


def concentration(dpair):
    """share of the total damage carried by the single worst pair"""
    tot = dpair[dpair > 0].sum()
    return float(dpair.max() / (tot + 1e-9))


ctrl = {}
k0 = 0
Ah0 = silence(k0)
removed_energy = float((Ahat[k0] ** 2).sum())
d0 = pair_damage(confusion(Ete, Ah0, yte)) - P0
ctrl["mode1"] = dict(drop=round(FULL - acc_from(Ete, Ah0, yte), 2),
                     conc=round(concentration(d0), 3))
rand_drops, rand_conc = [], []
for t in range(20):
    v = rng.normal(size=len(Ahat))
    v /= np.linalg.norm(v)
    Ah = Ahat - np.outer(v, v @ Ahat)               # remove a random direction
    scale = np.sqrt(removed_energy / max(((Ahat - Ah) ** 2).sum(), 1e-12))
    Ah = Ahat - scale * (Ahat - Ah)                 # matched removed energy
    d = pair_damage(confusion(Ete, Ah, yte)) - P0
    rand_drops.append(FULL - acc_from(Ete, Ah, yte))
    rand_conc.append(concentration(d))
ctrl["random_dir"] = dict(drop=round(float(np.mean(rand_drops)), 2),
                          conc=round(float(np.mean(rand_conc)), 3))
# the old edit: delete the single most-used prototype row
row = int(np.argmax((A ** 2).sum(1)))
Adel = A.copy(); Adel[row] = 0
pred_del = ((Fte - mu) @ Adel + (mu @ Adel + bias)).argmax(1)
CMd = np.zeros((10, 10))
for t, p in zip(yte, pred_del):
    CMd[t, p] += 1
dd = pair_damage(CMd) - P0
ctrl["prototype_row"] = dict(drop=round(FULL - 100.0 * float((pred_del == yte).mean()), 2),
                             conc=round(concentration(dd), 3))
print(f"[E3] concentration: mode {ctrl['mode1']['conc']} vs "
      f"random {ctrl['random_dir']['conc']} vs prototype {ctrl['prototype_row']['conc']}")


# ── E4: silence vs erase, tested by refitting a fresh readout ──
def fit_readout(Etr_, ytr_, Ete_, yte_, steps=1500, bs=512, lr=3e-2, snap=25):
    d = Etr_.shape[1]
    Aw = np.zeros((d, 10)); bw = np.zeros(10)
    mA = np.zeros_like(Aw); vA = np.zeros_like(Aw); mb = np.zeros(10); vb = np.zeros(10)
    rr = np.random.default_rng(0)
    curve, t = [], 0
    for s in range(steps):
        idx = rr.integers(0, len(Etr_), bs)
        lg = Etr_[idx] @ Aw + bw
        lg -= lg.max(1, keepdims=True)
        p = np.exp(lg); p /= p.sum(1, keepdims=True)
        p[np.arange(bs), ytr_[idx]] -= 1
        gA = Etr_[idx].T @ p / bs; gb = p.mean(0)
        t += 1
        for P_, G, M_, V_ in ((Aw, gA, mA, vA), (bw, gb, mb, vb)):
            M_ *= 0.9; M_ += 0.1 * G
            V_ *= 0.999; V_ += 0.001 * G * G
            P_ -= lr * (M_ / (1 - 0.9 ** t)) / (np.sqrt(V_ / (1 - 0.999 ** t)) + 1e-8)
        if s % snap == 0 or s == steps - 1:
            acc = 100.0 * float(((Ete_ @ Aw + bw).argmax(1) == yte_).mean())
            curve.append(round(acc, 2))
    return curve


recovery = {"modes": [], "pairs": []}
for k in (0, 1, 2):
    a, b = mode_rows[k]["worst_pair"]
    m_tr = (ytr == a) | (ytr == b)
    m_te = (yte == a) | (yte == b)
    ya = (ytr[m_tr] == b).astype(np.int64) * 0 + ytr[m_tr]
    # binary sub-problem on the pair the mode was holding
    keep = np.arange(len(Ahat)) != k
    cur_full = fit_readout(Etr[m_tr], ytr[m_tr], Ete[m_te], yte[m_te])
    cur_erase = fit_readout(Etr[m_tr][:, keep], ytr[m_tr],
                            Ete[m_te][:, keep], yte[m_te])
    recovery["modes"].append(dict(
        k=k, pair=[a, b], pair_names=[CLS[a], CLS[b]],
        silenced_then_refit=cur_full, erased_then_refit=cur_erase))
    print(f"[E4] mode {k+1} on {CLS[a]}/{CLS[b]}: refit after silencing "
          f"{cur_full[-1]:.2f}%, after erasing {cur_erase[-1]:.2f}%")

# ── E6: what does it cost to REALLY forget a distinction? ──
# Rank modes by how much they discriminate one class pair, delete the top r of
# them from the FEATURES, refit a fresh readout each time, and watch when the
# pair finally stops being recoverable, and what the removal costs elsewhere.
forget = []
A_, B2_ = mode_rows[0]["worst_pair"]
m_tr = (ytr == A_) | (ytr == B2_)
m_te = (yte == A_) | (yte == B2_)
disc = np.array([abs(cls_mean_full[k, A_] - cls_mean_full[k, B2_]) * np.sqrt(max(lam[k], 0))
                 for k in range(len(Ahat))])
order = np.argsort(disc)[::-1]
for r in (0, 1, 2, 4, 8, 16, 32, 64, 128):
    keep = np.setdiff1d(np.arange(len(Ahat)), order[:r])
    pair_curve = fit_readout(Etr[m_tr][:, keep], ytr[m_tr],
                             Ete[m_te][:, keep], yte[m_te], steps=900)
    rest_curve = fit_readout(Etr[:, keep], ytr, Ete[:, keep], yte, steps=900)
    # collateral: accuracy on the eight classes that were not targeted
    forget.append(dict(r=int(r), pair_acc=pair_curve[-1], all_acc=rest_curve[-1]))
    print(f"[E6] delete {r:3d} pair-discriminative modes -> "
          f"{CLS[A_]}/{CLS[B2_]} refit {pair_curve[-1]:.2f}%, whole task {rest_curve[-1]:.2f}%")

# ── E5: cumulative deletion, top-down vs bottom-up ──
ks = [0, 1, 2, 4, 8, 16, 32, 64, 128, 192, 255]
top_path, bot_path = [], []
for j in ks:
    Ah = Ahat.copy(); Ah[:j] = 0
    top_path.append(round(acc_from(Ete, Ah, yte), 2))
    Ah = Ahat.copy()
    if j > 0:
        Ah[-j:] = 0
    bot_path.append(round(acc_from(Ete, Ah, yte), 2))
print(f"[E5] delete top 8: {top_path[4]}%   delete bottom 128: {bot_path[8]}%")

# ── live payload: 800 test images in mode coordinates ──
sub = np.random.default_rng(0).choice(len(yte), 800, replace=False)
sub.sort()
X14 = (Xte.reshape(-1, 28, 28).reshape(-1, 14, 2, 14, 2).mean((2, 4)) * 255).astype(np.uint8)
KMAX = len(Ahat)          # ship every mode: the panels cut the true tail
live = dict(
    classes=CLS, y=yte[sub].tolist(), kmax=KMAX,
    E=np.round(Ete[sub][:, :KMAX], 2).tolist(),
    Ahat=np.round(Ahat[:KMAX], 4).tolist(),
    base=np.round(base, 4).tolist(),
    tail=np.zeros((len(sub), 10)).tolist(),
    thumbs=[X14[i].reshape(-1).tolist() for i in sub],
    full_acc=round(FULL, 2))

numbers = dict(full_acc=round(FULL, 2), n_modes=int(len(Ahat)),
               forget=forget, forget_pair=[CLS[A_], CLS[B2_]],
               pred_r=round(rho, 3), pred_rank_r=round(srho, 3),
               controls=ctrl, top_path=top_path, bot_path=bot_path, ks=ks,
               modes=[{kk: r[kk] for kk in ("k", "acc", "drop", "lam_share", "worst_pair")}
                      for r in mode_rows])

for name, obj in [("surgery", dict(full_acc=FULL, classes=CLS, modes=mode_rows)),
                  ("predict", dict(scatter=scatter, classes=CLS, r=rho, rank_r=srho)),
                  ("recovery", dict(**recovery, forget=forget,
                                    forget_pair=[CLS[A_], CLS[B2_]])), ("live", live), ("numbers", numbers)]:
    p = os.path.join(OUT, f"{name}.json")
    with open(p, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    print(f"{name}.json  {os.path.getsize(p)/1e6:.2f} MB")
print(json.dumps({k: v for k, v in numbers.items() if k != "modes"}, indent=1))
