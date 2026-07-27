"""GIFs + PNGs for the lazy-training companion.

Every moving thing is a real process:
  lazy-gram.gif       a Gram matrix crystallizing as real random Yat units are
                      added one batch at a time (the Monte Carlo estimate
                      concentrating), on 30 real Fashion-MNIST images
  lazy-race.gif       two readouts training for real inside this script, on
                      frozen banks of 64 and 512 units: accuracy through SGD
                      steps + test thumbnails flipping as the big bank's
                      readout learns
  lazy-protos.gif     the run's actual prototype snapshots through training
                      (bundle kgl_blog-yatlazy-traj): a random-init row that
                      moves without ever becoming pictures, a data-seeded row
                      whose garments survive training
  lazy-travel.gif     per-epoch median prototype travel for the three trained
                      arms, drawn through real training time
  lazy-ladder.png     the width ladder scoreboard (static: ten widths are ten
                      facts)
  lazy-movement.png   the anti-lazy power law, log-log, with the fitted slope
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = os.path.dirname(os.path.abspath(__file__))
PUB = os.path.join(HERE, "..", "public")
TRAJ = os.path.join(HERE, "results", "kgl_blog-yatlazy-traj")
CURVES = json.load(open(os.path.join(HERE, "..", "public", "lazy-training", "curves.json")))

INK, BLUE, ORANGE, GREEN, PURPLE = "#222", "#4a7fb3", "#c2553a", "#3a8f5e", "#9a4f9c"
plt.rcParams.update({"figure.facecolor": "#faf8f5", "axes.facecolor": "#faf8f5",
                     "font.size": 11, "axes.edgecolor": "#bbb"})

B0 = EPS0 = 0.5


def fmnist(n_train=10000, n_test=2000):
    import torchvision
    tr = torchvision.datasets.FashionMNIST("/tmp/fmnist", train=True, download=True)
    te = torchvision.datasets.FashionMNIST("/tmp/fmnist", train=False, download=True)
    Xtr = (tr.data.numpy().astype(np.float32) / 255.0).reshape(-1, 784)[:n_train]
    ytr = tr.targets.numpy()[:n_train]
    Xte = (te.data.numpy().astype(np.float32) / 255.0).reshape(-1, 784)[:n_test]
    yte = te.targets.numpy()[:n_test]
    return Xtr, ytr, Xte, yte


def yat_feats(X, W):
    dot = X @ W.T
    d2 = (X ** 2).sum(1, keepdims=True) + (W ** 2).sum(1) - 2 * dot
    return (dot + B0) ** 2 / (np.maximum(d2, 0) + EPS0)


def gif_gram():
    Xtr, ytr, _, _ = fmnist()
    idx = np.concatenate([np.where(ytr == c)[0][:10] for c in (0, 1, 7)])
    X = Xtr[idx]                                     # 30 images, 3 classes
    rng = np.random.default_rng(0)
    M = 2048
    W = rng.normal(0, 1 / np.sqrt(784), (M, 784)).astype(np.float32)
    F = yat_feats(X, W)                              # (30, M)
    ms = np.unique(np.round(np.exp(np.linspace(np.log(1), np.log(M), 48))).astype(int))
    fig, ax = plt.subplots(figsize=(6.4, 5.6))

    def draw(f):
        m = ms[min(f, len(ms) - 1)]
        G = (F[:, :m] @ F[:, :m].T) / m
        d = np.sqrt(np.maximum(np.diag(G), 1e-9))
        Gn = G / np.outer(d, d)
        ax.clear()
        ax.imshow(Gn, vmin=0, vmax=1, cmap="magma")
        for k in (10, 20):
            ax.axhline(k - 0.5, color="#fff", lw=0.8)
            ax.axvline(k - 0.5, color="#fff", lw=0.8)
        ax.set_xticks([4.5, 14.5, 24.5]); ax.set_xticklabels(["T-shirt", "Trouser", "Sneaker"])
        ax.set_yticks([4.5, 14.5, 24.5]); ax.set_yticklabels(["T-shirt", "Trouser", "Sneaker"], rotation=90, va="center")
        ax.set_title(f"the kernel, estimated by {m} frozen random neuron{'s' if m > 1 else ''}", fontsize=12)
        ax.set_xlabel("normalized Gram matrix of 30 real images; class blocks emerge as samples accumulate", fontsize=9)

    anim = FuncAnimation(fig, draw, frames=len(ms) + 6, interval=170)
    out = os.path.join(PUB, "lazy-gram.gif")
    anim.save(out, writer=PillowWriter(fps=6))
    plt.close(fig)
    print(out, f"{os.path.getsize(out)/1e6:.2f} MB")


def train_readout(F, y, Fte, yte, steps, bs=256, lr=3e-3, snap_every=8):
    """A real linear softmax readout trained by adam inside this script."""
    m, k = F.shape[1], 10
    rng = np.random.default_rng(0)
    A = np.zeros((m, k)); b = np.zeros(k)
    mA = np.zeros_like(A); vA = np.zeros_like(A); mb = np.zeros(k); vb = np.zeros(k)
    hist = []
    t = 0
    for s in range(steps):
        idx = rng.integers(0, len(F), bs)
        lg = F[idx] @ A + b
        lg -= lg.max(1, keepdims=True)
        p = np.exp(lg); p /= p.sum(1, keepdims=True)
        p[np.arange(bs), y[idx]] -= 1
        gA = F[idx].T @ p / bs; gb = p.mean(0)
        t += 1
        for P, G, M_, V in ((A, gA, mA, vA), (b, gb, mb, vb)):
            M_ *= 0.9; M_ += 0.1 * G
            V *= 0.999; V += 0.001 * G * G
            P -= lr * (M_ / (1 - 0.9 ** t)) / (np.sqrt(V / (1 - 0.999 ** t)) + 1e-8)
        if s % snap_every == 0 or s == steps - 1:
            pred = (Fte @ A + b).argmax(1)
            hist.append((s, 100 * (pred == yte).mean(), pred.copy()))
    return hist


def gif_race():
    Xtr, ytr, Xte, yte = fmnist()
    rng = np.random.default_rng(1)
    banks = {64: rng.normal(0, 1 / np.sqrt(784), (64, 784)).astype(np.float32),
             512: rng.normal(0, 1 / np.sqrt(784), (512, 784)).astype(np.float32)}
    hists = {}
    for m, W in banks.items():
        F, Fte = yat_feats(Xtr, W), yat_feats(Xte, W)
        hists[m] = train_readout(F, ytr, Fte, yte, steps=400, lr=3e-2)
    nF = len(hists[64])
    thumbs = list(range(12))
    fig = plt.figure(figsize=(8.8, 4.6))
    axC = fig.add_axes([0.08, 0.14, 0.55, 0.74])
    axT = [fig.add_axes([0.68 + (i % 4) * 0.078, 0.60 - (i // 4) * 0.24, 0.07, 0.21])
           for i in range(12)]

    def draw(f):
        axC.clear()
        for m, color in ((64, BLUE), (512, ORANGE)):
            h = hists[m][:f + 1]
            axC.plot([p[0] for p in h], [p[1] for p in h], color=color, lw=2,
                     label=f"m = {m} frozen units")
        axC.set_xlim(0, 400); axC.set_ylim(30, 90)
        axC.set_xlabel("SGD steps on the readout (features frozen)")
        axC.set_ylabel("held-out accuracy (%)")
        axC.legend(loc="lower right", fontsize=9)
        axC.set_title("two readouts, trained for real, on two frozen banks", fontsize=12)
        _, acc, pred = hists[512][min(f, nF - 1)]
        for i, axi in enumerate(axT):
            axi.clear()
            axi.imshow(1 - Xte[thumbs[i]].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            good = pred[thumbs[i]] == yte[thumbs[i]]
            for sp in axi.spines.values():
                sp.set_color(GREEN if good else ORANGE); sp.set_linewidth(2.2)
            axi.set_xticks([]); axi.set_yticks([])
        axT[0].set_title("m = 512 verdicts", fontsize=9, loc="left")

    anim = FuncAnimation(fig, draw, frames=nF + 5, interval=140)
    out = os.path.join(PUB, "lazy-race.gif")
    anim.save(out, writer=PillowWriter(fps=7))
    plt.close(fig)
    print(out, f"{os.path.getsize(out)/1e6:.2f} MB")


def gif_protos():
    zr = np.load(os.path.join(TRAJ, "traj_yat_trained_m64_s0.npz"))
    zd = np.load(os.path.join(TRAJ, "traj_yat_data_trained_m64_s0.npz"))
    Sr, Sd = zr["snaps"].astype(np.float32), zd["snaps"].astype(np.float32)
    Mr = zr["moves"].astype(np.float32)
    Md = zd["moves"].astype(np.float32)
    eps_list = list(range(0, Sr.shape[0], 2)) + [Sr.shape[0] - 1]
    fig, axes = plt.subplots(2, 8, figsize=(8.2, 3.1), dpi=92)

    def draw(f):
        e = eps_list[min(f, len(eps_list) - 1)]
        for j in range(8):
            for row, S in ((0, Sr), (1, Sd)):
                ax = axes[row, j]
                ax.clear()
                img = S[e, j].reshape(28, 28)
                lo, hi = img.min(), img.max()
                ax.imshow((img - lo) / (hi - lo + 1e-9), cmap="gray_r" if row else "gray")
                ax.set_xticks([]); ax.set_yticks([])
        mv_r = Mr[max(0, e - 1)].mean() if len(Mr) else 0
        mv_d = Md[max(0, e - 1)].mean() if len(Md) else 0
        axes[0, 0].set_ylabel("random init", fontsize=9)
        axes[1, 0].set_ylabel("data init", fontsize=9)
        fig.suptitle(f"epoch {e}: the same training, two initializations  "
                     f"(mean travel: random {mv_r:.1f}x its init norm, data {mv_d:.2f}x)",
                     fontsize=11)

    anim = FuncAnimation(fig, draw, frames=len(eps_list) + 4, interval=300)
    out = os.path.join(PUB, "lazy-protos.gif")
    anim.save(out, writer=PillowWriter(fps=4))
    plt.close(fig)
    print(out, f"{os.path.getsize(out)/1e6:.2f} MB")


def gif_travel():
    arms = [("yat_trained", ORANGE, "yat, random init"),
            ("yat_data_trained", BLUE, "yat, data init"),
            ("rbf_data_trained", GREEN, "gaussian, data init")]
    data = {}
    for arm, _, _ in arms:
        z = np.load(os.path.join(TRAJ, f"traj_{arm}_m64_s0.npz"))
        data[arm] = (np.median(z["moves"].astype(np.float32), axis=1),
                     z["acc"])
    nE = len(data["yat_trained"][0])
    fig, (axM, axA) = plt.subplots(1, 2, figsize=(8.8, 4.2))

    def draw(f):
        e = min(f, nE - 1)
        for ax in (axM, axA):
            ax.clear()
        for arm, color, label in arms:
            mv, acc = data[arm]
            axM.plot(np.arange(1, e + 2), mv[:e + 1], color=color, lw=2, label=label)
            axA.plot(np.arange(1, e + 2), acc[:e + 1], color=color, lw=2)
        axM.set_xlim(1, nE); axM.set_ylim(0, max(3.5, data["yat_trained"][0].max() * 1.1))
        axA.set_xlim(1, nE); axA.set_ylim(60, 92)
        axM.set_title("median prototype travel (x init norm)", fontsize=11)
        axA.set_title("held-out accuracy", fontsize=11)
        axM.set_xlabel("epoch"); axA.set_xlabel("epoch")
        axM.legend(fontsize=8, loc="upper left")

    anim = FuncAnimation(fig, draw, frames=nE + 5, interval=220)
    out = os.path.join(PUB, "lazy-travel.gif")
    anim.save(out, writer=PillowWriter(fps=5))
    plt.close(fig)
    print(out, f"{os.path.getsize(out)/1e6:.2f} MB")


def png_ladder():
    w = CURVES["widths"]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for arm, color, label, dash in [
            ("yat_frozen", BLUE, "yat, frozen random", "-"),
            ("yat_data", GREEN, "yat, frozen data-seeded", "-"),
            ("relu_frozen", PURPLE, "relu, frozen random", "-"),
            ("rbf_data", "#888", "gaussian, frozen data-seeded", "-"),
            ("yat_trained", ORANGE, "yat, everything trains", "--")]:
        a = CURVES["arms"][arm]
        ax.errorbar(w, a["mean"], yerr=a["std"], color=color, lw=2, ls=dash,
                    capsize=2, label=label)
    for v, lab in [(79, "raw-pixel head 79"), (83.3, "hand-built 83.3"), (85.7, "trained backbone 85.7")]:
        ax.axhline(v, color="#999", ls=":", lw=1)
        ax.text(w[-1], v + 0.2, lab, ha="right", fontsize=8, color="#666")
    ax.set_xscale("log"); ax.set_xticks(w[::2]); ax.set_xticklabels([str(x) for x in w[::2]])
    ax.set_xlabel("hidden units m"); ax.set_ylabel("held-out accuracy (%)")
    ax.set_ylim(58, 91)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("the width ladder: five arms, three seeds each, per-arm bracketed rates", fontsize=12)
    out = os.path.join(PUB, "lazy-ladder.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(out, f"{os.path.getsize(out)/1e6:.2f} MB")


def png_movement():
    w = np.array(CURVES["widths"], float)
    mv = np.array(CURVES["arms"]["yat_trained"]["move"], float)
    slope = CURVES["numbers"]["movement_slope"]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.loglog(w, mv, "o-", color=ORANGE, lw=2, label="measured travel")
    ref = mv[0] * (w / w[0]) ** 0.5
    ax.loglog(w, ref, ":", color="#888", label="a pure square-root law")
    ax.set_xlabel("hidden units m"); ax.set_ylabel("||W_end - W_0|| / ||W_0||")
    ax.set_title(f"the anti-lazy law: travel grows like m^{slope}", fontsize=12)
    ax.legend(fontsize=9)
    out = os.path.join(PUB, "lazy-movement.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(out, f"{os.path.getsize(out)/1e6:.2f} MB")


if __name__ == "__main__":
    import sys
    only = sys.argv[1] if len(sys.argv) > 1 else None
    todo = {"gram": gif_gram, "race": gif_race, "protos": gif_protos,
            "travel": gif_travel, "ladder": png_ladder, "movement": png_movement}
    for name, fn in todo.items():
        if only and name != only:
            continue
        try:
            fn()
        except FileNotFoundError as e:
            print(f"[skip {name}] {e}")
