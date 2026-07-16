"""Render the companion GIFs for a-network-that-conserves-energy from the REAL
run bundle (scripts/results/kgl_blog-hamiltonian-v2). Every moving thing is a
number from the run: the trajectories are the exported rollouts, the heatmap is
the exported HNN's scalar, the clouds are the exported seed-0 nets stepped on
the exported test points, the boundary sweep re-runs those nets at each depth.

Outputs (public/):
  hamnet-pendulum-race.gif    both learned pendulums + true energy, drawing in time
  hamnet-level-set.gif        the learned H landscape, trajectory riding one stripe
  hamnet-state-cloud.gif      both classifiers' hidden state, layer by layer
  hamnet-boundary-sweep.gif   both decision boundaries as depth sweeps 4 -> 128
plus a -preview.png first frame for each.

Local run is fine: pure plotting from the downloaded JSON (no training).
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.join(HERE, "results", "kgl_blog-hamiltonian-v2")
PUB = os.path.join(HERE, "..", "public")

BG, FG, MUT = "#faf8f5", "#1c1c1c", "#8a8a8a"
ORANGE, BLUE, GREEN, PURPLE, GOLD = "#c2553a", "#4a7fb3", "#3a8f5e", "#9a4f9c", "#d99a2b"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG,
                     "axes.edgecolor": MUT, "text.color": FG,
                     "axes.labelcolor": FG, "xtick.color": MUT, "ytick.color": MUT,
                     "font.size": 11})


def find(name):
    for root, _, files in os.walk(BUNDLE):
        if name in files:
            return os.path.join(root, name)
    raise SystemExit(f"missing {name}")


def save(anim, fig, name, fps):
    path = os.path.join(PUB, name)
    anim.save(path, writer=PillowWriter(fps=fps))
    kb = os.path.getsize(path) / 1024
    print(f"  {name}: {kb:.0f} KB")
    fig.savefig(os.path.join(PUB, name.replace(".gif", "-preview.png")), dpi=80)
    plt.close(fig)


# ── numpy mirrors of the trained models ──────────────────────────────────────

def affine(x, wb):
    w, b = np.asarray(wb[0]), np.asarray(wb[1])
    return x @ w + b


def hnn_energy(m, qp):
    h = np.tanh(affine(qp, m[0])); h = np.tanh(affine(h, m[1]))
    return affine(h, m[2])[..., 0]


DIM, T_TOTAL = 4, 6.0


def grad_v(net, q):
    a1 = affine(q, net["b1"]); h = np.tanh(a1)
    w1, w2 = np.asarray(net["b1"][0]), np.asarray(net["b2"][0])
    gh = (1 - h * h) * w2.sum(axis=1)
    return gh @ w1.T


def depth_step(net, kind, h, q, p):
    if kind == "ham":
        p = p - 0.5 * h * grad_v(net, q)
        q = q + h * p
        p = p - 0.5 * h * grad_v(net, q)
    else:
        z = np.concatenate([q, p], axis=1)
        f = affine(np.tanh(affine(z, net["b1"])), net["b2"])
        z = z + h * f
        q, p = z[:, :DIM], z[:, DIM:]
    return q, p


def net_forward(net, kind, X, L):
    h = T_TOTAL / L
    z = affine(X, net["enc"])
    q, p = z[:, :DIM], z[:, DIM:]
    for _ in range(L):
        q, p = depth_step(net, kind, h, q, p)
    return affine(np.concatenate([q, p], axis=1), net["dec"])


# ── GIF 1: the pendulum race ─────────────────────────────────────────────────

def gif_race(phys):
    tr = phys["trace"]
    t = np.array(tr["t"])
    E0 = phys["E0"]
    F = 90                                        # frames
    idx = np.linspace(2, len(t) - 1, F).astype(int)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.4))
    fig.suptitle("Two networks learned the same pendulum. Only one keeps its energy.", fontsize=12)
    ax1.set(xlim=(-3.2, 3.2), ylim=(-2.6, 2.6), xlabel="angle q", ylabel="momentum p")
    ax2.set(xlim=(0, t[-1]), ylim=(0, max(max(tr["baseline_E"]), E0 * 2) * 1.05),
            xlabel="time (s)", ylabel="true energy")
    ax2.axhline(E0, color=MUT, ls="--", lw=1)
    ax2.text(t[-1] * 0.99, E0, " starting energy", color=MUT, fontsize=9, va="bottom", ha="right")
    lb, = ax1.plot([], [], color=ORANGE, lw=1.4, label="plain MLP field")
    lh, = ax1.plot([], [], color=BLUE, lw=1.4, label="HNN (learned energy)")
    db, = ax1.plot([], [], "o", color=ORANGE, ms=6)
    dh, = ax1.plot([], [], "o", color=BLUE, ms=6)
    eb, = ax2.plot([], [], color=ORANGE, lw=1.8)
    eh, = ax2.plot([], [], color=BLUE, lw=1.8)
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)

    def frame(f):
        k = idx[f]
        lb.set_data(tr["baseline_q"][:k], tr["baseline_p"][:k])
        lh.set_data(tr["hnn_q"][:k], tr["hnn_p"][:k])
        db.set_data([tr["baseline_q"][k - 1]], [tr["baseline_p"][k - 1]])
        dh.set_data([tr["hnn_q"][k - 1]], [tr["hnn_p"][k - 1]])
        eb.set_data(t[:k], tr["baseline_E"][:k])
        eh.set_data(t[:k], tr["hnn_E"][:k])
        return lb, lh, db, dh, eb, eh

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(FuncAnimation(fig, frame, frames=F, blit=True), fig, "hamnet-pendulum-race.gif", fps=12)


# ── GIF 2: riding a level set of the learned scalar ──────────────────────────

def gif_levelset(phys):
    m = phys["models"]["hnn"]
    Q, P = np.meshgrid(np.linspace(-3.2, 3.2, 160), np.linspace(-2.6, 2.6, 130))
    Hgrid = hnn_energy(m, np.stack([Q.ravel(), P.ravel()], 1)).reshape(Q.shape)
    tr = phys["trace"]
    F = 80
    idx = np.linspace(2, len(tr["hnn_q"]) - 1, F).astype(int)

    fig, ax = plt.subplots(figsize=(7.2, 5))
    ax.set(xlabel="angle q", ylabel="momentum p",
           title="The scalar the HNN learned; the motion rides one level set")
    cf = ax.contourf(Q, P, Hgrid, levels=14, cmap="cividis")
    ax.contour(Q, P, Hgrid, levels=14, colors="white", linewidths=0.4, alpha=0.4)
    fig.colorbar(cf, ax=ax, label="learned H(q, p)")
    ln, = ax.plot([], [], color=GOLD, lw=2.2)
    dd, = ax.plot([], [], "o", color=GOLD, ms=7, mec="#3a2a08")

    def frame(f):
        k = idx[f]
        ln.set_data(tr["hnn_q"][:k], tr["hnn_p"][:k])
        dd.set_data([tr["hnn_q"][k - 1]], [tr["hnn_p"][k - 1]])
        return ln, dd

    fig.tight_layout()
    save(FuncAnimation(fig, frame, frames=F, blit=True), fig, "hamnet-level-set.gif", fps=10)


# ── GIF 3: the hidden state through depth ────────────────────────────────────

def gif_cloud(nets):
    key = {r["dataset"] + "_" + r["kind"]: r for r in nets if r["L"] == 16}
    dsp, dsh = key["moons_plain"], key["moons_ham"]
    X = np.array(dsp["viz_data"]["X"][:200]); y = np.array(dsp["viz_data"]["y"][:200])
    L, h = 16, T_TOTAL / 16
    clouds = {}
    for name, row, kind in [("plain", dsp, "plain"), ("ham", dsh, "ham")]:
        net = row["model_seed0"]
        z = affine(X, net["enc"]); q, p = z[:, :DIM], z[:, DIM:]
        hist = [(q.copy(), p.copy())]
        for _ in range(L):
            q, p = depth_step(net, kind, h, q, p)
            hist.append((q.copy(), p.copy()))
        clouds[name] = hist

    fig, axes_ = plt.subplots(1, 2, figsize=(9, 4.6))
    fig.suptitle("The hidden state through depth: free composition vs leapfrog", fontsize=12)
    R = 6
    for ax, name in zip(axes_, ["plain residual net", "leapfrog net"]):
        ax.set(xlim=(-R, R), ylim=(-R, R), xlabel="q$_1$", ylabel="p$_1$", title=name)
    cols = np.where(y == 0, GREEN, PURPLE)
    sc0 = axes_[0].scatter([], [], s=10, c=[])
    sc1 = axes_[1].scatter([], [], s=10, c=[])
    txt = fig.text(0.5, 0.02, "", ha="center", fontsize=10, color=MUT)
    F = 3 * (16 + 1)

    # learned energy per layer for the ham net (the quantity only it possesses)
    net_h = dsh["model_seed0"]

    def ham_energy(q, p):
        h1 = np.tanh(affine(q, net_h["b1"]))
        V = affine(h1, net_h["b2"]).sum(axis=1)
        return float((0.5 * (p ** 2).sum(1) + V).mean())

    def frame(f):
        l = min(f // 3, 16)
        for sc, name in [(sc0, "plain"), (sc1, "ham")]:
            q, p = clouds[name][l]
            sc.set_offsets(np.stack([np.clip(q[:, 0], -R, R), np.clip(p[:, 0], -R, R)], 1))
            sc.set_color(cols)
        E = ham_energy(*clouds["ham"][l])
        txt.set_text(f"layer {l}/16    leapfrog learned energy {E:+.3f}    plain: undefined (no scalar)")
        return sc0, sc1, txt

    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    save(FuncAnimation(fig, frame, frames=F, blit=False), fig, "hamnet-state-cloud.gif", fps=6)


# ── GIF 4: decision boundaries as depth sweeps ───────────────────────────────

def gif_boundary(nets):
    key = {r["dataset"] + "_" + r["kind"]: r for r in nets if r["L"] == 16}
    dsp, dsh = key["spirals_plain"], key["spirals_ham"]
    X = np.array(dsp["viz_data"]["X"][:220]); y = np.array(dsp["viz_data"]["y"][:220])
    R, G = 2.2, 70
    gx, gy = np.meshgrid(np.linspace(-R, R, G), np.linspace(R, -R, G))
    grid = np.stack([gx.ravel(), gy.ravel()], 1)
    LS = [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    maps = []
    for L in LS:
        row = []
        for dsx, kind in [(dsp, "plain"), (dsh, "ham")]:
            lg = net_forward(dsx["model_seed0"], kind, grid, L)
            prob = 1 / (1 + np.exp(lg[:, 0] - lg[:, 1]))
            te = net_forward(dsx["model_seed0"], kind, X, L)
            acc = float(((te[:, 1] > te[:, 0]).astype(int) == y).mean())
            row.append((prob.reshape(G, G), acc))
        maps.append(row)
        print(f"    L={L}: plain {row[0][1]:.3f}, ham {row[1][1]:.3f}")

    fig, axes_ = plt.subplots(1, 2, figsize=(9, 4.8))
    fig.suptitle("Trained at depth 16, run at depth 4 to 128 (same weights)", fontsize=12)
    ims, tts = [], []
    for ax, name in zip(axes_, ["plain residual net", "leapfrog net"]):
        im = ax.imshow(maps[0][0][0], extent=(-R, R, -R, R), cmap="PRGn_r",
                       vmin=0, vmax=1, alpha=0.75)
        ax.scatter(X[:, 0], X[:, 1], s=7, c=np.where(y == 0, GREEN, PURPLE), edgecolors="none")
        tt = ax.set_title(name)
        ax.set_xticks([]); ax.set_yticks([])
        ims.append(im); tts.append(tt)

    # slow sweep: hold each depth for a few frames
    seq = [i for i in range(len(LS)) for _ in range(5)]

    def frame(f):
        i = seq[f]
        for j in range(2):
            ims[j].set_data(maps[i][j][0])
            name = ["plain residual net", "leapfrog net"][j]
            mark = "  << train depth" if LS[i] == 16 else ""
            tts[j].set_text(f"{name}  depth {LS[i]}  acc {maps[i][j][1] * 100:.1f}%{mark}")
        return ims + tts

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(FuncAnimation(fig, frame, frames=len(seq), blit=False), fig, "hamnet-boundary-sweep.gif", fps=5)


def main():
    phys = json.load(open(find("physics.json")))
    nets = json.load(open(find("networks.json")))
    print("rendering GIFs from the real run:")
    gif_race(phys)
    gif_levelset(phys)
    gif_cloud(nets)
    gif_boundary(nets)


if __name__ == "__main__":
    main()
