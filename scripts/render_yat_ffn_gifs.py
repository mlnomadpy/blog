#!/usr/bin/env python3
"""Render the GIFs for the white-box FFN companion (mlp-block-is-a-representer-theorem-jax-flax-nnx).

Loads the trained Yat-FFN char-transformer that scripts/yat_ffn_whitebox.py exported to
public/mlp-representer/ (model.json manifest + weights.bin float32 blob) and re-runs the
exact numpy forward pass that the whitebox script asserted matches Flax to <1e-3. Every
pixel that moves is a real number from that trained model: real generations, real kernel
weights, real slot contributions. Nothing is retrained and nothing is fabricated.

GIFs (all written to public/):
  yat-ffn-memory.gif      READ:      the 256 memory slots as a 16x16 grid lighting up,
                                     slot by slot, during a real generation.
  yat-ffn-attribution.gif ATTRIBUTE: one next-token logit assembled slot by slot, with
                                     the remaining gap ticking down to 0.00 (exact sum).
  yat-ffn-gain-sweep.gif  EDIT:      the gain sweep on the newline slot unrolled; a real
                                     200-char generation types out at every gain while
                                     the newline fraction curve draws itself.
  yat-ffn-abstain.gif     ABSTAIN:   a string that degrades from Shakespeare into random
                                     characters; each token tints by its live peak kernel
                                     weight as a readout marches along.

Run: python3 scripts/render_yat_ffn_gifs.py [memory|attribution|gainsweep|abstain ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "public"
EXPORT = PUB / "mlp-representer"

# palette (matches the companion post's dark figure theme)
BG, PANEL, INK, MUTED, LINE = "#0e0d0b", "#16140f", "#e8e2d4", "#9a9282", "#3a352c"
POS, NEG, GOOD = "#b3661b", "#4a7fb3", "#7bbf5a"
DIM = "#4a4538"

plt.rcParams.update({
    "figure.facecolor": BG, "savefig.facecolor": BG, "text.color": INK,
    "axes.edgecolor": LINE, "font.size": 10,
})

# ── load the export ──────────────────────────────────────────────────────────
MDL = json.loads((EXPORT / "model.json").read_text())
V, T, D, HEADS, FF, LAYERS = MDL["V"], MDL["T"], MDL["D"], MDL["heads"], MDL["ff"], MDL["layers"]
LAST = MDL["lastLayer"]
ITOS = MDL["itos"]
STOI = {c: i for i, c in enumerate(ITOS)}

raw = np.fromfile(EXPORT / "weights.bin", dtype=np.float32)
tensors, off = {}, 0
for t in MDL["tensors"]:
    n = int(np.prod(t["shape"]))
    tensors[t["path"]] = raw[off:off + n].reshape(t["shape"]).astype(np.float64)
    off += n
assert off == raw.size, f"weights.bin size mismatch: consumed {off} of {raw.size}"

P = {
    "tok": tensors["tok"], "pos": tensors["pos"], "head": tensors["head"],
    "lnf": (tensors["lnf.scale"], tensors["lnf.bias"]),
    "blocks": [
        {
            "ln1": (tensors[f"b{L}.ln1.scale"], tensors[f"b{L}.ln1.bias"]),
            "ln2": (tensors[f"b{L}.ln2.scale"], tensors[f"b{L}.ln2.bias"]),
            "attn": {k: tensors[f"b{L}.attn.{k}"]
                     for k in ["q_k", "q_b", "k_k", "k_b", "v_k", "v_b", "o_k", "o_b"]},
            "ffn": {"W": tensors[f"b{L}.ffn.W"], "Vv": tensors[f"b{L}.ffn.Vv"],
                    "b": MDL["ffnScalars"][L]["b"], "eps": MDL["ffnScalars"][L]["eps"]},
        }
        for L in range(LAYERS)
    ],
}
U = P["head"]                                   # [D, V] unembedding
VVALS = P["blocks"][LAST]["ffn"]["Vv"]          # [D, FF] last-layer values

# ── the numpy forward pass (the whitebox script asserted this matches Flax) ──
HD = D // HEADS
SCALE = 1.0 / np.sqrt(HD)


def ln(x, g):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-6) * g[0] + g[1]


def mha(x, a):
    q = np.einsum("td,dhk->thk", x, a["q_k"]) + a["q_b"]
    k = np.einsum("td,dhk->thk", x, a["k_k"]) + a["k_b"]
    v = np.einsum("td,dhk->thk", x, a["v_k"]) + a["v_b"]
    s = np.einsum("qhk,shk->hqs", q, k) * SCALE
    Tn = x.shape[0]
    m = np.triu(np.ones((Tn, Tn), bool), 1)
    s = np.where(m[None], -1e30, s)
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s)
    aw = e / e.sum(-1, keepdims=True)
    o = np.einsum("hqs,shk->qhk", aw, v)
    return np.einsum("thk,hkd->td", o, a["o_k"]) + a["o_b"]


def yat_kw(x, f):                                # k(W_u, x): the memory weights [T, FF]
    dot = x @ f["W"].T
    d2 = (x ** 2).sum(-1, keepdims=True) + (f["W"] ** 2).sum(-1) - 2 * dot
    return (dot + f["b"]) ** 2 / (d2 + f["eps"])


def forward(prm, ids, kw_layer=None):
    """Logits [T, V]; if kw_layer is set, also that layer's kernel weights [T, FF]."""
    ids = np.asarray(ids)
    x = prm["tok"][ids] + prm["pos"][: len(ids)]
    kw_out = None
    for L in range(LAYERS):
        bl = prm["blocks"][L]
        x = x + mha(ln(x, bl["ln1"]), bl["attn"])
        h = ln(x, bl["ln2"])
        kw = yat_kw(h, bl["ffn"])
        if L == kw_layer:
            kw_out = kw
        x = x + kw @ bl["ffn"]["Vv"].T
    x = ln(x, prm["lnf"])
    return x @ prm["head"], kw_out


def residual_into_ffn(prm, ids, layer):          # x fed to the FFN of `layer` (post-LN)
    ids = np.asarray(ids)
    x = prm["tok"][ids] + prm["pos"][: len(ids)]
    for L in range(LAYERS):
        bl = prm["blocks"][L]
        x = x + mha(ln(x, bl["ln1"]), bl["attn"])
        h = ln(x, bl["ln2"])
        if L == layer:
            return h
        x = x + yat_kw(h, bl["ffn"]) @ bl["ffn"]["Vv"].T


def encode(s):
    return [STOI[c] for c in s]


def generate(prm, prompt, n=200, temp=0.8, seed=0, track_kw=False):
    """Real autoregressive sampling (same sampler as yat_ffn_whitebox.py)."""
    r = np.random.RandomState(seed)
    idx = encode(prompt)
    kws = []
    for _ in range(n):
        ctx = idx[-T:]
        logits, kw = forward(prm, ctx, kw_layer=(LAST if track_kw else None))
        if track_kw:
            kws.append(kw[-1])
        lg = logits[-1] / temp
        p = np.exp(lg - lg.max())
        p /= p.sum()
        idx.append(int(r.choice(V, p=p)))
    text = "".join(ITOS[i] for i in idx)
    return (text, np.asarray(kws)) if track_kw else text


def amplified(u, g, layer=LAST):                 # a copy of P with v_u -> g * v_u
    prm = {**P, "blocks": [dict(b) for b in P["blocks"]]}
    f = dict(prm["blocks"][layer]["ffn"])
    Vv = f["Vv"].copy()
    Vv[:, u] *= g
    f["Vv"] = Vv
    prm["blocks"][layer] = {**prm["blocks"][layer], "ffn": f}
    return prm


def slot_writes(u, k=6):                         # logit-lens label of slot u (last layer)
    return [ITOS[i] for i in np.argsort(-(VVALS[:, u] @ U))[:k]]


def glyph(c):                                    # printable stand-ins for whitespace
    return {"\n": "¶", " ": " "}.get(c, c)


def lerp_color(c0, c1, t):
    a, b = np.array(to_rgb(c0)), np.array(to_rgb(c1))
    return tuple(a + (b - a) * float(np.clip(t, 0, 1)))


def ease(t):
    return t * t * (3 - 2 * t)


def grab(fig):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return rgba


def save_gif(name, frames, fps, hold=12):
    out = PUB / f"{name}.gif"
    # inspection frames (not shipped)
    for tag, fr in [("early", frames[max(2, len(frames) // 8)]),
                    ("mid", frames[len(frames) // 2]), ("late", frames[-1])]:
        Image.fromarray(fr).save(f"/tmp/{name}-{tag}.png")
    frames = frames + [frames[-1]] * hold
    imageio.mimsave(out, frames, duration=1 / fps, loop=0,
                    palettesize=128, subrectangles=True)
    kb = out.stat().st_size / 1024
    print(f"wrote {out.name}: {len(frames)} frames, {kb:.0f} KB")
    return out


# ── sanity: the loaded export reproduces the whitebox script's numbers ───────
def sanity():
    for u in [198, 145, 248]:
        print(f"  slot {u:3d} writes {slot_writes(u)}   (manifest says {MDL['slotWrites'][u]})")
        assert slot_writes(u) == MDL["slotWrites"][u], "slot labels drifted from export"
    u_nl = int(np.argmax(VVALS.T @ U[:, STOI["\n"]]))
    print(f"  newline slot = {u_nl} (post says 248)")
    x = residual_into_ffn(P, encode("ROMEO:\nWhat"), LAST)[-1]
    kw = yat_kw(x[None], P["blocks"][LAST]["ffn"])[0]
    out = kw @ VVALS.T
    nxt = int((U.T @ out).argmax())
    push = kw * (VVALS.T @ U[:, nxt])
    top = np.argsort(-push)[:3]
    print(f"  attribution: FFN most promotes {ITOS[nxt]!r}; top slots {list(map(int, top))} "
          f"(post says 181, 202, 4)")
    return u_nl


# ═════════════════════════════════════════════════════════════════════════════
# GIF 1: READ. The 256 slots as a memory grid, read live during generation.
# ═════════════════════════════════════════════════════════════════════════════
def gif_memory():
    prompt = "ROMEO:\n"
    n_gen = 64
    text, kws = generate(P, prompt, n=n_gen, temp=0.8, seed=0, track_kw=True)
    body = text[len(prompt):]
    print(f"  generation: {text[:60]!r}...")
    kmax_all = float(kws.max())

    Wpx, Hpx = 680, 570
    frames = []
    for i in range(n_gen):
        kw = kws[i]                               # weights that produced character body[i]
        grid = kw.reshape(16, 16)
        top = int(kw.argmax())
        fig = plt.figure(figsize=(Wpx / 100, Hpx / 100), dpi=100, facecolor=BG)
        fig.text(0.5, 0.955, "Reading the memory, one character at a time",
                 ha="center", color=INK, fontsize=15, weight="bold")
        fig.text(0.5, 0.912,
                 "each cell is one (key, value) slot; brightness is its live weight k(Wᵤ, x)",
                 ha="center", color=MUTED, fontsize=10)

        ax = fig.add_axes([0.09, 0.24, 0.55, 0.62])
        ax.set_facecolor(PANEL)
        # gamma-brightened display of the real weights, quantized to 14 shades so the
        # GIF palette stays small (values in the readouts are raw)
        disp = np.round(((grid / kmax_all) ** 0.45) * 13) / 13
        ax.imshow(disp,
                  cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list(
                      "mem", [PANEL, POS, "#f0c98a"]), vmin=0, vmax=1,
                  origin="upper", interpolation="nearest")
        ty, tx = divmod(top, 16)
        ax.add_patch(plt.Rectangle((tx - 0.5, ty - 0.5), 1, 1, fill=False,
                                   ec=GOOD, lw=2.0))
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(LINE)
        ax.set_title("the last block's 256 memory slots", color=MUTED, fontsize=9.5, pad=6)

        # readout: the strongest slot and its logit-lens label
        wt = " ".join(repr(c) for c in slot_writes(top, 4))
        fig.text(0.68, 0.76, "strongest slot", color=MUTED, fontsize=9.5)
        fig.text(0.68, 0.70, f"slot {top}", color=GOOD, fontsize=15, weight="bold",
                 family="monospace")
        fig.text(0.68, 0.645, f"k = {kw[top]:.2f}", color=INK, fontsize=11,
                 family="monospace")
        fig.text(0.68, 0.575, "it writes:", color=MUTED, fontsize=9.5)
        fig.text(0.68, 0.52, wt, color=POS, fontsize=10.5, family="monospace")
        fig.text(0.68, 0.42, "slots lit above\nhalf the peak:", color=MUTED, fontsize=9.5)
        fig.text(0.68, 0.365, f"{int((kw > 0.5 * kw.max()).sum())} / 256",
                 color=INK, fontsize=13, family="monospace")

        # the generation so far, typed along the bottom
        shown = prompt + body[: i + 1]
        cols = 40
        axb = fig.add_axes([0.06, 0.055, 0.88, 0.145])
        axb.set_facecolor(PANEL)
        axb.set_xlim(-0.6, cols)
        axb.set_ylim(2.7, -0.7)
        axb.set_xticks([])
        axb.set_yticks([])
        for sp in axb.spines.values():
            sp.set_color(LINE)
        rows_txt = [shown[j:j + cols] for j in range(0, len(shown), cols)][-3:]
        for li, line in enumerate(rows_txt):
            for k, c in enumerate(line):
                last = (li == len(rows_txt) - 1) and (k == len(line) - 1)
                axb.text(k, li, glyph(c), color=(GOOD if last else INK),
                         fontsize=9.5, family="monospace", ha="center", va="center",
                         weight="bold" if last else "normal")
        axb.set_title("a real sample from the trained model, one grid readout per new character",
                      color=MUTED, fontsize=8.2, pad=4)
        frames.append(grab(fig))
        if (i + 1) % 20 == 0:
            print(f"  memory: {i + 1}/{n_gen} frames")
    save_gif("yat-ffn-memory", frames, fps=6, hold=10)


# ═════════════════════════════════════════════════════════════════════════════
# GIF 2: ATTRIBUTE. One next-token logit assembled slot by slot, gap -> 0.00.
# ═════════════════════════════════════════════════════════════════════════════
def gif_attribution():
    prompt = "ROMEO:\nWhat"
    x = residual_into_ffn(P, encode(prompt), LAST)[-1]
    kw = yat_kw(x[None], P["blocks"][LAST]["ffn"])[0]
    out = kw @ VVALS.T
    nxt = int((U.T @ out).argmax())
    target = float(out @ U[:, nxt])
    push = kw * (VVALS.T @ U[:, nxt])            # slot u's exact push toward token nxt
    order = np.argsort(-push)
    NTOP = 9
    tops = order[:NTOP]
    rest = float(push[order[NTOP:]].sum())
    print(f"  attribution: next char {ITOS[nxt]!r}, target logit push {target:.3f}, "
          f"top slot pushes {[round(float(push[u]), 2) for u in tops[:3]]}, rest {rest:.3f}")

    labels = [f"slot {u:3d}  k={kw[u]:5.2f}  writes " +
              " ".join(glyph(c) for c in slot_writes(u, 4)) for u in tops]
    labels.append(f"the other {FF - NTOP} slots, summed")
    values = [float(push[u]) for u in tops] + [rest]
    n_bars = len(values)
    xmax = max(max(values), target) * 1.18
    xmin = min(0.0, min(values) * 1.3) - 0.02 * xmax

    Wpx, Hpx = 880, 560
    frames = []
    GROW = 6                                     # frames per bar
    for step in range(n_bars * GROW + 8):
        if step < 8:                             # intro: context + empty axes
            done, t = -1, 0.0
        else:
            done, t = divmod(step - 8, GROW)
            t = ease((t + 1) / GROW)
        fig = plt.figure(figsize=(Wpx / 100, Hpx / 100), dpi=100, facecolor=BG)
        fig.text(0.5, 0.955, "One next-token logit, assembled slot by slot",
                 ha="center", color=INK, fontsize=15, weight="bold")
        fig.text(0.5, 0.909,
                 f"context …'ROMEO:¶What'  →  the FFN most promotes '{ITOS[nxt]}';  "
                 "each bar is one slot's exact share, k(Wᵤ, x)·(vᵤ·U[:,'e'])",
                 ha="center", color=MUTED, fontsize=9.5)

        ax = fig.add_axes([0.40, 0.14, 0.55, 0.70])
        ax.set_facecolor(PANEL)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(n_bars - 0.4, -0.6)
        ax.set_yticks([])
        ax.tick_params(colors=MUTED, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(LINE)
        ax.axvline(0, color=LINE, lw=0.8)
        ax.axvline(target, color=GOOD, lw=1.4, ls="--")
        ax.text(target, -0.52, f" full FFN push {target:.2f}", color=GOOD, fontsize=8.5,
                ha="left", va="bottom")

        run = 0.0
        for j in range(n_bars):
            if j < done:
                w = values[j]
            elif j == done:
                w = values[j] * t
            else:
                w = 0.0
            run += w
            c = POS if j < NTOP else NEG
            if j <= done:
                ax.barh(j, w, height=0.62, color=c, alpha=0.9 if j <= done else 0.0)
                ax.text(max(w, 0) + 0.015 * xmax, j, f"{values[j]:+.2f}" if j < done or t == 1.0 else "",
                        color=INK, fontsize=8, va="center", family="monospace")
            fig.text(0.045, 0.14 + 0.70 * (1 - (j + 0.6) / (n_bars + 0.2)),
                     labels[j], color=(INK if j <= done else DIM), fontsize=8.6,
                     family="monospace", va="center")

        gap = abs(target - run)
        fig.text(0.40, 0.055, "running sum of slot pushes:", color=MUTED, fontsize=9.5)
        fig.text(0.615, 0.055, f"{run:+.2f}", color=POS, fontsize=11, weight="bold",
                 family="monospace")
        fig.text(0.70, 0.055, "remaining gap:", color=MUTED, fontsize=9.5)
        fig.text(0.835, 0.055,
                 f"{gap:.2f}" if done < n_bars - 1 or t < 1.0 else "0.00  (exact)",
                 color=(GOOD if gap < 0.005 else INK), fontsize=11, weight="bold",
                 family="monospace")
        frames.append(grab(fig))
    print(f"  attribution: final gap {abs(target - sum(values)):.2e}")
    save_gif("yat-ffn-attribution", frames, fps=7, hold=16)


# ═════════════════════════════════════════════════════════════════════════════
# GIF 3: EDIT. The gain sweep on the newline slot, unrolled into its process.
# ═════════════════════════════════════════════════════════════════════════════
def gif_gainsweep(u_nl):
    prompt = "GLOUCESTER:\n"
    gains = [1, 2, 3, 4, 6, 8, 11, 15, 20, 26, 34]
    gens, fracs = [], []
    for g in gains:
        prm = P if g == 1 else amplified(u_nl, g)
        body = generate(prm, prompt, n=200, temp=0.8, seed=3)[len(prompt):]
        gens.append(body)
        fracs.append(body.count("\n") / len(body))
        print(f"  gain x{g:<3} newline fraction {fracs[-1]:.2f}  {body[:40]!r}")

    Wpx, Hpx = 880, 620
    cols, rows = 50, 4
    TYPE_F, HOLD_F = 7, 3                        # frames typing + holding per gain
    per = TYPE_F + HOLD_F
    frames = []
    ymax = 1.03
    for gi, g in enumerate(gains):
        body, frac = gens[gi], fracs[gi]
        for f in range(per):
            shown = min(len(body), int(np.ceil(len(body) * min(1.0, (f + 1) / TYPE_F))))
            fig = plt.figure(figsize=(Wpx / 100, Hpx / 100), dpi=100, facecolor=BG)
            fig.text(0.5, 0.958, "Turn up one readable slot and watch the model change its mind",
                     ha="center", color=INK, fontsize=15, weight="bold")
            fig.text(0.5, 0.916,
                     f"slot {u_nl} writes the newline; set vᵤ → g·vᵤ "
                     "and regenerate 200 characters (real samples, one per gain)",
                     ha="center", color=MUTED, fontsize=9.5)

            # text panel
            axt = fig.add_axes([0.06, 0.42, 0.88, 0.44])
            axt.set_facecolor(PANEL)
            axt.set_xlim(0, cols)
            axt.set_ylim(rows + 1.4, -0.6)
            axt.set_xticks([])
            axt.set_yticks([])
            for sp in axt.spines.values():
                sp.set_color(LINE)
            axt.text(0.4, 0.0, "".join(glyph(c) for c in prompt), color=MUTED,
                     fontsize=10, family="monospace", va="center")
            nl_typed = 0
            for ci in range(shown):
                c = body[ci]
                r, k = divmod(ci, cols)
                if r >= rows:
                    break
                is_nl = c == "\n"
                nl_typed += is_nl
                axt.text(0.4 + k, 1.0 + r, glyph(c),
                         color=(POS if is_nl else INK),
                         fontsize=10, family="monospace", va="center",
                         weight="bold" if is_nl else "normal")
            label = ("coherent" if frac < 0.15 else
                     "choppy" if frac < 0.5 else "flooded")
            axt.text(cols - 0.5, 0.0, f"gain x{g}   {label}", color=POS, fontsize=11,
                     weight="bold", family="monospace", ha="right", va="center")
            fig.text(0.06, 0.375, f"newlines typed so far: {nl_typed} / {shown}",
                     color=MUTED, fontsize=9.5, family="monospace")

            # the curve, drawing itself
            axc = fig.add_axes([0.09, 0.09, 0.85, 0.24])
            axc.set_facecolor(PANEL)
            axc.set_xlim(0, gains[-1] * 1.04)
            axc.set_ylim(-0.04, ymax)
            axc.tick_params(colors=MUTED, labelsize=8)
            for sp in axc.spines.values():
                sp.set_color(LINE)
            axc.set_xlabel(f"gain on slot {u_nl}", color=MUTED, fontsize=9)
            axc.set_ylabel("newline fraction", color=MUTED, fontsize=9)
            tt = ease(min(1.0, (f + 1) / TYPE_F))
            if gi > 0:
                axc.plot(gains[:gi + 1], fracs[:gi + 1], "-o", color=POS, lw=2.0,
                         ms=4, mec=BG, clip_on=False)
                px = gains[gi - 1] + (gains[gi] - gains[gi - 1]) * tt
                py = fracs[gi - 1] + (fracs[gi] - fracs[gi - 1]) * tt
            else:
                px, py = gains[0] * tt + 0.0, fracs[0] * tt
                axc.plot(gains[:1], fracs[:1], "o", color=POS, ms=4, mec=BG)
            if gi > 0:
                axc.plot([gains[gi - 1], px], [fracs[gi - 1], py], "-", color=POS, lw=2.0)
            axc.plot([px], [py], "o", color=GOOD, ms=7, mec=BG, zorder=5)
            frames.append(grab(fig))
        print(f"  gainsweep: gain x{g} done ({len(frames)} frames)")
    save_gif("yat-ffn-gain-sweep", frames, fps=7, hold=14)


# ═════════════════════════════════════════════════════════════════════════════
# GIF 4: ABSTAIN. Prose degrades into gibberish; the memory goes dark, live.
# ═════════════════════════════════════════════════════════════════════════════
def gif_abstain():
    # the opening of tinyshakespeare, the training corpus, sliced to 64 chars
    shake = ("First Citizen:\nBefore we proceed any further, hear me speak.\n\n"
             "All:\nSpeak, speak.\n")[:64]
    assert len(shake) == 64
    rr = np.random.RandomState(0)
    rand_ids = rr.randint(0, V, size=64)
    ids = encode(shake) + [int(i) for i in rand_ids]
    seq = shake + "".join(ITOS[i] for i in rand_ids)
    x = residual_into_ffn(P, ids, LAST)          # [128, D] real FFN inputs
    kw = yat_kw(x, P["blocks"][LAST]["ffn"])     # [128, 256]
    peak = kw.max(1)
    m_in, m_ood = peak[:64].mean(), peak[64:].mean()
    print(f"  abstain: mean peak weight, Shakespeare {m_in:.2f} vs random {m_ood:.2f}")
    vmax = float(peak.max())
    vref = float(np.percentile(peak, 85))        # display reference so tints are visible

    def shade(v):                                # peak weight -> color
        return lerp_color("#2c3a4c", GOOD, v / vref)

    Wpx, Hpx = 880, 560
    cols = 32
    frames = []
    N = len(seq)
    for i in range(N):
        fig = plt.figure(figsize=(Wpx / 100, Hpx / 100), dpi=100, facecolor=BG)
        fig.text(0.5, 0.955, "The memory goes dark when the prose does",
                 ha="center", color=INK, fontsize=15, weight="bold")
        fig.text(0.5, 0.912,
                 "each character tints by its peak memory weight, maxᵤ k(Wᵤ, x), "
                 "from a real forward pass",
                 ha="center", color=MUTED, fontsize=9.5)

        # the string, tinted as the readout passes
        axs = fig.add_axes([0.07, 0.50, 0.86, 0.36])
        axs.set_facecolor(PANEL)
        axs.set_xlim(-0.6, cols)
        axs.set_ylim(3.9, -0.9)
        axs.set_xticks([])
        axs.set_yticks([])
        for sp in axs.spines.values():
            sp.set_color(LINE)
        for ci, c in enumerate(seq):
            r, k = divmod(ci, cols)
            if ci <= i:
                col = shade(peak[ci])
                w = "bold" if ci == i else "normal"
            else:
                col, w = DIM, "normal"
            axs.text(k, r, glyph(c), color=col, fontsize=11.5, family="monospace",
                     ha="center", va="center", weight=w)
        cy, cx = divmod(i, cols)
        axs.add_patch(plt.Rectangle((cx - 0.5, cy - 0.5), 1.0, 1.0, fill=False,
                                    ec=INK, lw=1.1))
        axs.text(7.5, -0.72, "real Shakespeare", color=GOOD, fontsize=8.5, ha="center")
        axs.text(23.5, -0.72, "random characters", color=NEG, fontsize=8.5, ha="center")

        # the peak-weight trace, drawing itself under the string
        axc = fig.add_axes([0.09, 0.115, 0.84, 0.30])
        axc.set_facecolor(PANEL)
        axc.set_xlim(0, N - 1)
        axc.set_ylim(0, vmax * 1.08)
        axc.tick_params(colors=MUTED, labelsize=8)
        for sp in axc.spines.values():
            sp.set_color(LINE)
        axc.axvline(63.5, color=LINE, lw=1.0, ls="--")
        axc.set_xlabel("position in the string", color=MUTED, fontsize=9)
        axc.set_ylabel("peak memory weight", color=MUTED, fontsize=9)
        xs = np.arange(i + 1)
        axc.plot(xs[: min(i + 1, 64)], peak[: min(i + 1, 64)], "-", color=GOOD, lw=1.8)
        if i >= 64:
            axc.plot(np.arange(63, i + 1), peak[63: i + 1], "-", color=NEG, lw=1.8)
        axc.plot([i], [peak[i]], "o", color=INK, ms=5, mec=BG, zorder=5)
        msg = f"mean so far: Shakespeare {peak[:min(i + 1, 64)].mean():.2f}"
        if i >= 64:
            msg += f"   random {peak[64:i + 1].mean():.2f}"
        axc.text(0.98, 0.955, msg, transform=axc.transAxes, color=INK, fontsize=9.5,
                 family="monospace", va="top", ha="right",
                 bbox=dict(boxstyle="round,pad=0.3", fc=PANEL, ec=LINE, lw=0.8))
        frames.append(grab(fig))
        if (i + 1) % 32 == 0:
            print(f"  abstain: {i + 1}/{N} frames")
    save_gif("yat-ffn-abstain", frames, fps=9, hold=14)


if __name__ == "__main__":
    print("sanity: the loaded export reproduces the whitebox run")
    u_nl = sanity()
    want = set(sys.argv[1:]) or {"memory", "attribution", "gainsweep", "abstain"}
    if "memory" in want:
        gif_memory()
    if "attribution" in want:
        gif_attribution()
    if "gainsweep" in want:
        gif_gainsweep(u_nl)
    if "abstain" in want:
        gif_abstain()
