#!/usr/bin/env python3
"""Softmax vs L1-normalized sphere classifier, head-to-head.

Same encoder, same optimizer, same data; only the classification head differs.
Embeddings z and class prototypes w_c live on the unit sphere, so the score is
s = cos(z, w_c) in [-1, 1]. We compare training objectives:

  softmax     : p = softmax(s / tau)                    (exp + L1-normalize)
  L1-relu^b   : p = relu(s)^b      / sum                (no exp; sharpness via power b)
  L1-(1+cos)^b: p = ((1+s)/2)^b    / sum                (cosFormer-style shift)
  L1-yat      : p = s^2/(2-2s+eps) / sum                (the Yat kernel, normalized)

Loss is NLL = -log p_true in every case. argmax_c p == argmax_c s for all heads,
so any accuracy gap comes from how each *objective trains the encoder*, which is
exactly the question. We sweep each head's sharpness param and report the best.
"""
from __future__ import annotations
import jax, jax.numpy as jnp, optax, numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

jax.config.update("jax_enable_x64", False)

D_EMB, HID, STEPS, LR = 16, 128, 900, 3e-3
EPS = 1e-6

def load():
    X, y = load_digits(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    sc = StandardScaler().fit(Xtr)
    return (jnp.asarray(sc.transform(Xtr), jnp.float32), jnp.asarray(ytr),
            jnp.asarray(sc.transform(Xte), jnp.float32), jnp.asarray(yte), int(y.max() + 1))

def init(key, d_in, n_cls):
    k1, k2, k3 = jax.random.split(key, 3)
    he = lambda k, a, b: jax.random.normal(k, (a, b)) * np.sqrt(2.0 / a)
    return dict(W1=he(k1, d_in, HID), b1=jnp.zeros(HID),
                W2=he(k2, HID, D_EMB), b2=jnp.zeros(D_EMB),
                P=jax.random.normal(k3, (n_cls, D_EMB)))

def l2(x): return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

def scores(p, X):
    z = l2(jax.nn.relu(X @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"])
    return z @ l2(p["P"]).T                       # cos(z, w_c) in [-1, 1]

def head_probs(s, kind, hp):
    if kind == "softmax":
        return jax.nn.softmax(s / hp, axis=-1)
    if kind == "relu":
        phi = jnp.maximum(s, 0.0) ** hp
    elif kind == "shift":
        phi = ((1.0 + s) / 2.0) ** hp
    elif kind == "yat":
        phi = (s * s) / (2.0 - 2.0 * s + hp)
    phi = phi + EPS
    return phi / phi.sum(-1, keepdims=True)

def run(kind, hp, Xtr, ytr, Xte, yte, n_cls, seed):
    params = init(jax.random.key(seed), Xtr.shape[1], n_cls)
    opt = optax.adam(LR); state = opt.init(params)
    def loss_fn(p, X, y):
        prob = head_probs(scores(p, X), kind, hp)
        return -jnp.mean(jnp.log(prob[jnp.arange(y.shape[0]), y] + 1e-9))
    @jax.jit
    def step(p, st):
        l, g = jax.value_and_grad(loss_fn)(p, Xtr, ytr)
        up, st = opt.update(g, st); return optax.apply_updates(p, up), st, l
    for _ in range(STEPS):
        params, state, _ = step(params, state)
    s_te = scores(params, Xte)
    acc = float((s_te.argmax(-1) == yte).mean())
    # expected calibration error (10 bins) on the head's probabilities
    prob = np.asarray(head_probs(s_te, kind, hp)); conf = prob.max(-1); pred = prob.argmax(-1)
    correct = (pred == np.asarray(yte)).astype(float)
    ece = 0.0
    for lo in np.linspace(0, 1, 11)[:-1]:
        m = (conf >= lo) & (conf < lo + 0.1)
        if m.sum(): ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return acc, ece

SWEEPS = {
    "softmax":          [0.05, 0.1, 0.15, 0.2, 0.3, 0.5],
    "relu":             [1, 2, 4, 8, 16, 32],
    "shift":            [1, 2, 4, 8, 16, 32],
    "yat":              [0.5, 0.2, 0.1, 0.05, 0.02, 0.01],
}
LABEL = {"softmax": "softmax(cos/tau)", "relu": "relu(cos)^b / sum",
         "shift": "((1+cos)/2)^b / sum", "yat": "yat / sum"}

def main():
    Xtr, ytr, Xte, yte, n_cls = load()
    seeds = [0, 1, 2]
    print(f"digits: {Xtr.shape[0]} train / {Xte.shape[0]} test, {n_cls} classes, "
          f"d_emb={D_EMB}, {STEPS} steps, {len(seeds)} seeds\n")
    print(f"{'head':22s} {'best test acc':16s} {'@param':>8s} {'ECE':>6s}")
    print("-" * 56)
    for kind, hps in SWEEPS.items():
        best = None
        for hp in hps:
            accs, eces = zip(*[run(kind, hp, Xtr, ytr, Xte, yte, n_cls, s) for s in seeds])
            m, sd = float(np.mean(accs)), float(np.std(accs))
            if best is None or m > best[0]:
                best = (m, sd, hp, float(np.mean(eces)))
        m, sd, hp, ece = best
        print(f"{LABEL[kind]:22s} {m*100:6.2f} +/- {sd*100:4.2f}%   {hp:>7} {ece:6.3f}")

if __name__ == "__main__":
    main()
