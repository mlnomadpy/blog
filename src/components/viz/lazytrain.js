// Shared data + math for the lazy-training panels. The live panels compute
// real Yat features from a real Fashion-MNIST subset (14x14) and train the
// linear readout with hand-rolled gradients in the browser; the run panels
// read the exported bundle JSON.

import { grayGlyph } from './engine/draw.js';

const BASE = `${import.meta.env.BASE_URL ?? '/'}`.replace(/\/$/, '');

const cache = {};
export function load(name) {
  cache[name] ??= fetch(`${BASE}/lazy-training/${name}.json`).then((r) => r.json());
  return cache[name];
}

export const B0 = 0.5, EPS0 = 0.5; // the layer's frozen scalars (softplus init)

// deterministic PRNG so "reseed" is reproducible per seed value
export function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gauss(rand) {
  let u = 0, v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// a bank of m random prototypes in d dims, lecun-normal (the trained net's
// own init, frozen)
export function randomBank(m, d, seed) {
  const rand = mulberry32(seed);
  const W = new Float32Array(m * d);
  const s = 1 / Math.sqrt(d);
  for (let i = 0; i < W.length; i++) W[i] = gauss(rand) * s;
  return W;
}

// squared norms of a bank's prototypes: constant across inputs, so they are
// computed once instead of once per image (this halves the inner loop)
export function bankNorms(W, m, d) {
  const wn = new Float32Array(m);
  for (let u = 0; u < m; u++) {
    let ww = 0;
    const off = u * d;
    for (let a = 0; a < d; a++) { const w = W[off + a]; ww += w * w; }
    wn[u] = ww;
  }
  return wn;
}

// Yat features of one image (Float32Array d) against a bank (m x d).
// `wn` is the bank's squared norms; omit it and they are computed inline.
export function yatRow(x, W, m, d, out, wn) {
  out = out || new Float32Array(m);
  wn = wn || bankNorms(W, m, d);
  let xx = 0;
  for (let a = 0; a < d; a++) xx += x[a] * x[a];
  for (let u = 0; u < m; u++) {
    let dot = 0;
    const off = u * d;
    for (let a = 0; a < d; a++) dot += W[off + a] * x[a];
    const n = dot + B0;
    const d2 = Math.max(0, xx + wn[u] - 2 * dot);
    out[u] = (n * n) / (d2 + EPS0);
  }
  return out;
}

// features for a whole dataset (n x d, flat) -> n x m flat. Reads X in place
// (no per-image copy) and shares one prototype-norm table across the sweep.
export function yatFeatures(X, n, d, W, m, onProgress) {
  const F = new Float32Array(n * m);
  const wn = bankNorms(W, m, d);
  for (let i = 0; i < n; i++) {
    const xo = i * d, fo = i * m;
    let xx = 0;
    for (let a = 0; a < d; a++) { const v = X[xo + a]; xx += v * v; }
    for (let u = 0; u < m; u++) {
      let dot = 0;
      const off = u * d;
      for (let a = 0; a < d; a++) dot += W[off + a] * X[xo + a];
      const nn = dot + B0;
      const d2 = Math.max(0, xx + wn[u] - 2 * dot);
      F[fo + u] = (nn * nn) / (d2 + EPS0);
    }
    if (onProgress && i % 200 === 0) onProgress(i / n);
  }
  return F;
}

// A linear softmax head trained by hand-rolled Adam. State lives in the
// returned object; step(batchIdx) does one update and returns the loss.
// `stride` is the row pitch of the feature block, so a head can read the
// first m columns of a wider bank without any copy or recompute
export function makeHead(m, k, lr = 0.02, stride = m) {
  const A = new Float32Array(m * k), b = new Float32Array(k);
  const mA = new Float32Array(m * k), vA = new Float32Array(m * k);
  const mb = new Float32Array(k), vb = new Float32Array(k);
  // scratch buffers, allocated once: a training step used to allocate a fresh
  // gradient pair every call, which is the bulk of this loop's garbage
  const gA = new Float32Array(m * k), gb = new Float32Array(k);
  const lgBuf = new Float32Array(k);
  let t = 0;
  const b1 = 0.9, b2 = 0.999, eps = 1e-8;

  function logits(F, i, out) {
    const fo = i * stride;
    for (let c = 0; c < k; c++) out[c] = b[c];
    for (let u = 0; u < m; u++) {
      const f = F[fo + u];
      if (f === 0) continue;
      const ao = u * k;
      for (let c = 0; c < k; c++) out[c] += f * A[ao + c];
    }
  }

  function step(F, y, idx) {
    t++;
    gA.fill(0); gb.fill(0);
    const lg = lgBuf;
    let loss = 0;
    for (const i of idx) {
      logits(F, i, lg);
      let mx = -Infinity;
      for (let c = 0; c < k; c++) mx = Math.max(mx, lg[c]);
      let Z = 0;
      for (let c = 0; c < k; c++) { lg[c] = Math.exp(lg[c] - mx); Z += lg[c]; }
      const fo = i * stride;
      for (let c = 0; c < k; c++) {
        const p = lg[c] / Z;
        const g = p - (y[i] === c ? 1 : 0);
        if (y[i] === c) loss -= Math.log(Math.max(p, 1e-12));
        gb[c] += g;
        lg[c] = g;                       // reuse the row as the gradient vector
      }
      // one pass over the features per sample instead of one per class
      for (let u = 0; u < m; u++) {
        const f = F[fo + u];
        if (f === 0) continue;
        const ao = u * k;
        for (let c = 0; c < k; c++) gA[ao + c] += lg[c] * f;
      }
    }
    const n = idx.length;
    const c1 = 1 - Math.pow(b1, t), c2 = 1 - Math.pow(b2, t);
    const upd = (P, G, M, V) => {
      for (let j = 0; j < P.length; j++) {
        const g = G[j] / n;
        M[j] = b1 * M[j] + (1 - b1) * g;
        V[j] = b2 * V[j] + (1 - b2) * g * g;
        P[j] -= lr * (M[j] / c1) / (Math.sqrt(V[j] / c2) + eps);
      }
    };
    upd(A, gA, mA, vA);
    upd(b, gb, mb, vb);
    return loss / n;
  }

  function predict(F, i) {
    logits(F, i, lgBuf);
    let bi = 0;
    for (let c = 1; c < k; c++) if (lgBuf[c] > lgBuf[bi]) bi = c;
    return bi;
  }

  // accuracy over a whole feature block, allocation-free
  function accuracy(F, y, n) {
    let ok = 0;
    for (let i = 0; i < n; i++) if (predict(F, i) === y[i]) ok++;
    return (100 * ok) / n;
  }

  return { A, b, step, predict, accuracy };
}

// draw a small grayscale image; `key` identifies the pixels so the sprite is
// painted once and blitted thereafter (see engine/draw.js)
export function drawGlyph(ctx, img, side, x, y, w, h, invert, key) {
  grayGlyph(ctx, img, side, x, y, w, h, { invert, key });
}
