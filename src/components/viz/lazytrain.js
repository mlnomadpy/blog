// Shared data + math for the lazy-training panels. The live panels compute
// real Yat features from a real Fashion-MNIST subset (14x14) and train the
// linear readout with hand-rolled gradients in the browser; the run panels
// read the exported bundle JSON.

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

// Yat features of one image (Float32Array d) against a bank (m x d)
export function yatRow(x, W, m, d, out) {
  out = out || new Float32Array(m);
  let xx = 0;
  for (let a = 0; a < d; a++) xx += x[a] * x[a];
  for (let u = 0; u < m; u++) {
    let dot = 0, ww = 0;
    const off = u * d;
    for (let a = 0; a < d; a++) { const w = W[off + a]; dot += w * x[a]; ww += w * w; }
    const n = dot + B0;
    const d2 = Math.max(0, xx + ww - 2 * dot);
    out[u] = (n * n) / (d2 + EPS0);
  }
  return out;
}

// features for a whole dataset (n x d, flat) -> n x m flat
export function yatFeatures(X, n, d, W, m, onProgress) {
  const F = new Float32Array(n * m);
  const x = new Float32Array(d), row = new Float32Array(m);
  for (let i = 0; i < n; i++) {
    for (let a = 0; a < d; a++) x[a] = X[i * d + a];
    yatRow(x, W, m, d, row);
    F.set(row, i * m);
    if (onProgress && i % 200 === 0) onProgress(i / n);
  }
  return F;
}

// A linear softmax head trained by hand-rolled Adam. State lives in the
// returned object; step(batchIdx) does one update and returns the loss.
export function makeHead(m, k, lr = 0.02) {
  const A = new Float32Array(m * k), b = new Float32Array(k);
  const mA = new Float32Array(m * k), vA = new Float32Array(m * k);
  const mb = new Float32Array(k), vb = new Float32Array(k);
  let t = 0;
  const b1 = 0.9, b2 = 0.999, eps = 1e-8;

  function logits(F, i, out) {
    for (let c = 0; c < k; c++) {
      let s = b[c];
      for (let u = 0; u < m; u++) s += F[i * m + u] * A[u * k + c];
      out[c] = s;
    }
  }

  function step(F, y, idx) {
    t++;
    const gA = new Float32Array(m * k), gb = new Float32Array(k);
    const lg = new Float32Array(k);
    let loss = 0;
    for (const i of idx) {
      logits(F, i, lg);
      let mx = -Infinity;
      for (let c = 0; c < k; c++) mx = Math.max(mx, lg[c]);
      let Z = 0;
      for (let c = 0; c < k; c++) { lg[c] = Math.exp(lg[c] - mx); Z += lg[c]; }
      for (let c = 0; c < k; c++) {
        const p = lg[c] / Z;
        const g = p - (y[i] === c ? 1 : 0);
        if (y[i] === c) loss -= Math.log(Math.max(p, 1e-12));
        gb[c] += g;
        for (let u = 0; u < m; u++) gA[u * k + c] += g * F[i * m + u];
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
    const lg = new Float32Array(k);
    logits(F, i, lg);
    let bi = 0;
    for (let c = 1; c < k; c++) if (lg[c] > lg[bi]) bi = c;
    return bi;
  }

  return { A, b, step, predict };
}

// draw a 14x14 grayscale image into a canvas rect (via offscreen, dpr-safe)
export function drawGlyph(ctx, img, side, x, y, w, h, invert) {
  const off = document.createElement('canvas');
  off.width = side; off.height = side;
  const octx = off.getContext('2d'), im = octx.createImageData(side, side);
  let lo = Infinity, hi = -Infinity;
  for (const v of img) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  for (let i = 0; i < side * side; i++) {
    let t = (img[i] - lo) / (hi - lo + 1e-9);
    if (invert) t = 1 - t;
    const g = Math.round(255 * t);
    im.data[i * 4] = g; im.data[i * 4 + 1] = g; im.data[i * 4 + 2] = g;
    im.data[i * 4 + 3] = 255;
  }
  octx.putImageData(im, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, x, y, w, h);
}
