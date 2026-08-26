// Shared data + math for the patches-in-conversation panels.
// The panels run the REAL trained mixer (seed 0 of kgl_blog-mixer-v1) forward
// in the browser: patch embed, then the weight-tied block applied r times,
// sphere renorm after every residual add, exactly the training-time math.

const BASE = `${import.meta.env.BASE_URL ?? '/'}`.replace(/\/$/, '');
const cache = {};
export function load(name) {
  cache[name] ??= fetch(`${BASE}/patches-in-conversation/${name}.json`).then((r) => r.json());
  return cache[name];
}

const P = 8, T = 16, DP = 64;

export function prep(w) {
  if (w._p) return w._p;
  const flat = (m) => {
    const r = m.length, c = m[0].length, a = new Float32Array(r * c);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) a[i * c + j] = m[i][j];
    return a;
  };
  w._p = {
    m: w.m, We: flat(w.We), Wt: flat(w.Wt), Wc: flat(w.Wc), A: flat(w.A),
    bias: new Float32Array(w.bias),
    nWe: rowNorms(flat(w.We), DP), nWt: rowNorms(flat(w.Wt), T),
    nWc: rowNorms(flat(w.Wc), w.m),
  };
  return w._p;
}
function rowNorms(W, d) {
  const n = W.length / d, out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < d; j++) s += W[i * d + j] * W[i * d + j];
    out[i] = s;
  }
  return out;
}

// yat map of one vector x (dim d) against W (m x d): out[i] = (x.wi+b)^2/(|x-wi|^2+eps)
function yatVec(x, W, W2, d, m, b, eps, out) {
  let x2 = 0;
  for (let j = 0; j < d; j++) x2 += x[j] * x[j];
  for (let i = 0; i < m; i++) {
    let dot = 0;
    for (let j = 0; j < d; j++) dot += x[j] * W[i * d + j];
    const d2 = Math.max(x2 + W2[i] - 2 * dot, 0);
    out[i] = (dot + b) * (dot + b) / (d2 + eps);
  }
}
function sphereRow(h, off, d) {
  let n = 0;
  for (let j = 0; j < d; j++) n += h[off + j] * h[off + j];
  n = 1 / Math.max(Math.sqrt(n), 1e-6);
  for (let j = 0; j < d; j++) h[off + j] *= n;
}

export function imageTokens(img) {
  // img: 1024 grayscale ints (row-major 32x32) -> Float32Array (16 x 64), /255
  const t = new Float32Array(T * DP);
  for (let gy = 0; gy < 4; gy++) for (let gx = 0; gx < 4; gx++) {
    const ti = gy * 4 + gx;
    for (let py = 0; py < P; py++) for (let px = 0; px < P; px++) {
      t[ti * DP + py * P + px] = img[(gy * P + py) * 32 + gx * P + px] / 255;
    }
  }
  return t;
}

// The full walk: states h_r (T x m) for r = 0..rMax, plus logits per depth.
export function walk(w, img, rMax = 8) {
  const p = prep(w);
  const tok = imageTokens(img);
  let h = new Float32Array(T * p.m);
  const buf = new Float32Array(p.m), tbuf = new Float32Array(T);
  for (let i = 0; i < T; i++) {
    yatVec(tok.subarray(i * DP, (i + 1) * DP), p.We, p.nWe, DP, p.m,
           w.be[0], w.be[1], buf);
    h.set(buf, i * p.m);
    sphereRow(h, i * p.m, p.m);
  }
  const states = [h.slice()], logitsAt = [readout(p, w, h)];
  const col = new Float32Array(T);
  for (let r = 1; r <= rMax; r++) {
    // token mix: per channel, the 16-vector across patches through the T->T map
    const t = new Float32Array(T * p.m);
    for (let c = 0; c < p.m; c++) {
      for (let i = 0; i < T; i++) col[i] = h[i * p.m + c];
      yatVec(col, p.Wt, p.nWt, T, T, w.bt[0], w.bt[1], tbuf);
      for (let i = 0; i < T; i++) t[i * p.m + c] = tbuf[i];
    }
    for (let i = 0; i < T; i++) {
      for (let j = 0; j < p.m; j++) h[i * p.m + j] += w.a_t * t[i * p.m + j];
      sphereRow(h, i * p.m, p.m);
    }
    // channel mix: per token through the m->m map
    for (let i = 0; i < T; i++) {
      yatVec(h.subarray(i * p.m, (i + 1) * p.m), p.Wc, p.nWc, p.m, p.m,
             w.bc[0], w.bc[1], buf);
      for (let j = 0; j < p.m; j++) h[i * p.m + j] += w.a_c * buf[j];
      sphereRow(h, i * p.m, p.m);
    }
    states.push(h.slice());
    logitsAt.push(readout(p, w, h));
  }
  return { states, logitsAt };
}

function readout(p, w, h) {
  const pool = new Float32Array(p.m);
  for (let i = 0; i < T; i++) for (let j = 0; j < p.m; j++) pool[j] += h[i * p.m + j] / T;
  const lg = new Float32Array(100);
  for (let c = 0; c < 100; c++) {
    let s = p.bias[c];
    for (let j = 0; j < p.m; j++) s += pool[j] * p.A[j * 100 + c];
    lg[c] = s;
  }
  return lg;
}

// token cosine gram at a state (tokens are unit vectors, so dot = cosine)
export function gram(state, m) {
  const G = new Float32Array(T * T);
  for (let i = 0; i < T; i++) for (let j = i; j < T; j++) {
    let s = 0;
    for (let k = 0; k < m; k++) s += state[i * m + k] * state[j * m + k];
    G[i * T + j] = G[j * T + i] = s;
  }
  return G;
}

// who feeds patch i at this state: d(tokmix out_i)/d(u_j), averaged over
// channels, scaled by the learned step a_t. The analytic gradient of the
// kernel map, evaluated on the real state, no approximation.
export function influence(w, state, iPatch) {
  const p = prep(w);
  const inf = new Float32Array(T);
  const u = new Float32Array(T);
  for (let c = 0; c < p.m; c++) {
    for (let j = 0; j < T; j++) u[j] = state[j * p.m + c];
    let u2 = 0, dot = 0;
    for (let j = 0; j < T; j++) u2 += u[j] * u[j];
    for (let j = 0; j < T; j++) dot += u[j] * p.Wt[iPatch * T + j];
    const D = Math.max(u2 + p.nWt[iPatch] - 2 * dot, 0) + w.bt[1];
    const num = dot + w.bt[0];
    for (let j = 0; j < T; j++) {
      const g = 2 * num * p.Wt[iPatch * T + j] / D
              - num * num * 2 * (u[j] - p.Wt[iPatch * T + j]) / (D * D);
      inf[j] += (w.a_t * g) / p.m;
    }
  }
  return inf;
}

export function top5(lg) {
  const idx = Array.from({ length: 100 }, (_, i) => i);
  idx.sort((a, b) => lg[b] - lg[a]);
  return idx.slice(0, 5);
}
