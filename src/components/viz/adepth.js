// Shared data + math for the depth-on-demand panels. The exported seed-0
// leapfrog nets run live: same forward, same gradient of the learned
// potential, and the same step-doubling controller the run used.

const BASE = `${import.meta.env.BASE_URL ?? '/'}`.replace(/\/$/, '');
const DIM = 4, T_TOTAL = 6.0, L_TRAIN = 16;

const cache = {};
export function load(name) {
  cache[name] ??= fetch(`${BASE}/depth-on-demand/${name}.json`).then((r) => r.json());
  return cache[name];
}

export { DIM, T_TOTAL, L_TRAIN };

function affine(x, wb) {
  const [W, b] = wb, out = new Array(b.length).fill(0);
  for (let j = 0; j < b.length; j++) {
    let s = b[j];
    for (let i = 0; i < x.length; i++) s += x[i] * W[i][j];
    out[j] = s;
  }
  return out;
}

// dV/dq for V(q) = sum(tanh(q W1 + b1) W2 + b2)
function gradV(m, q) {
  const h = affine(q, m.b1).map(Math.tanh);
  const W1 = m.b1[0], W2 = m.b2[0];
  const g = new Array(DIM).fill(0);
  for (let j = 0; j < h.length; j++) {
    let wsum = 0;
    for (let k = 0; k < DIM; k++) wsum += W2[j][k];
    const c = (1 - h[j] * h[j]) * wsum;
    for (let i = 0; i < DIM; i++) g[i] += c * W1[i][j];
  }
  return g;
}

export function leap(m, h, q, p) {
  let g = gradV(m, q);
  p = p.map((v, i) => v - 0.5 * h * g[i]);
  q = q.map((v, i) => v + h * p[i]);
  g = gradV(m, q);
  p = p.map((v, i) => v - 0.5 * h * g[i]);
  return [q, p];
}

export function encode(m, x) {
  const z = affine(x, m.enc);
  return [z.slice(0, DIM), z.slice(DIM)];
}

export function decode(m, q, p) {
  return affine([...q, ...p], m.dec);
}

export function forwardFixed(m, x, L) {
  const h = T_TOTAL / L;
  let [q, p] = encode(m, x);
  for (let l = 0; l < L; l++) [q, p] = leap(m, h, q, p);
  return decode(m, q, p);
}

// step doubling; returns logits, work (all leapfrog evals incl. probes),
// accepted committed steps, and the trajectory of accepted (q, h) pairs
export function renderAdaptive(m, x, tol, collect = false) {
  let [q, p] = encode(m, x);
  let t = 0, h = T_TOTAL / L_TRAIN, work = 0, accepted = 0;
  const hMin = T_TOTAL / 4096, traj = collect ? [{ q: [...q], h }] : null;
  let guard = 0;
  while (t < T_TOTAL - 1e-9 && guard++ < 4000) {
    h = Math.min(h, T_TOTAL - t);
    const [q1, p1] = leap(m, h, q, p);
    const [qh, ph] = leap(m, h / 2, q, p);
    const [q2, p2] = leap(m, h / 2, qh, ph);
    work += 3;
    let err = 0;
    for (let i = 0; i < DIM; i++) {
      err = Math.max(err, Math.abs(q1[i] - q2[i]), Math.abs(p1[i] - p2[i]));
    }
    if (err > tol && h > hMin) { h /= 2; continue; }
    q = q2; p = p2; accepted += 2; t += h;
    if (collect) traj.push({ q: [...q], h, err });
    if (err < tol / 4) h = Math.min(h * 2, T_TOTAL / 4);
  }
  return { logits: decode(m, q, p), work, accepted, traj };
}
