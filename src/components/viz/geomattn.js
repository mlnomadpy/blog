// Shared data + math for the geometry-of-attention panels. The sandbox panels
// compute both attention laws live from their formulas over a 2-D plane of
// keys; the run panels read the exported bundle JSON from the trained models.

const BASE = `${import.meta.env.BASE_URL ?? '/'}`.replace(/\/$/, '');

const cache = {};
export function load(name) {
  cache[name] ??= fetch(`${BASE}/the-geometry-of-attention/${name}.json`).then((r) => r.json());
  return cache[name];
}

// toy scalars for the 2-D panels: b at the trained models' softplus(0) = log 2
// initialization; the softening set below the body spacing, the regime where
// every body owns the ground it stands on (the trained models learn both,
// per head)
export const B = Math.log(2);
export const EPS = 0.25;

export const dotp = (a, b) => a[0] * b[0] + a[1] * b[1];

// the two compatibility laws, 2-D toy versions of the run's formulas
export function scoreDot(q, k, dh = 2) {
  return dotp(q, k) / Math.sqrt(dh);
}

export function scoreYat(q, k, b = B, eps = EPS) {
  const n = dotp(q, k) + b;
  const dx = q[0] - k[0], dy = q[1] - k[1];
  return (n * n) / (dx * dx + dy * dy + eps);
}

export function softmaxRow(logits) {
  const m = Math.max(...logits);
  const e = logits.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export function l1Row(kappas) {
  const s = kappas.reduce((a, b) => a + b, 0);
  return { w: kappas.map((v) => v / (s + 1e-9)), mass: s };
}

// normalized entropy of a weight row: 0 = one-hot, 1 = uniform
export function normEntropy(w) {
  const n = w.length;
  if (n < 2) return 0;
  let h = 0;
  for (const p of w) if (p > 1e-12) h -= p * Math.log(p);
  return h / Math.log(n);
}

// winner index and full weight rows for one query against a key list
export function rows(q, keys, b = B, eps = EPS) {
  const logits = keys.map((k) => scoreDot(q, k));
  const kappas = keys.map((k) => scoreYat(q, k, b, eps));
  const wSoft = softmaxRow(logits);
  const { w: wKer, mass } = l1Row(kappas);
  const arg = (w) => w.indexOf(Math.max(...w));
  return { wSoft, wKer, mass, winSoft: arg(wSoft), winKer: arg(wKer) };
}

// is point p strictly inside the convex hull of pts (2-D)? Andrew monotone
// chain hull, then a sign test on every hull edge.
export function convexHull(pts) {
  const P = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const half = (list) => {
    const h = [];
    for (const p of list) {
      while (h.length >= 2 && cross(h[h.length - 2], h[h.length - 1], p) <= 0) h.pop();
      h.push(p);
    }
    return h;
  };
  const lower = half(P), upper = half(P.slice().reverse());
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}

export function insideHull(p, hull) {
  if (hull.length < 3) return false;
  for (let i = 0; i < hull.length; i++) {
    const a = hull[i], b = hull[(i + 1) % hull.length];
    if ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) <= 0) return false;
  }
  return true;
}
