// Shared data + math for the Yat-attention panels. The kernel panels compute
// the two score functions live from their formulas; the run panels read the
// exported bundle JSON.

const BASE = `${import.meta.env.BASE_URL ?? '/'}`.replace(/\/$/, '');

const cache = {};
export function load(name) {
  cache[name] ??= fetch(`${BASE}/attention-is-a-compatibility-kernel/${name}.json`).then((r) => r.json());
  return cache[name];
}

export const EPS = 1.0;

export function dotp(a, b) {
  return a[0] * b[0] + a[1] * b[1];
}

// the two scores, 2-D toy versions of the run's formulas
export function scoreBilinear(q, k, dh = 2) {
  return dotp(q, k) / Math.sqrt(dh);
}

export function scoreYat(q, k) {
  const d = dotp(q, k);
  const dx = q[0] - k[0], dy = q[1] - k[1];
  return (d * d) / (dx * dx + dy * dy + EPS);
}

export function softmaxRow(logits) {
  const m = Math.max(...logits);
  const e = logits.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export function kappaRow(kappas) {
  const s = kappas.reduce((a, b) => a + b, 0);
  return { w: kappas.map((v) => v / (s + 1e-9)), mass: s };
}
