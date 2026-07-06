// Shared data + kernel-math module for the survival post's interactive panels.
// One memoized load of the exported bundle in public/yat-deepsurv/, plus the tiny
// bit of Yat-kernel arithmetic the panels re-run live (attribution, OOD score,
// Nadaraya-Watson curve). Every number originates in scripts/yat_deepsurv.py; the
// panels only re-slice and re-weight it, never invent it.

import { loadJSON } from './engine/io.js';

const BASE = import.meta.env.BASE_URL.replace(/\/$/, '');
const DIR = `${BASE}/yat-deepsurv`;

let _proto = null, _attr = null, _ood = null, _edit = null, _embed = null, _curves = null, _metrics = null;

export const loadProto = () => (_proto ||= loadJSON(`${DIR}/prototypes.json`));
export const loadAttr = () => (_attr ||= loadJSON(`${DIR}/attributions.json`));
export const loadOod = () => (_ood ||= loadJSON(`${DIR}/ood.json`));
export const loadEdit = () => (_edit ||= loadJSON(`${DIR}/edit.json`));
export const loadEmbed = () => (_embed ||= loadJSON(`${DIR}/embedding.json`));
export const loadCurves = () => (_curves ||= loadJSON(`${DIR}/curves.json`));
export const loadMetrics = () => (_metrics ||= loadJSON(`${DIR}/metrics.json`));

// Yat kernel of a single patient x against a bank of prototypes W (rows), b, eps.
// phi_u(x) = (w_u . x + b)^2 / (||x - w_u||^2 + eps).  x, W rows in NORMALIZED space.
export function yatFeatures(W, x, b, eps) {
  const K = W.length, out = new Float64Array(K);
  for (let u = 0; u < K; u++) {
    const w = W[u];
    let dot = 0, d2 = 0;
    for (let j = 0; j < w.length; j++) {
      dot += w[j] * x[j];
      const dj = x[j] - w[j];
      d2 += dj * dj;
    }
    const num = dot + b;
    out[u] = (num * num) / (d2 + eps);
  }
  return out;
}

// Normalize a raw (clinical-unit) patient vector using stored train mu/sd.
export function normalize(xClin, mu, sd) {
  return xClin.map((v, j) => (v - mu[j]) / sd[j]);
}

// Signed prototype contributions c_u = a_u * phi_u(x); they sum to the log-risk.
export function contributions(phi, readout) {
  const K = phi.length, c = new Float64Array(K);
  let sum = 0;
  for (let u = 0; u < K; u++) { c[u] = readout[u] * phi[u]; sum += c[u]; }
  return { contrib: c, logrisk: sum };
}
