// Shared data + compute for the kernel-distillation panels. One memoized fetch of
// distill.json (and the garment sprite) for every panel, the class palette, and
// the small kernel math the panels share: temperature softmax over the real
// exported teacher logits and the class-similarity kernel S(T) built from them.
// The matrix work runs on the engine compute backend.
import { np, ensureJax } from './engine/jax.js';
import { loadJSON, loadImage, whenVisible } from './engine/io.js';

const BASE = import.meta.env.BASE_URL.replace(/\/$/, '');
const DIR = `${BASE}/kernel-distill`;

export const CLASS_COL = ['#b3661b', '#4a7fb3', '#3a8f5e', '#9a4f9c', '#c2553a', '#5a7d3a', '#2f8f8f', '#a06a2a', '#7a5fc0', '#c0892a'];
export const CLASSES = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot'];

export const loadKD = () => loadJSON(`${DIR}/distill.json`);
export const samplesImg = () => loadImage(`${DIR}/samples.png`);

// softmax of one logit row at temperature T (10 numbers; plain JS is exact here)
export function softmaxT(z, T) {
  const s = z.map((v) => v / T), m = Math.max(...s);
  const e = s.map((v) => Math.exp(v - m)), Z = e.reduce((a, v) => a + v, 0);
  return e.map((v) => v / Z);
}

// the class-similarity kernel S(T) = E[p pᵀ] over the n exported test logits,
// diagonal-normalized so S_cc = 1; the matmul runs on the compute backend.
export function classKernel(logits, T) {
  const P = np.array(logits.map((z) => softmaxT(z, T)));
  const S = np.matmul(np.transpose(P.ref), P).js();       // 10x10, sum over examples
  const n = logits.length;
  const d = S.map((r, i) => Math.sqrt(r[i] / n) + 1e-12);
  return S.map((r, i) => r.map((v, j) => v / n / (d[i] * d[j])));
}

export { ensureJax, whenVisible };
