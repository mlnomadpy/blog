// Shared compute for the edit-a-network visualizations. The Yat kernel runs on
// the engine's compute backend; the heavy full-bank evaluation is offloaded to a
// Web Worker so it runs in parallel and never blocks the page, with a main-thread
// fallback if workers are unavailable. All three panels share ONE memoized load
// and ONE kernel evaluation, deferred until a panel is run.
import { np, ensureJax } from './engine/jax.js';

const BASE = import.meta.env.BASE_URL.replace(/\/$/, '');
const D = 784, NPROTO = 200;
const loadImg = (src) => new Promise((res) => { const im = new Image(); im.onload = () => res(im); im.onerror = () => res(null); im.src = src; });

// one 28x28 tile of a grayscale sprite as a length-784 Float32Array in [0,1]
export function tilePixels(img, cols, idx) {
  const c = idx % cols, r = Math.floor(idx / cols);
  const cv = document.createElement('canvas'); cv.width = 28; cv.height = 28;
  const cx = cv.getContext('2d', { willReadFrequently: true });
  cx.drawImage(img, c * 28, r * 28, 28, 28, 0, 0, 28, 28);
  const d = cx.getImageData(0, 0, 28, 28).data, v = new Float32Array(D);
  for (let i = 0; i < D; i++) v[i] = d[i * 4] / 255;       // grayscale: R == G == B
  return v;
}

// all n tiles as one flat Float32Array(n*784), via a single getImageData
export function extractFlat(img, cols, n) {
  const rows = Math.ceil(n / cols), Wd = cols * 28, Hd = rows * 28;
  const cv = document.createElement('canvas'); cv.width = Wd; cv.height = Hd;
  const cx = cv.getContext('2d', { willReadFrequently: true });
  cx.drawImage(img, 0, 0);
  const d = cx.getImageData(0, 0, Wd, Hd).data, out = new Float32Array(n * D);
  for (let t = 0; t < n; t++) {
    const tc = (t % cols) * 28, tr = Math.floor(t / cols) * 28, o = t * D;
    for (let py = 0; py < 28; py++) { const base = ((tr + py) * Wd + tc) * 4; for (let px = 0; px < 28; px++) out[o + py * 28 + px] = d[base + px * 4] / 255; }
  }
  return out;
}

// flat [n*D] -> nested rows (for the main-thread jax-js path)
const toRows = (flat, n) => { const r = []; for (let i = 0; i < n; i++) r.push(flat.subarray(i * D, i * D + D)); return r; };
export const rowOf = (flat, i) => flat.subarray(i * D, i * D + D);

// Yat kernel of every row of X against every prototype W, on jax-js (main thread).
// X: rows of length D, W: rows of length D  ->  [n][K] activations.
export function yatKernel(X, W, b, eps) {
  const n = X.length, K = W.length;
  const Xt = np.array(X.map((r) => Array.from(r))), Wt = np.array(W.map((r) => Array.from(r)));
  const dot = np.matmul(Xt.ref, np.transpose(Wt.ref));            // [n,K]
  const xn = np.reshape(np.sum(Xt.ref.mul(Xt), 1), [n, 1]);       // ||x||^2
  const wn = np.reshape(np.sum(Wt.ref.mul(Wt), 1), [1, K]);       // ||W||^2
  const dist2 = np.add(xn, wn).sub(dot.ref.mul(2));               // [n,K]
  const lin = dot.add(b);                                         // dot, final use
  return lin.ref.mul(lin).div(dist2.add(eps)).js();
}

// per-class strongest activation, [n][10], from a [n][K] kernel and length-K vote
export function perClassMax(ker, vote, nClasses = 10) {
  const n = ker.length;
  const out = Array.from({ length: n }, () => new Float64Array(nClasses).fill(-Infinity));
  for (let i = 0; i < n; i++) { const row = ker[i], o = out[i];
    for (let u = 0; u < vote.length; u++) { const c = vote[u]; if (row[u] > o[c]) o[c] = row[u]; } }
  return out;
}

// run cb once, when `el` first scrolls near the viewport (nothing computes on mount)
export function whenVisible(el, cb, margin = '700px') {
  if (typeof IntersectionObserver === 'undefined') { cb(); return; }
  const io = new IntersectionObserver((es) => { if (es[0].isIntersecting) { io.disconnect(); cb(); } }, { rootMargin: margin });
  io.observe(el);
}

// load sprites + json + extract pixels ONCE, shared across all panels
let _data = null;
export function loadEditData() {
  if (_data) return _data;
  _data = (async () => {
    const [protos, bank] = await Promise.all([loadImg(`${BASE}/yat-edit/protos.png`), loadImg(`${BASE}/yat-edit/testbank.png`)]);
    const d = await fetch(`${BASE}/yat-edit/edit.json`).then((r) => r.json());
    const vote = new Int8Array(NPROTO); for (let c = 0, k = 0; c < 10; c++) for (let j = 0; j < d.per; j++) vote[k++] = c;
    const protoFlat = extractFlat(protos, d.per, NPROTO);
    const bankFlat = extractFlat(bank, d.bankCols, d.nbank);
    return { protos, bank, d, vote, protoFlat, bankFlat, protoRows: toRows(protoFlat, NPROTO) };
  })();
  return _data;
}

// evaluate the full [nbank x 10] per-class scores ONCE, in a Worker (parallel),
// falling back to the main thread. Memoized and shared.
let _pm = null;
export function loadKernel() {
  if (_pm) return _pm;
  _pm = (async () => {
    const s = await loadEditData();
    let pm;
    try { pm = await kernelInWorker(s); }
    catch { pm = await kernelOnMainThread(s); }            // graceful fallback
    return { ...s, pm };
  })();
  return _pm;
}

function kernelInWorker(s) {
  return new Promise((resolve, reject) => {
    if (typeof Worker === 'undefined') { reject(new Error('no worker')); return; }
    let w;
    try { w = new Worker(new URL('./yatedit.worker.js', import.meta.url), { type: 'module' }); }
    catch (e) { reject(e); return; }
    const done = (fn) => { clearTimeout(timer); w.onmessage = w.onerror = null; w.terminate(); fn(); };
    const timer = setTimeout(() => done(() => reject(new Error('worker timeout'))), 20000);
    w.onerror = (e) => done(() => reject(e.error || new Error('worker error')));
    w.onmessage = (e) => {
      if (e.data.error) { done(() => reject(new Error(e.data.error))); return; }
      const flat = new Float32Array(e.data.pm), n = e.data.n, pm = [];
      for (let i = 0; i < n; i++) { const r = new Float64Array(10); for (let c = 0; c < 10; c++) r[c] = flat[i * 10 + c]; pm.push(r); }
      done(() => resolve(pm));
    };
    // structured-clone copies (no transfer) so the shared buffers stay usable
    w.postMessage({ bank: s.bankFlat, proto: s.protoFlat, n: s.d.nbank, D, K: NPROTO, b: s.d.b, eps: s.d.eps, vote: s.vote });
  });
}

async function kernelOnMainThread(s) {
  await ensureJax();
  return perClassMax(yatKernel(toRows(s.bankFlat, s.d.nbank), s.protoRows, s.d.b, s.d.eps), s.vote);
}

export { ensureJax };
