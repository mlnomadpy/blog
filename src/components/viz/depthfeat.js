// Shared data + live compute for the depth-by-construction post. One memoized
// fetch of depth.json and the sample sprite, plus a faithful JS port of the
// layer-1 and layer-2 feature math in scripts/handbuilt_depth.py, so every map a
// panel shows is computed live from the raw image. Every number traces to that
// script.
import { loadJSON, loadImage } from './engine/io.js';

const BASE = import.meta.env.BASE_URL.replace(/\/$/, '');
const DIR = `${BASE}/handbuilt-depth`;

export const CLASS_COL = ['#b3661b', '#4a7fb3', '#3a8f5e', '#9a4f9c', '#c2553a', '#5a7d3a', '#2f8f8f', '#a06a2a', '#7a5fc0', '#c0892a'];
// orientation hue ramp (6 edges) + corner, matching the script's channel order
export const CHAN_COL = ['#c2553a', '#c77d2a', '#5a7d3a', '#2f8f8f', '#4a7fb3', '#7a5fc0', '#a06a2a'];
export const KIND_COL = { junction: '#c2553a', continuation: '#4a7fb3', bend: '#9a4f9c', stripe: '#3a8f5e' };

export const loadDepth = () => loadJSON(`${DIR}/depth.json`);
export const samplesImg = () => loadImage(`${DIR}/samples.png`);

// ── layer 1, per-pixel: Sobel gradients + soft orientation binning + corner,
// identical math to channel_maps() in scripts/handbuilt_depth.py (the C4 recipe).
export function l1maps(im, d) {
  const N = 28, NB = d.NB, NCH = d.nChan, PI = Math.PI;
  const mod = (v, m) => ((v % m) + m) % m;
  const at = (y, x) => im[Math.min(N - 1, Math.max(0, y)) * N + Math.min(N - 1, Math.max(0, x))];
  const maps = []; for (let ch = 0; ch < NCH; ch++) maps.push(new Float32Array(N * N));
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const a = at(y - 1, x - 1), b = at(y - 1, x), c = at(y - 1, x + 1), l = at(y, x - 1), r = at(y, x + 1), g = at(y + 1, x - 1), h = at(y + 1, x), i = at(y + 1, x + 1);
    const gx = -a + c - 2 * l + 2 * r - g + i, gy = -a - 2 * b - c + g + 2 * h + i;
    const mag = Math.sqrt(gx * gx + gy * gy) + 1e-6, ang = mod(Math.atan2(gy, gx), PI);
    const p = y * N + x;
    for (let ch = 0; ch < NB; ch++) { const dst = Math.abs(mod(ang - ch * PI / NB + PI / 2, PI) - PI / 2); maps[ch][p] = Math.max(0, Math.min(1, 1 - dst / (PI / NB))) * mag; }
    maps[NB][p] = Math.abs(gx) * Math.abs(gy);
  }
  return maps;                                        // NCH arrays of 28*28
}

// mean-pool a 28x28 map 2x2 down to the 14x14 cell grid the layer-2 detectors read
export function toCells(map) {
  const C = 14, out = new Float32Array(C * C);
  for (let y = 0; y < C; y++) for (let x = 0; x < C; x++)
    out[y * C + x] = (map[(2 * y) * 28 + 2 * x] + map[(2 * y) * 28 + 2 * x + 1] + map[(2 * y + 1) * 28 + 2 * x] + map[(2 * y + 1) * 28 + 2 * x + 1]) / 4;
  return out;
}

// out[y][x] = A[y+dy][x+dx], zero outside (integer shift on a size x size grid)
function shiftInt(A, size, dy, dx) {
  const out = new Float32Array(size * size);
  for (let y = 0; y < size; y++) {
    const sy = y + dy; if (sy < 0 || sy >= size) continue;
    for (let x = 0; x < size; x++) { const sx = x + dx; if (sx < 0 || sx >= size) continue; out[y * size + x] = A[sy * size + sx]; }
  }
  return out;
}

// bilinear fractional shift, matching shift_frac() in the script
export function shiftFrac(A, size, dy, dx) {
  const y0 = Math.floor(dy), x0 = Math.floor(dx), fy = dy - y0, fx = dx - x0;
  const a = shiftInt(A, size, y0, x0), b = shiftInt(A, size, y0, x0 + 1), c = shiftInt(A, size, y0 + 1, x0), e = shiftInt(A, size, y0 + 1, x0 + 1);
  const out = new Float32Array(size * size);
  for (let i = 0; i < out.length; i++) out[i] = (1 - fy) * (1 - fx) * a[i] + (1 - fy) * fx * b[i] + fy * (1 - fx) * c[i] + fy * fx * e[i];
  return out;
}

// ── layer 2: one named combination detector evaluated on the cell grid, the
// min-AND one-cell design the script exports (l2_maps with t=1, rule='min').
export function l2map(cellsByChan, spec, d) {
  const C = 14, NB = d.NB;
  const minOf = (A, B2) => { const o = new Float32Array(C * C); for (let i = 0; i < o.length; i++) o[i] = Math.min(A[i], B2[i]); return o; };
  const maxOf = (A, B2) => { const o = new Float32Array(C * C); for (let i = 0; i < o.length; i++) o[i] = Math.max(A[i], B2[i]); return o; };
  if (spec.kind === 'junction') return minOf(cellsByChan[spec.b1], cellsByChan[spec.b2]);
  if (spec.kind === 'continuation') {
    const [dx, dy] = d.contour[spec.b1];
    return minOf(cellsByChan[spec.b1], shiftFrac(cellsByChan[spec.b1], C, dy, dx));
  }
  if (spec.kind === 'stripe') {
    const th = d.centersDeg[spec.b1] * Math.PI / 180, gdx = Math.cos(th), gdy = Math.sin(th);
    return minOf(cellsByChan[spec.b1], maxOf(shiftFrac(cellsByChan[spec.b1], C, gdy * 2, gdx * 2),
                                             shiftFrac(cellsByChan[spec.b1], C, -gdy * 2, -gdx * 2)));
  }
  const b = spec.b1, [dx, dy] = d.contour[b];
  return minOf(cellsByChan[b], maxOf(shiftFrac(cellsByChan[(b + 1) % NB], C, dy, dx),
                                     shiftFrac(cellsByChan[(b + NB - 1) % NB], C, dy, dx)));
}

// center-crop 14x14 to 12x12 and mean-pool 3x3 to the 4x4 layer-2 regions (pool_l2)
export function poolL2(map) {
  const out = new Float32Array(16);
  for (let gy = 0; gy < 4; gy++) for (let gx = 0; gx < 4; gx++) {
    let s = 0;
    for (let dy = 0; dy < 3; dy++) for (let dx = 0; dx < 3; dx++) s += map[(1 + gy * 3 + dy) * 14 + (1 + gx * 3 + dx)];
    out[gy * 4 + gx] = s / 9;
  }
  return out;
}

// read one 28x28 tile out of the 10-column sample sprite into a [0,1] array
export function tileFromSprite(img, k) {
  const c = document.createElement('canvas'); c.width = 28; c.height = 28;
  const cx = c.getContext('2d', { willReadFrequently: true });
  cx.drawImage(img, (k % 10) * 28, ((k / 10) | 0) * 28, 28, 28, 0, 0, 28, 28);
  const px = cx.getImageData(0, 0, 28, 28).data, out = new Float32Array(28 * 28);
  for (let i = 0; i < 784; i++) out[i] = px[i * 4] / 255;
  return out;
}

// render a scalar map to an offscreen canvas (dark base -> tint), for drawImage
// scaling into the panel; putImageData stays on the offscreen so DPR is safe.
export function mapToCanvas(vals, size, tint, gamma = 0.6) {
  const c = document.createElement('canvas'); c.width = size; c.height = size;
  const cx = c.getContext('2d'), img = cx.createImageData(size, size);
  let mx = 1e-6; for (let i = 0; i < vals.length; i++) if (vals[i] > mx) mx = vals[i];
  const [tr, tg, tb] = tint;
  for (let i = 0; i < vals.length; i++) {
    const t = Math.pow(Math.max(0, vals[i]) / mx, gamma);
    img.data[i * 4] = (18 + (tr - 18) * t) | 0;
    img.data[i * 4 + 1] = (16 + (tg - 16) * t) | 0;
    img.data[i * 4 + 2] = (14 + (tb - 14) * t) | 0;
    img.data[i * 4 + 3] = 255;
  }
  cx.putImageData(img, 0, 0);
  return c;
}

export function parseHex(v) {
  const s = v.replace('#', '');
  return [parseInt(s.slice(0, 2), 16), parseInt(s.slice(2, 4), 16), parseInt(s.slice(4, 6), 16)];
}
