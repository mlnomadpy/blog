// Live, in-browser forward pass over the REAL trained Yat-FFN transformer (the one
// scripts/yat_ffn_whitebox.py trains on tinyshakespeare). The weights are exported to
// public/mlp-representer/{model.json, weights.bin}; this module runs the whole network
// in plain JS so the explainer's panels read/attribute/edit/abstain on the real memory
// as the reader types. The math mirrors the script's numpy reference forward, which is
// asserted equal to Flax before export, so this port is faithful. No replay: every
// number below is computed here from the loaded weights.
import { loadJSON, loadFloat32 } from './engine/io.js';

const BASE = import.meta.env.BASE_URL.replace(/\/$/, '');
const DIR = `${BASE}/mlp-representer`;

let _model = null;
export function loadModel() {
  if (_model) return _model;
  _model = Promise.all([loadJSON(`${DIR}/model.json`), loadFloat32(`${DIR}/weights.bin`)])
    .then(([meta, buf]) => {
      const P = {}; let off = 0;
      for (const t of meta.tensors) {
        const n = t.shape.reduce((a, b) => a * b, 1);
        P[t.path] = buf.subarray(off, off + n); off += n;
      }
      // precompute ||W_u||^2 per layer for the kernel denominator
      const Wn2 = [];
      for (let L = 0; L < meta.layers; L++) {
        const W = P[`b${L}.ffn.W`], ff = meta.ff, D = meta.D, a = new Float64Array(ff);
        for (let u = 0; u < ff; u++) { let s = 0; for (let i = 0; i < D; i++) { const w = W[u * D + i]; s += w * w; } a[u] = s; }
        Wn2.push(a);
      }
      const stoi = {}; meta.itos.forEach((c, i) => { stoi[c] = i; });
      return { meta, P, Wn2, stoi };
    });
  return _model;
}

export function encode(m, str) {
  const ids = []; for (const c of str) if (c in m.stoi) ids.push(m.stoi[c]); return ids;
}
export const decode = (m, ids) => ids.map((i) => m.meta.itos[i]).join('');

function layernorm(x, T, D, scale, bias) {
  const o = new Float64Array(T * D);
  for (let t = 0; t < T; t++) {
    let mean = 0; for (let i = 0; i < D; i++) mean += x[t * D + i]; mean /= D;
    let v = 0; for (let i = 0; i < D; i++) { const d = x[t * D + i] - mean; v += d * d; } v /= D;
    const inv = 1 / Math.sqrt(v + 1e-6);
    for (let i = 0; i < D; i++) o[t * D + i] = (x[t * D + i] - mean) * inv * scale[i] + bias[i];
  }
  return o;
}

function mha(a, T, D, H, HD, P, L) {
  const qk = P[`b${L}.attn.q_k`], qb = P[`b${L}.attn.q_b`], kk = P[`b${L}.attn.k_k`], kb = P[`b${L}.attn.k_b`];
  const vk = P[`b${L}.attn.v_k`], vb = P[`b${L}.attn.v_b`], ok = P[`b${L}.attn.o_k`], ob = P[`b${L}.attn.o_b`];
  const proj = (kern, bias) => {                          // a[T,D] -> [T,H,HD]
    const o = new Float64Array(T * H * HD);
    for (let t = 0; t < T; t++) for (let h = 0; h < H; h++) for (let k = 0; k < HD; k++) {
      let s = bias[h * HD + k]; for (let i = 0; i < D; i++) s += a[t * D + i] * kern[i * H * HD + h * HD + k];
      o[t * H * HD + h * HD + k] = s;
    }
    return o;
  };
  const q = proj(qk, qb), k = proj(kk, kb), v = proj(vk, vb);
  const scale = 1 / Math.sqrt(HD), out = new Float64Array(T * D);
  const sc = new Float64Array(T);
  for (let h = 0; h < H; h++) for (let i = 0; i < T; i++) {
    let mx = -1e30;
    for (let j = 0; j <= i; j++) { let s = 0; for (let kk2 = 0; kk2 < HD; kk2++) s += q[i * H * HD + h * HD + kk2] * k[j * H * HD + h * HD + kk2]; sc[j] = s * scale; if (sc[j] > mx) mx = sc[j]; }
    let Z = 0; for (let j = 0; j <= i; j++) { sc[j] = Math.exp(sc[j] - mx); Z += sc[j]; }
    for (let kk2 = 0; kk2 < HD; kk2++) {
      let acc = 0; for (let j = 0; j <= i; j++) acc += (sc[j] / Z) * v[j * H * HD + h * HD + kk2];
      // out[i,:] += acc * o_k[h, kk2, :]
      for (let d = 0; d < D; d++) out[i * D + d] += acc * ok[h * HD * D + kk2 * D + d];
    }
  }
  for (let i = 0; i < T; i++) for (let d = 0; d < D; d++) out[i * D + d] += ob[d];
  return out;
}

// the Yat FFN: out(x) = Σ_u k(W_u,x) v_u. Returns the output AND the kernel weights kw.
// An optional edit scales the value of one slot (v_u *= gain) at this layer only.
function yatffn(a, T, D, ff, P, Wn2, b, eps, L, edit) {
  const W = P[`b${L}.ffn.W`], Vv = P[`b${L}.ffn.Vv`];     // W[ff,D], Vv[D,ff]
  const out = new Float64Array(T * D), kw = new Float64Array(T * ff);
  const gain = (edit && edit.layer === L) ? edit.gain : 1, slot = edit ? edit.slot : -1;
  for (let t = 0; t < T; t++) {
    let xn = 0; for (let i = 0; i < D; i++) { const xi = a[t * D + i]; xn += xi * xi; }
    for (let u = 0; u < ff; u++) {
      let dot = 0; const off = u * D; for (let i = 0; i < D; i++) dot += a[t * D + i] * W[off + i];
      const num = dot + b, k = (num * num) / (xn + Wn2[u] - 2 * dot + eps);
      kw[t * ff + u] = k;
      const g = (u === slot) ? gain : 1, kg = k * g;
      for (let d = 0; d < D; d++) out[t * D + d] += kg * Vv[d * ff + u];
    }
  }
  return { out, kw };
}

// full forward over ids; returns last-position logits/probs/pred and the captured
// FFN state (kw, out) at `captureLayer` (default last), plus per-token peak weights.
export function forward(m, ids, opts = {}) {
  const { meta, P, Wn2 } = m, { V, D, heads, ff, layers, lastLayer } = meta, HD = D / heads;
  const cap = opts.captureLayer ?? lastLayer, T = ids.length;
  let x = new Float64Array(T * D);
  const tok = P.tok, pos = P.pos;
  for (let t = 0; t < T; t++) { const id = ids[t]; for (let i = 0; i < D; i++) x[t * D + i] = tok[id * D + i] + pos[t * D + i]; }
  let captured = null;
  for (let L = 0; L < layers; L++) {
    const a1 = layernorm(x, T, D, P[`b${L}.ln1.scale`], P[`b${L}.ln1.bias`]);
    const ao = mha(a1, T, D, heads, HD, P, L);
    for (let i = 0; i < T * D; i++) x[i] += ao[i];
    const a2 = layernorm(x, T, D, P[`b${L}.ln2.scale`], P[`b${L}.ln2.bias`]);
    const sc = meta.ffnScalars[L];
    const { out, kw } = yatffn(a2, T, D, ff, P, Wn2[L], sc.b, sc.eps, L, opts.edit);
    if (L === cap) captured = { a2, kw, out, b: sc.b, eps: sc.eps };
    for (let i = 0; i < T * D; i++) x[i] += out[i];
  }
  const xf = layernorm(x, T, D, P['lnf.scale'], P['lnf.bias']);
  const head = P.head, logits = new Float64Array(V), last = (T - 1) * D;
  let mx = -1e30;
  for (let v = 0; v < V; v++) { let s = 0; for (let i = 0; i < D; i++) s += xf[last + i] * head[i * V + v]; logits[v] = s; if (s > mx) mx = s; }
  let Z = 0; const probs = new Float64Array(V);
  for (let v = 0; v < V; v++) { probs[v] = Math.exp(logits[v] - mx); Z += probs[v]; }
  let pred = 0; for (let v = 0; v < V; v++) { probs[v] /= Z; if (probs[v] > probs[pred]) pred = v; }
  // per-token peak kernel weight + write norm at the captured layer
  const peak = new Float64Array(T), wnorm = new Float64Array(T);
  for (let t = 0; t < T; t++) {
    let pmx = 0; for (let u = 0; u < ff; u++) { const k = captured.kw[t * ff + u]; if (k > pmx) pmx = k; } peak[t] = pmx;
    let n = 0; for (let d = 0; d < D; d++) { const o = captured.out[t * D + d]; n += o * o; } wnorm[t] = Math.sqrt(n);
  }
  return { logits, probs, pred, captured, peak, wnorm, T };
}

// decompose the last token's FFN output into the slots that wrote it (exact: it is a sum).
export function attribution(m, ids, opts = {}) {
  const { meta, P } = m, { ff, D, V } = meta, r = forward(m, ids, opts);
  const L = opts.captureLayer ?? meta.lastLayer, Vv = P[`b${L}.ffn.Vv`], head = P.head;
  const kw = r.captured.kw, base = (r.T - 1) * ff;
  const pred = r.pred;
  // push of slot u toward the predicted token: k_u * (v_u . U[:,pred])
  const rows = [];
  for (let u = 0; u < ff; u++) {
    let vu = 0; for (let d = 0; d < D; d++) vu += Vv[d * ff + u] * head[d * V + pred];
    rows.push({ u, k: kw[base + u], push: kw[base + u] * vu, writes: meta.slotWrites[u] });
  }
  rows.sort((a, b) => Math.abs(b.push) - Math.abs(a.push));
  return { ...r, pred, rows };
}

// greedy/sampled continuation; applies the edit at every step
export function generate(m, ids, n, temp = 0.8, opts = {}, seed = 1) {
  let s = seed; const rnd = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; };
  const T = m.meta.T, out = ids.slice();
  for (let step = 0; step < n; step++) {
    const ctx = out.slice(Math.max(0, out.length - T));
    const { probs } = forward(m, ctx, opts);
    // temperature sample
    let r = rnd(), acc = 0, pick = 0; const V = m.meta.V;
    const logT = 1 / temp, p2 = new Float64Array(V); let Z = 0;
    for (let v = 0; v < V; v++) { p2[v] = Math.pow(probs[v], logT); Z += p2[v]; }
    for (let v = 0; v < V; v++) { acc += p2[v] / Z; if (r <= acc) { pick = v; break; } }
    out.push(pick);
  }
  return out;
}

// a few interpretable slots (for the edit picker): the strongest writer of each target token
export function namedSlots(m, tokens) {
  const Vv = m.P[`b${m.meta.lastLayer}.ffn.Vv`], head = m.P.head, { ff, D, V } = m.meta;
  return tokens.map(({ tok, label }) => {
    const t = m.stoi[tok]; if (t === undefined) return null;
    let best = 0, bv = -1e30;
    for (let u = 0; u < ff; u++) { let s = 0; for (let d = 0; d < D; d++) s += Vv[d * ff + u] * head[d * V + t]; if (s > bv) { bv = s; best = u; } }
    return { u: best, label, tok, writes: m.meta.slotWrites[best] };
  }).filter(Boolean);
}
