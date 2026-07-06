// Shared data + compute for the "skip connections are half of Newton" panels.
// One memoized fetch per JSON export of scripts/momentum_resnet.py, plus the
// forward pass of the exported width-2 residual nets so the browser can
// re-integrate the real trained network live (decision fields, accuracy at a
// turned inertia dial, exact rewinds). All array math runs on the engine's
// compute backend; weights are held on-device once per net and reused via .ref.
import { np, ensureJax } from './engine/jax.js';
import { loadJSON } from './engine/io.js';

const BASE = import.meta.env.BASE_URL.replace(/\/$/, '');
const DIR = `${BASE}/momentum-resnet`;

export const loadTraj = () => loadJSON(`${DIR}/trajectories.json`);
export const loadCliff = () => loadJSON(`${DIR}/cliff.json`);
export const loadInertia = () => Promise.all([loadJSON(`${DIR}/inertia.json`), ensureJax()]).then(([d]) => d);

// class colours: class 0 (inner disk / first arm) orange, class 1 blue
export const CLS = ['#b3661b', '#4a7fb3'];

// ── live forward pass of an exported net on the compute backend ──────────────
// A runner holds one net's weights on-device. probs(Xjs, mu) runs the real
// residual dynamics  v <- mu v + (1-mu) F_l(x);  x <- x + h v  exactly as
// trained (mu can be overridden to re-integrate at a different damping) and
// returns the class-1 probability per row. Bounded loop, nothing leaks: every
// op consumes its inputs, weights survive via .ref.
const _runners = new Map();
export function netRunner(data, muKey) {
  const id = String(muKey);
  if (_runners.has(id)) return _runners.get(id);
  const net = data.nets[id], L = net.W1.length, h = data.h;
  const W1 = net.W1.map((w) => np.array(w));          // L × [2,H]
  const b1 = net.b1.map((b) => np.array([b]));        // L × [1,H]
  const W2 = net.W2.map((w) => np.array(w));          // L × [H,2]
  const b2 = net.b2.map((b) => np.array([b]));        // L × [1,2]
  const Wr = np.array(net.Wr), br = np.array([net.br]);
  const runner = {
    mu: net.mu, acc: net.acc, L, h,
    probs(Xjs, mu = net.mu) {
      let x = np.array(Xjs);
      let v = np.zeros([Xjs.length, 2]);
      for (let l = 0; l < L; l++) {
        const f = np.matmul(np.tanh(np.matmul(x.ref, W1[l].ref).add(b1[l].ref)), W2[l].ref).add(b2[l].ref);
        v = v.mul(mu).add(f.mul(1 - mu));
        x = x.add(v.ref.mul(h));
      }
      const lg = np.matmul(x, Wr.ref).add(br.ref).js();
      v.dispose();
      return lg.map((r) => { const m = Math.max(r[0], r[1]); return Math.exp(r[1] - m) / (Math.exp(r[0] - m) + Math.exp(r[1] - m)); });
    },
  };
  _runners.set(id, runner);
  return runner;
}

// ── double-precision JS forward/backward for the exact-rewind demo ────────────
// Plain JS numbers are float64, so this runs the trained dynamics at higher
// precision than training itself; the backward pass inverts it exactly.
function blockF(net, l, x) {
  const H = net.b1[l].length, f = [net.b2[l][0], net.b2[l][1]];
  for (let j = 0; j < H; j++) {
    const t = Math.tanh(x[0] * net.W1[l][0][j] + x[1] * net.W1[l][1][j] + net.b1[l][j]);
    f[0] += t * net.W2[l][j][0]; f[1] += t * net.W2[l][j][1];
  }
  return f;
}

// forward: the whole (x, v) trajectory, [L+1] entries of {x:[2], v:[2]}
export function rollForward(net, h, x0, mu = net.mu) {
  const L = net.W1.length;
  let x = [x0[0], x0[1]], v = [0, 0];
  const out = [{ x: [...x], v: [...v] }];
  for (let l = 0; l < L; l++) {
    const f = blockF(net, l, x);
    v = [mu * v[0] + (1 - mu) * f[0], mu * v[1] + (1 - mu) * f[1]];
    x = [x[0] + h * v[0], x[1] + h * v[1]];
    out.push({ x: [...x], v: [...v] });
  }
  return out;
}

// backward: from (xL, vL) recover every earlier state exactly (needs mu > 0).
// Returns [L+1] entries; entry l is the reconstructed state after block l.
export function rollBackward(net, h, xL, vL, mu = net.mu) {
  const L = net.W1.length;
  let x = [xL[0], xL[1]], v = [vL[0], vL[1]];
  const out = new Array(L + 1); out[L] = { x: [...x], v: [...v] };
  for (let l = L - 1; l >= 0; l--) {
    const xp = [x[0] - h * v[0], x[1] - h * v[1]];
    const f = blockF(net, l, xp);
    v = [(v[0] - (1 - mu) * f[0]) / mu, (v[1] - (1 - mu) * f[1]) / mu];
    x = xp;
    out[l] = { x: [...x], v: [...v] };
  }
  return out;
}

export { ensureJax };
