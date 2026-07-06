// Shared data + compute for the "edit a fixed point" panels. One memoized fetch of
// model.json (the trained base operator + the constructed edit: anchor prototypes,
// their A-columns, the calibrated readout scale) and an on-device compute object
// that runs the exact training-time iteration in the browser:
//
//     F(z; x) = tanh( A · φ_W(z) + U·x + z0 ),   φ_W(z)_i = (⟨z,w_i⟩ + b)² / (‖z-w_i‖² + ε)
//     z_{k+1} = (1-β) z_k + β F(z_k; x)   →   z*
//
// The taught class reads the settled state through the same kernel: s₃ = α·max_k φ(z*, c_k).
// The dynamics edit appends the anchors c_k to W and columns γ·ĉ_k to A; `makeOp(gain)`
// builds that edited operator at any gain multiple (gain 0 = base, 1 = the shipped edit),
// which is what the certificate panel dials. sigma() runs power iteration on JᵀJ with
// ANALYTIC Jacobian-vector products derived from the operator's formula (no finite
// differences), so the ‖J_F‖₂ certificate shown is the real spectral norm estimate.
// Numbers from scripts/yat_deq_edit.py.
import { np, ensureJax } from './engine/jax.js';
import { loadJSON } from './engine/io.js';

const BASE = import.meta.env.BASE_URL.replace(/\/$/, '');
const URL = `${BASE}/yat-deq-edit/model.json`;

// class colours: moons 0/1, the trained blob 2, the TAUGHT class 3 (accent)
export const CLS = ['#4a7fb3', '#c2553a', '#3a8f5e', '#b3661b'];
export const loadReady = () => Promise.all([loadJSON(URL), ensureJax()]).then(([m]) => m);

let _cmp = null;
export function compute(model) {
  if (_cmp) return _cmp;
  const P = model.params, E = model.edit, D = model.dims.d, M = model.dims.m;
  const beta = model.solver.beta, b = P.b, eps = P.eps;

  // persistent device copies of the base weights (reused via .ref, never disposed)
  const Wnp = np.array(P.W);
  const baseOp = {
    M,
    WT: np.transpose(Wnp.ref),                            // D×M
    AT: np.transpose(np.array(P.A)),                      // M×D
    w2: np.sum(np.square(Wnp), 1).reshape([1, M]),        // 1×M
  };
  const UinT = np.transpose(np.array(P.Uin));             // d_in×D
  const CT = np.transpose(np.array(P.C));                 // D×ncls
  const z0 = np.array([P.z0]);                            // 1×D
  const cb = np.array([P.cb]);                            // 1×ncls
  const mean = np.array([model.pca.mean]);
  const basisT = np.transpose(np.array(model.pca.basis)); // D×2

  // the edit, on-device: anchors (m×D) + their unit directions for the A columns
  const ancT = np.transpose(np.array(E.anchors));         // D×m
  const anc2 = np.sum(np.square(np.array(E.anchors)), 1).reshape([1, E.m_teach]);
  const ancUnitJS = E.anchors.map((r) => { const n = Math.hypot(...r); return r.map((v) => v / n); });

  // build the operator with the dynamics edit at absolute gain γ (0 -> the base
  // operator). Returns fresh device arrays; caller disposes via op.dispose().
  const makeOp = (gammaAbs) => {
    if (!gammaAbs) return { ...baseOp, dispose() {} };
    const Wjs = P.W.concat(E.anchors);
    const W2 = np.array(Wjs);
    const cols = np.array(ancUnitJS).mul(gammaAbs);        // m×D
    const A2T = np.concatenate([np.transpose(np.array(P.A)), cols], 0); // (M+m)×D
    const op = {
      M: M + E.m_teach,
      WT: np.transpose(W2.ref),
      AT: A2T,
      w2: np.sum(np.square(W2), 1).reshape([1, M + E.m_teach]),
      dispose() { op.WT.dispose(); op.AT.dispose(); op.w2.dispose(); },
    };
    return op;
  };
  const editOp = makeOp(E.gamma);                          // the shipped (certified) edit

  // φ_W(Z) against an operator's bank: n×M
  const feats = (op, Z, n) => {
    const z2 = np.sum(np.square(Z.ref), 1).reshape([n, 1]);
    const dot = np.matmul(Z, op.WT.ref);
    const num = dot.ref.add(b);
    return num.ref.mul(num).div(z2.add(op.w2.ref).sub(dot.mul(2)).add(eps));
  };
  // one application of the operator: F(Z; xU) = tanh(φ·Aᵀ + xU + z0)
  const applyDev = (op, Z, xU, n) =>
    np.tanh(np.matmul(feats(op, Z, n), op.AT.ref).add(xU.ref).add(z0.ref));

  _cmp = { model, D, M, beta, b, eps, baseOp, editOp, makeOp, UinT, CT, z0, cb,
           mean, basisT, ancT, anc2, applyDev, feats };
  return _cmp;
}

// A stateful iterator over one batch under a chosen operator. step() advances the
// damped-Picard iteration; scores4() reads the three trained logits + the taught
// class's anchor score off the current state. Always dispose().
export function makeRunner(cmp, op, Xjs, Z0js = null) {
  const n = Xjs.length;
  const alpha = cmp.model.edit.alpha;
  const xU = np.matmul(np.array(Xjs), cmp.UinT.ref);
  let Z = Z0js ? np.array(Z0js) : np.zeros([n, cmp.D]);
  const advance = (o) => {
    const Fz = cmp.applyDev(o, Z.ref, xU.ref, n);
    return Z.ref.mul(1 - cmp.beta).add(Fz.mul(cmp.beta));
  };
  return {
    n,
    step(o = op) { const out = advance(o); Z.dispose(); Z = out; return this; },
    stepR(o = op) {
      const out = advance(o);
      const rn = np.sqrt(np.sum(np.square(out.ref.sub(Z.ref)), 1)).js();
      Z.dispose(); Z = out; return rn;
    },
    solve(iters = 80, o = op) { for (let k = 0; k < iters; k++) this.step(o); return this; },
    proj() { return np.matmul(Z.ref.sub(cmp.mean.ref), cmp.basisT.ref).js(); },
    // [l0, l1, l2, s3] per sample: trained linear logits + the constructed anchor score
    scores4() {
      const lg = np.matmul(Z.ref, cmp.CT.ref).add(cmp.cb.ref).js();
      const kA = yatK(cmp, Z.ref, n).js();
      return lg.map((r, i) => [...r, alpha * Math.max(...kA[i])]);
    },
    // distance of each state to its nearest anchor (the taught class's wells)
    anchorDist() {
      const z2 = np.sum(np.square(Z.ref), 1).reshape([n, 1]);
      const dot = np.matmul(Z.ref, cmp.ancT.ref);
      const d2 = z2.add(cmp.anc2.ref).sub(dot.mul(2)).js();
      return d2.map((r) => Math.sqrt(Math.max(0, Math.min(...r))));
    },
    stateJS() { return Z.ref.js(); },
    dispose() { Z.dispose(); xU.dispose(); },
  };
}

// the Yat kernel of a batch of states against the anchors: n×m
function yatK(cmp, Zref, n) {
  const z2 = np.sum(np.square(Zref.ref), 1).reshape([n, 1]);
  const dot = np.matmul(Zref, cmp.ancT.ref);
  const num = dot.ref.add(cmp.b);
  return num.ref.mul(num).div(z2.add(cmp.anc2.ref).sub(dot.mul(2)).add(cmp.eps));
}

// ── the certificate: per-sample ‖J_F‖₂ at z*, power iteration on JᵀJ ─────────────
// Analytic products from φ_i(z) = n_i²/q_i, n_i = ⟨z,w_i⟩+b, q_i = ‖z−w_i‖²+ε:
//   ∂φ_i/∂z = (G1+G2)_i w_i − G2_i z,   G1 = 2n/q,  G2 = 2n²/q²
//   J v  = D ⊙ [((G1+G2) ⊙ (V Wᵀ) − G2 ⊙ ⟨z,v⟩) Aᵀ]          D = 1 − tanh²(pre)
//   Jᵀu  = (c ⊙ (G1+G2)) W − rowsum(c ⊙ G2) ⊙ z,             c = (D ⊙ u) A
// Everything is a batched matmul on the device; nothing is approximated.
export function sigma(cmp, op, Xjs, { solveIters = 70, powerIters = 9 } = {}) {
  const n = Xjs.length, Mo = op.M;
  const xU = np.matmul(np.array(Xjs), cmp.UinT.ref);
  let Z = np.zeros([n, cmp.D]);
  for (let k = 0; k < solveIters; k++) {
    const Fz = cmp.applyDev(op, Z.ref, xU.ref, n);
    const out = Z.ref.mul(1 - cmp.beta).add(Fz.mul(cmp.beta));
    Z.dispose(); Z = out;
  }
  // pieces of the Jacobian at z*
  const z2 = np.sum(np.square(Z.ref), 1).reshape([n, 1]);
  const dot = np.matmul(Z.ref, op.WT.ref);                       // n×M  ⟨z,w⟩
  const num = dot.ref.add(cmp.b);                                // n×M  n_i
  const q = z2.add(op.w2.ref).sub(dot.mul(2)).add(cmp.eps);      // n×M  q_i
  const G1 = num.ref.mul(2).div(q.ref);                          // n×M
  const G2 = num.ref.mul(num.ref).mul(2).div(q.ref.mul(q.ref));  // n×M
  const G12 = G1.add(G2.ref);                                    // n×M  (G1 consumed)
  const K = num.ref.mul(num).div(q);                             // n×M  φ at z*
  const pre = np.matmul(K, op.AT.ref).add(xU.ref).add(cmp.z0.ref);
  const t = np.tanh(pre);
  const Dact = np.ones([n, cmp.D]).sub(t.ref.mul(t));            // n×D
  const W = np.transpose(op.WT.ref);                             // M×D
  const A = np.transpose(op.AT.ref);                             // D×M

  // each takes ownership of its argument (consumes it exactly once)
  const Jv = (V) => {
    const vw = np.matmul(V.ref, op.WT.ref);                      // n×M ⟨v,w⟩
    const zv = np.sum(Z.ref.mul(V), 1).reshape([n, 1]);          // n×1 ⟨z,v⟩ (consumes V)
    const dphi = G12.ref.mul(vw).sub(G2.ref.mul(zv));            // n×M
    return Dact.ref.mul(np.matmul(dphi, op.AT.ref));             // n×D
  };
  const Jtu = (U) => {
    const c = np.matmul(Dact.ref.mul(U), A.ref);                 // n×M (consumes U)
    const s = np.sum(c.ref.mul(G2.ref), 1).reshape([n, 1]);      // n×1
    return np.matmul(c.mul(G12.ref), W.ref).sub(Z.ref.mul(s));   // n×D
  };

  let V = np.array(Xjs.map(() => Array.from({ length: cmp.D }, () => Math.random() - 0.5)));
  let nv = np.sqrt(np.sum(np.square(V.ref), 1)).reshape([n, 1]);
  V = V.div(nv.add(1e-9));
  let sig = null;
  for (let it = 0; it < powerIters; it++) {
    const JtJv = Jtu(Jv(V.ref));
    if (sig) sig.dispose();
    sig = np.sqrt(np.sqrt(np.sum(np.square(JtJv.ref), 1)).add(1e-12));  // ≈ ‖J‖₂ per sample
    const nn2 = np.sqrt(np.sum(np.square(JtJv.ref), 1)).reshape([n, 1]);
    const Vn = JtJv.div(nn2.add(1e-9));
    V.dispose(); V = Vn;
  }
  const out = sig.js();
  for (const a of [xU, Z, G2, G12, Dact, W, A, V]) a.dispose();
  return out;
}

// ── off-main-thread certificate ──────────────────────────────────────────────
// sigma() is ~10^9 multiplies per call (70 solve + 9 power iterations). The
// certificate dial re-runs it on every gain setting, so we push it to a Web
// Worker to keep the slider responsive, with a synchronous main-thread fallback
// if workers are unavailable. One shared worker; requests are keyed by an id so a
// stale reply from a superseded gain is ignored. The worker replays the exact
// same math (certgauge.worker.js), so the returned slopes are identical.
let _sigWorker, _sigPending = new Map(), _sigReqId = 0, _sigBroken = false;
function ensureSigWorker() {
  if (_sigBroken) return null;
  if (_sigWorker) return _sigWorker;
  if (typeof Worker === 'undefined') { _sigBroken = true; return null; }
  try { _sigWorker = new Worker(new URL('./certgauge.worker.js', import.meta.url), { type: 'module' }); }
  catch { _sigBroken = true; return null; }
  _sigWorker.onmessage = (e) => {
    const { reqId, slopes, error } = e.data, p = _sigPending.get(reqId);
    if (!p) return; _sigPending.delete(reqId);
    if (error) { p.reject(new Error(error)); return; }
    p.resolve(Array.from(new Float32Array(slopes)));
  };
  _sigWorker.onerror = () => {
    _sigBroken = true;
    for (const p of _sigPending.values()) p.reject(new Error('worker error'));
    _sigPending.clear(); try { _sigWorker.terminate(); } catch {} _sigWorker = null;
  };
  return _sigWorker;
}

// ── off-main-thread one-shot solve ───────────────────────────────────────────
// A settle-to-equilibrium solve is dozens of iterations of two batched matmuls
// (~10^8 multiplies for a 48-input batch). The one-shot setup solves that a panel
// runs when it scrolls into view push through here so the page never hitches, with
// a synchronous fallback. The worker replays the exact iteration, so z* and its
// projection are identical.
let _solWorker, _solPending = new Map(), _solReqId = 0, _solBroken = false;
function ensureSolveWorker() {
  if (_solBroken) return null;
  if (_solWorker) return _solWorker;
  if (typeof Worker === 'undefined') { _solBroken = true; return null; }
  try { _solWorker = new Worker(new URL('./yatdeqsolve.worker.js', import.meta.url), { type: 'module' }); }
  catch { _solBroken = true; return null; }
  _solWorker.onmessage = (e) => {
    const { reqId, z, proj, error } = e.data, p = _solPending.get(reqId);
    if (!p) return; _solPending.delete(reqId);
    if (error) { p.reject(new Error(error)); return; }
    p.resolve({ z, proj });
  };
  _solWorker.onerror = () => {
    _solBroken = true;
    for (const p of _solPending.values()) p.reject(new Error('worker error'));
    _solPending.clear(); try { _solWorker.terminate(); } catch {} _solWorker = null;
  };
  return _solWorker;
}

// Solve a batch Xjs to equilibrium under an operator, off the main thread. `which`
// is 'base' | 'edit' | 'gain' (for 'gain', pass gammaAbs). Resolves with
// { z, proj } identical to makeRunner(cmp, op, Xjs).solve(iters) then stateJS()/proj().
export function solveAsync(cmp, which, Xjs, iters = 80, gammaAbs = 0) {
  const w = ensureSolveWorker();
  const sync = () => {
    const op = which === 'edit' ? cmp.editOp : which === 'gain' ? cmp.makeOp(gammaAbs) : cmp.baseOp;
    const r = makeRunner(cmp, op, Xjs).solve(iters, op);
    const out = { z: r.stateJS(), proj: r.proj() };
    r.dispose(); if (which === 'gain') op.dispose();
    return out;
  };
  if (!w) return Promise.resolve(sync());
  const reqId = ++_solReqId;
  return new Promise((resolve) => {
    _solPending.set(reqId, { resolve, reject: () => resolve(sync()) });   // graceful fallback
    w.postMessage({ model: cmp.model, op: which, gammaAbs, X: Xjs, iters, reqId });
  });
}

// Estimate the per-sample ‖J_F‖₂ certificate at edit gain `gammaAbs` off the main
// thread. Resolves with the same array sigma(cmp, makeOp(gammaAbs), Xjs) returns.
export function sigmaAsync(cmp, gammaAbs, Xjs, opts = {}) {
  const w = ensureSigWorker();
  if (!w) {
    const op = cmp.makeOp(gammaAbs), r = sigma(cmp, op, Xjs, opts); op.dispose();
    return Promise.resolve(r);
  }
  const reqId = ++_sigReqId;
  return new Promise((resolve, reject) => {
    _sigPending.set(reqId, {
      resolve,
      reject: (err) => {                                    // graceful fallback
        try { const op = cmp.makeOp(gammaAbs), r = sigma(cmp, op, Xjs, opts); op.dispose(); resolve(r); }
        catch { reject(err); }
      },
    });
    w.postMessage({ model: cmp.model, gammaAbs, X: Xjs, reqId, ...opts });
  });
}
