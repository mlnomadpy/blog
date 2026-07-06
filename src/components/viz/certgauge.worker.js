// Off-main-thread contraction-certificate estimator for the "edit a fixed point"
// certificate dial. Given the trained model params and an edit gain, this rebuilds
// the exact edited operator and runs the same power-iteration on JᵀJ that the main
// thread would (70 damped-Picard solve steps to reach z*, then 9 power iterations
// with analytic Jacobian-vector products). That is ~10^9 multiplies per gain
// setting; running it here keeps the dial responsive and the page from freezing.
//
// The math below is a verbatim port of compute()/makeOp()/sigma() in yatdeqedit.js
// (same jax-js cpu backend, same float32 interpreter), so the returned slopes are
// bit-for-bit what the synchronous path produces. Do not "optimize" the math here;
// a worker is a threading change, not a numerical one.
import { numpy as np, init, defaultDevice } from '@jax-js/jax';

const ready = init('cpu').then(() => { defaultDevice('cpu'); return true; });

// Build the on-device compute object from the raw model JSON (verbatim from
// yatdeqedit.js compute()). Only the pieces sigma() needs are kept.
function buildCmp(model) {
  const P = model.params, D = model.dims.d, M = model.dims.m;
  const beta = model.solver.beta, b = P.b, eps = P.eps;
  const Wnp = np.array(P.W);
  const baseOp = {
    M,
    WT: np.transpose(Wnp.ref),                            // D×M
    AT: np.transpose(np.array(P.A)),                      // M×D
    w2: np.sum(np.square(Wnp), 1).reshape([1, M]),        // 1×M
  };
  const UinT = np.transpose(np.array(P.Uin));             // d_in×D
  const z0 = np.array([P.z0]);                            // 1×D
  const E = model.edit;
  const ancUnitJS = E.anchors.map((r) => { const n = Math.hypot(...r); return r.map((v) => v / n); });

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

  const feats = (op, Z, n) => {
    const z2 = np.sum(np.square(Z.ref), 1).reshape([n, 1]);
    const dot = np.matmul(Z, op.WT.ref);
    const num = dot.ref.add(b);
    return num.ref.mul(num).div(z2.add(op.w2.ref).sub(dot.mul(2)).add(eps));
  };
  const applyDev = (op, Z, xU, n) =>
    np.tanh(np.matmul(feats(op, Z, n), op.AT.ref).add(xU.ref).add(z0.ref));

  return { D, M, beta, b, eps, baseOp, makeOp, UinT, z0, applyDev, feats };
}

// verbatim port of sigma() in yatdeqedit.js
function sigma(cmp, op, Xjs, { solveIters = 70, powerIters = 9 } = {}) {
  const n = Xjs.length;
  const xU = np.matmul(np.array(Xjs), cmp.UinT.ref);
  let Z = np.zeros([n, cmp.D]);
  for (let k = 0; k < solveIters; k++) {
    const Fz = cmp.applyDev(op, Z.ref, xU.ref, n);
    const out = Z.ref.mul(1 - cmp.beta).add(Fz.mul(cmp.beta));
    Z.dispose(); Z = out;
  }
  const z2 = np.sum(np.square(Z.ref), 1).reshape([n, 1]);
  const dot = np.matmul(Z.ref, op.WT.ref);                       // n×M  ⟨z,w⟩
  const num = dot.ref.add(cmp.b);                                // n×M  n_i
  const q = z2.add(op.w2.ref).sub(dot.mul(2)).add(cmp.eps);      // n×M  q_i
  const G1 = num.ref.mul(2).div(q.ref);                          // n×M
  const G2 = num.ref.mul(num.ref).mul(2).div(q.ref.mul(q.ref));  // n×M
  const G12 = G1.add(G2.ref);                                    // n×M
  const K = num.ref.mul(num).div(q);                             // n×M  φ at z*
  const pre = np.matmul(K, op.AT.ref).add(xU.ref).add(cmp.z0.ref);
  const t = np.tanh(pre);
  const Dact = np.ones([n, cmp.D]).sub(t.ref.mul(t));            // n×D
  const W = np.transpose(op.WT.ref);                             // M×D
  const A = np.transpose(op.AT.ref);                             // D×M

  const Jv = (V) => {
    const vw = np.matmul(V.ref, op.WT.ref);                      // n×M ⟨v,w⟩
    const zv = np.sum(Z.ref.mul(V), 1).reshape([n, 1]);          // n×1 ⟨z,v⟩
    const dphi = G12.ref.mul(vw).sub(G2.ref.mul(zv));            // n×M
    return Dact.ref.mul(np.matmul(dphi, op.AT.ref));             // n×D
  };
  const Jtu = (U) => {
    const c = np.matmul(Dact.ref.mul(U), A.ref);                 // n×M
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
    sig = np.sqrt(np.sqrt(np.sum(np.square(JtJv.ref), 1)).add(1e-12));
    const nn2 = np.sqrt(np.sum(np.square(JtJv.ref), 1)).reshape([n, 1]);
    const Vn = JtJv.div(nn2.add(1e-9));
    V.dispose(); V = Vn;
  }
  const out = sig.js();
  for (const a of [xU, Z, G2, G12, Dact, W, A, V]) a.dispose();
  return out;
}

self.onmessage = async (e) => {
  try {
    await ready;
    const { model, gammaAbs, X, reqId } = e.data;
    const cmp = buildCmp(model);
    const op = cmp.makeOp(gammaAbs);
    const slopes = sigma(cmp, op, X);
    op.dispose();
    const buf = Float32Array.from(slopes);
    self.postMessage({ reqId, slopes: buf.buffer }, [buf.buffer]);
  } catch (err) {
    self.postMessage({ reqId: e.data && e.data.reqId, error: String(err && err.message || err) });
  }
};
