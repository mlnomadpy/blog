// Off-main-thread fixed-point solver for the "edit a fixed point" panels. Given the
// trained model JSON, which operator to use (base or the shipped edit), and a batch
// of inputs, it runs the same damped-Picard iteration to the equilibrium z* and
// returns the settled states plus their 2-D PCA projection. Each solve is dozens of
// iterations of two batched matmuls over a 48-input batch (~10^8 multiplies); running
// it here keeps the panel from freezing while it scrolls into view.
//
// The math is a verbatim port of compute()/makeRunner().solve() in yatdeqedit.js
// (same jax-js cpu float32 interpreter), so z* and its projection are identical to
// the synchronous path. Do not alter the math; a worker is a threading change only.
import { numpy as np, init, defaultDevice } from '@jax-js/jax';

const ready = init('cpu').then(() => { defaultDevice('cpu'); return true; });

// verbatim port of compute() in yatdeqedit.js (only the solve pieces are kept)
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
  const mean = np.array([model.pca.mean]);
  const basisT = np.transpose(np.array(model.pca.basis)); // D×2
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

  const editOp = makeOp(E.gamma);
  return { D, M, beta, b, eps, baseOp, editOp, makeOp, UinT, z0, mean, basisT, applyDev, feats };
}

// verbatim port of makeRunner().solve(iters) + stateJS()/proj() in yatdeqedit.js
function solve(cmp, op, Xjs, iters) {
  const n = Xjs.length;
  const xU = np.matmul(np.array(Xjs), cmp.UinT.ref);
  let Z = np.zeros([n, cmp.D]);
  for (let k = 0; k < iters; k++) {
    const Fz = cmp.applyDev(op, Z.ref, xU.ref, n);
    const out = Z.ref.mul(1 - cmp.beta).add(Fz.mul(cmp.beta));
    Z.dispose(); Z = out;
  }
  const z = Z.ref.js();
  const proj = np.matmul(Z.sub(cmp.mean.ref), cmp.basisT.ref).js();
  xU.dispose();
  return { z, proj };
}

self.onmessage = async (e) => {
  try {
    await ready;
    const { model, op: which, gammaAbs, X, iters, reqId } = e.data;
    const cmp = buildCmp(model);
    let op;
    if (which === 'edit') op = cmp.editOp;
    else if (which === 'gain') op = cmp.makeOp(gammaAbs);
    else op = cmp.baseOp;
    const out = solve(cmp, op, X, iters);
    if (which === 'gain') op.dispose();
    self.postMessage({ reqId, z: out.z, proj: out.proj });
  } catch (err) {
    self.postMessage({ reqId: e.data && e.data.reqId, error: String(err && err.message || err) });
  }
};
