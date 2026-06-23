// Off-main-thread Yat-kernel evaluation. Receives the flat test-bank and
// prototype pixels, evaluates the full kernel on jax-js (cpu backend, the same
// reference interpreter the engine pins), reduces to per-class strongest
// activations, and posts back a compact [n*10] Float32 buffer. Running here keeps
// the ~10^8-multiply matmul off the UI thread, so the page never freezes.
import { numpy as np, init, defaultDevice } from '@jax-js/jax';

const ready = init('cpu').then(() => { defaultDevice('cpu'); return true; });

self.onmessage = async (e) => {
  try {
    await ready;
    const { bank, proto, n, D, K, b, eps, vote } = e.data;
    const X = np.reshape(np.array(Array.from(bank)), [n, D]);
    const W = np.reshape(np.array(Array.from(proto)), [K, D]);
    const dot = np.matmul(X.ref, np.transpose(W.ref));             // [n,K] = x·Wᵀ
    const xn = np.reshape(np.sum(X.ref.mul(X), 1), [n, 1]);        // ||x||²
    const wn = np.reshape(np.sum(W.ref.mul(W), 1), [1, K]);        // ||W||²
    const dist2 = np.add(xn, wn).sub(dot.ref.mul(2));              // ||x-W||²
    const lin = dot.add(b);
    const ker = lin.ref.mul(lin).div(dist2.add(eps)).js();        // (dot+b)²/(dist²+eps)

    const pm = new Float32Array(n * 10).fill(-Infinity);          // per-class max
    for (let i = 0; i < n; i++) { const row = ker[i], o = i * 10;
      for (let u = 0; u < K; u++) { const c = vote[u]; if (row[u] > pm[o + c]) pm[o + c] = row[u]; } }
    self.postMessage({ pm: pm.buffer, n }, [pm.buffer]);          // transfer back (cheap)
  } catch (err) {
    self.postMessage({ error: String(err && err.message || err) });
  }
};
