// jax.js — the engine's compute backend. All blog ML math runs on real jax-js
// (@jax-js/jax): NumPy/JAX in the browser. This module owns the one async bit
// (device init) and the glue that lets a synchronous vizkit spec use it:
// kick `withJax` in setup(), guard draw() with `drawJaxLoading`.
//
//   import { np, nn, withJax, drawJaxLoading } from './engine/jax.js';
//   setup(api){ /* build JS data into api.state */ withJax(api, recompute); }
//   draw(api){ if (!api.state.jaxReady) return drawJaxLoading(api); ... }
//
// We pin the `cpu` device. jax-js's faster wasm/webgpu backends read tensor data
// back synchronously through a `SharedArrayBuffer`, which the browser only exposes
// on cross-origin-isolated pages (COOP/COEP) — and COEP would break giscus and our
// other cross-origin embeds site-wide. The `cpu` backend is the reference JS
// interpreter: its synchronous read is a plain buffer slice (no SharedArrayBuffer),
// so it runs in every browser with zero isolation. It's still real jax-js — true
// NumPy semantics, real LU-based `linalg.solve`, real `nn.softmax` — just
// interpreted rather than SIMD-compiled, which is irrelevant at our problem sizes.
//
// Reminder: arrays have Rust-like move semantics — every op consumes its inputs.
// Reuse a value -> take `.ref`. A forward pass that allocates fresh and reads out
// once per call never leaks; only long training loops that reuse params do.
import { numpy as np, nn, init, defaultDevice, tree, random } from '@jax-js/jax';
import { text } from './draw.js';

let _ready = null;
// memoised init. `cpu` is registered at import time, so init('cpu') resolves
// immediately (no WebGPU adapter request, no Wasm workers); we then pin it default.
export const ensureJax = () => (_ready ||= init('cpu').then(() => { defaultDevice('cpu'); return true; }));

// Kick init from a vizkit setup(); marks api.state.jaxReady and re-renders when
// the device is live (all jax ops are synchronous after init resolves).
export function withJax(api, compute) {
  api.state.jaxReady = false;
  ensureJax().then(() => { if (!api.state) return; api.state.jaxReady = true; if (compute) compute(api); api.render(); });
}

// standard loading frame to show until the device is ready
export function drawJaxLoading(api) {
  const { ctx, colors, size } = api;
  text(ctx, 'initializing jax-js…', size.W / 2, size.H / 2, colors,
    { align: 'center', baseline: 'middle', color: colors.muted, size: 12, mono: true });
}

export { np, nn, tree, random };
