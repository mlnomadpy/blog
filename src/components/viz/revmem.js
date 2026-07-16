// Shared data + math for the backprop-without-the-memory panels.
// wall.json carries the run's measured numbers (memory, fidelity, training).
// The momentum-net demo math runs live: forward v' = mu v + f(x), x' = x + v',
// inverse x = x' - v', v = (v' - f(x)) / mu. Float32 is emulated exactly with
// Math.fround so the rewind pays the same arithmetic the GPU pays.

const BASE = `${import.meta.env.BASE_URL ?? '/'}`.replace(/\/$/, '');

let cache = null;
export async function load() {
  if (cache) return cache;
  const r = await fetch(`${BASE}/backprop-without-the-memory/wall.json`);
  cache = await r.json();
  return cache;
}

// deterministic small MLP field f: R^2 -> R^2 (the algebra holds for ANY f;
// weights are a fixed seeded draw, stated in captions)
function mulberry(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function makeField(seed = 7, hidden = 16, scale = 1.4) {
  const rnd = mulberry(seed);
  const g = () => (rnd() * 2 - 1);
  const W1 = Array.from({ length: hidden }, () => [g() * scale, g() * scale]);
  const b1 = Array.from({ length: hidden }, () => g() * 0.4);
  const W2 = Array.from({ length: 2 }, () => Array.from({ length: hidden }, () => g() * scale / Math.sqrt(hidden)));
  return { W1, b1, W2, hidden };
}

const f32 = Math.fround;

// f(x) with every intermediate rounded to float32
export function fieldF32(F, x) {
  const h = new Array(F.hidden);
  for (let j = 0; j < F.hidden; j++) {
    h[j] = f32(Math.tanh(f32(f32(F.W1[j][0] * x[0]) + f32(F.W1[j][1] * x[1]) + F.b1[j])));
  }
  const o = [0, 0];
  for (let k = 0; k < 2; k++) {
    let s = 0;
    for (let j = 0; j < F.hidden; j++) s = f32(s + f32(F.W2[k][j] * h[j]));
    o[k] = f32(s * 0.15);
  }
  return o;
}

// forward L momentum steps in float32; returns the list of (x, v) states
export function forwardF32(F, x0, v0, mu, L) {
  let x = [f32(x0[0]), f32(x0[1])], v = [f32(v0[0]), f32(v0[1])];
  const states = [{ x: [...x], v: [...v] }];
  for (let l = 0; l < L; l++) {
    const f = fieldF32(F, x);
    v = [f32(f32(mu * v[0]) + f[0]), f32(f32(mu * v[1]) + f[1])];
    x = [f32(x[0] + v[0]), f32(x[1] + v[1])];
    states.push({ x: [...x], v: [...v] });
  }
  return states;
}

// rewind from the endpoint only, in float32: the (1/mu)^L amplification is real
export function rewindF32(F, xL, vL, mu, L) {
  let x = [f32(xL[0]), f32(xL[1])], v = [f32(vL[0]), f32(vL[1])];
  const states = [{ x: [...x], v: [...v] }];
  for (let l = 0; l < L; l++) {
    x = [f32(x[0] - v[0]), f32(x[1] - v[1])];
    const f = fieldF32(F, x);
    v = [f32(f32(v[0] - f[0]) / mu), f32(f32(v[1] - f[1]) / mu)];
    states.push({ x: [...x], v: [...v] });
  }
  return states.reverse();
}

// damped / undamped pendulum step (semi-implicit Euler), gamma = friction
export function pendStep(q, p, dt, gamma) {
  p = p - dt * (Math.sin(q) + gamma * p);
  q = q + dt * p;
  return [q, p];
}
// time-reversed step: negate dt; friction keeps its sign (dissipation does not reverse)
export function pendStepBack(q, p, dt, gamma) {
  q = q - dt * p;
  p = p + dt * (Math.sin(q) + gamma * p);
  return [q, p];
}
