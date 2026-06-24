// nbody.js, shared gravity bookkeeping used by DirectNBody, FieldMesh and the
// combined BookkeepingFour viz. A physically honest toy, not a GR simulator:
//   direct = softened Newtonian N-body force summation (pure JS, O(N^2));
//   field  = Newtonian particle-mesh gravity: deposit mass density on a fixed
//            grid, solve a discrete Poisson equation with jax-js, sample -grad Phi.
import { text, parseColor } from './draw.js';
import { np } from './jax.js';

export const TAU = Math.PI * 2;
export const MAX_N = 72;
export const GRID = 13;
export const CELLS = GRID * GRID;
const DT = 0.032;
const G = 0.018;
const EPS2 = 0.018;
const FIELD_GAIN = 0.095;
const DAMP = 0.999;
const BOUND = 0.94;
export const GRID_TOUCHES_PER_BODY = 8; // 4 bilinear deposits + 4 bilinear samples

// ── setup ──
export function setupBodies(api) {
  const { pos, vel, mass } = initialBodies(MAX_N);
  api.state.pairPos = pos.slice();
  api.state.pairVel = vel.slice();
  api.state.meshPos = pos.slice();
  api.state.meshVel = vel.slice();
  api.state.mass = mass;
}

export function setupField(api) {
  api.state.laplace = poissonMatrix(GRID);
  api.state.phi = new Float64Array(CELLS);
  api.state.rho = new Float64Array(CELLS);
  api.state.grad = new Float64Array(CELLS * 2);
  api.state.energy = 0;
}

// ── one physics tick (each side advances its own copy of the bodies) ──
export function stepDirect(s, N) {
  const a = directAcceleration(s.pairPos, s.mass, N);
  integrate(s.pairPos, s.pairVel, a, N);
}

export function stepField(s, N) {
  const field = fieldAcceleration(s.meshPos, s.mass, N, s.laplace);
  s.phi = field.phi;
  s.rho = field.rho;
  s.grad = field.grad;
  integrate(s.meshPos, s.meshVel, field.accel, N);
  s.energy = field.fieldEnergy;
}

// ── panels ──
export function drawDirectPanel(ctx, box, colors, state, N) {
  panelTitle(ctx, 'direct Newtonian N-body', 'sum every softened pairwise force', box, colors.blue, colors);
  drawDomain(ctx, box, colors);
  drawPairEdges(ctx, box, state.pairPos, N, colors);
  drawBodies(ctx, box, state.pairPos, state.mass, N, colors, false);
  text(ctx, 'work grows like N(N-1)', box.x + box.w / 2, box.y + box.h - 14, colors, { align: 'center', size: 11, color: colors.muted });
}

export function drawFieldPanel(ctx, box, colors, state, N) {
  panelTitle(ctx, 'particle-mesh field solve', 'deposit rho, solve del^2 Phi, read -grad Phi', box, colors.accent, colors);
  drawField(ctx, box, state.phi, state.grad, colors);
  drawBodies(ctx, box, state.meshPos, state.mass, N, colors, true);
  text(ctx, 'body work linear in N once the grid is fixed', box.x + box.w / 2, box.y + box.h - 14, colors, { align: 'center', size: 11, color: colors.muted });
}

// ── physics ──
// reused acceleration scratch buffers: the consumed region [0,2N) is fully
// overwritten each call and read immediately by integrate, so reuse is exact and
// avoids a Float64Array allocation every animation frame.
const _accDirect = new Float64Array(MAX_N * 2);
const _accSample = new Float64Array(MAX_N * 2);

function directAcceleration(pos, mass, N) {
  const acc = _accDirect;
  for (let i = 0; i < N; i++) {
    const xi = pos[2 * i], yi = pos[2 * i + 1];
    let ax = 0, ay = 0;
    for (let j = 0; j < N; j++) {
      if (i === j) continue;
      const dx = pos[2 * j] - xi, dy = pos[2 * j + 1] - yi;
      const r2 = dx * dx + dy * dy + EPS2;
      const inv = 1 / (r2 * Math.sqrt(r2));
      ax += G * mass[j] * dx * inv;
      ay += G * mass[j] * dy * inv;
    }
    acc[2 * i] = ax;
    acc[2 * i + 1] = ay;
  }
  return acc;
}

function fieldAcceleration(pos, mass, N, laplace) {
  const rho = depositDensity(pos, mass, N);
  const rhs = Array.from(rho, (v) => 4 * Math.PI * G * v);
  for (let y = 0; y < GRID; y++) {
    for (let x = 0; x < GRID; x++) {
      if (x === 0 || y === 0 || x === GRID - 1 || y === GRID - 1) rhs[idx(x, y)] = 0;
    }
  }
  // Discrete Poisson equation: del^2 Phi = 4 pi G rho, Dirichlet boundary Phi=0.
  // The grid is fixed, so this solve cost does not grow with N in the toy.
  const phiRaw = np.linalg.solve(np.array(laplace), np.array(rhs)).js();
  const phi = Float64Array.from(phiRaw);
  const grad = gradient(phi);
  const accel = sampleFieldAcceleration(pos, grad, N);
  let e = 0;
  for (let i = 0; i < CELLS; i++) e += rho[i] * phi[i];
  return { rho, phi, grad, accel, fieldEnergy: e };
}

function depositDensity(pos, mass, N) {
  const rho = new Float64Array(CELLS);
  for (let i = 0; i < N; i++) {
    const gx = toGrid(pos[2 * i]);
    const gy = toGrid(pos[2 * i + 1]);
    const x0 = Math.floor(gx), y0 = Math.floor(gy);
    const tx = gx - x0, ty = gy - y0;
    splat(rho, x0, y0, (1 - tx) * (1 - ty) * mass[i]);
    splat(rho, x0 + 1, y0, tx * (1 - ty) * mass[i]);
    splat(rho, x0, y0 + 1, (1 - tx) * ty * mass[i]);
    splat(rho, x0 + 1, y0 + 1, tx * ty * mass[i]);
  }
  const cellArea = Math.pow(2 / (GRID - 1), 2);
  for (let i = 0; i < rho.length; i++) rho[i] /= cellArea;
  return rho;
}

function gradient(phi) {
  const grad = new Float64Array(CELLS * 2);
  const h = 2 / (GRID - 1);
  for (let y = 0; y < GRID; y++) {
    for (let x = 0; x < GRID; x++) {
      const xm = Math.max(0, x - 1), xp = Math.min(GRID - 1, x + 1);
      const ym = Math.max(0, y - 1), yp = Math.min(GRID - 1, y + 1);
      const gx = (phi[idx(xp, y)] - phi[idx(xm, y)]) / ((xp - xm) * h || h);
      const gy = (phi[idx(x, yp)] - phi[idx(x, ym)]) / ((yp - ym) * h || h);
      grad[2 * idx(x, y)] = -gx * FIELD_GAIN;
      grad[2 * idx(x, y) + 1] = -gy * FIELD_GAIN;
    }
  }
  return grad;
}

function sampleFieldAcceleration(pos, grad, N) {
  const acc = _accSample;
  for (let i = 0; i < N; i++) {
    const gx = toGrid(pos[2 * i]);
    const gy = toGrid(pos[2 * i + 1]);
    const x0 = Math.floor(gx), y0 = Math.floor(gy);
    const tx = gx - x0, ty = gy - y0;
    const a00 = readGrad(grad, x0, y0);
    const a10 = readGrad(grad, x0 + 1, y0);
    const a01 = readGrad(grad, x0, y0 + 1);
    const a11 = readGrad(grad, x0 + 1, y0 + 1);
    acc[2 * i] = lerp(lerp(a00.x, a10.x, tx), lerp(a01.x, a11.x, tx), ty);
    acc[2 * i + 1] = lerp(lerp(a00.y, a10.y, tx), lerp(a01.y, a11.y, tx), ty);
  }
  return acc;
}

function integrate(pos, vel, acc, N) {
  for (let i = 0; i < N; i++) {
    const k = 2 * i;
    vel[k] = (vel[k] + acc[k] * DT) * DAMP;
    vel[k + 1] = (vel[k + 1] + acc[k + 1] * DT) * DAMP;
    pos[k] += vel[k] * DT;
    pos[k + 1] += vel[k + 1] * DT;
    for (const d of [0, 1]) {
      const q = k + d;
      if (pos[q] > BOUND) { pos[q] = BOUND; vel[q] *= -0.65; }
      if (pos[q] < -BOUND) { pos[q] = -BOUND; vel[q] *= -0.65; }
    }
  }
}

// ── drawing ──
function drawDomain(ctx, box, colors) {
  const r = plotRegion(box);
  ctx.fillStyle = colors.cellBg;
  ctx.fillRect(r.x, r.y, r.w, r.h);
  ctx.strokeStyle = colors.border;
  ctx.strokeRect(r.x + 0.5, r.y + 0.5, r.w - 1, r.h - 1);
  ctx.strokeStyle = rgba(colors.blue, 0.12);
  ctx.lineWidth = 1;
  for (let i = 1; i < 5; i++) {
    const x = r.x + (i / 5) * r.w;
    const y = r.y + (i / 5) * r.h;
    ctx.beginPath();
    ctx.moveTo(x, r.y);
    ctx.lineTo(x, r.y + r.h);
    ctx.moveTo(r.x, y);
    ctx.lineTo(r.x + r.w, y);
    ctx.stroke();
  }
}

function drawPairEdges(ctx, box, pos, N, colors) {
  const r = plotRegion(box);
  const maxEdges = Math.min(900, N * (N - 1));
  ctx.lineWidth = 0.5;
  for (let e = 0; e < maxEdges; e++) {
    const i = (e * 37) % N;
    const j = (e * 83 + 11) % N;
    if (i === j) continue;
    const a = mapPoint(r, pos[2 * i], pos[2 * i + 1]);
    const b = mapPoint(r, pos[2 * j], pos[2 * j + 1]);
    ctx.strokeStyle = rgba(colors.blue, 0.035 + 0.12 * ((e % 19) / 18));
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }
}

function drawField(ctx, box, phi, grad, colors) {
  const r = plotRegion(box);
  let lo = Infinity, hi = -Infinity;
  for (const v of phi) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const range = hi - lo || 1;
  const cw = r.w / GRID, ch = r.h / GRID;
  for (let y = 0; y < GRID; y++) {
    for (let x = 0; x < GRID; x++) {
      const t = (phi[idx(x, y)] - lo) / range;
      ctx.fillStyle = mix(colors.blue, colors.accent, t);
      ctx.globalAlpha = 0.12 + 0.42 * t;
      ctx.fillRect(r.x + x * cw, r.y + y * ch, cw + 0.5, ch + 0.5);
    }
  }
  ctx.globalAlpha = 1;
  ctx.strokeStyle = colors.border;
  ctx.strokeRect(r.x + 0.5, r.y + 0.5, r.w - 1, r.h - 1);
  ctx.strokeStyle = rgba(colors.accent, 0.42);
  ctx.lineWidth = 1;
  for (let y = 1; y < GRID - 1; y += 2) {
    for (let x = 1; x < GRID - 1; x += 2) {
      const p = mapPoint(r, -1 + 2 * x / (GRID - 1), -1 + 2 * y / (GRID - 1));
      const gx = grad[2 * idx(x, y)], gy = grad[2 * idx(x, y) + 1];
      const s = 220;
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(p.x + gx * s, p.y - gy * s);
      ctx.stroke();
    }
  }
}

function drawBodies(ctx, box, pos, mass, N, colors, fieldSide) {
  const r = plotRegion(box);
  for (let i = 0; i < N; i++) {
    const p = mapPoint(r, pos[2 * i], pos[2 * i + 1]);
    const rad = 2.8 + mass[i] * 2.2;
    ctx.fillStyle = i === N - 1 ? colors.accent : mix(fieldSide ? colors.accent : colors.blue, colors.green, ((i * 17) % 100) / 100);
    ctx.beginPath();
    ctx.arc(p.x, p.y, rad, 0, TAU);
    ctx.fill();
  }
}

function panelTitle(ctx, head, sub, box, color, colors) {
  const cx = box.x + box.w / 2;
  text(ctx, head, cx, box.y + 2, colors, { align: 'center', size: 14, weight: '700', color });
  text(ctx, sub, cx, box.y + 24, colors, { align: 'center', size: 10.8, color: colors.muted });
}

// ── cost models ──
export function directCost(n) { return n * (n - 1); }
export function meshVariableCost(n) { return GRID_TOUCHES_PER_BODY * n + CELLS; }

// ── geometry / helpers ──
function initialBodies(n) {
  const pos = new Float64Array(n * 2);
  const vel = new Float64Array(n * 2);
  const mass = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const a = i * 2.399963;
    const r = 0.12 + 0.76 * Math.sqrt((i + 0.5) / n);
    const x = Math.cos(a) * r;
    const y = Math.sin(a) * r;
    const m = 0.6 + ((i * 29) % 100) / 120;
    const tang = 0.18 / Math.sqrt(r + 0.18);
    pos[2 * i] = x;
    pos[2 * i + 1] = y;
    vel[2 * i] = -y * tang + 0.006 * Math.sin(i * 1.7);
    vel[2 * i + 1] = x * tang + 0.006 * Math.cos(i * 1.3);
    mass[i] = m;
  }
  return { pos, vel, mass };
}

function poissonMatrix(g) {
  const h = 2 / (g - 1);
  const invH2 = 1 / (h * h);
  const A = Array.from({ length: g * g }, () => Array(g * g).fill(0));
  for (let y = 0; y < g; y++) {
    for (let x = 0; x < g; x++) {
      const k = idx(x, y);
      if (x === 0 || y === 0 || x === g - 1 || y === g - 1) {
        A[k][k] = 1;
        continue;
      }
      A[k][k] = -4 * invH2;
      A[k][idx(x - 1, y)] = invH2;
      A[k][idx(x + 1, y)] = invH2;
      A[k][idx(x, y - 1)] = invH2;
      A[k][idx(x, y + 1)] = invH2;
    }
  }
  return A;
}

function plotRegion(box) {
  const top = box.y + 44;
  const bottom = box.y + box.h - 26;
  const side = Math.max(72, Math.min(box.w - 24, bottom - top));
  return { x: box.x + (box.w - side) / 2, y: top + (bottom - top - side) / 2, w: side, h: side };
}

function mapPoint(r, x, y) {
  return { x: r.x + ((x + 1) / 2) * r.w, y: r.y + ((1 - y) / 2) * r.h };
}

function toGrid(x) { return clamp01((x + 1) / 2) * (GRID - 1.001); }

function splat(rho, x, y, v) {
  x = Math.max(0, Math.min(GRID - 1, x));
  y = Math.max(0, Math.min(GRID - 1, y));
  rho[idx(x, y)] += v;
}

function readGrad(grad, x, y) {
  x = Math.max(0, Math.min(GRID - 1, x));
  y = Math.max(0, Math.min(GRID - 1, y));
  const k = idx(x, y);
  return { x: grad[2 * k], y: grad[2 * k + 1] };
}

function idx(x, y) { return y * GRID + x; }
function lerp(a, b, t) { return a + (b - a) * t; }
function clamp01(x) { return Math.max(0, Math.min(1, x)); }

export function fmt(n) {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(n >= 1e4 ? 0 : 1)}K`;
  return `${n}`;
}

function rgba(hex, a) { const p = parseColor(hex); return `rgba(${p[0]},${p[1]},${p[2]},${a})`; }
function mix(a, b, t) {
  const pa = parseColor(a), pb = parseColor(b);
  t = clamp01(t);
  return `rgb(${Math.round(pa[0] + (pb[0] - pa[0]) * t)},${Math.round(pa[1] + (pb[1] - pa[1]) * t)},${Math.round(pa[2] + (pb[2] - pa[2]) * t)})`;
}
