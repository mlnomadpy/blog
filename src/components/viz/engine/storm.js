// storm.js — shared attention bookkeeping used by SoftmaxStorm, LinearReservoir
// and the combined BookkeepingFour viz. Two pure-canvas panels (no jax needed):
//   storm     = exact softmax attention as an all-pairs web (visible work ~ N x N);
//   reservoir = linear attention pouring keys/values into a fixed feature state
//               phi(K)^T V that every query reads (token work ~ N x m).
// Both renderers are fraction-based so they fit any box, including a 2x2 quadrant.
import { text, arrow, parseColor } from './draw.js';

const TWO_PI = Math.PI * 2;
export const MAX_N = 96;
export const MAX_M = 24;

export function drawStormPanel(ctx, box, colors, iter, N) {
  const cx = box.x + box.w * 0.5;
  const cy = box.y + box.h * 0.54;
  const R = Math.min(box.w, box.h * 1.1) * 0.3;
  const pts = tokenCircle(cx, cy, R, N, iter * 0.003);

  text(ctx, 'exact attention', cx, box.y + 2, colors, { align: 'center', size: 14, weight: '700', color: colors.blue });
  text(ctx, 'every token asks every token', cx, box.y + 24, colors, { align: 'center', size: 11.5, color: colors.muted });

  const maxEdges = Math.min(1200, N * N);
  ctx.lineWidth = 0.55;
  for (let e = 0; e < maxEdges; e++) {
    const i = (e * 37 + iter) % N;
    const j = (e * 91 + 17) % N;
    if (i === j) continue;
    const a = pts[i], b = pts[j];
    const glow = 0.04 + 0.16 * Math.abs(Math.sin((e + iter) * 0.021));
    ctx.strokeStyle = rgba(colors.blue, glow);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  for (let i = 0; i < N; i++) {
    const p = pts[i];
    ctx.fillStyle = i === N - 1 ? colors.accent : colors.blue;
    ctx.beginPath();
    ctx.arc(p.x, p.y, i === N - 1 ? 4.5 : 3, 0, TWO_PI);
    ctx.fill();
  }

  text(ctx, 'score matrix grows with the square of tokens', cx, box.y + box.h - 14, colors, { align: 'center', size: 11, color: colors.muted });
}

export function drawReservoirPanel(ctx, box, colors, iter, N, m, phase) {
  const cx = box.x + box.w * 0.5;
  const bankW = Math.min(150, box.w * 0.52);
  const bank = {
    x: cx - bankW / 2,
    y: box.y + box.h * 0.30,
    w: bankW,
    h: Math.max(44, box.h * 0.22),
  };
  const tokenY = box.y + box.h * 0.82;

  text(ctx, 'linear attention', cx, box.y + 2, colors, { align: 'center', size: 14, weight: '700', color: colors.accent });
  text(ctx, 'tokens pour into a fixed feature state', cx, box.y + 24, colors, { align: 'center', size: 11.5, color: colors.muted });

  drawFeatureBank(ctx, bank, m, phase, colors);
  text(ctx, 'φ(K)^T V', cx, bank.y - 16, colors, { align: 'center', mono: true, size: 12, weight: '700', color: colors.accent });

  const tokenCount = Math.min(N, 26);
  const span = box.w - 54;
  const startX = box.x + 27;
  const rowsBank = Math.max(1, Math.ceil(m / 6) - 1 || 1);
  for (let i = 0; i < tokenCount; i++) {
    const x = startX + (tokenCount === 1 ? 0.5 : i / (tokenCount - 1)) * span;
    const y = tokenY + Math.sin(i * 0.8 + iter * 0.05) * 4;
    ctx.fillStyle = i === tokenCount - 1 ? colors.blue : colors.green;
    ctx.beginPath();
    ctx.arc(x, y, i === tokenCount - 1 ? 4.5 : 3, 0, TWO_PI);
    ctx.fill();
    const slot = i % m;
    const tx = bank.x + bank.w * (0.12 + 0.76 * ((slot % 6) / 5));
    const ty = bank.y + bank.h * (0.12 + 0.76 * (Math.floor(slot / 6) / rowsBank));
    const pulse = ((iter * 0.025 + i / tokenCount) % 1);
    const px = x + (tx - x) * pulse;
    const py = y + (ty - y) * pulse;
    ctx.strokeStyle = rgba(colors.green, 0.14);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, y - 5);
    ctx.lineTo(tx, ty);
    ctx.stroke();
    ctx.fillStyle = colors.green;
    ctx.beginPath();
    ctx.arc(px, py, 2.1, 0, TWO_PI);
    ctx.fill();
  }

  const qx = startX + span;
  arrow(ctx, qx, tokenY - 14, bank.x + bank.w, bank.y + bank.h * 0.52, colors.blue, 1.7);
  text(ctx, 'query reads state', qx - 16, tokenY - 30, colors, { align: 'right', size: 10.5, color: colors.blue });

  text(ctx, 'feature state changes; its shape stays m x d', cx, box.y + box.h - 14, colors, { align: 'center', size: 11, color: colors.muted });
}

function tokenCircle(cx, cy, R, n, rot) {
  const pts = [];
  for (let i = 0; i < n; i++) {
    const a = rot + TWO_PI * i / n;
    pts.push({ x: cx + Math.cos(a) * R, y: cy + Math.sin(a) * R });
  }
  return pts;
}

function drawFeatureBank(ctx, box, m, phase, colors) {
  ctx.fillStyle = colors.cellBg;
  ctx.fillRect(box.x, box.y, box.w, box.h);
  ctx.strokeStyle = colors.border;
  ctx.strokeRect(box.x + 0.5, box.y + 0.5, box.w - 1, box.h - 1);
  const cols = 6, rows = Math.ceil(m / cols);
  const cw = (box.w - 10) / cols, ch = (box.h - 10) / rows;
  for (let i = 0; i < m; i++) {
    const c = i % cols, r = Math.floor(i / cols);
    const x = box.x + 5 + c * cw, y = box.y + 5 + r * ch;
    const v = 0.25 + 0.75 * Math.abs(Math.sin(phase * 8 + i * 0.9));
    ctx.fillStyle = mix(colors.cellBg, colors.accent, v);
    ctx.fillRect(x, y, cw - 2, ch - 2);
  }
}

export function fmt(n) {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(n >= 1e4 ? 0 : 1)}K`;
  return `${n}`;
}

function rgba(hex, a) { const p = parseColor(hex); return `rgba(${p[0]},${p[1]},${p[2]},${a})`; }
function mix(a, b, t) {
  const pa = parseColor(a), pb = parseColor(b);
  t = Math.max(0, Math.min(1, t));
  return `rgb(${Math.round(pa[0] + (pb[0] - pa[0]) * t)},${Math.round(pa[1] + (pb[1] - pa[1]) * t)},${Math.round(pa[2] + (pb[2] - pa[2]) * t)})`;
}
