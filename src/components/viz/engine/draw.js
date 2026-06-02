// draw.js — canvas drawing primitives shared by every viz. All take a 2-D ctx
// already transformed to CSS pixels. Colours come from vizkit's theme reader.

export const PALETTE = ['#b3661b', '#4a7fb3', '#3a8f5e', '#9a4f9c', '#c2553a', '#5a5f66'];

// split a region into a cols×rows grid of inner boxes (with a small gap)
export function grid(W, H, cols, rows, gap = 8) {
  const cells = []; const pw = W / cols, ph = H / rows;
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++)
    cells.push({ x: c * pw + gap / 2, y: r * ph + gap / 2, w: pw - gap, h: ph - gap });
  return cells;
}

// a framed panel; returns a disc mapper for [-1,1]^2 -> canvas (header reserves top px)
export function panel(ctx, box, colors, { header = 0 } = {}) {
  ctx.fillStyle = colors.cellBg; ctx.fillRect(box.x, box.y, box.w, box.h);
  ctx.strokeStyle = colors.border; ctx.lineWidth = 1; ctx.strokeRect(box.x + 0.5, box.y + 0.5, box.w - 1, box.h - 1);
  const ix = box.x, iy = box.y + header, iw = box.w, ih = box.h - header;
  const side = Math.min(iw, ih) - 8, ox = ix + (iw - side) / 2, oy = iy + (ih - side) / 2;
  const cx = ox + side / 2, cy = oy + side / 2, R = side * 0.42;
  return { cx, cy, R, side, x: ox, y: oy, map: (p) => [cx + p[0] * R, cy - p[1] * R] };
}

export function unitCircle(ctx, m, colors) { ctx.beginPath(); ctx.arc(m.cx, m.cy, m.R, 0, 2 * Math.PI); ctx.strokeStyle = colors.border; ctx.lineWidth = 1; ctx.stroke(); }

export function dot(ctx, x, y, r, color) { ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fill(); }
export function cross(ctx, x, y, r, color) { ctx.strokeStyle = color; ctx.lineWidth = 1.3; ctx.beginPath(); ctx.moveTo(x - r, y - r); ctx.lineTo(x + r, y + r); ctx.moveTo(x + r, y - r); ctx.lineTo(x - r, y + r); ctx.stroke(); }
// shape: 0 -> filled circle, 1 -> cross
export function mark(ctx, x, y, r, shape, color) { shape ? cross(ctx, x, y, r, color) : dot(ctx, x, y, r, color); }

// parse a CSS colour ('#rgb', '#rrggbb', 'rgb(...)') to [r,g,b]
export function parseColor(v) {
  v = (v || '').trim();
  if (v[0] === '#') { let s = v.slice(1); if (s.length === 3) s = s.split('').map((c) => c + c).join(''); return [parseInt(s.slice(0, 2), 16), parseInt(s.slice(2, 4), 16), parseInt(s.slice(4, 6), 16)]; }
  const m = v.match(/\d+(\.\d+)?/g); return m ? [+m[0], +m[1], +m[2]] : [0, 0, 0];
}

// N×N matrix heatmap (flat row-major A) in a square inside box; lerps cellBg→accent.
// opts: {activeRow, gamma, title, colors override via base/hi}
export function heatmap(ctx, box, A, N, colors, { activeRow = -1, gamma = 0.6 } = {}) {
  const side = Math.min(box.w, box.h); const ox = box.x + (box.w - side) / 2, oy = box.y + (box.h - side) / 2;
  const base = parseColor(colors.cellBg), hi = parseColor(colors.accent);
  ctx.fillStyle = colors.cellBg; ctx.fillRect(ox, oy, side, side);
  const cell = side / N;
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    const a = A[i * N + j]; const t = Math.min(1, Math.pow(Math.max(0, a) * N * 0.7, gamma));
    ctx.fillStyle = `rgb(${Math.round(base[0] + (hi[0] - base[0]) * t)},${Math.round(base[1] + (hi[1] - base[1]) * t)},${Math.round(base[2] + (hi[2] - base[2]) * t)})`;
    ctx.fillRect(ox + j * cell, oy + i * cell, cell + 0.5, cell + 0.5);
  }
  if (activeRow >= 0) { ctx.strokeStyle = colors.fg; ctx.lineWidth = 1.4; ctx.strokeRect(ox + 0.5, oy + activeRow * cell + 0.5, side - 1, cell); }
  ctx.strokeStyle = colors.border; ctx.lineWidth = 1; ctx.strokeRect(ox + 0.5, oy + 0.5, side - 1, side - 1);
  return { ox, oy, side, cell };
}

export function arrow(ctx, x0, y0, x1, y1, color, w = 2) {
  ctx.strokeStyle = color; ctx.lineWidth = w; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
  const a = Math.atan2(y1 - y0, x1 - x0), h = 7;
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x1 - h * Math.cos(a - 0.4), y1 - h * Math.sin(a - 0.4));
  ctx.moveTo(x1, y1); ctx.lineTo(x1 - h * Math.cos(a + 0.4), y1 - h * Math.sin(a + 0.4)); ctx.stroke();
}

export function text(ctx, str, x, y, colors, { size = 11, weight = '', mono = false, align = 'left', baseline = 'top', color } = {}) {
  ctx.fillStyle = color || colors.fg; ctx.font = `${weight} ${size}px ${mono ? 'ui-monospace, monospace' : 'ui-sans-serif, system-ui'}`;
  ctx.textAlign = align; ctx.textBaseline = baseline; ctx.fillText(str, x, y);
}

// line plot of one or more series in a framed box; each: {data:[], color, scale:(v)=>0..1}
export function sparkline(ctx, box, series, colors, { dashedMid = false } = {}) {
  ctx.fillStyle = colors.cellBg; ctx.fillRect(box.x, box.y, box.w, box.h);
  ctx.strokeStyle = colors.border; ctx.lineWidth = 1; ctx.strokeRect(box.x + 0.5, box.y + 0.5, box.w - 1, box.h - 1);
  if (dashedMid) { const mid = box.y + box.h / 2; ctx.strokeStyle = colors.border; ctx.setLineDash([2, 2]); ctx.beginPath(); ctx.moveTo(box.x, mid); ctx.lineTo(box.x + box.w, mid); ctx.stroke(); ctx.setLineDash([]); }
  for (const s of series) { const n = s.data.length; if (n < 2) continue; ctx.strokeStyle = s.color; ctx.lineWidth = 1.5; ctx.beginPath();
    for (let i = 0; i < n; i++) { const x = box.x + (i / (n - 1)) * box.w, y = box.y + box.h - Math.max(0, Math.min(1, s.scale(s.data[i]))) * (box.h - 4) - 2; i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); } ctx.stroke(); }
}
