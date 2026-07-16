// Shared data access for the "you don't have to solve a kernel machine"
// panels: one memoized fetch per exported asset (wall.json, scale.json,
// compose.json, panel.json + the Fashion-MNIST sprite), plus the count / time /
// memory formatters the panels share. Every number the panels show comes from
// these files; nothing is hardcoded, so a fuller run that overwrites them flows
// straight through.
import { loadJSON, loadImage } from './engine/io.js';

const BASE = import.meta.env.BASE_URL.replace(/\/$/, '');
const DIR = `${BASE}/you-dont-have-to-solve-a-kernel-machine`;

// fetch JSON once and never reject: a missing export resolves to null so the
// panel can render its pending state instead of throwing.
const _safe = new Map();
function safeJSON(url) {
  if (!_safe.has(url)) _safe.set(url, fetch(url).then((r) => (r.ok ? r.json() : null)).catch(() => null));
  return _safe.get(url);
}

export const loadWall = () => loadJSON(`${DIR}/wall.json`);
export const loadScale = () => loadJSON(`${DIR}/scale.json`);
export const loadPanel = () => Promise.all([
  safeJSON(`${DIR}/panel.json`),
  loadImage(`${DIR}/panel.png`),
  safeJSON(`${DIR}/compose.json`),
]).then(([meta, img, compose]) => ({ meta, img, compose }));

// ── formatters ──
export function fmtCount(n) {
  n = Math.round(n);
  if (n >= 1e6) return (n / 1e6).toFixed(n % 1e6 ? 1 : 0) + 'M';
  if (n >= 1e4) return Math.round(n / 1e3) + 'k';
  if (n >= 1e3) return n % 1e3 ? (n / 1e3).toFixed(1) + 'k' : n / 1e3 + 'k';
  return String(n);
}

export function humanTime(s) {
  if (!Number.isFinite(s)) return '?';
  if (s < 1e-3) return (s * 1e6).toFixed(0) + ' µs';
  if (s < 1) return (s * 1e3).toFixed(s < 0.01 ? 1 : 0) + ' ms';
  if (s < 90) return s.toFixed(s < 10 ? 1 : 0) + ' s';
  if (s < 5400) return (s / 60).toFixed(1) + ' min';
  if (s < 172800) return (s / 3600).toFixed(1) + ' h';
  if (s < 3.156e7) return (s / 86400).toFixed(1) + ' days';
  return (s / 3.156e7).toFixed(1) + ' years';
}

export function humanGB(gb) {
  if (gb >= 1024) return (gb / 1024).toFixed(1) + ' TB';
  if (gb >= 1) return gb.toFixed(gb < 10 ? 1 : 0) + ' GB';
  if (gb >= 1e-3) return (gb * 1024).toFixed(1) + ' MB';
  return (gb * 1048576).toFixed(0) + ' KB';
}
