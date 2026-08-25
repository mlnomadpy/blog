// Shared data + helpers for the interrogation-protocol panels. Everything the
// panels show is recomputed here from the audit's exported quantities.

import { grayGlyph, sprite } from './engine/draw.js';

const BASE = `${import.meta.env.BASE_URL ?? '/'}`.replace(/\/$/, '');

const cache = {};
export function load(name) {
  cache[name] ??= fetch(`${BASE}/yat-protocol/${name}.json`).then((r) => r.json());
  return cache[name];
}

export function thumb(ctx, img, x, y, w, h, key) {
  grayGlyph(ctx, img, 14, x, y, w, h, { invert: true, key });
}

// A signed image (a concept picture is a difference of prototypes, so it has
// both signs): blue where negative, orange where positive.
export function signedGlyph(ctx, img, side, x, y, w, h, key) {
  const paint = (octx) => {
    const im = octx.createImageData(side, side);
    let mx = 1e-12;
    for (const v of img) mx = Math.max(mx, Math.abs(v));
    for (let i = 0; i < side * side; i++) {
      const t = img[i] / mx;
      const a = Math.min(1, Math.abs(t));
      const pos = t >= 0;
      im.data[i * 4] = Math.round(250 - (pos ? 60 : 180) * a);
      im.data[i * 4 + 1] = Math.round(248 - (pos ? 170 : 130) * a);
      im.data[i * 4 + 2] = Math.round(244 - (pos ? 210 : 60) * a);
      im.data[i * 4 + 3] = 255;
    }
    octx.putImageData(im, 0, 0);
  };
  const src = key == null
    ? (() => { const c = document.createElement('canvas'); c.width = c.height = side; paint(c.getContext('2d')); return c; })()
    : sprite(key, side, side, paint);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(src, x, y, w, h);
}

// Rebuild the network from its top-k concepts: project the logit vector onto
// the leading k concept axes. Exact, and exact at k = 10 by construction.
export function rebuildFromConcepts(L, E, base, k) {
  const n = L.length, C = base.length;
  const pred = new Int8Array(n);
  const acc = new Float64Array(C);
  for (let i = 0; i < n; i++) {
    const l = L[i];
    acc.fill(0);
    for (let j = 0; j < k; j++) {
      let d = 0;
      for (let c = 0; c < C; c++) d += l[c] * E[c][j];   // coordinate along concept j
      for (let c = 0; c < C; c++) acc[c] += d * E[c][j];
    }
    let bi = 0;
    for (let c = 1; c < C; c++) if (acc[c] + base[c] > acc[bi] + base[bi]) bi = c;
    pred[i] = bi;
  }
  return pred;
}

export const accuracy = (pred, y) => {
  let ok = 0;
  for (let i = 0; i < pred.length; i++) if (pred[i] === y[i]) ok++;
  return (100 * ok) / pred.length;
};

// R^2 between two standardized score vectors (the channel ledger's statistic)
export function r2(a, b) {
  let sa = 0, sb = 0, sab = 0, saa = 0, sbb = 0;
  const n = a.length;
  for (let i = 0; i < n; i++) { sa += a[i]; sb += b[i]; }
  const ma = sa / n, mb = sb / n;
  for (let i = 0; i < n; i++) {
    const da = a[i] - ma, db = b[i] - mb;
    sab += da * db; saa += da * da; sbb += db * db;
  }
  const r = sab / Math.sqrt(saa * sbb + 1e-30);
  return r * r;
}
