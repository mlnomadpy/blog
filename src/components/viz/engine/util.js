// util.js, tiny shared utilities: memoization for pure recompute, and flat
// typed-matrix helpers to replace nested JS arrays in hot numeric code.

// Memoize a pure function on a primitive cache key. Bounded LRU-ish (drops the
// oldest when over `max`). Use for per-frame recompute whose inputs rarely change
//, e.g. a kernel lookup keyed by a slider value.
export function memo(fn, keyFn = (...a) => a.join('|'), max = 24) {
  const cache = new Map();
  return (...args) => {
    const k = keyFn(...args);
    const hit = cache.get(k);
    if (hit !== undefined || cache.has(k)) return hit;
    const v = fn(...args);
    cache.set(k, v);
    if (cache.size > max) cache.delete(cache.keys().next().value);
    return v;
  };
}

// Flat row-major Float64 matrix (single allocation, cache-friendly) to replace
// Array.from({length:n}, () => Array(m)) patterns in eigensolvers / Poisson grids.
export const mat = (rows, cols, fill = 0) => {
  const a = new Float64Array(rows * cols);
  if (fill) a.fill(fill);
  return { rows, cols, a };
};
export const at = (m, i, j) => m.a[i * m.cols + j];
export const put = (m, i, j, v) => { m.a[i * m.cols + j] = v; };
export const addTo = (m, i, j, v) => { m.a[i * m.cols + j] += v; };
