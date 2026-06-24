// pointer.js, one place for pointer math and drag wiring, so components stop
// hand-rolling getBoundingClientRect + pointer capture + drag state. Coordinates
// are returned in CSS pixels relative to the canvas, which is the same space
// vizkit's draw() works in (api.size.W/H), so a hit-test against draw geometry
// just works.

// Pointer event → {x, y} in CSS pixels relative to the canvas top-left.
export function localPoint(canvas, e) {
  const r = canvas.getBoundingClientRect();
  return { x: e.clientX - r.left, y: e.clientY - r.top };
}

// Index of the first rect {x,y,w,h} containing p, else -1.
export function hitTest(p, rects) {
  for (let i = 0; i < rects.length; i++) {
    const r = rects[i];
    if (p.x >= r.x && p.x <= r.x + r.w && p.y >= r.y && p.y <= r.y + r.h) return i;
  }
  return -1;
}

// Wire drag + hover on a canvas with pointer capture. Callbacks receive
// (point, event); onHover fires on every move, onMove only while a drag is active.
// Idempotent (guards against double-binding under Astro view transitions).
export function drag(canvas, { onStart, onMove, onEnd, onHover, cursor } = {}) {
  if (canvas.dataset.vkDrag) return; canvas.dataset.vkDrag = '1';
  let active = false;
  const down = (e) => {
    active = true; canvas.setPointerCapture?.(e.pointerId);
    if (cursor) canvas.style.cursor = cursor;
    onStart && onStart(localPoint(canvas, e), e);
  };
  const move = (e) => {
    const p = localPoint(canvas, e);
    onHover && onHover(p, e);
    if (active && onMove) onMove(p, e);
  };
  const end = (e) => {
    if (!active) return; active = false;
    try { canvas.releasePointerCapture?.(e.pointerId); } catch {}
    onEnd && onEnd(localPoint(canvas, e), e);
  };
  canvas.addEventListener('pointerdown', down);
  canvas.addEventListener('pointermove', move);
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);
}
