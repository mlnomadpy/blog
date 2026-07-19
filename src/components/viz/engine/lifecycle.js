// lifecycle.js, teardown + play-gate helpers for the bundled-module bespoke viz
// (ContrastiveLab, SimoExplorer). Those files hand-rolled their own theme
// MutationObserver, ResizeObserver, IntersectionObserver and an rAF loop and never
// cleaned up, so an Astro view-transition swap left the detached canvas's observers
// firing and its loop running forever. `onDetach` registers a cleanup that runs once
// the node leaves the DOM.
//
// The registry (`window.__vizReg`) and the sweep that empties it on `astro:after-swap`
// / `astro:page-load` are shared with the inline bespoke viz and set up in
// BaseLayout.astro (the sweep runs at every navigation, exactly when nodes detach).
// We only ensure the registry Set exists here so import order does not matter; the
// sweep listeners are BaseLayout's responsibility (guarded by window.__vizSweepHooked).
//
//   import { onDetach, readyPulse, clearReadyPulse } from './engine/lifecycle.js';
//   onDetach(canvas, () => { cancelAnimationFrame(raf); themeObs.disconnect(); ro.disconnect(); io.disconnect(); });
//   readyPulse(playBtn);          // one-time "press play" nudge on the still preview
//   ... in play(): clearReadyPulse(playBtn);

export function onDetach(node, cleanup) {
  if (typeof window === 'undefined') return () => {};
  const R = window.__vizReg || (window.__vizReg = new Set());
  const entry = { n: node, c: cleanup };
  R.add(entry);
  return () => R.delete(entry);
}

// The pulse keyframe (.viz-play-ready) lives in global.css so both module and inline
// bespoke viz share it; here we just toggle the class.
export function readyPulse(btn) { if (btn) btn.classList.add('viz-play-ready'); }
export function clearReadyPulse(btn) { if (btn) btn.classList.remove('viz-play-ready'); }
