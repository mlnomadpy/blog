// theme.js, the canvas colour palette, read from CSS custom properties, plus a
// helper to re-read it when the site theme toggles. Single source for vizkit and
// for the bespoke (non-vizkit) components that used to reimplement readColors.
export function readColors() {
  const s = getComputedStyle(document.documentElement);
  const v = (n, f) => (s.getPropertyValue(n).trim() || f);
  return { accent: v('--accent', '#b3661b'), blue: '#4a7fb3', green: '#3a8f5e', muted: v('--fg-muted', '#5a5f66'), faint: v('--fg-faint', '#8d8d8d'), border: v('--border', '#e4e1d6'), fg: v('--fg', '#1a1a1a'), bg: v('--bg', '#fbf8f1'), cellBg: v('--bg-elev', '#fff') };
}

// Call cb whenever the site theme attribute changes; returns a disconnect fn.
export function watchTheme(cb) {
  const obs = new MutationObserver(cb);
  obs.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  return () => obs.disconnect();
}
