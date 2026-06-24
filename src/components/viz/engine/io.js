// Shared async I/O for visualizations. Fetches each JSON or image at most once
// per URL and hands every caller the same promise, so a page with many panels
// that read the same asset downloads and parses it a single time instead of once
// per panel. Also a small viewport trigger so panels defer their work until they
// are scrolled near.
const _json = new Map();
const _img = new Map();

export function loadJSON(url) {
  if (!_json.has(url)) _json.set(url, fetch(url).then((r) => r.json()));
  return _json.get(url);
}

export function loadImage(url) {
  if (!_img.has(url)) _img.set(url, new Promise((res) => {
    const im = new Image(); im.onload = () => res(im); im.onerror = () => res(null); im.src = url;
  }));
  return _img.get(url);
}

// run cb once, when `el` first scrolls near the viewport (nothing loads on mount)
export function whenVisible(el, cb, margin = '700px') {
  if (typeof IntersectionObserver === 'undefined') { cb(); return; }
  const io = new IntersectionObserver((es) => { if (es[0].isIntersecting) { io.disconnect(); cb(); } }, { rootMargin: margin });
  io.observe(el);
}
