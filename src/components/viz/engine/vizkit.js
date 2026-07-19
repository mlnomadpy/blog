// vizkit.js, the harness. Give it a control spec + setup/step/draw and it
// renders the control bar and canvas, runs the play/step/reset/speed loop, and
// wires dpr, theme, resize, and visibility (autoplay/pause). Multi-instance and
// Astro-view-transition safe. A new viz is just a short spec object.
//
//   defineViz('myviz', {
//     title, height, autoplay, stepsPerFrame, maxStep,
//     controls: [ {type:'play'},{type:'step'},{type:'reset'},{type:'speed'},
//                 {type:'range', name, label, min,max,step,value, int?, reset?, fmt?},
//                 {type:'select', name, label, options:[{value,label}], value, reset?},
//                 {type:'toggle', name, label, value},
//                 {type:'button', name, label, reset?},
//                 {type:'readout', name} ],
//     setup(api){}, step(api){}, draw(api){},
//   })
// api: { get(name), set(name,v), state, colors, ctx, size:{W,H}, iter, readout(name,txt) }
import { readColors } from './theme.js';

const CSS = `
.vk-cell{position:relative;}
.vk-cell h4{margin:0 0 .4rem;}
.vk-controls{display:flex;flex-direction:column;gap:7px;padding:6px 0 10px;}
.vk-row{display:flex;flex-wrap:wrap;align-items:center;gap:8px 10px;}
.vk-play,.vk-btn,.vk-s,.vk-tool{appearance:none;font-family:inherit;line-height:1;transition:background-color .12s ease,border-color .12s ease,color .12s ease,transform .08s ease;}
.vk-play{background:var(--accent);color:#fff;border:1px solid var(--accent);border-radius:6px;padding:6px 12px;font-size:12px;cursor:pointer;min-width:76px;}
.vk-btn{background:var(--bg-elev,#fff);color:var(--fg-muted);border:1px solid var(--border);border-radius:6px;padding:6px 10px;font-size:12px;cursor:pointer;}
.vk-play:hover,.vk-btn:hover,.vk-s:hover,.vk-tool:hover{border-color:var(--accent);color:var(--fg);}
.vk-play:hover{color:#fff;filter:brightness(.97);}
.vk-play:active,.vk-btn:active,.vk-s:active,.vk-tool:active{transform:translateY(1px);}
.vk-play:focus-visible,.vk-btn:focus-visible,.vk-s:focus-visible,.vk-tool:focus-visible,.vk-controls input:focus-visible,.vk-controls select:focus-visible{outline:2px solid color-mix(in srgb,var(--accent) 55%,transparent);outline-offset:2px;}
.vk-spd{display:inline-flex;align-items:center;gap:4px;color:var(--fg-muted);font-size:12px;}
.vk-s{background:var(--bg-elev,#fff);color:var(--fg-muted);border:1px solid var(--border);border-radius:5px;padding:4px 7px;font-size:11px;cursor:pointer;font-family:ui-monospace,monospace;}
.vk-s.active{background:var(--accent);color:#fff;border-color:var(--accent);}
.vk-controls label{display:inline-flex;align-items:center;gap:6px;color:var(--fg-muted);font-size:12px;}
.vk-range{display:inline-grid!important;grid-template-columns:auto minmax(90px,1fr) minmax(34px,auto);align-items:center;gap:5px 7px;min-width:min(230px,100%);}
.vk-range-name{white-space:nowrap;}
.vk-controls input[type=range]{accent-color:var(--accent);width:100%;min-width:86px;}
.vk-controls input[type=checkbox]{accent-color:var(--accent);}
.vk-controls select,.vk-controls input[type=number],.vk-controls input[type=text]{background:var(--bg-elev,#fff);color:var(--fg);border:1px solid var(--border);border-radius:5px;padding:3px 5px;font-size:12px;font-family:inherit;}
.vk-text input[type=text]:focus{outline:none;border-color:var(--accent);}
.vk-val{font-family:ui-monospace,monospace;font-size:11px;color:var(--fg-muted);min-width:30px;text-align:right;font-variant-numeric:tabular-nums;}
.vk-readout{margin-left:auto;font-family:ui-monospace,monospace;font-size:11px;color:var(--fg-muted);white-space:nowrap;font-variant-numeric:tabular-nums;}
.vk-tools{display:inline-flex;align-items:center;gap:4px;margin-left:4px;}
.vk-tool{width:28px;height:28px;display:inline-grid;place-items:center;background:var(--bg-elev,#fff);color:var(--fg-muted);border:1px solid var(--border);border-radius:6px;font-size:13px;cursor:pointer;}
.vk-cell canvas{width:100%;display:block;border-radius:6px;touch-action:manipulation;}
.vk-cell figcaption{margin-top:.6rem;font-size:.85rem;line-height:1.55;color:var(--fg-muted);}
.vk-annotation{margin-top:.7rem;padding:10px 14px;border:1px solid var(--border);border-radius:6px;background:var(--bg);font-size:14px;line-height:1.55;color:var(--fg);}
.vk-help{margin-top:4px;font-size:10.5px;color:var(--fg-muted);opacity:.78;}
.vk-toast{position:absolute;right:10px;bottom:10px;z-index:4;padding:5px 8px;border:1px solid var(--border);border-radius:6px;background:var(--bg-elev,#fff);color:var(--fg-muted);font-size:11px;box-shadow:0 6px 20px rgba(0,0,0,.08);opacity:0;transform:translateY(4px);pointer-events:none;transition:opacity .16s ease,transform .16s ease;}
.vk-toast.show{opacity:1;transform:translateY(0);}
.viz:fullscreen,.viz:-webkit-full-screen{background:var(--bg);padding:18px;overflow:auto;}
.viz:fullscreen .vk-cell,.viz:-webkit-full-screen .vk-cell{max-width:min(1280px,100%);margin:0 auto;}
.viz:fullscreen .vk-cell canvas,.viz:-webkit-full-screen .vk-cell canvas{min-height:min(74vh,760px);}
.vk-play.vk-ready{animation:vk-ready-pulse 1.6s ease-out 3;}
@keyframes vk-ready-pulse{0%{box-shadow:0 0 0 0 color-mix(in srgb,var(--accent) 55%,transparent);}70%,100%{box-shadow:0 0 0 6px color-mix(in srgb,var(--accent) 0%,transparent);}}
@media (prefers-reduced-motion:reduce){.vk-play.vk-ready{animation:none;}}
@media (max-width:680px){.vk-range{grid-template-columns:1fr minmax(96px,1.4fr) minmax(34px,auto);width:100%;}.vk-readout{width:100%;margin-left:0;}.vk-tools{margin-left:auto;}}
`;
function injectCSS() { if (document.getElementById('vizkit-css')) return; const s = document.createElement('style'); s.id = 'vizkit-css'; s.textContent = CSS; document.head.appendChild(s); }

function readNumber(v) {
  if (v == null || v === '') return undefined;
  const n = Number.parseFloat(String(v).trim());
  return Number.isFinite(n) ? n : undefined;
}

function cssNumber(styles, name) {
  return readNumber(styles.getPropertyValue(name));
}

function resolveHeight(root, spec, width) {
  const styles = getComputedStyle(root);
  const ds = root.dataset || {};
  const base = readNumber(ds.height) ?? cssNumber(styles, '--viz-height') ?? spec.height ?? 360;
  const desktop = readNumber(ds.heightDesktop) ?? cssNumber(styles, '--viz-height-desktop') ?? spec.heightDesktop;
  const mobile = readNumber(ds.heightMobile) ?? cssNumber(styles, '--viz-height-mobile') ?? spec.heightMobile;
  const responsive = width > 0 && width < 680 ? mobile : desktop;
  const scale = readNumber(ds.heightScale) ?? cssNumber(styles, '--viz-height-scale') ?? spec.heightScale ?? 1;
  const min = readNumber(ds.minHeight) ?? cssNumber(styles, '--viz-min-height') ?? spec.minHeight ?? 160;
  const max = readNumber(ds.maxHeight) ?? cssNumber(styles, '--viz-max-height') ?? spec.maxHeight ?? 520;
  return Math.max(min, Math.min(max, Math.round((responsive ?? base) * scale)));
}

function downloadCanvas(canvas, filename) {
  const save = (url) => {
    const a = document.createElement('a');
    a.href = url; a.download = filename; a.rel = 'noopener';
    document.body.appendChild(a); a.click(); a.remove();
    if (url.startsWith('blob:')) URL.revokeObjectURL(url);
  };
  if (canvas.toBlob) canvas.toBlob((blob) => blob && save(URL.createObjectURL(blob)), 'image/png');
  else save(canvas.toDataURL('image/png'));
}

function showToast(root, text) {
  let el = root.querySelector('.vk-toast');
  if (!el) { el = document.createElement('div'); el.className = 'vk-toast'; root.appendChild(el); }
  el.textContent = text; el.classList.add('show');
  clearTimeout(el._vkTimer);
  el._vkTimer = setTimeout(() => el.classList.remove('show'), 1300);
}

function mountOne(root, specOrFactory) {
  if (root.dataset.vkReady) return; root.dataset.vkReady = '1';
  // spec may be a factory (root) => spec, so each instance can configure its
  // controls/title from its own data-* attributes (e.g. per-loss panels).
  const spec = typeof specOrFactory === 'function' ? specOrFactory(root) : specOrFactory;
  injectCSS();
  const cell = root.querySelector('.viz-cell') || root; cell.classList.add('vk-cell');
  const caption = root.dataset.caption || '';
  const initialHeight = resolveHeight(root, spec, root.getBoundingClientRect().width || 0);

  if (spec.title) { const h = document.createElement('h4'); h.textContent = spec.title; cell.appendChild(h); }
  const bar = document.createElement('div'); bar.className = 'vk-controls'; cell.appendChild(bar);
  const canvas = document.createElement('canvas');
  canvas.height = initialHeight;
  canvas.style.height = initialHeight + 'px';
  canvas.setAttribute('role', 'img');
  canvas.setAttribute('aria-label', spec.title || 'Interactive visualization');
  cell.appendChild(canvas);
  let annEl = null;
  if (spec.annotation) { annEl = document.createElement('div'); annEl.className = 'vk-annotation'; cell.appendChild(annEl); }
  if (caption) { const fc = document.createElement('figcaption'); fc.textContent = caption; cell.appendChild(fc); }
  const ctx = canvas.getContext('2d');

  const vals = {}, readouts = {}, controlEls = {}; let speedMul = 1;
  let hasLoop = !!spec.step;       // true for physics viz (spec.step) or a timeline range (range.play)
  let timeline = null, lastTs = 0; // timeline: a range that a Play button auto-advances
  let playing = false, raf = 0, iter = 0, acc = 0, colors = readColors();
  let didSetup = false, renderRaf = 0; // setup is deferred to first viewport entry; paints coalesce into one rAF
  const reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  // Poster mode: a heavy viz can ship a pre-rendered PNG (data-poster) that is
  // shown, with ZERO compute, until the reader opts in (Play, or first interaction
  // for a pure-interactive viz). `?vizposter=1` / window.__VIZ_NO_POSTER forces the
  // live path so the generator can screenshot the real frame. everPlayed gates the
  // one-time "press play" pulse on the button.
  let everPlayed = false;
  const posterName = (root.dataset.poster || '').trim();
  const noPosterFlag = (() => { try { return new URLSearchParams(location.search).has('vizposter') || !!window.__VIZ_NO_POSTER; } catch { return false; } })();
  let usePoster = !!posterName && !noPosterFlag; // "still until intent": defer setup entirely
  let posterImg = null, posterReady = false, posterFailed = false;

  const api = {
    get: (n) => vals[n], set: (n, v) => { vals[n] = v; }, state: {}, iter: 0,
    get colors() { return colors; }, ctx, canvas, size: { W: 0, H: 0 },
    get playing() { return playing; },
    render: () => render(),
    readout: (n, txt) => { if (readouts[n]) readouts[n].textContent = txt; },
    note: (html) => { if (annEl) annEl.innerHTML = html; },
    // programmatically set another control's value (updates DOM + label, no reset)
    setControl: (n, v) => { const e = controlEls[n]; if (!e) { vals[n] = v; return; } vals[n] = e.cast ? e.cast(v) : v; if (e.input) e.input.value = v; if (e.span && e.fmt) e.span.textContent = e.fmt(vals[n]); },
  };

  // ── build control bar ──
  let row = document.createElement('div'); row.className = 'vk-row'; bar.appendChild(row);
  const firstRow = row;
  const newRow = () => { row = document.createElement('div'); row.className = 'vk-row'; bar.appendChild(row); };
  const fit = () => {
    const fsEl = document.fullscreenElement || document.webkitFullscreenElement;
    if (!fsEl || fsEl !== root) {
      const targetHeight = resolveHeight(root, spec, root.getBoundingClientRect().width || canvas.getBoundingClientRect().width || 0);
      if (Math.abs(parseFloat(canvas.style.height || '0') - targetHeight) > 0.5) canvas.style.height = targetHeight + 'px';
    } else {
      canvas.style.height = '';
    }
    const rect = canvas.getBoundingClientRect(); const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, spec.maxDpr || 2));
    api.size.W = rect.width; api.size.H = rect.height; canvas.width = Math.round(rect.width * dpr); canvas.height = Math.round(rect.height * dpr); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); };
  // renderNow paints synchronously; it no-ops before the deferred setup has run
  // (theme/resize observers can fire while the panel is still below the fold).
  const renderNow = () => { if (!didSetup) return; if (api.size.W === 0) fit(); ctx.setTransform(1, 0, 0, 1, 0, 0); const dpr = canvas.width / api.size.W; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, api.size.W, api.size.H); api.iter = iter; spec.draw(api); };
  // render (the public api.render, and every interaction/observer paint) coalesces
  // into a single rAF, so a burst of pointermove/resize/theme events during a drag
  // repaints at most once per frame instead of flooding the main thread.
  const render = () => { if (renderRaf || !didSetup) return; renderRaf = requestAnimationFrame(() => { renderRaf = 0; renderNow(); }); };
  // ── transitions: api.tween(key, target, ms) eases a value toward target across
  // control changes and drives re-renders until it settles; snapped on (re)setup.
  let tweens = {}, tweenRaf = 0;
  const tnow = () => (typeof performance !== 'undefined' && performance.now ? performance.now() : 0);
  const driveTweens = () => { if (tweenRaf) return; const step = () => { tweenRaf = 0; renderNow(); let active = false; for (const k in tweens) if (tweens[k].p < 1) active = true; if (active) tweenRaf = requestAnimationFrame(step); }; tweenRaf = requestAnimationFrame(step); };
  api.tween = (key, target, ms = 300) => {
    if (!Number.isFinite(target)) { tweens[key] = { to: target, val: target, p: 1 }; return target; }
    const now = tnow(); let t = tweens[key];
    if (!t || !Number.isFinite(t.val)) { tweens[key] = { from: target, to: target, val: target, start: now, ms, p: 1 }; return target; }
    if (t.to !== target) { t.from = t.val; t.to = target; t.start = now; t.ms = ms; t.p = 0; }
    const p = t.ms > 0 ? Math.min(1, (now - t.start) / t.ms) : 1; t.p = p; const e = p * p * (3 - 2 * p);
    t.val = t.from + (t.to - t.from) * e; if (p < 1) driveTweens(); return t.val;
  };
  const doSetup = () => { tweens = {}; api.state = {}; iter = 0; api.iter = 0; acc = 0; spec.setup && spec.setup(api); };
  const stepN = (k) => { for (let i = 0; i < k; i++) { spec.step(api); iter++; } api.iter = iter; };
  const stop = () => { playing = false; if (playBtn) playBtn.textContent = '▶ Play'; cancelAnimationFrame(raf); };
  // advance a timeline range by real elapsed time, snapping the slider to its value
  const advanceTimeline = (ts) => {
    if (!lastTs) lastTs = ts;
    const dt = Math.min(ts - lastTs, 100); lastTs = ts;     // clamp dt so a backgrounded tab does not jump
    const tl = timeline, span = tl.max - tl.min || 1;
    tl.pos += span * (dt / tl.period) * speedMul * tl.dir;
    if (tl.pos >= tl.max) { if (tl.loop === 'pingpong') { tl.pos = tl.max; tl.dir = -1; } else if (tl.loop) { tl.pos = tl.min + ((tl.pos - tl.min) % span); } else { tl.pos = tl.max; api.setControl(tl.name, tl.int ? Math.round(tl.pos) : tl.pos); renderNow(); stop(); return; } }
    else if (tl.pos <= tl.min && tl.dir < 0) { tl.pos = tl.min; tl.dir = 1; }
    api.setControl(tl.name, tl.int ? Math.round(tl.pos) : tl.pos);
  };
  const loop = (ts) => { if (!playing) return;
    if (!canvas.isConnected) { stop(); return; } // a view-transition swap detached us mid-play
    if (timeline && !spec.step) { advanceTimeline(ts); if (!playing) return; renderNow(); raf = requestAnimationFrame(loop); return; }
    acc += (spec.stepsPerFrame || 1) * speedMul;
    while (acc >= 1) { spec.step(api); iter++; acc -= 1; if (spec.maxStep && iter >= spec.maxStep) { acc = 0; break; } }
    renderNow(); if (spec.maxStep && iter >= spec.maxStep) { stop(); return; } raf = requestAnimationFrame(loop); };
  const play = () => { if (!hasLoop) return;
    ensureSetup();                                  // a poster viz has deferred setup until now
    everPlayed = true; if (playBtn) playBtn.classList.remove('vk-ready');
    if (timeline && !spec.step) { if (timeline.pos >= timeline.max && timeline.loop !== 'pingpong') timeline.pos = timeline.min; timeline.dir = 1; }
    else if (spec.maxStep && iter >= spec.maxStep) doReset();
    lastTs = 0; playing = true; if (playBtn) playBtn.textContent = '❚❚ Pause'; raf = requestAnimationFrame(loop); };
  const doReset = () => { stop(); doSetup(); renderNow(); };

  let playBtn = null;
  for (const c of spec.controls || []) {
    if (c.type === 'play') { playBtn = document.createElement('button'); playBtn.className = 'vk-play'; playBtn.textContent = '▶ Play'; playBtn.onclick = () => playing ? stop() : play(); row.appendChild(playBtn); }
    else if (c.type === 'step') { const b = document.createElement('button'); b.className = 'vk-btn'; b.textContent = c.label || 'Step'; b.onclick = () => { stop(); stepN(1); render(); }; row.appendChild(b); }
    else if (c.type === 'reset') { const b = document.createElement('button'); b.className = 'vk-btn'; b.textContent = c.label || 'Reset'; b.onclick = doReset; row.appendChild(b); }
    else if (c.type === 'speed') { const wrap = document.createElement('span'); wrap.className = 'vk-spd'; wrap.append('speed');
      (c.presets || [0.5, 1, 2, 4]).forEach(m => { const b = document.createElement('button'); b.className = 'vk-s' + (m === 1 ? ' active' : ''); b.textContent = m + '×';
        b.onclick = () => { speedMul = m; wrap.querySelectorAll('.vk-s').forEach(x => x.classList.remove('active')); b.classList.add('active'); }; wrap.appendChild(b); }); row.appendChild(wrap); }
    else if (c.type === 'newrow') { newRow(); }
    else if (c.type === 'range') {
      // a range with `play` gets a Play button that auto-advances it (a timeline)
      if (c.play) { hasLoop = true; timeline = { name: c.name, min: +c.min, max: +c.max, step: +c.step, int: !!c.int, period: c.play.period || 4000, loop: c.play.loop ?? false, pos: c.int ? Math.round(+c.value) : +c.value, dir: 1 };
        playBtn = document.createElement('button'); playBtn.className = 'vk-play'; playBtn.textContent = '▶ Play'; playBtn.onclick = () => playing ? stop() : play(); row.appendChild(playBtn); }
      const lab = document.createElement('label'); lab.className = 'vk-range';
      const name = document.createElement('span'); name.className = 'vk-range-name'; name.textContent = c.label || c.name;
      const inp = document.createElement('input'); inp.type = 'range'; inp.min = c.min; inp.max = c.max; inp.step = c.step; inp.value = c.value;
      const span = document.createElement('span'); span.className = 'vk-val'; const fmt = c.fmt || (v => v);
      vals[c.name] = c.int ? Math.round(+c.value) : +c.value; span.textContent = fmt(vals[c.name]);
      controlEls[c.name] = { input: inp, span, fmt, cast: c.int ? (v => Math.round(+v)) : (v => +v) };
      inp.setAttribute('aria-label', c.label || c.name);
      inp.oninput = () => { vals[c.name] = c.int ? Math.round(+inp.value) : +inp.value; span.textContent = fmt(vals[c.name]); if (c.play) { if (playing) stop(); timeline.pos = vals[c.name]; timeline.dir = 1; } spec.onControl && spec.onControl(c.name, vals[c.name], api); if (c.reset) doReset(); else if (!playing) render(); };
      lab.appendChild(name); lab.appendChild(inp); lab.appendChild(span); row.appendChild(lab); }
    else if (c.type === 'select') { const lab = document.createElement('label'); lab.append((c.label || c.name) + ' ');
      const sel = document.createElement('select'); (c.options || []).forEach(o => { const op = document.createElement('option'); op.value = o.value; op.textContent = o.label; sel.appendChild(op); });
      sel.value = c.value != null ? c.value : (c.options[0] && c.options[0].value); vals[c.name] = sel.value;
      controlEls[c.name] = { input: sel };
      sel.onchange = () => { vals[c.name] = sel.value; spec.onControl && spec.onControl(c.name, vals[c.name], api); if (c.reset !== false) doReset(); else if (!playing) render(); }; lab.appendChild(sel); row.appendChild(lab); }
    else if (c.type === 'number') { const lab = document.createElement('label'); lab.append((c.label || c.name) + ' ');
      const inp = document.createElement('input'); inp.type = 'number'; if (c.min != null) inp.min = c.min; if (c.max != null) inp.max = c.max; inp.value = c.value; inp.style.width = (c.width || 54) + 'px'; vals[c.name] = +c.value;
      controlEls[c.name] = { input: inp, cast: v => +v };
      inp.onchange = () => { vals[c.name] = +inp.value; spec.onControl && spec.onControl(c.name, vals[c.name], api); if (c.reset !== false) doReset(); else if (!playing) render(); }; lab.appendChild(inp); row.appendChild(lab); }
    else if (c.type === 'text') { const lab = document.createElement('label'); lab.className = 'vk-text'; if (c.label) lab.append((c.label) + ' ');
      const inp = document.createElement('input'); inp.type = 'text'; inp.value = c.value != null ? c.value : ''; if (c.placeholder) inp.placeholder = c.placeholder; inp.style.width = (c.width || 200) + 'px';
      if (c.maxLength) inp.maxLength = c.maxLength; inp.setAttribute('aria-label', c.label || c.name); vals[c.name] = inp.value;
      controlEls[c.name] = { input: inp, cast: v => '' + v };
      let deb; inp.oninput = () => { vals[c.name] = inp.value; clearTimeout(deb); deb = setTimeout(() => { spec.onControl && spec.onControl(c.name, vals[c.name], api); if (!playing) render(); }, c.debounce ?? 120); };
      lab.appendChild(inp); row.appendChild(lab); }
    else if (c.type === 'toggle') { const lab = document.createElement('label'); const inp = document.createElement('input'); inp.type = 'checkbox'; inp.checked = !!c.value; vals[c.name] = !!c.value;
      inp.onchange = () => { vals[c.name] = inp.checked; if (!playing) render(); }; lab.appendChild(inp); lab.append(' ' + (c.label || c.name)); row.appendChild(lab); }
    else if (c.type === 'button') { const b = document.createElement('button'); b.className = 'vk-btn'; b.textContent = c.label || c.name;
      b.onclick = () => { if (c.reset) doReset(); else { spec.onAction && spec.onAction(c.name, api); if (!playing) render(); } }; row.appendChild(b); }
    else if (c.type === 'readout') { const span = document.createElement('span'); span.className = 'vk-readout'; span.setAttribute('aria-live', 'polite'); readouts[c.name] = span; row.appendChild(span); }
  }

  if (spec.tools !== false) {
    const tools = document.createElement('span'); tools.className = 'vk-tools';
    const shot = document.createElement('button');
    shot.type = 'button'; shot.className = 'vk-tool'; shot.textContent = '⇩';
    shot.title = 'Download this plot as PNG'; shot.setAttribute('aria-label', 'Download plot as PNG');
    shot.onclick = () => { renderNow(); downloadCanvas(canvas, `${root.classList[1] || 'viz'}-${Date.now()}.png`); showToast(root, 'PNG saved'); };
    tools.appendChild(shot);
    if (root.requestFullscreen || root.webkitRequestFullscreen) {
      const fs = document.createElement('button');
      fs.type = 'button'; fs.className = 'vk-tool'; fs.textContent = '⛶';
      fs.title = 'Toggle fullscreen'; fs.setAttribute('aria-label', 'Toggle fullscreen');
      fs.onclick = () => {
        const active = document.fullscreenElement === root || document.webkitFullscreenElement === root;
        if (active) (document.exitFullscreen || document.webkitExitFullscreen).call(document);
        else (root.requestFullscreen || root.webkitRequestFullscreen).call(root);
      };
      tools.appendChild(fs);
    }
    firstRow.appendChild(tools);
  }

  if (spec.help !== false) {
    const help = document.createElement('div');
    help.className = 'vk-help';
    help.textContent = spec.step ? 'Shortcuts: Space play/pause, S step, R reset.' : (hasLoop ? 'Shortcut: Space play/pause.' : 'Interactive controls update the plot live.');
    bar.appendChild(help);
  }

  // ── poster (pre-rendered still) ──
  // A one-time nudge: pulse the Play button while the still preview waits, cleared
  // for good on the first play (see play()).
  const markReady = () => { if (playBtn && hasLoop && !everPlayed) playBtn.classList.add('vk-ready'); };
  const posterSrc = () => {
    const base = (import.meta.env.BASE_URL || '/').replace(/\/$/, '');
    const dark = document.documentElement.dataset.theme === 'dark';
    return `${base}/viz-posters/${posterName}${dark ? '-dark' : ''}.png`;
  };
  // paint the poster to fill the canvas (cover-fit, no distortion); owns its own fit()
  // so the resize/theme paths can call it directly.
  const paintPoster = () => {
    if (!posterReady || !posterImg) return;
    fit();
    ctx.clearRect(0, 0, api.size.W, api.size.H);
    const iw = posterImg.naturalWidth, ih = posterImg.naturalHeight;
    if (!iw || !ih) return;
    const s = Math.max(api.size.W / iw, api.size.H / ih);
    const w = iw * s, h = ih * s;
    ctx.drawImage(posterImg, (api.size.W - w) / 2, (api.size.H - h) / 2, w, h);
    markReady();
  };
  // theme-correct fallback when the poster image is missing or still loading; keeps
  // setup deferred so a heavy panel never computes on scroll, just shows a hint.
  const drawPlaceholder = () => {
    fit();
    ctx.clearRect(0, 0, api.size.W, api.size.H);
    ctx.fillStyle = colors.muted || '#888';
    ctx.font = '13px ui-monospace, monospace';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(hasLoop ? '▶  press play' : 'click to load', api.size.W / 2, api.size.H / 2);
    markReady();
  };
  const loadPoster = () => {
    posterReady = false; posterFailed = false;
    const img = new Image(); posterImg = img;
    img.onload = () => { if (img !== posterImg) return; posterReady = true; if (!didSetup) paintPoster(); };
    img.onerror = () => { if (img !== posterImg) return; posterFailed = true; if (!didSetup) drawPlaceholder(); }; // stay deferred
    img.src = posterSrc();
  };
  // repaint the current still (image if loaded, else placeholder); does not re-fetch.
  const paintStill = () => { if (posterReady && posterImg) paintPoster(); else drawPlaceholder(); };
  // paint the still on viewport entry, kicking off the (lazy, theme-specific) load once.
  const showStill = () => { if (posterReady) paintPoster(); else if (posterFailed) drawPlaceholder(); else { drawPlaceholder(); loadPoster(); } };
  const showingPoster = () => usePoster && !didSetup;

  // setup() is deferred to first viewport entry (see the IntersectionObserver
  // below), so nothing computes at page load; a page of panels stays responsive
  // and each panel does its work only as it scrolls near. A poster viz defers even
  // this until Play / first interaction.
  const ensureSetup = () => { if (didSetup) return; fit(); doSetup(); didSetup = true; renderNow(); markReady(); };

  // A poster viz wakes on intent: a loop viz on Play (see play()); a pure-interactive
  // one on the first pointer or focus so its controls compute live.
  if (usePoster) {
    const wake = () => { if (!didSetup) ensureSetup(); };
    canvas.addEventListener('pointerdown', wake);
    bar.addEventListener('pointerdown', wake);
    root.addEventListener('focusin', wake);
  }

  // keyboard shortcuts on the focusable figure root: space=play/pause, s=step, r=reset
  root.addEventListener('keydown', (e) => {
    if (e.key === ' ' && hasLoop) { e.preventDefault(); ensureSetup(); playing ? stop() : play(); }
    else if ((e.key === 's' || e.key === 'S') && spec.step) { ensureSetup(); stop(); stepN(1); renderNow(); }
    else if ((e.key === 'r' || e.key === 'R') && spec.step) { ensureSetup(); doReset(); }
  });

  // Observers hold closures over this panel; disconnect them once its canvas
  // leaves the DOM (Astro view transitions swap nodes) so they do not pile up.
  const observers = [];
  const alive = () => { if (canvas.isConnected) return true; stop(); observers.forEach((o) => o.disconnect()); document.removeEventListener('fullscreenchange', onFullscreen); document.removeEventListener('webkitfullscreenchange', onFullscreen); return false; };

  const themeObs = new MutationObserver(() => { if (!alive()) return; colors = readColors(); if (showingPoster()) loadPoster(); else render(); });
  themeObs.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  const resizeObs = new ResizeObserver(() => { if (!alive()) return; if (showingPoster()) paintStill(); else { fit(); render(); } });
  resizeObs.observe(canvas);
  let kicked = false;
  const onFullscreen = () => { colors = readColors(); if (showingPoster()) paintStill(); else { fit(); renderNow(); } };
  document.addEventListener('fullscreenchange', onFullscreen);
  document.addEventListener('webkitfullscreenchange', onFullscreen);
  // Preload margin: the still preview is ready by the time the panel scrolls in,
  // without doing the work for every panel up front. Nothing auto-plays: a loop viz
  // paints its first frame (or poster) and waits for Play, unless it explicitly opts
  // in with autoplay:true.
  const io = new IntersectionObserver((e) => {
    if (!alive()) return;
    if (e[0].isIntersecting) {
      if (showingPoster()) showStill();
      else if (!didSetup) ensureSetup(); else { fit(); renderNow(); }
      if (spec.step && spec.autoplay === true && !reduceMotion && !kicked) { kicked = true; play(); }
    } else stop();
  }, { rootMargin: '400px 0px' });
  io.observe(canvas);
  observers.push(themeObs, resizeObs, io);
}

export function defineViz(className, spec) {
  const boot = () => document.querySelectorAll('.' + className).forEach((r) => mountOne(r, spec));
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
  document.addEventListener('astro:page-load', boot);
  document.addEventListener('astro:after-swap', boot);
}
