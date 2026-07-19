// Poster generator for the heaviest viz. A viz opts in with `data-poster="<name>"`
// on its figure (see vizkit.js); this captures a still PNG per theme into
// public/viz-posters/, which the harness then shows (with ZERO compute) until the
// reader presses Play. Reuses shot.mjs's headless-Chrome-over-CDP plumbing.
//
//   npm run posters                # builds (with drafts) + generates every poster
//   node scripts/gen-posters.mjs   # against an already-running `astro preview`
//
// It loads each poster page with ?vizposter=1 (which forces the LIVE render path so
// we screenshot the real frame, not the poster), sets the theme, scrolls to trip the
// lazy setup, presses Play on loop viz for an evolved still, then crops each canvas.
import { writeFileSync, mkdirSync, readdirSync, readFileSync, existsSync, statSync } from 'node:fs';
import { spawn } from 'node:child_process';
import { join, relative } from 'node:path';

const ROOT = new URL('..', import.meta.url).pathname;
const DIST = join(ROOT, 'dist');
const OUT = join(ROOT, 'public', 'viz-posters');
const BASE = '/blog';
const PREVIEW = 'http://localhost:4321';
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const CDP_PORT = 9334;
const THEMES = ['light', 'dark'];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ── find every built page that contains a data-poster figure, map to its route ──
function findPosterPages() {
  const pages = [];
  const walk = (dir) => {
    for (const ent of readdirSync(dir, { withFileTypes: true })) {
      const p = join(dir, ent.name);
      if (ent.isDirectory()) walk(p);
      else if (ent.name === 'index.html') {
        const html = readFileSync(p, 'utf8');
        if (html.includes('data-poster="')) {
          const route = '/' + relative(DIST, dir).split('\\').join('/');
          pages.push(route === '/' ? '' : route);
        }
      }
    }
  };
  walk(DIST);
  return pages;
}

// ── minimal CDP client over a single tab ──
async function connectTab(url) {
  const tab = await (await fetch(`http://localhost:${CDP_PORT}/json/new?${encodeURIComponent(url)}`, { method: 'PUT' })).json();
  const ws = new WebSocket(tab.webSocketDebuggerUrl);
  await new Promise((res) => { ws.onopen = res; });
  let id = 0; const pending = new Map();
  ws.onmessage = (m) => { const msg = JSON.parse(m.data); if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); } };
  const send = (method, params = {}) => new Promise((res) => { const i = ++id; pending.set(i, res); ws.send(JSON.stringify({ id: i, method, params })); });
  const evalJs = async (expr, awaitPromise = false) => (await send('Runtime.evaluate', { expression: expr, awaitPromise, returnByValue: true })).result?.result?.value;
  const closeTab = () => { ws.close(); return fetch(`http://localhost:${CDP_PORT}/json/close/${tab.id}`).catch(() => {}); };
  return { send, evalJs, closeTab };
}

async function capturePage(route, theme) {
  const url = `${PREVIEW}${BASE}${route}?vizposter=1`;
  const t = await connectTab(url);
  const shots = [];
  try {
    await t.send('Page.enable');
    await t.send('Runtime.enable');
    await sleep(900);
    // pin the theme before the viz set up, so their first frame is drawn in it
    await t.evalJs(`(()=>{document.documentElement.setAttribute('data-theme','${theme}');try{localStorage.setItem('theme','${theme}');}catch(e){}})()`);
    // scroll the whole page to trip every lazy IntersectionObserver, then back
    await t.evalJs(`(async()=>{const h=document.body.scrollHeight;for(let y=0;y<h;y+=560){window.scrollTo(0,y);await new Promise(r=>setTimeout(r,45));}window.scrollTo(0,0);})()`, true);
    await sleep(1200);
    const names = await t.evalJs(`JSON.stringify([...document.querySelectorAll('.viz[data-poster]')].map(f=>f.dataset.poster))`);
    for (const name of JSON.parse(names || '[]')) {
      const sel = `document.querySelector('.viz[data-poster="${name}"]')`;
      await t.evalJs(`${sel}?.querySelector('canvas')?.scrollIntoView({block:'center'})`);
      await sleep(500);
      // loop viz: press Play and let it evolve into an inviting still; static viz settle as-is
      const hasPlay = await t.evalJs(`!!${sel}?.querySelector('.vk-play')`);
      if (hasPlay) { await t.evalJs(`${sel}.querySelector('.vk-play').click()`); await sleep(2600); await t.evalJs(`${sel}.querySelector('.vk-play').click()`); }
      else { await sleep(900); }
      const rect = JSON.parse(await t.evalJs(`(()=>{const c=${sel}?.querySelector('canvas');if(!c)return'null';const r=c.getBoundingClientRect();return JSON.stringify({x:r.x+scrollX,y:r.y+scrollY,w:r.width,h:r.height});})()`) || 'null');
      if (!rect || rect.w < 4 || rect.h < 4) { console.log(`  skip ${name} (no size)`); continue; }
      const shot = await t.send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true, clip: { x: rect.x, y: rect.y, width: rect.w, height: rect.h, scale: 1 } });
      const file = join(OUT, `${name}${theme === 'dark' ? '-dark' : ''}.png`);
      writeFileSync(file, Buffer.from(shot.result.data, 'base64'));
      shots.push(name);
      console.log(`  wrote ${name}${theme === 'dark' ? '-dark' : ''}.png  (${Math.round(rect.w)}x${Math.round(rect.h)})`);
    }
  } finally { await t.closeTab(); }
  return shots;
}

async function main() {
  if (!existsSync(DIST)) { console.error('No dist/. Run `SHOW_DRAFTS=true npm run build` first (or `npm run posters`).'); process.exit(1); }
  mkdirSync(OUT, { recursive: true });
  const pages = findPosterPages();
  console.log(`Found ${pages.length} page(s) with data-poster viz.`);

  // reuse a running preview if present, else spawn `astro preview`
  let preview = null;
  const up = async () => { try { await fetch(`${PREVIEW}${BASE}/`); return true; } catch { return false; } };
  if (!(await up())) {
    console.log('Starting astro preview…');
    preview = spawn('npm', ['run', 'preview'], { cwd: ROOT, stdio: 'ignore' });
    for (let i = 0; i < 60 && !(await up()); i++) await sleep(500);
    if (!(await up())) { console.error('preview did not come up'); preview?.kill(); process.exit(1); }
  }

  const chrome = spawn(CHROME, ['--headless=new', `--remote-debugging-port=${CDP_PORT}`, '--disable-gpu', '--hide-scrollbars', '--force-color-profile=srgb', '--window-size=1500,2200', '--no-first-run', '--no-default-browser-check', 'about:blank'], { stdio: 'ignore' });
  const cleanup = () => { try { chrome.kill('SIGKILL'); } catch {} try { preview?.kill(); } catch {} };
  process.on('exit', cleanup);
  for (let i = 0; i < 50; i++) { try { await fetch(`http://localhost:${CDP_PORT}/json/version`); break; } catch { await sleep(150); } }

  const done = new Set();
  for (const theme of THEMES) {
    console.log(`\n== theme: ${theme} ==`);
    for (const route of pages) { console.log(`page ${route || '/'}`); for (const n of await capturePage(route, theme)) done.add(n); }
  }
  console.log(`\nDone. ${done.size} unique poster(s) × ${THEMES.length} theme(s) → public/viz-posters/`);
  cleanup();
  process.exit(0);
}

main().catch((e) => { console.error(e); process.exit(1); });
