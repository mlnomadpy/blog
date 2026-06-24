// Headless-Chrome screenshot tool for verifying visualizations, over the DevTools
// Protocol (Node 22 has global WebSocket + fetch, so no puppeteer needed).
// Usage: node scripts/shot.mjs <url> [selector=canvas] [outPrefix=/tmp/shot] [maxN=12]
// Scrolls each matching element into view (to trip the IntersectionObserver lazy
// load), waits for compute/raf, and writes one cropped PNG per element.
import { writeFileSync } from 'node:fs';
import { spawn } from 'node:child_process';

const [url, selector = 'canvas', outPrefix = '/tmp/shot', maxN = '12'] = process.argv.slice(2);
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const PORT = 9333;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const chrome = spawn(CHROME, [
  '--headless=new', `--remote-debugging-port=${PORT}`, '--disable-gpu',
  '--hide-scrollbars', '--force-color-profile=srgb', '--window-size=1500,2200',
  '--no-first-run', '--no-default-browser-check', 'about:blank',
], { stdio: 'ignore' });

const cleanup = () => { try { chrome.kill('SIGKILL'); } catch {} };
process.on('exit', cleanup);

async function cdp() {
  // wait for the debugger endpoint
  let ver;
  for (let i = 0; i < 50; i++) {
    try { ver = await (await fetch(`http://localhost:${PORT}/json/version`)).json(); break; } catch { await sleep(150); }
  }
  // open a fresh page target
  const tab = await (await fetch(`http://localhost:${PORT}/json/new?${encodeURIComponent(url)}`, { method: 'PUT' })).json();
  const ws = new WebSocket(tab.webSocketDebuggerUrl);
  await new Promise((res) => { ws.onopen = res; });

  let id = 0; const pending = new Map();
  ws.onmessage = (m) => { const msg = JSON.parse(m.data); if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); } };
  const send = (method, params = {}) => new Promise((res) => { const i = ++id; pending.set(i, res); ws.send(JSON.stringify({ id: i, method, params })); });

  await send('Page.enable');
  await send('Runtime.enable');
  await sleep(1500);
  // scroll the whole page once to trip every lazy IntersectionObserver, then top
  await send('Runtime.evaluate', { expression: `(async()=>{const h=document.body.scrollHeight;for(let y=0;y<h;y+=600){window.scrollTo(0,y);await new Promise(r=>setTimeout(r,40));}window.scrollTo(0,0);})()`, awaitPromise: true });
  await sleep(2500);

  const sel = JSON.stringify(selector);
  const count = (await send('Runtime.evaluate', { expression: `document.querySelectorAll(${sel}).length`, returnByValue: true })).result.result.value;
  const n = Math.min(count, +maxN);
  console.log(`found ${count} elements for selector "${selector}", capturing ${n}`);

  for (let i = 0; i < n; i++) {
    // bring the nth element into view so a lazy component renders + computes, let
    // it settle, THEN measure its page rect (post-layout, not stale) and capture
    await send('Runtime.evaluate', { expression: `document.querySelectorAll(${sel})[${i}]?.scrollIntoView({block:'center'})` });
    await sleep(1200);
    const r = JSON.parse((await send('Runtime.evaluate', {
      expression: `(()=>{const el=document.querySelectorAll(${sel})[${i}]; if(!el) return 'null'; const r=el.getBoundingClientRect(); return JSON.stringify({x:r.x+scrollX,y:r.y+scrollY,w:r.width,h:r.height});})()`,
      returnByValue: true,
    })).result.result.value);
    if (!r || r.w < 4 || r.h < 4) { console.log(`skip ${i} (no size)`); continue; }
    const shot = await send('Page.captureScreenshot', {
      format: 'png', captureBeyondViewport: true,
      clip: { x: r.x, y: r.y, width: r.w, height: r.h, scale: 1 },
    });
    const path = `${outPrefix}-${i}.png`;
    writeFileSync(path, Buffer.from(shot.result.data, 'base64'));
    console.log(`wrote ${path}  (${Math.round(r.w)}x${Math.round(r.h)})`);
  }
  ws.close();
}

cdp().then(() => { cleanup(); process.exit(0); }).catch((e) => { console.error(e); cleanup(); process.exit(1); });
