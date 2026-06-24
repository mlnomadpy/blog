// Health-check every visualization on a page: navigate, scroll through to trip
// the lazy IntersectionObservers, then for each canvas sample its pixels and flag
// any that rendered blank/near-uniform (a broken or non-rendering viz). Also
// catches console errors. Node 22 CDP client, headless Chrome, no puppeteer.
// Usage: node scripts/vizcheck.mjs <url>
import { spawn } from 'node:child_process';

const url = process.argv[2];
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const PORT = +(process.env.CDP_PORT || 9344);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const chrome = spawn(CHROME, [
  '--headless=new', `--remote-debugging-port=${PORT}`, '--disable-gpu',
  '--hide-scrollbars', '--force-color-profile=srgb', '--window-size=1500,2200',
  '--no-first-run', '--no-default-browser-check', 'about:blank',
], { stdio: 'ignore' });
const cleanup = () => { try { chrome.kill('SIGKILL'); } catch {} };
process.on('exit', cleanup);

async function main() {
  for (let i = 0; i < 50; i++) { try { await (await fetch(`http://localhost:${PORT}/json/version`)).json(); break; } catch { await sleep(150); } }
  const tab = await (await fetch(`http://localhost:${PORT}/json/new?${encodeURIComponent(url)}`, { method: 'PUT' })).json();
  const ws = new WebSocket(tab.webSocketDebuggerUrl);
  await new Promise((res) => { ws.onopen = res; });
  let id = 0; const pending = new Map(); const errors = [];
  ws.onmessage = (m) => {
    const msg = JSON.parse(m.data);
    if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); }
    if (msg.method === 'Runtime.consoleAPICalled' && msg.params.type === 'error') errors.push(msg.params.args.map((a) => a.value || a.description || '').join(' '));
    if (msg.method === 'Runtime.exceptionThrown') errors.push('EXCEPTION: ' + (msg.params.exceptionDetails?.exception?.description || msg.params.exceptionDetails?.text || ''));
  };
  const send = (method, params = {}) => new Promise((res) => { const i = ++id; pending.set(i, res); ws.send(JSON.stringify({ id: i, method, params })); });

  await send('Page.enable'); await send('Runtime.enable');
  await sleep(1200);
  await send('Runtime.evaluate', { expression: `(async()=>{const h=document.body.scrollHeight;for(let y=0;y<h;y+=500){scrollTo(0,y);await new Promise(r=>setTimeout(r,90));}scrollTo(0,0);})()`, awaitPromise: true });
  await sleep(2500);

  const sel = 'canvas';
  const count = (await send('Runtime.evaluate', { expression: `document.querySelectorAll('${sel}').length`, returnByValue: true })).result.result.value;
  const out = [];
  for (let i = 0; i < count; i++) {
    // scroll the nth canvas into view so a lazy/animating viz actually paints,
    // settle, measure its rect, capture, and use PNG byte-density as a blank flag
    await send('Runtime.evaluate', { expression: `document.querySelectorAll('${sel}')[${i}]?.scrollIntoView({block:'center'})` });
    await sleep(900);
    const meta = JSON.parse((await send('Runtime.evaluate', {
      expression: `(()=>{const el=document.querySelectorAll('${sel}')[${i}]; if(!el)return 'null'; const r=el.getBoundingClientRect(); const fig=el.closest('figure'); const cls=fig?[...fig.classList].filter(x=>x!=='viz').join('.'):''; const title=(fig?.querySelector('h4')?.textContent||'').slice(0,46); return JSON.stringify({x:r.x+scrollX,y:r.y+scrollY,w:r.width,h:r.height,cls,title});})()`,
      returnByValue: true,
    })).result.result.value);
    if (!meta || meta.w < 4 || meta.h < 4) { out.push({ i, blank: true, info: 'no-size' }); continue; }
    const shot = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true, clip: { x: meta.x, y: meta.y, width: meta.w, height: meta.h, scale: 1 } });
    const bytes = Buffer.from(shot.result.data, 'base64').length;
    const density = bytes / (meta.w * meta.h);   // blank/solid PNGs compress to near-nothing
    out.push({ i, cls: meta.cls, title: meta.title, w: Math.round(meta.w), h: Math.round(meta.h), kb: Math.round(bytes / 102.4) / 10, density: Math.round(density * 1000) / 1000, blank: density < 0.03 });
  }
  ws.close();
  const blanks = out.filter((o) => o.blank);
  console.log(JSON.stringify({ url: url.split('/blog/')[1], n: out.length, blanks: blanks.length, errors: [...new Set(errors)].slice(0, 10), canvases: out }, null, 1));
}
main().then(() => { cleanup(); process.exit(0); }).catch((e) => { console.error(e); cleanup(); process.exit(1); });
