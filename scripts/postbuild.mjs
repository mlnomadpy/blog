// After `astro build`:
//   1. Rearrange `dist/` so the site lives under `dist/blog/`
//      (CNAME + .nojekyll stay at the deploy root).
//   2. Run pagefind against dist/blog/ so the search index lives at
//      /blog/pagefind/.
//
// `base: '/blog'` in astro.config.mjs makes Astro emit /blog/... URLs but
// the static files are written to dist/ root, so we shift them under blog/.

import { promises as fs } from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';

const root = path.resolve('./dist');
const blogDir = path.join(root, 'blog');

// robots.txt must be reachable at the apex (https://tahabouhsine.com/robots.txt)
// for crawlers to discover it, so we keep it at the deploy root too.
const KEEP_AT_ROOT = new Set(['CNAME', '.nojekyll', 'blog', 'robots.txt']);

await fs.mkdir(blogDir, { recursive: true });

for (const entry of await fs.readdir(root)) {
  if (KEEP_AT_ROOT.has(entry)) continue;
  await fs.rename(path.join(root, entry), path.join(blogDir, entry));
}

console.log('[postbuild] moved site into dist/blog/, kept CNAME + .nojekyll at dist/');

await new Promise((resolve, reject) => {
  const proc = spawn(
    'npx',
    ['--no-install', 'pagefind', '--site', blogDir, '--output-subdir', 'pagefind'],
    { stdio: 'inherit' },
  );
  proc.on('exit', (code) =>
    code === 0 ? resolve(undefined) : reject(new Error(`pagefind exited with code ${code}`)),
  );
});

console.log('[postbuild] pagefind index written to dist/blog/pagefind/');
