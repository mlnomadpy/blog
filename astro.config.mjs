import { defineConfig } from 'astro/config';
import fs from 'node:fs';
import path from 'node:path';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeSlug from 'rehype-slug';
import rehypeAutolinkHeadings from 'rehype-autolink-headings';

// Tag pages with fewer than TAG_MIN published posts are thin content: they
// rarely rank and dilute crawl/quality signals. The route noindexes them
// (see src/pages/tag/[tag].astro) and we drop them from the sitemap here so
// Search Console never sees a "submitted URL marked noindex" conflict. Kept in
// sync with the `< 3` threshold in the tag route.
const TAG_MIN = 3;
function thinTags() {
  const dir = './src/content/blog';
  const count = {};
  for (const f of fs.readdirSync(dir)) {
    if (!/\.mdx?$/.test(f)) continue;
    const fm = fs.readFileSync(path.join(dir, f), 'utf8').split(/^---$/m)[1] || '';
    if (/^draft:\s*true/m.test(fm)) continue;
    const m = fm.match(/^tags:\s*\[([^\]]*)\]/m);
    if (!m) continue;
    for (const t of m[1].split(',').map((x) => x.trim().replace(/^["']|["']$/g, '')).filter(Boolean))
      count[t] = (count[t] || 0) + 1;
  }
  return new Set(Object.entries(count).filter(([, n]) => n < TAG_MIN).map(([t]) => t));
}
const THIN_TAGS = thinTags();

export default defineConfig({
  site: 'https://tahabouhsine.com',
  base: '/blog',
  trailingSlash: 'ignore',
  // Posts moved from the blog root into the writeups section keep their old
  // URLs alive via redirects, so inbound links and search rankings survive.
  // Destinations need the /blog base spelled out: Astro applies `base` to the
  // redirect source path but not the destination string.
  redirects: {
    '/ai-illiteracy-pt1': '/blog/writeups/ai-illiteracy-pt1',
    '/poem-0-1': '/blog/writeups/poem-0-1',
  },
  // Web Workers (the off-thread Yat-kernel compute in the viz) import jax-js,
  // which code-splits; ES module workers are required for that.
  vite: { worker: { format: 'es' } },
  integrations: [
    mdx(),
    sitemap({
      // Drop thin (noindex'd) tag pages from the sitemap.
      filter: (page) => {
        const m = page.match(/\/tag\/([^/]+)\/?$/);
        return m ? !THIN_TAGS.has(decodeURIComponent(m[1])) : true;
      },
    }),
  ],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [
      [rehypeKatex, { strict: false }],
      rehypeSlug,
      [
        rehypeAutolinkHeadings,
        {
          behavior: 'append',
          content: {
            type: 'element',
            tagName: 'span',
            properties: { className: ['heading-anchor'], 'aria-hidden': 'true' },
            children: [{ type: 'text', value: '#' }],
          },
          properties: { className: ['heading-anchor-link'], ariaLabel: 'Link to heading' },
        },
      ],
    ],
    shikiConfig: {
      themes: { light: 'github-light', dark: 'github-dark-dimmed' },
      wrap: true,
    },
  },
});
