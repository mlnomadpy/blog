// llms.txt — a curated, machine-readable index of the blog for LLMs and
// AI search engines (the convention from llmstxt.org). It gives ChatGPT,
// Perplexity, Claude, Gemini, and AI Overviews a clean map of every post with a
// one-line summary, so the content is easy to discover, extract, and cite.
// Served at /blog/llms.txt (referenced from robots.txt).
import { getCollection } from 'astro:content';
import {
  SITE_TITLE,
  SITE_DESCRIPTION,
  SITE_AUTHOR,
  SITE_AUTHOR_URL,
  SITE_AUTHOR_JOB_TITLE,
  SITE_SAMEAS,
} from '../consts';
import { postFilter } from '../utils/posts';
import type { APIContext } from 'astro';

const fmtDate = (d: Date) => d.toISOString().slice(0, 10);
const oneLine = (s: string | undefined) => (s ?? '').replace(/\s+/g, ' ').trim();

export async function GET(context: APIContext) {
  const site = context.site!;                       // https://tahabouhsine.com
  const abs = (p: string) => new URL(p, site).toString();
  const posts = (await getCollection('blog', postFilter)).sort(
    (a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf(),
  );

  const out: string[] = [];
  out.push(`# ${SITE_TITLE}`);
  out.push('');
  out.push(`> ${oneLine(SITE_DESCRIPTION)}`);
  out.push('');
  out.push(
    `Long-form machine-learning research notes by ${SITE_AUTHOR} (${SITE_AUTHOR_JOB_TITLE}). ` +
      `Every post is built around live, in-browser interactive visualizations: the math actually runs in the page, ` +
      `not as pre-rendered figures. Recurring threads: neural-network interpretability, kernel methods and RKHS, ` +
      `contrastive learning, attention mechanisms, the Welch bound and simplex/ETF geometry, and white-box architectures.`,
  );
  out.push('');

  out.push('## Posts');
  out.push('');
  for (const p of posts) {
    const desc = oneLine(p.data.seoDescription ?? p.data.description);
    out.push(
      `- [${p.data.title}](${abs(`/blog/${p.id}/`)}) (${fmtDate(p.data.pubDate)})` +
        (desc ? `: ${desc}` : ''),
    );
  }
  out.push('');

  out.push('## About & feeds');
  out.push(`- [About ${SITE_AUTHOR}](${abs('/blog/about/')}): author background and research interests.`);
  out.push(`- [Homepage](${abs('/blog/')}): all posts.`);
  out.push(`- [RSS feed](${abs('/blog/rss.xml')})`);
  out.push(`- [Sitemap](${abs('/blog/sitemap-index.xml')})`);
  out.push('');

  out.push('## Author profiles');
  out.push(`- ${SITE_AUTHOR_URL}`);
  for (const s of SITE_SAMEAS) out.push(`- ${s}`);
  out.push('');

  return new Response(out.join('\n'), {
    headers: {
      'Content-Type': 'text/plain; charset=utf-8',
      'Cache-Control': 'public, max-age=3600',
    },
  });
}
