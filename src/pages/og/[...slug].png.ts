import type { APIRoute, GetStaticPaths } from 'astro';
import { getCollection } from 'astro:content';
import satori from 'satori';
import { Resvg } from '@resvg/resvg-js';
import { html } from 'satori-html';
import fs from 'node:fs/promises';
import path from 'node:path';
import { SITE_TITLE, SITE_AUTHOR } from '../../consts';
import { postFilter } from '../../utils/posts';

// ─── font loading ───────────────────────────────────────────────────
// Satori supports WOFF via opentype.js, so we load Lora directly from
// fontsource. These reads happen once per build, not per request.
const fontsDir = path.resolve('./node_modules/@fontsource/lora/files');
const [loraRegular, loraBold] = await Promise.all([
  fs.readFile(path.join(fontsDir, 'lora-latin-400-normal.woff')),
  fs.readFile(path.join(fontsDir, 'lora-latin-700-normal.woff')),
]);

// ─── helpers ────────────────────────────────────────────────────────
const truncate = (s: string, n: number) => (s.length > n ? s.slice(0, n - 1).trimEnd() + '…' : s);
const escapeHtml = (s: string) =>
  s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

// Lora's latin subset doesn't include arrows or the mathematical-minus
// codepoint, so substitute the common ones with ASCII fallbacks. Anything
// missing here will render as a .notdef box.
const sanitizeForOg = (s: string): string =>
  s
    // Map characters Lora's latin subset doesn't cover to readable ASCII.
    // Avoid '<'/'>' in the substitutions because they survive into the
    // HTML-escape pass and satori-html does not decode entities.
    .replace(/[→⟶➝➞]/g, ' to ')
    .replace(/[←⟵]/g, ' from ')
    .replace(/[↔⇔]/g, ' ↔ ')
    .replace(/[⇒⟹]/g, ' implies ')
    .replace(/[⇐⟸]/g, ' if ')
    .replace(/[−]/g, '-') // U+2212 minus sign → ASCII hyphen
    .replace(/[∞]/g, 'inf')
    .replace(/[⊥]/g, ' perp ')
    .replace(/\s{2,}/g, ' '); // collapse double spaces introduced by replacements

// ─── path generation ────────────────────────────────────────────────
export const getStaticPaths: GetStaticPaths = async () => {
  const posts = await getCollection('blog', postFilter);
  return posts.map((post) => ({
    params: { slug: post.id.replace(/\.(md|mdx)$/, '') },
    props: { post },
  }));
};

// ─── render ─────────────────────────────────────────────────────────
export const GET: APIRoute = async ({ props }) => {
  const post = (props as any).post;
  const title = sanitizeForOg(post.data.title as string);
  const description = sanitizeForOg((post.data.description as string | undefined) ?? '');
  const tags = ((post.data.tags as string[] | undefined) ?? []).slice(0, 4);
  const dateStr = new Date(post.data.pubDate).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });

  // Title size shrinks for long headlines so they always fit two lines.
  const titleSize = title.length > 60 ? 56 : title.length > 40 ? 64 : 76;

  // Build the OG card as HTML; satori-html parses it into satori's AST.
  // Flex layout is required at every level for satori to compute geometry.
  const markup = html(`
    <div style="
      height: 100%;
      width: 100%;
      display: flex;
      flex-direction: column;
      background: #fbf8f1;
      padding: 64px 80px;
      position: relative;
      font-family: 'Lora';
      color: #1a1a1a;
    ">
      <div style="
        position: absolute;
        left: 0;
        top: 0;
        width: 14px;
        height: 100%;
        background: #b3661b;
        display: flex;
      "></div>

      <div style="
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 22px;
        color: #b3661b;
      ">
        <div style="display: flex; width: 10px; height: 10px; background: #b3661b; border-radius: 999px;"></div>
        <div style="display: flex;">${escapeHtml(SITE_TITLE)}</div>
      </div>

      <div style="
        display: flex;
        flex-direction: column;
        flex: 1;
        justify-content: center;
        gap: 28px;
      ">
        <div style="
          display: flex;
          font-size: ${titleSize}px;
          font-weight: 700;
          line-height: 1.12;
          color: #1a1a1a;
        ">${escapeHtml(title)}</div>

        ${
          description
            ? `<div style="
                display: flex;
                font-size: 26px;
                line-height: 1.4;
                color: #5a5f66;
              ">${escapeHtml(truncate(description, 180))}</div>`
            : ''
        }
      </div>

      <div style="
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 22px;
        color: #5a5f66;
      ">
        <div style="display: flex; gap: 10px; align-items: center;">
          ${tags
            .map(
              (t) => `<div style="
                display: flex;
                padding: 6px 14px;
                background: #f3eee0;
                border-radius: 999px;
                color: #b3661b;
                font-size: 18px;
              ">#${escapeHtml(t)}</div>`,
            )
            .join('')}
        </div>
        <div style="display: flex; gap: 16px; align-items: center;">
          <div style="display: flex;">${escapeHtml(SITE_AUTHOR)}</div>
          <div style="display: flex; width: 4px; height: 4px; background: #8d8d8d; border-radius: 999px;"></div>
          <div style="display: flex;">${escapeHtml(dateStr)}</div>
        </div>
      </div>
    </div>
  `);

  const svg = await satori(markup as any, {
    width: 1200,
    height: 630,
    fonts: [
      { name: 'Lora', data: loraRegular, weight: 400, style: 'normal' },
      { name: 'Lora', data: loraBold, weight: 700, style: 'normal' },
    ],
  });

  const png = new Resvg(svg, {
    fitTo: { mode: 'width', value: 1200 },
  })
    .render()
    .asPng();

  return new Response(png, {
    headers: {
      'Content-Type': 'image/png',
      'Cache-Control': 'public, max-age=31536000, immutable',
    },
  });
};
