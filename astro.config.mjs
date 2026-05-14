import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeSlug from 'rehype-slug';
import rehypeAutolinkHeadings from 'rehype-autolink-headings';

export default defineConfig({
  site: 'https://tahabouhsine.com',
  base: '/blog',
  trailingSlash: 'ignore',
  integrations: [mdx(), sitemap()],
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
