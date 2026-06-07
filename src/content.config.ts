import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const blog = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/blog' }),
  schema: z.object({
    title: z.string(),
    description: z.string().optional(),
    // Short (<=160 char) meta description for SERP / OG / Twitter / JSON-LD. The
    // long `description` stays the on-page excerpt (cards, RSS, OG image); this is
    // the search-engine-facing version. Falls back to `description` when omitted.
    seoDescription: z.string().optional(),
    pubDate: z.coerce.date(),
    updatedDate: z.coerce.date().optional(),
    tags: z.array(z.string()).default([]),
    keywords: z.array(z.string()).optional(),
    categories: z.array(z.string()).default([]),
    draft: z.boolean().default(false),
    heroImage: z.string().optional(),
    // Declared on an explainer post; value is the slug (id) of its runnable JAX
    // companion. Drives the paired callout on each post and the nested link on
    // the home page. Only set this on the main/explainer side — the reverse link
    // is derived.
    companion: z.string().optional(),
    // Bibliography rendered at the end of the post. Each entry is one cited work;
    // `key` (optional) lets you anchor inline citations to it (#ref-<key>).
    references: z
      .array(
        z.object({
          key: z.string().optional(),
          authors: z.string(),
          year: z.union([z.number(), z.string()]),
          title: z.string(),
          venue: z.string().optional(),
          url: z.string().optional(),
          arxiv: z.string().optional(),
          doi: z.string().optional(),
        }),
      )
      .optional(),
  }),
});

export const collections = { blog };
