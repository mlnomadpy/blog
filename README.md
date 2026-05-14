# Records of the !mmortal Data Scientist

Astro + MDX blog. Math via KaTeX. Comments via giscus. Hosted on GitHub Pages at
`https://tahabouhsine.com/blog`.

## Local development

```bash
npm install
npm run dev      # http://localhost:4321/blog/
npm run build    # outputs to ./dist
npm run preview  # serve ./dist locally
```

## Writing a post

Drop an `.mdx` (or `.md`) file in `src/content/blog/`. The filename becomes the URL slug.

```mdx
---
title: My new post
description: One-line summary used for SEO and the index page.
pubDate: 2026-05-14
tags: [math, ai]
draft: false
---

Inline math: $e^{i\pi} + 1 = 0$.

$$
\int_0^\infty e^{-x^2}\,dx = \frac{\sqrt{\pi}}{2}
$$
```

### Theorems, lemmas, proofs

```mdx
import Theorem from '../../components/Theorem.astro';

<Theorem kind="theorem" number="2.1" title="Spectral theorem">
Every Hermitian operator has a real spectrum.
</Theorem>

<Theorem kind="proof">
Standard; see Reed & Simon.
</Theorem>
```

`kind` accepts: `theorem`, `lemma`, `definition`, `proposition`, `corollary`,
`remark`, `example`, `proof`.

### Custom JS / visualizations

Any `<script is:inline>` block in an MDX file is shipped verbatim. Drop in
D3, Three.js, Observable Plot, or hand-rolled canvas code. See
`src/content/blog/hello-math.mdx` for a working example.

For heavier visualizations, install a UI framework integration (`npx astro add react`)
and import `.tsx` components directly into MDX.

## What you need to do

These steps must happen outside the repo — I can't do them for you:

### 1. Point the custom domain at GitHub Pages

In your DNS for `tahabouhsine.com`, add **A records** at the apex pointing to:

```
185.199.108.153
185.199.109.153
185.199.110.153
185.199.111.153
```

(Or a single **ALIAS/ANAME** record to `mlnomadpy.github.io.` if your DNS host supports it.)

The repo already contains `public/CNAME` with `tahabouhsine.com`, so GitHub Pages
will serve the site at `https://tahabouhsine.com/`. Because `astro.config.mjs`
sets `base: '/blog'`, every page lives under `/blog/...` — i.e. the home page
is `https://tahabouhsine.com/blog/`. A request to `https://tahabouhsine.com/`
will 404 unless something else (e.g. a separate root site repo) serves it.

> **If you want a different URL** (e.g. `blog.tahabouhsine.com` on a subdomain,
> or the blog at the root `tahabouhsine.com/`), change `base` in `astro.config.mjs`
> and the DNS records accordingly. Subdomain is simpler — just a `CNAME` record
> pointing `blog` → `mlnomadpy.github.io.` and set `base: '/'`.

### 2. Enable GitHub Pages

On `github.com/mlnomadpy/blog` → **Settings → Pages**:

- **Source**: GitHub Actions
- Save.

The `.github/workflows/deploy.yml` workflow will run on push to `master` and
deploy `dist/` automatically.

When the domain field shows your CNAME, tick **Enforce HTTPS**.

### 3. Enable comments (giscus)

1. On `github.com/mlnomadpy/blog`, go to **Settings → General → Features** and
   enable **Discussions**.
2. Install the [giscus GitHub App](https://github.com/apps/giscus) on the repo.
3. Visit <https://giscus.app>, plug in `mlnomadpy/blog`, choose a Discussion
   category (e.g. create one called `Comments`), and copy the
   `data-repo-id` and `data-category-id` values it generates.
4. Paste them into `src/consts.ts`:

   ```ts
   export const GISCUS = {
     repo: 'mlnomadpy/blog',
     repoId: 'R_kgD...',           // ← from giscus.app
     category: 'Comments',
     categoryId: 'DIC_kwDO...',    // ← from giscus.app
     ...
   };
   ```

   Until both IDs are filled in, posts render a placeholder note instead of the
   comments widget.

### 4. (Optional) Branch name

The workflow triggers on `master` or `main`. The current branch is
`mlnomadpy/hartford`; merge to `master` before expecting deploys.

## Project layout

```
.
├── astro.config.mjs          # site URL, base path, MDX, KaTeX
├── src/
│   ├── consts.ts             # site title, giscus IDs
│   ├── content.config.ts     # blog collection schema
│   ├── content/blog/         # posts (.mdx)
│   ├── components/
│   │   ├── BaseHead.astro
│   │   ├── Header.astro
│   │   ├── Footer.astro
│   │   ├── Theorem.astro     # theorem/lemma/proof boxes
│   │   ├── Callout.astro
│   │   └── Giscus.astro      # comments
│   ├── layouts/
│   │   ├── BaseLayout.astro
│   │   └── PostLayout.astro
│   ├── pages/
│   │   ├── index.astro       # post list
│   │   ├── about.astro
│   │   ├── rss.xml.ts        # RSS feed
│   │   └── [...slug].astro   # dynamic post route
│   └── styles/global.css
├── public/
│   ├── CNAME                 # tahabouhsine.com
│   ├── .nojekyll
│   └── favicon.svg
└── .github/workflows/deploy.yml
```
