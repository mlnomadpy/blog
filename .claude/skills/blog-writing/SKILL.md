---
name: blog-writing
description: >-
  How to write the theory-ML blog posts in this repo (src/content/blog/*.mdx) and
  their interactive visualizations (src/components/viz/). Use this skill WHENEVER
  writing, rewriting, editing, reviewing, or planning a blog post, building or
  fixing a viz component, writing a JAX companion, or working on the Yat-kernel
  series, even when the user just says "improve this post", "build the next post",
  "write the companion", "fix this viz", or "make the writing better" without
  naming any rule. It encodes the hard rules that are repeatedly forgotten (no em
  dashes in reader-facing text, the explainer-vs-JAX-companion split, every number
  from a real run, fresh viz per post, never surfacing "jax-js" to readers), the
  writing craft (motivation-first, question-driven, physics-as-bridge), the engine
  viz pitfalls, and the screenshot-verification workflow. Consult it before
  drafting prose or viz, not after.
---

# Writing the blog

These posts are not papers. A paper proves; a blog **seduces a reader into
understanding**. Stating true facts in sequence convinces no one to read on. The
single recurring failure is reverting to results-section voice (claim, evidence,
claim) and to mechanism-before-motivation. Everything below exists to prevent
that and to stop re-litigating rules we already settled.

## Orient first (one minute that saves a rewrite)

Before drafting anything, read the two canonical docs in this repo and the prior
posts in the relevant arc:

- **`docs/writing-style.md`**, the craft guide: the diagnosis, 7 principles, the
  physics-bridge playbook, the paragraph-opening checklist, the per-post audit.
- **`docs/blog-series.md`**, the series bible: house conventions, the catalog,
  the arc beats, the **concept ledger** (what is already explained), open threads.

Check the concept ledger before you explain anything. If a concept is "spent"
(prototype-as-picture, the kernel-as-wells, construct-the-head, OOD abstention,
etc.), do **not** re-derive it. Reference the prior post in one clause and move on.
Re-explaining bloats the post and reads as padding.

## The two formats, never mixed

| | Explainer (the main post) | JAX companion (paired post) |
| --- | --- | --- |
| content | concept + **math** + **interactive engine viz** | **real Python JAX / Flax NNX**, code-first |
| figures | no static images, no GIFs | **GIFs** from a `scripts/render_*.py` |
| viz | built on `src/components/viz/engine/` | none |
| link | `companion:` frontmatter -> the companion | linked back from the explainer |

Putting code or GIFs in an explainer, or engine viz in a companion, has cost
rewrites. They are different artifacts.

**A JAX companion is not done until it has at least one GIF.** A code-only
companion is incomplete, no matter how good the code is. This rule is easy to drop
under scope pressure ("ship the code now, add GIFs later"); that rationalisation
has shipped half-built companions. Render the GIF in the same pass that writes the
code, or the companion is not finished.

## Writing craft (the part that backslides)

The full guide is `docs/writing-style.md`; the load-bearing rules:

- **Motivation before mechanism.** Open every post and most sections on a concrete
  tension the reader feels, not the abstraction or the formula that resolves it.
- **Run on questions.** Plant a real question, let the section answer it, let that
  answer raise the next. Never open a section on its own conclusion.
- **One narrator, one thread.** Each section inherits the last ("we just saw X,
  which is strange, because Y"), never a cold restart from a bare fact.
- **Intuition first, math as confirmation.** The equation arrives after the reader
  already believes the picture, as "here is why that is exactly true."
- **Name the surprise.** Build the expectation, then break it (the random net at
  74%, the hand-built net at 83.3%). Wonder over usefulness.
- **Physics as the bridge, not a footnote.** The reader cannot picture 784
  dimensions; the physical picture (masses, wells, fields, basins, phase changes)
  is how you rescue the intuition. Make the correspondence exact, and let the
  metaphor predict (superposition -> order-independence) before the math confirms.
  Keep one physical world across the series.

**Paragraph-opening smells (rewrite these):** opening with the conclusion ("Because
X, Y"), an instruction ("Start with", "Train a", "Take", "Consider", "Note that"),
a bare subject-noun fact ("The prototypes are"), a stage direction ("The clean way
to see it is", "It is worth", "Look at"), or an announcement ("This is the
thesis"). Open instead on a question, a turn ("But..."), a concrete image, or a
stake. Run the section-opening test: read the first sentence of every section in
isolation; if it could open a paper subsection, rewrite it.

## Hard rules (non-negotiable, they have all cost rewrites)

- **No em dashes anywhere in reader-facing text** (prose, captions, on-canvas
  readouts, titles). Replace a prose em dash with a comma, a colon, or a
  restructure. Em-dash *glyphs* are fine and must stay: legend line-markers
  (`— cos θ` meaning a solid line) and "no value yet" placeholders (`>—<`).
- **Every number is from a real run.** The run lives in `scripts/`, is named in
  the post, and reproduces. Never invent or round-guess a result.
- **Never surface "jax-js"** to readers. Frame compute as the post's real (Python)
  computation run live in the browser.
- **`draft: true` until the user says publish.** Do not commit/publish unprompted.
- `seoTitle` <= 60 chars, `seoDescription` <= 160 chars. Explainer gets a date-only
  `pubDate`; its companion gets the same date with a `Txx:00` time so it sorts just
  after.

## Visualizations (explainer only)

- **Fresh per post.** Never reuse another post's viz component; build new ones. A
  new post earns new visuals, and not redundant with concepts already shown.
- **One concept per viz.** Do not pack teach+forget+vote into one panel. Aim for
  4+ focused panels, each a different visual idiom.
- **Build on the engine.** Use `defineViz` (vizkit.js) and the shared helpers in
  `engine/draw.js` (`gate`, `loadingText`, `scatter`, `heatmapField`, `axes`,
  `legend`, `colorbar`, `headline`, `frame`), `engine/pointer.js` (`drag`,
  `hitTest`), and `engine/theme.js` (`readColors`). Do not hand-roll readColors,
  the load gate, or drag wiring; that duplication is exactly what these exist to
  remove.
- **Compute live on jax-js** via `engine/jax.js`; defer until the panel is scrolled
  to or clicked (no autoplay, no idle render loop). Heavy work goes in a Web Worker
  (see `yatedit.worker.js`).
- A data pipeline in `scripts/*.py` dumps `public/<slug>/` assets (JSON + sprite
  PNGs); a shared `<slug>.js` module memoizes the loads and runs the kernel math.
  Follow the pattern of `trainfeat.js` / `handbuilt.js`.

## Engine viz pitfalls (check every new viz against these)

These three bugs recur and are invisible until you screenshot on a real device:

- **Canvas collapse.** A canvas with `width:100%` and no CSS height has
  `clientHeight` 0, so it renders blank (and can crash with a negative radius). For
  a bespoke component, pin it once: `c.style.height = (+c.getAttribute('height') ||
  H) + 'px'`. `defineViz` components are handled by the engine.
- **Negative `arc` radius.** Any `ctx.arc(x, y, R, ...)` whose `R` is derived from
  `min(w, h)` can go negative when the box is tiny, throwing `IndexSizeError` and
  blanking the panel. Clamp it: `Math.max(0, R)`.
- **`putImageData` ignores the DPR transform.** On a retina display (dpr 2) it
  mis-places and mis-sizes the image; headless Chrome is dpr 1, so it hides the
  bug. Render glyphs/fields with `drawImage` (from an offscreen canvas) or vector
  ops, or use the engine's `heatmapField`.

## Verify before declaring done (you cannot judge a viz from text)

1. **Build green.** `npm run build`, or `SHOW_DRAFTS=true npm run build` for a
   draft post (drafts are excluded from the normal build and from preview).
2. **Health-check.** `node scripts/vizcheck.mjs <url>` flags blank canvases and
   console errors for every viz on a page. Serve first:
   `SHOW_DRAFTS=true npm run build && npm run preview` then hit the localhost URL.
3. **Screenshot and JUDGE.** `node scripts/shot.mjs <url> <selector> <outprefix>`
   writes one cropped PNG per element; read them. "Renders without error" is not
   "good." Ask: does it teach the idea, is it crisp (not an abstract blob), legible,
   not redundant with another panel? If it is weak, rebuild it.
4. **Grep then look** for em dashes in reader-facing strings (`grep $'—'`),
   and confirm "jax-js" is nowhere in captions/readouts.
5. **Companion has its GIF(s).** A JAX companion is not done until at least one
   `<figure class="jax-fig">` points at a real `public/*.gif` rendered by a
   `scripts/render_*_gif.py`. Check it: `grep '\.gif' src/content/blog/<companion>.mdx`
   and confirm the file exists in `public/`. Code-only is incomplete; do not ship it.

Both scripts are Node-22 + headless Chrome over the DevTools Protocol (no
puppeteer). They are the only way to see the actual rendered output here.

## Publishing (only when asked)

Flip `draft: false`, commit the post + its viz components + `public/<slug>/` assets
+ `scripts/` + GIFs, push, open a PR to `master`, merge. Merging `master`
auto-deploys to `gh-pages`. End commit messages with the standard co-author line.
