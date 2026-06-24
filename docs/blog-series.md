# Blog series bible

Working reference for the blog at `src/content/blog/`. Records what exists, the
conventions we write to, the story arc of the active (Yat-kernel) series, a
ledger of which concepts are already "spent" so we stop re-explaining them, and
the open threads for future posts.

_Last updated: 2026-06-22. `D` = draft._

---

## 1. House conventions (read before writing a post)

These are hard rules we have converged on. Breaking them has cost rewrites.

**Two formats, never mixed.**
- **Explainer** (the main post): explains the concept with **math** and **interactive visualizations built on our engine** (`src/components/viz/engine/`, vizkit + jax-js). It is the post a reader lands on. No static figures, no GIFs.
- **JAX companion** (separate paired post): the **real Python JAX / Flax NNX** implementation, **code-first**, with **GIFs** rendered by a Python script. No engine/interactive viz. Linked via the `companion:` frontmatter field on the explainer.

**Visualizations (explainer).**
- Built fresh for each post. **Never reuse** another post's viz component; build new ones (protects the creativity of each post).
- **Not redundant** with concepts already shown in earlier posts. A new post earns new visuals.
- Compute runs live on **jax-js** through `engine/jax.js`; for heavy work use a Web Worker (see `yatedit.worker.js`) so the page never blocks. Defer compute until the panel is scrolled to / clicked (no autoplay, no idle render loop).
- **Never surface "jax-js"** in reader-facing captions/titles/readouts. Frame it as the post's real computation run live in the browser.
- One concept per visualization. Do not pack teach + forget + vote into one panel. A post wants several focused panels (4+ is a good target), each a different visual idiom.
- Physics-inspired framing is encouraged where honest (the Yat denominator is a softened inverse-square law; prototypes are masses; classification is falling into a basin; superposition gives order-independence).

**GIFs (companion).** Rendered by a `scripts/render_*.py`. Pace them slowly (about half the first instinct), give them real UI (legend, labels, titles), keep file size ~<1.6 MB.

**Always true.**
- **Every number is from a real run.** The run lives in `scripts/`, is cited in the post, and reproduces.
- **No em dashes** anywhere in reader-facing text (prose, captions, on-canvas readouts).
- `seoTitle` (optional) <= 60 chars for long titles; `seoDescription` <= 160 chars.
- Posts stay `draft: true` until the user says publish.

**Publishing.** Flip `draft: false`, commit the post + its viz components + `public/` assets + `scripts/` + GIFs, push, open a PR to `master`, merge. Merging `master` auto-deploys to `gh-pages` (`.github/workflows/deploy.yml`). The site is `https://tahabouhsine.com/blog`. Sitemap: `https://tahabouhsine.com/blog/sitemap-index.xml`.

**Pairing/ordering.** Explainer gets a date-only `pubDate` (sorts at 00:00); its companion gets the same date with a `Txx:00` time so it sorts just after.

---

## 2. The catalog (29 posts, three arcs)

### Arc A: representation geometry, contrastive learning, latent space
| status | explainer | companion |
| --- | --- | --- |
| live | activations-are-bad-for-geometry | -- |
| live | opposite-is-not-different | -- |
| live | not-all-infinities-are-equal | -- |
| live | untangling-the-moons | organizing-randomness-jax |
| live | welch-bound-good-latent-space | welch-bound-jax-analysis |
| live | readout-as-convex-combination | convex-readout-jax-flax-nnx |
| live | latent-on-the-spectrum | latent-on-the-spectrum-jax |
| live | three-states-of-information | three-states-of-information-jax |
| **D** | modality-gap-complementary | -- |
| **D** | simo2-geometry-by-construction | -- |

### Arc B: attention as a kernel
| status | explainer | companion |
| --- | --- | --- |
| live | attention-is-a-kernel | attention-is-kernel-jax-flax-nnx |
| live | what-an-mlp-knows | -- |
| live | cheap-attention-is-linear-attention | linear-attention-jax-flax-nnx |
| live | why-attention-needs-qk-projections | qk-projections-jax-flax-nnx |

### Arc C: the Yat kernel / prototype neuron (the active story)
| status | explainer | companion |
| --- | --- | --- |
| live | what-a-finite-kernel-buys-an-mlp | yat-mlp-jax-flax-nnx |
| live | your-neuron-is-a-picture | yat-mlp-fmnist-jax-flax-nnx |
| live | edit-a-network-by-hand | edit-a-network-jax-flax-nnx |
| **D** | train-the-features | (companion not built) |

---

## 3. Arc C story, beat by beat (the part we are actively building)

The through-line: **put the Yat kernel `phi(x) = (w.x + b)^2 / (||x - w||^2 + eps)` where a neuron's activation was, and the neuron stops being a direction and becomes a prototype, a point in input space. That one change makes the network legible, editable, and constructable.** All on Fashion-MNIST.

### C0. what-a-finite-kernel-buys-an-mlp  (+ yat-mlp-jax-flax-nnx)
The entry point. Replace the activation with a finite, explicit, positive-definite kernel (the Yat kernel) and an MLP becomes a kernel machine: locality, attribution, geometry, capacity control, a writable feature map. General argument, 2D toys.

### C1. your-neuron-is-a-picture  (+ yat-mlp-fmnist-jax-flax-nnx)
Title: "Your Neuron Is a Direction. It Should Be a Picture." A direction is not a referent you can point at; a Yat prototype lives in input space and on images is literally a picture. Beats: every neuron is a picture; build it by hand with no training (**79%**, sum-vote; **~79%** nearest-prototype); the network explains itself (convex attribution); its geometry is a map you can walk; reading the mistakes; **where the pictures come from** (legibility is decided at init, by seeding prototypes on data); warm-start helps low-data and imbalance. Companion: build/train YatMLP in Flax NNX, prototypes as images, prediction as a vote, **OOD abstention** ("it knows when it doesn't know", MNIST as OOD), random-init stays noise, UMAP prototype trajectories.

### C2. edit-a-network-by-hand  (+ edit-a-network-jax-flax-nnx)
Title: "Your Network Is a List of Pictures. You Can Edit It." If a neuron is a picture, a network is a list, and a list you edit by hand. Beats: the score algebra `s = A^T phi`, `s_c = sum over class c's prototypes`; **the kernel as a landscape of wells** (the denominator is a softened inverse-square law, `eps` is the N-body softening length; prototypes are masses; classification is falling into a basin; superposition); **teach** a class = append rows, no training (Bag/Boot **95%/94%**, old classes invariant, incremental == from-scratch **79.4% == 79.4%**); **forget** a class = delete rows, exact unlearning (Sandal **64% -> 0%**, other nine **81.1% -> 81.3%**); why it is hard for ordinary nets (entangled `z_c = w_c . h(x)`); the construction-vs-optimization boundary (tease). Math grounded in the per-class **kernel mean embedding** `mu_c = sum Phi(W_u)`. Four fresh jax-js viz: **KernelVote** (the molecule), **AttractorField** (2D physics sandbox: drop particles into basins, delete a well), **TeachByExample** (empty memory bins, test images flip red->green), **ForgetMatch** (each image linked to its nearest prototype; delete a class, only its images re-route). Companion: teach = `jnp.concatenate`, forget = boolean mask, three physics GIFs (settling, basins forming, a basin reclaimed). Scripts: `yat_editable_fmnist.py`, `render_yat_edit_gifs.py`, `render_yat_edit_assets.py`.

### C3. train-the-features  (DRAFT, on hold)
Title: "You Only Have to Train the Features." Construct the Yat head on **learned features** instead of pixels. Real numbers (`construct_vs_optimize.py`): constructed head **83.2%** vs trained head **85.7%** on a converged CNN backbone; **74%** on a random backbone whose trained head is at chance; raw-pixel head 79%; edits survive in feature space (Sandal 93 -> 0, others 82.2 -> 82.5; teach Bag/Boot 95/95, all-10 83.2). Thesis: the representation is the only thing you must train; the classifier and its edits are furniture you place. **Status:** judged thin (mostly re-treads "construct the head, no training" with "on features" as the only new beat). Has one viz (`RepresentationFold`, the representation untangling live). Candidate to fold into a stronger post or drop.

---

## 4. Concept ledger (what is already spent, do not re-explain)

| concept | established in |
| --- | --- |
| direction vs prototype; neuron as a picture | C1 |
| no-training hand-built classifier (place prototypes + one-hot) | C1, C2 |
| Yat kernel is positive-definite / Mercer; RKHS feature map; representer theorem; centers live in input space (so pictures) | C0, C1 |
| self-explaining prediction / convex attribution | C0, C1 |
| **OOD abstention, "knows when it doesn't know"** (bounded kernel) | C1 + companion |
| teach / forget / class-incremental / exact unlearning | C2 |
| attractor field, inverse-square wells, softening length, basins, superposition, order-independence | C2 |
| kernel mean embedding `mu_c`; score algebra | C2 |
| construct the head on learned features; construction-vs-optimization boundary | C3 (draft) |

Reuse these only by reference (link the prior post), never by re-derivation.

---

## 5. Open threads / candidate next posts

Audited against the catalog; these are genuinely uncovered.

1. **Adversarial robustness: "a bounded neuron is hard to fool."** Locality means an attacker must push the input out of the true basin and into a wrong one, a large move, where a direction-neuron flips on an imperceptible nudge. Unifies with OOD (attack = climbing out of a basin; OOD = sitting between basins), ties straight back to the AttractorField. **Risk:** apparent robustness can be gradient masking. Must verify with FGSM + PGD on a Yat net vs a matched ReLU net (accuracy-vs-perturbation curves) before committing. Strongest candidate.
2. **Few-shot / one-shot: learn a class from a single picture.** Place one prototype. Safe, definitely works, a striking extension of editability; pairs naturally with C3's "features make placement better." Good fallback if robustness does not hold.
3. **Calibration, deeply.** Softmax overconfidence on garbage vs the Yat field going quiet. Only touched so far (C1); could be the OOD/robustness post's quantitative half.
4. **Constructing the backbone itself.** The deeper-construction tease at the end of C3: how far down, layer by layer, can construction reach before you must optimize.

---

## 6. Where things live

- Posts: `src/content/blog/<slug>.mdx`
- Engine viz components: `src/components/viz/*.astro` (+ shared helpers like `yatedit.js`, `yatedit.worker.js`)
- Viz engine: `src/components/viz/engine/` (`vizkit.js`, `jax.js`, `draw.js`, `nanograd.js`)
- Experiments + renderers: `scripts/*.py`
- Public assets (sprites, JSON, GIFs): `public/` and `public/<slug>/`
- Deploy: merge to `master` -> `.github/workflows/deploy.yml` -> `gh-pages`
