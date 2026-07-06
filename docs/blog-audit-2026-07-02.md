# Full blog audit, 2026-07-02

Scope: all 39 posts in `src/content/blog/` (18 explainer/companion pairs + standalones,
3 drafts), their viz components, GIFs, scripts, and the series bible. Method: mechanical
sweeps (em dashes, jax-js, SEO lengths, GIF sizes, link integrity) plus a deep read of
every post against `docs/writing-style.md` and the blog-writing skill, plus a runtime
pass (build, vizcheck, screenshots) on the newest draft and three recent posts.

**Overall verdict.** The catalog is healthy where it is newest. The build is green (147
pages, zero errors), vizcheck found zero blank canvases and zero console errors on every
page checked, there are no em dashes anywhere reader-facing, and the recent posts
(where-does-a-weight-live, you-dont-have-to-train-the-features, the fixed-point draft)
are close to the house standard. The problems concentrate in four places: (1) a handful
of genuine correctness bugs in live posts, the worst being a sum-vs-max readout mismatch
that runs through the whole Yat arc; (2) a systematic "jax-js" branding violation across
~8 explainers; (3) companions that shipped under the GIF bar (one has zero); (4) a series
bible that is two weeks and one entire mini-arc out of date.

---

## 1. Correctness bugs (fix before anything cosmetic)

1. **Sum-vs-max readout mismatch across Arc C (the big one).**
   `scripts/yat_editable_fmnist.py` computes every C2 number with `mode='max'`
   (nearest-prototype per class), but `edit-a-network-by-hand.mdx` derives everything
   from the linear sum readout `s = A^T phi` and `edit-a-network-jax-flax-nnx.mdx`
   shows sum-readout code whose own comment concedes it (`# ~79% with a
   nearest-prototype readout`, line ~96). Running the shown code gives ~68%, not 79%.
   The same bug appears in `train-the-features.mdx` (lines 68-70 show a sum; the
   83.2/74 numbers come from `.max(1)` in `construct_vs_optimize.py`). It also
   collides with C1, which published 68% sum-vote vs 79% max and stated "a linear
   layer cannot express a max", and with the bible's C1 beat ("79%, sum-vote").
   Fix: pick one readout per post, make code/algebra/numbers agree, add the missing
   assert, and correct the bible.

2. **The three-states pair contradicts itself.** Explainer line 108: alignment
   resolves first (ORGANIZED), uniformity later breaks symmetry (STRUCTURED).
   Companion line 159 + GIF caption 189: uniformity resolves fast (ORGANIZED),
   alignment slowly (STRUCTURED). One of them is wrong for its own run; reconcile the
   mapping against the actual render scripts.

3. **Fixed-point draft: the "training length" line is mislabeled.**
   `yat_deq_maze.py` trains at `T_TRAIN = 30`, but `MazeExtrapolate.astro:71` draws
   the dashed "train length" at 22 and the post binds 54% to it (lines 146-148).
   Measured: 27x27 is 54.2% at 22 iters, 76.0% at the true training length 30. Fix
   post + component + the bible's C5 beat. Also soften "never exceeds 0.92"
   (measured 0.9226).

4. **`not-all-infinities-are-equal.mdx:97`**: 0.9/0.01 = 90x, stated as "one hundred
   times"; the adjacent caption says the correct thing.

5. **`cheap-attention-is-linear-attention.mdx:288`** cites "the companion post" for an
   argument made by the earlier explainer (`attention-is-a-kernel`), not its companion.

6. **`untangling-the-moons.mdx:120,138`**: "eight datasets" — the dropdown has ten.

7. **Live post links a draft**: `latent-on-the-spectrum.mdx:166` (and the "three
   posts, one picture" count at 219) links `/blog/modality-gap-complementary/`,
   which is `draft: true` — a dead link in production until that draft ships.

8. **Editing artifacts**: duplicated teach sentence at `edit-a-network-by-hand.mdx:93-95`;
   double colon-stub opener at `readout-as-convex-combination.mdx:149-151`.

## 2. Number-provenance failures (the "every number from a real run" rule)

- **`what-an-mlp-knows.mdx`** is the worst offender: all knockout/naming numbers
  (51/256, 86.5%→42.7%, 43.7pp, 1.6pp, 0.81M) trace only to a PDF in
  `public/papers/`, and `KnockoutBars.astro:70-71` hardcodes admitted
  approximations ("Approximations consistent with paper text"). Building its missing
  JAX companion would fix both (see §6).
- **`your-neuron-is-a-picture.mdx`**: 82.6/85.9 exist only in
  `public/yat-fmnist/meta.json` (no script produces that directory); the warm-start
  numbers (65.5→75.3 vs 72.0; recall 48→59) appear in no script at all; the
  companion quotes different headline numbers (82.1/84.5) without acknowledgment.
- **`construct_vs_optimize.py` is cited in neither C3 nor C4**, despite sourcing
  85.7/83.2/74 in both. The C4 companion's displayed code also omits the placed
  `eps = median(d2)*0.1`, so the shown code does not reproduce the printed 83.3%.
- **`mlp-block-is-a-representer-theorem.mdx`** never names `yat_ffn_whitebox.py`
  (only the companion does).
- Smaller: `three-states.mdx:130` "recent line of work" has no citation;
  `latent-on-the-spectrum.mdx:205` "Look at a real CLIP space" has no run;
  `what-a-finite-kernel-buys-an-mlp.mdx:144` "markedly smaller generalization gap"
  traces to nothing; `yat-mlp-jax-flax-nnx.mdx` names zero of its seven render
  scripts and its 25%-vs-50% lazy-loading comparison rides on an arbitrary 0.18
  threshold vs a sign test.

## 3. Hard-rule violations (mechanical, high-count)

**"jax-js" surfaced to readers — the most widespread violation in the audit.**
- In captions (17 hits, 5 posts): `edit-a-network-by-hand` (1),
  `readout-as-convex-combination` (4), `cheap-attention-is-linear-attention` (7),
  `what-a-finite-kernel-buys-an-mlp` (4), `why-attention-needs-qk-projections` (4).
- In on-canvas readouts (`· jax-js`) of 13 components across three more explainers:
  KernelSmoother:87, AttentionPatterns:91, RBFExpansion:68 (all three panels of
  `attention-is-a-kernel`), FeatureLift:93, GramReconstruction:86,
  RandomFourierBuild:99, RandomFeatureKernel:133, KernelPartition:124,
  LinearAttentionApprox:124, SymSkewScore:120, RoPERelative:97, LowRankBudget:92,
  InductionPattern:108.
- Same-spirit engine branding: "nanograd autodiff engine" in captions at
  `what-a-finite-kernel-buys-an-mlp.mdx:105,140` and `three-states.mdx:106`;
  "the engine's symmetric eigendecomposition" at `readout-as-convex-combination.mdx:277`.
  One sweep fixes all of these: replace with "computed live" / "runs live in your
  browser".

**GIF budget (~1.6 MB cap), 9 offenders:**
`gravity-attention-bookkeeping.gif` **5.66 MB** (and `loading="eager"`),
`jax-welch/simplex-descent.gif` 2.01, `jax-welch/rank-collapse.gif` 1.72,
`jax-welch/welch-descent.gif` 1.70, `linear-attention-pipeline.gif` 1.72,
`yat-vs-relu-progression.gif` 1.69, `edit-teach.gif` 1.67, plus two at the line
(1.61, 1.60).

**Companion GIF floor (3 minimum, 5-6 shipped standard):**
- `mlp-block-is-a-representer-theorem-jax-flax-nnx`: **0 GIFs** (two static PNGs,
  both summaries not processes) — the largest single format violation in the catalog.
- `linear-attention-jax-flax-nnx`: 2 GIFs, both oversized, no verification code,
  trains on placeholder data.
- `qk-projections-jax-flax-nnx`: 2 GIFs; its most original section (rank budget,
  gauge freedom, RoPE diagonal) is entirely unanimated; also trains on nothing.
- `convex-readout-jax-flax-nnx`: 2 GIFs; its "hull check" comment is backed by a
  max-abs printout that tests neither hull nor span (line 157-158).
- At the floor (3) with visible gaps: `yat-mlp-fmnist-jax-flax-nnx` (no GIF for OOD
  abstention, a bible-promised beat), `edit-a-network-jax-flax-nnx` (all three GIFs
  one idiom; no figure of the actual row edit; the promised "diff the weights"
  proof never happens), `where-does-a-weight-live-jax-flax-nnx` (fine, one easy
  fourth GIF available).

**Frontmatter / conventions:**
- seoDescription over 160: `train-the-features-jax-flax-nnx` (172),
  `where-does-a-weight-live-jax-flax-nnx` (164).
- pubDate pairing convention broken widely: attention pair a month apart
  (05-14 vs 06-15), untangling/organizing 3 days apart, C3 pair inverted
  (explainer T18:00 after companion T17:00), latent-spectrum and three-states
  explainers share a byte-identical timestamp, welch pair has no time offset,
  several explainers carry times where date-only is the rule. CiteAs date mismatch
  in `linear-attention-jax-flax-nnx` (06-01 vs 05-31).
- `modality-gap-complementary` title is 62 chars with no seoTitle (draft, minor).

## 4. Series-level contradictions (reader-facing coherence)

1. `opposite-is-not-different` claims CLIP "optimized the wrong target for years";
   `untangling-the-moons` §8 explicitly corrects this (equilibrium is fine, dynamics
   is the issue) but the earlier post was never reconciled. Add the
   equilibrium-vs-dynamics distinction or a forward pointer.
2. Both posts repeat an imprecise antipodal-pairs claim (`opposite`:102,
   `untangling`:319): "mutually antiparallel" sets max out at 2 vectors; what is
   meant is d mutually orthogonal antipodal pairs / the Gram-PSD infeasibility.
3. `modality-gap-complementary` (draft) engineers away the init-born gap that
   `not-all-infinities-are-equal:146` (live) reports from Liang 2022; needs one
   reconciling paragraph.
4. Same-day redundancy (06-07): three GIFs across the latent-spectrum and
   three-states pairs all train a tiny classifier and watch spectra/cosines
   collapse; alignment/uniformity is re-explained after being spent in the welch post.
5. The C4 explainer re-displays the full Yat kernel equation (spent C0-C2);
   `mlp-block-is-a-representer-theorem` re-teaches Geva key-value memory and OOD
   abstention without linking `readout-as-convex-combination` or C1.

## 5. The bible is out of date (`docs/blog-series.md`)

- Catalog says 29 posts; there are 39. The entire weight/representer mini-arc
  (where-does-a-weight-live, what-can-a-weight-be, mlp-block-is-a-representer-theorem,
  three pairs, June 25-27) is missing from the catalog.
- C3/C4 are marked draft; both are live. C1's beat numbers ("79% sum-vote") are
  stale/wrong (see §1.1). C5's "22-iter training budget" carries the §1.3 error.
- Suggested arc placement for the missing cluster: an RKHS-foundations interlude
  between Arcs B and C (where-does → what-can → mlp-block as the B×C capstone),
  with `readout-as-convex-combination` re-annotated as the capstone's seed.
- Ledger needs new rows: convex/conic/affine/linear taxonomy + non-identifiability;
  direction-vs-place at RKHS level + representer build; price-list /
  eigenvalue-decay / Sobolev-vs-Gaussian / sphere harmonics; FFN as exact key-value
  memory (read/attribute/edit/abstain); and the C5 row is present but its numbers
  need the 22→30 fix.
- New open thread to register: `why-attention-needs-qk-projections` closes on "the
  real kernel version of attention... the question these posts have been heading
  toward" with nowhere to go — the Yat-attention post.

## 6. What is missing (posts and companions)

1. **`what-an-mlp-knows` needs a JAX companion more than any other post.** It is the
   only Arc B explainer without one, its experiment exists only as a PDF, and a
   companion (train the Yat trunk in Flax NNX, reproduce 86.5→42.7 and the 51-unit
   naming, export real JSON to re-drive KnockoutBars) fixes its provenance violation
   and yields 5-6 natural GIFs (footprints emerging over training, knockout bars
   from a real run, prototype drift, slot surgery).
2. **The fixed-point companion** (bible: "companion pending"). The scripts already
   contain all the math. GIF slate: residual collapsing on a log axis while the PCA
   state rolls to z*; scattered starts contracting ~0.66/step (Banach); the boundary
   freezing with accuracy plateau at 6; the settle-time map/histogram filling in;
   **implicit differentiation** (the `jax.custom_vjp` adjoint solved by the same
   iteration — the concept the post states but never shows); the 27x27 flood-fill
   climbing 76%→99.5% plus the r=0.98 scatter accumulating.
3. **Draft finish lines:**
   - `modality-gap-complementary` (closest to shippable, and a live post links to
     it): reconcile the init-born gap with `not-all-infinities`; re-run `gap.py` at
     the sharp temperature or soften "through zero and negative" (the README says
     it only reaches the random line); name `toy.py`; craft pass on 5 flagged
     openers; vizcheck the four bespoke canvases; then publish.
   - `simo2-geometry-by-construction` (rightly stalled): blocked on the SimO2/AFCL
     paper's ε-collapse crux; also reuses the welch post's `ConeEffect` (hard-rule
     violation — cut it and link), has zero backing scripts (create
     `scripts/jax-simo2/` with a b-sweep confirming the closed-form regimes), and
     is structured as a numbered rebuttal rather than a narrative.
4. **Unclosed promises inside live posts:** the weight's "which training images is
   it near?" question (`where-does-a-weight-live`) is never answered; the "diff the
   weights to prove nothing else moved" promise (C2) is never delivered; the
   mlp-block post promises "the whole transformer explains itself" but attention
   never appears in any panel; `three-states`' practical thesis (probes see the
   transition coming) is never demonstrated; the welch companion's checklist skips
   the margin/noise constraint entirely.

## 7. Craft findings (the pattern, then the worst posts)

The failure mode is exactly the one `docs/writing-style.md` diagnoses: sections that
open like paper subsections. Across the catalog the section-opening test fails most in:

- **`activations-are-bad-for-geometry`** (weakest post in the catalog): definition
  lede, paper roadmap at line 72, 5/9 section openers fail, only 3 panels, no
  physics bridge for the most physical material in Arc A, and its central claim
  (rank loss compounds with depth) has neither experiment nor viz.
- **`attention-is-a-kernel`**: thesis-first opening ("The claim of this piece, in
  one sentence"), 5/8 openers fail, 3 panels, and the punchline (the MLP has no
  kernel) is never shown, only asserted.
- **`what-an-mlp-knows`**: 6/10 openers fail; its crown-jewel surprise (the
  x-knockout at 43.7pp vs 1.6pp) is delivered as a measurement paragraph instead of
  a sprung trap.
- **`readout-as-convex-combination`**: instruction-plus-equation lede ("Write the
  hidden activation vector..."), two endings ("The point" lands before the best
  section), and its sharpest claim (non-identifiability of attributions) is
  introduced as "a subtlety worth stating plainly".
- **Citation-first survey sections** recur: welch:218,233; three-states:120-131;
  all six historical sections of `untangling-the-moons` open "[Authors]'s [Paper]
  did X".
- **Surprise economics in C3/C4**: both explainers spend their headline number in
  the hook (74% and 83.3%), and C3's closer pre-spends C4's number, so no in-body
  reveal has tension left. The C1 opener flagged in the style doc
  ("None of this is a method bolted on", your-neuron:146) survives verbatim.
- **Comma-splice overload**: the no-em-dash rule is being satisfied by substituting
  comma pileups instead of restructuring (worst in `latent-on-the-spectrum`, e.g.
  line 219, and the 120-word frontmatter descriptions).
- Smell-list openers worth a single sweep: "Start with" (welch:94, cheap:122,156,
  what-can:66, organizing:102), "It is worth" (where-does:55, not-all:116,
  modality:239), "Here is..." (welch:144, latent:97, three-states:112, mlp:77,
  where-does:75), "The clean way to see it" (qk:158, modality:194).

## 8. Runtime health (verified today)

- `SHOW_DRAFTS=true npm run build`: exit 0, 147 pages, no warnings.
- vizcheck: 0 blank canvases, 0 console errors on `your-network-is-a-fixed-point`
  (7 canvases), `mlp-block-is-a-representer-theorem` (4), `what-can-a-weight-be`
  (4), `you-dont-have-to-train-the-features` (6).
- Fixed-point viz screenshot verdicts: all seven render and teach; polish items:
  RecursionLoop wastes the bottom ~35% of its canvas; MazeFloodFill's slider max
  (60) disagrees with the iter counter (36); MazeExtrapolate's "train length"
  label is occluded by the default budget line (both at 22) and carries the 22-vs-30
  error; ContractionPull trajectories huddle in one quadrant (shared bounds).

## 9. Best new-viz ideas from the audit (top picks)

Explainer panels (live, engine-based, one concept each):
1. **PlateauProbes** (three-states) — loss flatlining on top; intra-class variance,
   inter-class cosine, distance-to-ETF resolving underneath, live. The post's thesis,
   made watchable.
2. **OrderShuffle** (edit-a-network) — teach classes in random orders live; accuracy
   traces braid to the same endpoint; max |Δs| between orderings reads exactly 0.
   The missing 79.4==79.4 visual, and superposition made falsifiable.
3. **GaugeSpinner** (qk-projections) — (W_Q M, W_K M^-T) heatmaps churn while the
   score matrix sits frozen; "only the product is identified" made visceral.
4. **NoDistanceScatter** (attention-is-a-kernel) — kernel layer vs ReLU MLP fed the
   same pairs; one scatter is monotone, the other structureless. The post's punchline,
   currently unillustrated.
5. **DepthRankDecay** (activations) — live SVD of the end-to-end Jacobian as a depth
   slider grows; toggle residuals and watch the rank floor appear. The core claim,
   currently asserted.
6. **SimplexEmergence** (opposite-is-not-different) — live CE training of n class
   vectors; the pairwise-cosine heatmap converges onto -1/(n-1). The post's best
   prose claim as its centerpiece.
7. **Two halves, one colorbar** (mlp-block) — one prompt; live attention weights over
   tokens beside live memory weights over slots, same idiom. The thesis panel the
   post lacks.
8. **ModeAnatomy** (latent-on-the-spectrum) — step through eigenmodes; per-class
   loadings recolor live from eigh. "Mode 1 = animals-vs-vehicles" as a readable fact.
9. **Neighbors of a weight** (where-does-a-weight-live) — rank ||f - phi(x_i)||_H
   live and highlight the weight's nearest training points; answers the post's own
   unanswered question.
10. **BifurcationCurve** (simo2, when unblocked) — settled pairwise cosine vs b with
    the closed-form pitchfork overlaid; empirical dots landing on theory.

Companion GIFs (real-process, matplotlib):
1. **ts-staircase** — deep-linear loss steps + per-mode gains switching on (Saxe
   exact case); the missing plateau GIF.
2. **rowLedger / bitForBit** (edit companion) — W as a thumbnail strip: teach
   concatenates rows on, forget excises a block; then |W_after - W_before[keep]| as
   a wall of exact zeros. Delivers the "diff the weights" promise.
3. **OOD histogram forming** (C1 companion) — Fashion vs MNIST max-match
   distributions separating over training.
4. **error-vs-m** (linear-attention companion) — exact vs implied attention matrix
   with relative error falling like 1/sqrt(m); the companion's missing quantitative
   anchor.
5. **gain sweep animated + abstention token-by-token + attribution recomposition +
   memory heatmap during generation** (mlp-block companion) — four GIFs, all
   computable from the existing script/model, taking it from 0 to standard.
6. **implicit-diff residual** (fixed-point companion) — the backward fixed-point
   solve decaying; the one concept the post states but never shows.

## 10. Suggested execution order

1. Correctness: sum-vs-max across Arc C (posts + script + bible), three-states
   contradiction, fixed-point 22→30, the four small factual fixes (§1.4-1.8).
2. One sweep: de-brand jax-js/nanograd (17 captions + 13 readouts + 3 captions),
   trim 2 seoDescriptions, fix pubDate pairings, shrink the 7 oversized GIFs.
3. Provenance: create/name the missing scripts (yat-fmnist exporter, cite
   construct_vs_optimize.py in C3+C4, add eps line to the C4 companion snippet).
4. Companions below the bar: mlp-block (0→4 GIFs), linear-attention, qk-projections,
   convex-readout; then the what-an-mlp-knows companion (new post) and the
   fixed-point companion (gates its publish).
5. Bible refresh: catalog +10 posts, ledger rows, C1/C5 numbers, register the
   Yat-attention open thread.
6. Craft passes, highest-leverage first: activations-are-bad-for-geometry,
   attention-is-a-kernel, what-an-mlp-knows, readout-as-convex-combination lede,
   the smell-list opener sweep, C3/C4 surprise economics.
7. Drafts: finish modality-gap (unblocks the production dead link), keep simo2
   parked on the paper, ship fixed-point after its companion.
