# Blog series bible

Working reference for the blog at `src/content/blog/`. Records what exists, the
conventions we write to, the story arc of the active (Yat-kernel) series, a
ledger of which concepts are already "spent" so we stop re-explaining them, and
the open threads for future posts.

_Last updated: 2026-07-19 (added the draft `the-geometry-of-attention`, the attention-territory post, to the catalog, beat log, and concept ledger). `D` = draft._

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
- **Title-pattern watch.** Two title families are at their ceiling; do not extend either. The negation pair "You Only Have to Train the Features" / "You Don't Even Have to Train the Features" (train-the-features, you-dont-have-to-train-the-features) is a *deliberate* coupled escalation across adjacent C3/C4 posts, keep it as the pair it is but add no third. The possessive parallel "Your Neuron Is a Direction. It Should Be a Picture." / "Your Network Is a Stack of Layers. It Could Be a Fixed Point." (your-neuron-is-a-picture, your-network-is-a-fixed-point) is likewise intentional, keep it at two. Any new post takes a title from a different mold. (The C9 draft was the third negation title and broke the rule; retitled to "One Kernel, Fitted Twice" 2026-07-16.)

**LR fairness in any head-to-head.** Whenever a post compares a Yat/kernel model
against a baseline, each side gets a per-variant LR sweep + best-epoch selection
(the survival-trial protocol). Yat variants typically want ~3-10x the softmax LR
(see the yat-attention-training skill); reusing the baseline's LR under-trains the
kernel side. The yat-attention post shipped without this and was audited after the
fact (bundles kgl_blog-yatattn-lrsweep + -fair): the table survived, but only
because the kernel's quality happens to be LR-flat. Do the sweep BEFORE publishing.

**Publishing.** Flip `draft: false`, commit the post + its viz components + `public/` assets + `scripts/` + GIFs, push, open a PR to `master`, merge. Merging `master` auto-deploys to `gh-pages` (`.github/workflows/deploy.yml`). The site is `https://tahabouhsine.com/blog`. Sitemap: `https://tahabouhsine.com/blog/sitemap-index.xml`.

**Pairing/ordering.** Explainer gets a date-only `pubDate` (sorts at 00:00); its companion gets the same date with a `Txx:00` time so it sorts just after.

---

## 2. The catalog (46 posts, five arcs)

Reader-facing series navigation lives in `src/data/series.ts` (SeriesNav on every
post); keep that file in sync with this catalog when a post publishes.

### Arc A: representation geometry, contrastive learning, latent space
| status | explainer | companion |
| --- | --- | --- |
| live | activations-are-bad-for-geometry | -- |
| live | opposite-is-not-different | -- |
| live | not-all-infinities-are-equal | -- |
| live | untangling-the-moons | organizing-randomness-jax |
| live | welch-bound-good-latent-space | welch-bound-jax-analysis |
| live | latent-on-the-spectrum | latent-on-the-spectrum-jax |
| live | three-states-of-information | three-states-of-information-jax |
| live | distillation-is-kernel-transfer | distillation-is-kernel-transfer-jax-flax-nnx |
| **D** | modality-gap-complementary | -- |
| **D** | simo2-geometry-by-construction | -- |

### Arc B: attention as a kernel
| status | explainer | companion |
| --- | --- | --- |
| live | attention-is-a-kernel | attention-is-kernel-jax-flax-nnx |
| live | what-an-mlp-knows | -- (companion wanted; would also fix its number provenance, see open thread 5) |
| live | cheap-attention-is-linear-attention | linear-attention-jax-flax-nnx |
| live | why-attention-needs-qk-projections | qk-projections-jax-flax-nnx |
| live | attention-is-a-compatibility-kernel | attention-is-a-compatibility-kernel-jax-flax-nnx |
| **D** | the-geometry-of-attention | the-geometry-of-attention-jax-flax-nnx |

### Arc B.5: weights in kernel space (the RKHS-foundations interlude)
Sits between Arcs B and C: what a weight is once everything is a kernel.
`readout-as-convex-combination` seeds the question; `mlp-block` is the B x C
capstone (the Yat prototype story returned to the transformer).
| status | explainer | companion |
| --- | --- | --- |
| live | readout-as-convex-combination | convex-readout-jax-flax-nnx |
| live | where-does-a-weight-live | where-does-a-weight-live-jax-flax-nnx |
| live | what-can-a-weight-be | what-can-a-weight-be-jax-flax-nnx |
| live | mlp-block-is-a-representer-theorem | mlp-block-is-a-representer-theorem-jax-flax-nnx |
| live | regularization-is-a-price-list | regularization-is-a-price-list-jax-flax-nnx |

### Arc C: the Yat kernel / prototype neuron (the active story)
| status | explainer | companion |
| --- | --- | --- |
| live | what-a-finite-kernel-buys-an-mlp | yat-mlp-jax-flax-nnx |
| live | your-neuron-is-a-picture | yat-mlp-fmnist-jax-flax-nnx |
| live | edit-a-network-by-hand | edit-a-network-jax-flax-nnx |
| live | train-the-features | train-the-features-jax-flax-nnx |
| live | you-dont-have-to-train-the-features | handbuilt-features-jax-flax-nnx |
| live | depth-by-construction | depth-by-construction-jax-flax-nnx |
| live | calibration-of-a-bounded-net | calibration-of-a-bounded-net-jax-flax-nnx |
| live | a-risk-model-that-names-its-reasons | a-risk-model-that-names-its-reasons-jax-flax-nnx |
| live | your-network-is-a-fixed-point | your-network-is-a-fixed-point-jax-flax-nnx |
| live | edit-a-fixed-point | edit-a-fixed-point-jax-flax-nnx |
| live | survival-model-on-trial | survival-model-on-trial-jax-flax-nnx |
| **D** | you-dont-have-to-solve-a-kernel-machine | you-dont-have-to-solve-a-kernel-machine-jax-flax-nnx |

### Arc D: networks as integrators (new, opened 2026-07-04)
The standing move: numerical analysis as an architecture catalog. Each
structure-preserving integrator (symplectic, reversible, adaptive-step,
energy-conserving) is a candidate network, and its conservation law is a
testable prediction about trained hidden states.
| status | explainer | companion |
| --- | --- | --- |
| live | skip-connections-are-half-of-newton | momentum-resnet-jax-flax-nnx |
| live | transformers-with-a-velocity-ledger | transformers-with-a-velocity-ledger-jax-flax-nnx |
| live | a-network-that-conserves-energy | a-network-that-conserves-energy-jax-flax-nnx |
| live | backprop-without-the-memory | backprop-without-the-memory-jax-flax-nnx |
| live | depth-on-demand | depth-on-demand-jax-flax-nnx |

---

## 3. Arc C story, beat by beat (the part we are actively building)

The through-line: **put the Yat kernel `phi(x) = (w.x + b)^2 / (||x - w||^2 + eps)` where a neuron's activation was, and the neuron stops being a direction and becomes a prototype, a point in input space. That one change makes the network legible, editable, and constructable.** All on Fashion-MNIST.

### C0. what-a-finite-kernel-buys-an-mlp  (+ yat-mlp-jax-flax-nnx)
The entry point. Replace the activation with a finite, explicit, positive-definite kernel (the Yat kernel) and an MLP becomes a kernel machine: locality, attribution, geometry, capacity control, a writable feature map. General argument, 2D toys.

### C1. your-neuron-is-a-picture  (+ yat-mlp-fmnist-jax-flax-nnx)
Title: "Your Neuron Is a Direction. It Should Be a Picture." A direction is not a referent you can point at; a Yat prototype lives in input space and on images is literally a picture. Beats: every neuron is a picture; build it by hand with no training (**68%** sum-vote; **79%** nearest-prototype, and the nearest-prototype rule is the one behind every quoted accuracy in C2/C3); the network explains itself (convex attribution); its geometry is a map you can walk; reading the mistakes; **where the pictures come from** (legibility is decided at init, by seeding prototypes on data); warm-start helps low-data and imbalance. Companion: build/train YatMLP in Flax NNX, prototypes as images, prediction as a vote, **OOD abstention** ("it knows when it doesn't know", MNIST as OOD), random-init stays noise, UMAP prototype trajectories.

### C2. edit-a-network-by-hand  (+ edit-a-network-jax-flax-nnx)
Title: "Your Network Is a List of Pictures. You Can Edit It." If a neuron is a picture, a network is a list, and a list you edit by hand. Beats: the score algebra `s = A^T phi`, `s_c = sum over class c's prototypes`; **the kernel as a landscape of wells** (the denominator is a softened inverse-square law, `eps` is the N-body softening length; prototypes are masses; classification is falling into a basin; superposition); **teach** a class = append rows, no training (Bag/Boot **95%/94%**, old classes invariant, incremental == from-scratch **79.4% == 79.4%**); **forget** a class = delete rows, exact unlearning (Sandal **64% -> 0%**, other nine **81.1% -> 81.3%**); why it is hard for ordinary nets (entangled `z_c = w_c . h(x)`); the construction-vs-optimization boundary (tease). Math grounded in the per-class **kernel mean embedding** `mu_c = sum Phi(W_u)`. Four fresh jax-js viz: **KernelVote** (the molecule), **AttractorField** (2D physics sandbox: drop particles into basins, delete a well), **TeachByExample** (empty memory bins, test images flip red->green), **ForgetMatch** (each image linked to its nearest prototype; delete a class, only its images re-route). Companion: teach = `jnp.concatenate`, forget = boolean mask, three physics GIFs (settling, basins forming, a basin reclaimed). Scripts: `yat_editable_fmnist.py`, `render_yat_edit_gifs.py`, `render_yat_edit_assets.py`.

### C3. train-the-features  (LIVE)
Title: "You Only Have to Train the Features." Construct the Yat head on **learned features** instead of pixels. Real numbers (`construct_vs_optimize.py`): constructed head **83.2%** vs trained head **85.7%** on a converged CNN backbone; **74%** on a random backbone whose trained head is at chance; raw-pixel head 79%; edits survive in feature space (Sandal 93 -> 0, others 82.2 -> 82.5; teach Bag/Boot 95/95, all-10 83.2). Thesis: the representation is the only thing you must train; the classifier and its edits are furniture you place. **Status:** rebuilt into a 7-act post with five fresh viz (FeatureVsPixel, TrainedVsBuilt, RandomSurprise, BoundaryCurve, RepresentationFold + the interpretability pair); the "what is a feature / what is trained vs built" foundation answered the earlier thinness.

### C4. you-dont-have-to-train-the-features  (LIVE, companion handbuilt-features-jax-flax-nnx)
Title: "You Don't Even Have to Train the Features." The capstone: build the FEATURES by hand too. Hand-engineered detectors (6 oriented-edge channels + 1 corner, the HOG/SIFT lineage) pooled over a 7x7 patch grid = 343 NAMED dimensions, fed to the same constructed Yat head. Real numbers (`handbuilt_vision.py`): hand-built features + built head **83.3%** (matches the trained backbone's 83.2%, vs trained head 85.7%, raw pixels 79%, nearest-centroid 75.6%). Thesis: construction reaches all the way to a fully readable ~83%; the last 2.4 points are the only thing learning buys (data-specific features). Five fresh viz: **DetectorBank** (the hand-built filter glyphs), **DecomposeImage** (image -> 7 named energy maps), **AssignedDimensions** (the centerpiece: 7x7 grid x 7 detectors = 343 named axes, "every axis is a sentence"), **HandBuiltVote** (full forward pass traced, kernel live), **ConstructionLadder** (the four-rung accuracy ladder, blue=built/orange=trained). Closes open thread #4 (constructing the backbone). Assets in `public/handbuilt/`, shared module `handbuilt.js`.

---

### C5. your-network-is-a-fixed-point  (LIVE, 2026-07-08; companion still wanted)
Title: "Your Network Is a Stack of Layers. It Could Be a Fixed Point." Share ONE Yat operator across depth (recursive / weight-tied), and the stack collapses into a single equation whose answer is a fixed point `z* = F(z*; x)`, `F(z;x) = tanh(A·phi_W(z) + Ux + z0)`. Trained into a contraction (spectral penalty on `||J_F||`), so the fixed point is unique and reached from anywhere; depth becomes iteration count and the network halts when the residual stops (adaptive depth per input). This is a Yat **deep-equilibrium model** (Bai et al.) + implicit differentiation (constant memory, no unrolling). Real numbers (`scripts/yat_deq.py`, two moons): test **98.2%** from **1700** params shared across all depth, **24** prototypes, 32-D state; `||J||₂` mean **0.66** / max **0.92** (contraction), residual to **4e-7**, accuracy plateau by depth **6**, settle-times 14 (easy) to 65 (boundary), median 20. Physics: the iteration IS the falling; a contraction = a single basin, the state rolls to one rest point (extends the C2 attractor world into state-space dynamics). Math is explicit: the recursion unrolled `z_L = F(F(...F(z0)))`, shared-alpha vs the ordinary per-layer-alpha stack, and the implicit-diff solve `(I - J_Fᵀ)u = ∂L/∂z*`. **The depth motivation is a second experiment** (`scripts/yat_deq_maze.py`): the SAME shared Yat operator as a weight-tied recurrent conv (5-cell neighbourhood, wall-masked) solving grid **reachability by propagation**. Trained on 11×11 with **30 unrolled iterations** (`T_TRAIN = 30`), it **extrapolates** to 27×27 (never seen) at **99.5%** by iterating more (at the training length 30: 27×27=76%; at a short 22-iter budget: 27×27=54%, 21×21=71%, 15×15=92%); iterations-to-settle vs BFS radius r=**0.98**. This is the "why go deep" the moons can't give (a fixed L-stack can only propagate L hops; a recursion has no ceiling). Seven fresh live viz: **RecursionLoop** (one operator fed back vs a fixed A0..A5 stack, real 32-D state strip), **RelaxToEquilibrium**, **ContractionPull** (spread shrinks ~2/3/step), **DepthDial** (boundary freezes, accuracy plateau@6), **AdaptiveDepthMap**, **MazeFloodFill** (live flood-fill to a settled reachable map), **MazeExtrapolate** (accuracy-vs-iterations per size + the r=0.98 adaptive scatter). Modules `yatdeq.js` (moons, fully on-device jax-js with .ref discipline) + `yatdeqmaze.js` (maze); assets `public/yat-deq/model.json`, `public/yat-deq-maze/model.json`. **Companion (JAX + GIFs) not yet built.**

### C4.5. depth-by-construction  (LIVE, 2026-07-04)
Title: "How Far Down Can You Build?" Closes open thread "depth by construction":
a hand-designed layer 2 on C4's features (15 junctions + 6 continuations + 6 bends
+ 6 stripes = 33 one-sentence detectors, min-AND on a 14x14 cell grid, 4x4 re-pool;
`scripts/handbuilt_depth.py`). Layer 1 reproduces **83.3%**; layer 2 alone **78.8%**
(real signal, near the raw-pixel baseline with the raw edges thrown away); layers
1+2 **82.9%**, best of six designs **83.5%** (+0.2). The honest punchline: the
constructed ANDs are *synonyms* of what the kernel head already reads, and the wall
is combinatorial selection (7 names -> 224 pairwise -> ~4,630 three-way): picking the
few non-redundant combinations jointly against data IS training. Viz: JunctionAssembly,
NameAnAxis2, DepthLadder, VocabularyWall.

### C6. calibration-of-a-bounded-net  (LIVE, 2026-07-04)
The post bets against its own series and reports the loss: the Yat MLP's softmax
probabilities are WORSE calibrated than the matched ReLU's (ECE **6.2% vs 1.8%**,
fitted T **1.68 vs 1.17**, still behind after scaling; `scripts/yat_calibration.py`,
seeds 0/1/2), because the kernel's huge dynamic range (median top logit 28.9 vs 7.2)
saturates the softmax. C1's honesty survives *pre-softmax*: max kernel score separates
Fashion from MNIST at AUROC **0.916 +/- 0.008** across seeds while the softmax channel
swings 0.81-0.93. Sharpest claim: calibration lives in the head; the kernel's honesty
lives in field magnitude, and the softmax's shift invariance provably deletes it.
Spends: reliability diagram/ECE, temperature scaling, softmax shift-invariance,
selective prediction. Viz: ReliabilityStaircase, TemperatureDial, ConfidenceLedger,
FieldVsSoftmax, CostOfOverconfidence.

### A-capstone. distillation-is-kernel-transfer  (LIVE, 2026-07-04)
Title: "Distillation Is a Geometry, Not an Answer Key." The teacher's softened
outputs serialize a class-similarity kernel S(T) = E[p p^T]; a student trained on
NOTHING but that kernel's pairwise relations (no labels; `scripts/kernel_distill.py`)
recovers about two thirds of the random-to-labels linear-probe gap (83.7-84.9% vs
77.5 random / 87.0 labels) and out-organizes the label-trained student on the
nearest-centroid probe (**81.2 vs 80.0**, near the teacher's 81.7). The student's
latent spectrum grows into the *transferred* kernel (CKA **0.976**, L1 0.24), not the
teacher's private latent (0.47), which the teacher itself only matches at 0.854:
latent-on-the-spectrum's theorem observed as a training dynamic. Temperature is the
knob choosing which kernel goes on the wire (offdiag 0.03/0.30/0.90 at T=1/4/16).
Spends: distillation-as-kernel-transfer, relational-loss isolation, spectrum/confusion
inheritance. Viz: WhatALogitLeaks, KernelHandoff, SpectrumInheritance, TemperatureLens,
InheritanceScale.

### C5.5. edit-a-fixed-point  (LIVE, 2026-07-08)
Title: "Edit One Operator, Edit Every Depth." The collision of C2's row-editability
with C5's weight-tied fixed point (`scripts/yat_deq_edit.py`, run on Kaggle;
`public/yat-deq-edit/report.json`). Editability survives but SPLITS BY ROW KIND:
readout rows keep C2's exact proofs (taught recall **85.8%**, max change in old
logits **exactly 0**, bit-for-bit undo), dynamics rows re-carve the flow at every
depth (**100%** recall, margin 3.83) but downgrade invariance from proof to
measurement (drift over 520 fixed points: median **0.02**, max 1.47, **3 flips**)
and can break the contraction certificate (**1.44**, rescued to **0.980** by one
bisected gain on the new rows). New concepts: certificate as load-bearing wall;
**edit evaporation** (finite-depth edits erased by the contraction, excursion back
to ~1e-7); silenced-vs-erased (masked trained knowledge resurrectable at 100% by 8
anchor rows). Viz: EquilibriumReshape, CertificateGauge, UntouchedProbe,
EvaporatingEdit, OneEditEveryDepth, ForgetResurrect. Open thread: true erasure of
trained knowledge = retraining by another name.

### C7. a-risk-model-that-names-its-reasons  (LIVE, 2026-07-05; + companion)
Title: "A Risk Model That Names Its Reasons." The Yat toolkit spent on a new domain,
clinical survival analysis. DeepSurv = neural Cox log-risk on the partial likelihood
with right-censoring; a Yat trunk makes h(x) = sum a_u phi_u(x) a vote over
**prototype patients**. On METABRIC (n=1904, 42% censored, `scripts/yat_deepsurv.py`,
run on Kaggle) it MATCHES standard DeepSurv on C-index (**0.627 vs 0.623**) and beats
it on integrated Brier (**0.199 vs 0.225**) and Cox NLL (**5.64 vs 6.84**):
interpretability at no accuracy cost, slightly better calibrated. Beats: exact convex
attribution ("your risk is high because you resemble these patients", sums to h within
1e-7); the clinical risk landscape (attractor field); a Nadaraya-Watson survival curve
built from a patient's prototype-neighbors; **cohort deletion** as an exact closed-form
readout edit (delta h = -a_u phi_u(x): 1.02 on the deleted 28-patient cohort vs 0.024
on everyone else, framed as "exact and local", not "provably unchanged");
OOD abstention via kernel-max (7.1 real vs 3.7 stranger). Carries a "research
illustration, not a clinical tool" note. Six fresh viz (PatientPrototypes,
WhoDoYouLookLike, RiskLandscape, SurvivalCurveFromNeighbors, ForgetACohort,
AbstainOnStrangers) + six GIFs. Spends nothing new-derived: references prototype-as-picture
(C1), attractor field (C2), convex attribution (C0/C1), teach/forget (C2), OOD
abstention (C1), calibration (C6), RKHS/representer (C0) by link.

### C8. survival-model-on-trial  (LIVE, 2026-07-09; + companion)
Title: "The White-Box Survival Model on Trial." C7's model retried at scale: five
clinical datasets (METABRIC, WHAS500, GBSG, SUPPORT, FLCHAIN) against Cox, penalized
Cox, RSF, and a standard MLP DeepSurv, LR-fair (per-model LR sweep + best-epoch
selection, `scripts/deepsurv_trial.py`, Kaggle, bundle
`scripts/results/kgl_blog-deepsurv-trial-v2/`). The claim is the existence proof:
a deep Yat-kernel survival model trains with plain gradient descent, no solve, and
lands in the pack on C-index (FLCHAIN 0.909, METABRIC 0.621, GBSG/SUPPORT in the
pack, WHAS500 its worst case at 0.679 vs Cox 0.765) while inheriting what the
others cannot give: exact convex attribution, prototype ablation (K=6 kmeans 0.629
vs random 0.605 on METABRIC), calibration read off reliability gaps (0.008-0.078
except WHAS500's 0.265), OOD abstention (perm ~0.66 / extreme 0.63 AUROC; subgroup
<0.5 and cross-dataset 0.044 reported as failures of the detector), and prototype
plausibility (8/24 within the covariate ranges, median distance 1.45). Viz:
TrialForest, TrialAblation, TrialCalibration, TrialOOD, TrialPlausibility.
Companion survival-model-on-trial-jax-flax-nnx (2 GIFs: KM tertiles separating,
prototypes migrating; 5 PNG scoreboards). Spends: everything from C7 by link;
new-derived: the LR-fairness protocol and the five-dataset viability table.

### C9. you-dont-have-to-solve-a-kernel-machine  (DRAFT, 2026-07-09; + companion)
Title: "One Kernel, Fitted Twice" (slug still `you-dont-have-to-solve-a-kernel-machine`). The thesis under the whole Arc C
existence-proof frame, stated at the kernel level: the O(n³) Gram solve that emptied
the kernel field is *plumbing*, not the deal. Take ONE Mercer kernel (the Yat/IMQ
kernel, arXiv 2605.03262) and fit it twice, once by the exact solve (kernel ridge via
Cholesky, float64), once by plain gradient descent on a small bank of prototypes
(`scripts/kernel_solve_wall.py`, Kaggle, LR-swept + best-epoch, 3 seeds). California
Housing (4,000 districts): exact solve RMSE **0.491** (4,000 coeffs, 0.7 s) vs
descended K=64 **0.512 ± 0.002** (640 params); prediction correlation on held-out
districts **r = 0.95** across K=16/64/128, i.e. the same function reached two ways.
Then descent walks through three walls the solve dies at: (1) a *measured memory wall*
at **n = 16,000** rows (Gram outgrows 16 GB; cubic projection past it), (2) Covertype
**511,012** rows the solve cannot enter whole (its 64k Gram = 33 GB) so it gets the
biggest subsample that fits, **85.5%** at 16k, then goes silent, while the descended
net minibatches (512) through the full set and its capacity ladder K=64→1,024 climbs
to **85.9% ± 0.1**, past the solve's best, (3) the end-to-end conv trunk the solve
*constitutionally cannot be* (a solve is a closed procedure, not a layer). Random
features (Rahimi–Recht 2007) is the control at the bottom of the ladder: the escape
that kept the gradient step by giving up the kernel object. Viz (fresh, live jax-js,
`solvewall.js`): TheWall (measured cost vs cubic projection), PastTheWall (Covertype
accuracy-vs-rows, solve's shaded dead zone), SameMachineTwice (r=0.95 agreement),
WhatEachMachineSees (prototypes vs held rows). Companion
you-dont-have-to-solve-a-kernel-machine-jax-flax-nnx (Cholesky solve + Flax NNX
descended module side by side, the timing wall, 511k minibatching, the conv trunk).
Spends: RKHS/representer/Mercer + centers-in-input-space (C0/C1), prototype-as-picture
(C1), the "kernel machine by plain gradient descent, no solve" existence-proof frame
that all of Arc C rides on; new-derived: the solve-is-plumbing argument, the three
walls (memory / scale / composition), r=0.95 solve-vs-descend equivalence, random
features as the historical escape.

### D1. skip-connections-are-half-of-newton  (LIVE, 2026-07-04; opens Arc D)
A skip connection x + f(x) is forward Euler: depth=time, hidden state=position,
residual branch=vector field; Euler is HALF of Newton, there is no velocity anywhere.
Physics anchor (`scripts/momentum_resnet.py`, pure-math part): Kepler orbit, dt=0.02,
Euler gains **+68.3%** energy over 20 orbits (apoapsis 1.0 -> 3.37) while leapfrog
holds **0.016%**. Network result: no cliff exists in the data; the finding is
an **exactness ceiling**: a flow-like plain net (L>=32) never reaches 100% on rings
in any of 6 runs (non-crossing obstruction) while the momentum net (v <- mu v + f(x);
x <- x + v) posts **100.0%** in all seeds; spirals L128 is a reliability story (96.2%
with seed spread 92.9-98.3 vs 98.1% with 0.6-pt spread). Momentum nets are
algebraically invertible, and the rewind fails INFORMATIVELY: backward steps divide
by mu, amplifying float noise by (1/mu)^L (3.7e-4 at mu=0.9 float32, 54 at 0.6,
3.7e10 at 0.3): friction is the thief of memory. Spends: depth=time dictionary,
non-crossing/homeomorphism obstruction, Euler-vs-leapfrog energy drift, mu-friction
irreversibility. Viz: OrbitIntegrator, DepthTrajectories, InertiaDial, RewindExact,
StabilityCliff. Companion momentum-resnet-jax-flax-nnx (6 GIFs: orbits, depth-lapse, velocity ledger, crystallize, rewind + (1/mu)^L sweep, exactness ceiling).

### D2. transformers-with-a-velocity-ledger  (LIVE, 2026-07-09; + companion)
Title: "Transformers With a Velocity Ledger." The pre-norm Transformer residual
stream IS forward Euler (x += Attn(norm x); x += MLP(norm x)), so D1's dictionary
transfers wholesale. Experiment (`scripts/velocity_ledger.py`, Kaggle, bundle
`scripts/results/kgl_blog-velocity-ledger-v2/`): four parameter-matched char-level
GPTs (2,724,864 params each; plain / velocity ledger v = mu v + (1-mu) F(x) /
ngpt-lite / ngpt+ledger), 3 seeds, dropout, best-val early stopping. The result
splits: on quality they tie (best-val 1.433 / 1.435 / 1.420 / 1.456), and the
ngpt+ledger synthesis LOSES, momentum on the sphere hurts; on dynamics the ledger
changes everything: residual-stream path length through depth 128 -> 58 (ledger)
vs 48 (ngpt) vs 32 (ngpt+ledger), mean turning angle 75° -> 29°. Same destination,
gentler road, zero extra parameters. Related work spent: Momentum Streams (arXiv
2605.24425), YuriiFormer (arXiv 2601.23236), Momentum Transformer (arXiv
2208.00579), Lu et al. (arXiv 1906.02762), ODE Transformer (arXiv 2203.09176),
nGPT (arXiv 2410.01131) as the first-order-on-a-sphere contrast. Viz: QualityTie,
PathLengthBars, ResidualStreamPath, PathStraightens, VelocityThroughDepth
(engine/velocityledger.js). Companion transformers-with-a-velocity-ledger-jax-flax-nnx
(1 GIF: path straightening over training; 4 PNGs). Spends: depth=time and
velocity-ledger dictionary from D1; new-derived: depth telemetry (path length +
turning angle per sub-update) as an instrument.

### D3. a-network-that-conserves-energy  (LIVE, 2026-07-16; + companion)
Title: "A Network That Conserves Energy." The symplectic entry of the Arc D
catalog (Greydanus et al. 2019 HNN + a leapfrog residual block). Part A: fit a
pendulum's field two ways from the same arrows (`scripts/hamiltonian_net.py`,
Kaggle, bundle `kgl_blog-hamiltonian-v2`): a free 2-output MLP (MSE 4.4e-05)
vs an HNN (one scalar H, field = its rotated gradient; MSE 3.1e-05, the
constraint is information). Integrated: free field drifts **36%** of E0, HNN
holds **0.6%** (the rollout drift is a chaotic functional of the fit, run-
sensitive; the ~60x structural gap is the claim, and prose numbers must match
the shipped bundle whose weights the panels run). dH/dt = 0 identically by
antisymmetry: a loss says please, the architecture says cannot. Part B: hidden
state (q,p) in R^4 each, block = kick-drift-kick leapfrog of a learned V(q),
h = T/L, weight-tied, plain GD. Scoreboard (3 seeds, parameter-matched plain
residual net): moons 100.0/99.9, rings 100.0/99.9, spirals 98.8 vs 92.9±5.4
(free net ahead, stated straight; existence proof). What the law buys: (1) the
learned energy is a row the plain net does not possess (undefined, not
unmeasured), held in an **h² band**: 0.05-0.19 at L=16 collapsing to
0.003-0.011 at L=64, a x16-22 shrink for a x4 step refinement, the textbook
second-order signature measured inside a trained classifier; (2) **depth =
resolution**: trained at 16, run at 64, plain drops spirals 98.8->68.0 and
rings 100->86 while leapfrog holds 92.9->92.4 and 99.8->99.8; BOTH nets are
weight-tied, so the extrapolation lives in the integrator (fixed T, steps
that mean time), not the tying, sharpening C5's lesson; (3) leapfrog is
algebraically invertible (tees D4). Does NOT buy: accuracy, or a state-norm
bound (both nets' RMS grows, stated). Physics bridge: marble-on-terrain
dictionary table; V is the series' landscape world made into the model itself.
Four fresh live panels running the real exported weights (hamnet.js):
PendulumRace (both fields integrated live, draggable start, pre-integrated
first paint), LearnedEnergyMap (banded H landscape, click to drop on a level
set), DepthLedger (state clouds through depth + the energy row only one net
has), FinerTime (live accuracy + boundary re-render at depths 4->128).
Companion (code verified by `hamiltonian_nnx_check.py`, Kaggle CPU): HNNField /
PlainField / LeapfrogNet in NNX, depth as lax.scan; 4 GIFs from the run
(pendulum race, level-set ride, state clouds with the undefined-row readout,
boundary sweep 4->128). Gotcha recorded: v1 exported an unshuffled
single-class viz sample; v2 fixed (balanced perm). Spends: depth=time (D1),
existence-proof framing; new-derived: conservation-by-construction,
symplectic/leapfrog block, h² shadow-Hamiltonian band, depth-as-resolution
extrapolation, learned-energy-as-missing-row.

### D4. backprop-without-the-memory  (LIVE, 2026-07-16; + companion)
Title: "Backprop Without the Memory." Spends D1's invertibility: the momentum
block's exact inverse lets the backward pass RECOMPUTE the trajectory instead
of storing it (a custom_vjp whose residuals are only the endpoint).
Experiment `scripts/reversible_memory.py` (Kaggle GPU, bundle
`kgl_blog-revmem-v1`), three parts. E1 memory: XLA memory_analysis of the
compiled step (each depth in a fresh process; allocator peak agrees):
standard 13.4/44.8/170.7/674.0 MB at L=8/32/128/512 vs reversible FLAT 3.2 MB;
price 24% step time at L=512. E2 fidelity: gradient cosine (rewind vs stored,
identical weights/batch) = 1.000000 inside the noise budget, dying exactly
where (1/mu)^L x eps32 ~ 1 predicts (napkin L* = ln(1/eps)/ln(1/mu): mu=.6
predicts 31, measured 0.30 at L=32; mu=.3 predicts 13; mu=.9 predicts 151,
collapsed by 128 as accumulation outpaces the single-step estimate). E3:
trains at L=64/mu=.9: 81.6% vs 79.7%, one seed each, stated. Physics bridge:
friction breaks time-reversal (damped pendulum cannot rewind, dissipation
destroys information); mu IS D1's friction; Maclaurin et al. 2015 stored the
lost bits. Arc payoff: conservation, reversibility, constant-memory training =
one property, three coats; the frictionless leapfrog (D3) rewinds by
subtraction, no cliff. Panels (fresh, revmem.js): TwoLedgers (store-vs-
recompute mechanics animated, meters = real MB), FrictionArrow (live damped/
undamped pendulum time-reversal), ExactRewind (12 particles through 64 layers
in REAL float32 via Math.fround, mu select, rewind lands/shears live),
BudgetCliff (9 measured cosines vs the pre-drawn napkin verticals, hover).
Companion: the actual custom_vjp quoted from the script, fresh-process +
memory_analysis measurement discipline, 3 GIFs (ledger mechanics, float32
rewind shear at 3 mus, error-growth riding the dashed budget) + 2 PNGs (wall,
cliff). Spends: velocity ledger + (1/mu)^L + friction-thief-of-memory (D1),
conservation/leapfrog (D3); new-derived: recompute-vs-store bookkeeping,
custom_vjp-by-inversion, the noise-budget napkin L*, memory_analysis as
instrument, dissipation-destroys-information bridge.

### D5. depth-on-demand  (LIVE, 2026-07-16; + companion)
Title: "Depth on Demand." D3's depth-as-resolution taken to its integrator
conclusion: an error controller (step doubling, Hairer) re-renders the TRAINED
leapfrog classifier to tolerance at inference, no retraining; depth becomes a
per-input expenditure. Experiment `scripts/adaptive_depth.py` (Kaggle CPU,
bundle `kgl_blog-adaptive-v2` with seed-0 weight export; v1 numbers identical).
E1 fidelity: agreement with a 256-step uniform reference 99-100% at ALL
tolerances (0.3..0.003), ~16 accepted steps at the loosest vs the reference's
256. Accuracy flat across the whole dial on moons/rings (99.9-100). E2 the
power law: accepted steps ~ tol^(-1/3) (measured log-log slope 0.32, predicted
1/3 from the leapfrog's 2nd order, h^3 local error), the sibling of D3's h^2
band, now in the compute bill. E3 spread: max/min work 3-4x per input (spirals
tol .003: 111..282). E4 the NEGATIVE (kept honest): work does NOT correlate
with classification difficulty (r between -0.2 and +0.3, no consistent sign,
all seeds/datasets); effort maps the STIFFNESS of the learned flow, not the
margin; explicitly contrasted with C5's residual halting (distance-to-
convergence, which did track the boundary): two adaptive-depth signals, two
different physical quantities, neither is the margin. Wrinkle stated: on
spirals the fully-resolved flow and the training render disagree on a few
percent of inputs (3-seed mean 94.7 vs fixed 97.4, direction varies by seed):
trained-at-16 calibrates to the 16-step render. Panels (fresh, adepth.js runs
the exported weights + the same controller live): StepDoubler (probe/accept/
reject with breathing h bar), TolDial (live re-render of 120 inputs per dial
setting + the measured power-law chart), PersonalBudget (run's margin-vs-work
scatter, hover, near-zero r printed), EffortField (live effort field over the
plane vs the boundary overlay). Companion: controller code quoted, work-vs-
accepted accounting note, 3 GIFs (controller stepping, h breathing at 3 tols,
effort field swept) + 2 PNGs (tol-sweep small multiples, power law). Spends:
depth-as-resolution + leapfrog (D3), residual halting (C5) by link;
new-derived: error-controlled inference rendering, tol^(-1/3) law in a
network, effort-measures-stiffness-not-difficulty.

### BxC. attention-is-a-compatibility-kernel  (LIVE, 2026-07-16; + companion)
Title: "The Kernel Between the Roles" (slug attention-is-a-compatibility-kernel).
The Arc B x Arc C bridge the qk-projections post teed up: keep the Q/K roles,
kernelize the compatibility BETWEEN them, s_ij = kappa(f_Q x_i, f_K x_j) with
the Yat kernel. kappa >= 0 (Mercer: squared-linear x IMQ, Schur; proof sketch
in-post per the never-hedge rule), so NO softmax anywhere: weights =
kappa/sum kappa, a literal Nadaraya-Watson smoother; no max-trick, no shift
gauge; the row mass sum kappa survives as a channel softmax cannot represent.
Experiment `scripts/yat_attention.py` (Kaggle GPU; quality bundle
`kgl_blog-yatattn-v1` 3 seeds, telemetry bundle `kgl_blog-yatattn-telem` with
checkpointed attention maps on a fixed window). Existence proof: parameter-
matched char-GPTs (2,719,169 params EXACTLY, 6L/4H/D192/T128, 12k steps),
softmax best-val 1.4923+-0.0060 vs yat 1.5131+-0.0071 (1.4% gap, stated
straight). TWO BELIEFS MEASURED DEAD (kept, not buried): (a) "bounded scores"
FALSE: trained kappa reaches 443,508 vs softmax logits ~35 (kappa(q,q) =
||q||^4/eps and norms grow); the honest claim is no-exp-downstream (4e5 passes
a polynomial ratio; e^35 overflows f32); (b) mass-as-token-confidence NULL:
AUROC 0.509/0.501/0.508. Map reading (measured): yat routing is systematically
MORE DIFFUSE (normalized entropy 0.74 vs 0.48; sharp rows 0% vs 22%) because
exp is a soft-argmax (log-scale ratios) while polynomial ratios cannot
concentrate; plausible source of the 1.4%; sharpening dial named ((q.k)^(2m)).
SLAY (arXiv 2602.04915) = the linear-time payoff. LR-FAIRNESS AUDIT (2026-07-17,
bundles kgl_blog-yatattn-lrsweep + kgl_blog-yatattn-fair): sweep over 3e-4..1e-2
x both variants; softmax's optimum IS 3e-4 (fair), yat's single-seed optimum 3e-3
(1.5007) but 3 seeds there = 1.5163+-0.0023, within run noise (~0.01) of the
published 1.5131, so the table stands; the real finding is the robustness
asymmetry (softmax diverges at 1e-2 to 2.58, yat's whole column sits in a 1%
band; they tie at every LR except softmax's optimum), now in the post.
CANONICAL-KERNEL PROMOTION (2026-07-17, user: b=0 loses universality per the
paper, b/eps must be learned per head): the post now DEFINES the kernel as
(q.k+b)^2/(d^2+eps), b/eps>0 learned per head via softplus init log 2 (+48
params, stated); universality paragraph added (numerator expands to quadratic+
linear+constant in alignment; b>0 buys the paper's density theorem, b=0 loses
it). Headline contest: softmax@3e-4 1.4923+-0.0060 vs kernel@3e-3
1.5089+-0.0020, gap 1.1%; the b=0/eps=1 run demoted to ablation (1.5131,
score_max 443,508 vs 61,221). Telemetry re-measured on the canonical kernel
(bundle kgl_blog-yatattn-b-telem, also exports learned scalars): entropy story
survives (0.73 vs softmax 0.48, sharp 0.2% vs 22%); DRIFT STORY now measured:
b wanders [0.22, 3.40] from log 2, eps SPLITS per head (most -> ~0, sharp local
routers; two late heads -> 6.4/12.7, near-global averagers): "each head chooses
what kind of statistician to be". The earlier "eps grows to tame the peak"
mechanism claim was WRONG (most eps shrank) and was removed; only measured
facts remain. QK-ECTOMY (2026-07-17/18, user: "the kernel is not bilinear, we
can omit QK"): GOAT tiers added as a section of the SAME post (user chose one
post over a series). goat_v (no Q/K, keep V; strict self-mask j<i REQUIRED,
diagonal is the kernel row-max) and goat_nov (no Q/K/V; W_O + 2 scalars/head
only, zero attention cache). Bracketed-at-both-ends LR sweeps (3e-4..1e-1;
user forced measuring the low cells, which beat 1e-3, a non-monotone pocket
inference would have missed): optima goat_v@3e-2, goat_nov@1e-2; LR-optimum
progression softmax 3e-4 -> kernel 3e-3 -> nov 1e-2 -> v 3e-2 (100x spread).
3 seeds: goat_v 1.5293+-0.0041 (2,274,545 params, -16%), goat_nov
1.5343+-0.0075 (2,052,209, -25%); V barely earns its keep once Q/K gone.
Telemetry (kgl_blog-goat-telem): on raw slices eps GROWS (mean 13.8, vs ->0
on projections) and b spans [0, 10.2]; entropy stays diffuse 0.72. Bundles:
kgl_blog-goat-{lrsweep,lrsweep2,lrsweep-low,lrsweep-hi,v-v1,nov-v1,telem}. Panels (fresh, yatattn.js):
ScoreSurface (drag the query, both score landscapes live), GaugeAndMass (the
shift gauge freezing softmax bars vs the live mass readout), QualityTieBxC
(six real curves + nulls), HeadsAtWork (real checkpointed maps, scrub
training). Companion: the two-branch attention module quoted, 3 GIFs (curves,
score-orbit, reading-traversal) + 2 PNGs (depth grid, checkpoint grid);
the entropy check lives in render_yatattn_gifs.py. Spends: attention-is-a-kernel, qk-projections,
cheap-attention, calibration's shift-invariance, prototype geometry, Mercer
memory; new-derived: no-softmax normalization, the mass channel, the
diffuseness mechanism, boundedness-vs-training distinction.

### BxC addendum: the exponential's second job (2026-07-18, user asked to add exp-of-kernel + scalar viz + normalizer swap)
Section "The exponential's second job" added to the post. Findings: (1) exp-of-kernel
(softmax(kappa) not kappa/sum) HELPS: yat_b_exp 1.4968+-0.0054 TIES softmax (1.4923),
goat_nov_exp 1.5158 beats its L1 twin (1.5343); goat_v_exp worse+unstable (1.5559+-0.043).
So the exponential had TWO jobs: positivity (kernel replaces) + SHARPENING (L1 can't, exp-on-
kernel can); char-LM wants near-one-hot copy heads. Bundles kgl_blog-yatexp-{lr-a,lr-b,edges,
edges2,b-seeds,gv-seeds,gn-seeds}, LR-bracketed both ends (exp arms inherit softmax's LR
fragility: yat_b_exp diverges at 3e-3, optimum 3e-4 -> the normalizer sets LR appetite, not
the score). (2) SCALAR LANDSCAPE (yatattn-scalars.png, all 5 variants, per L x head): L1-trained
heads drive eps->0 (sharpen INSIDE kernel), exp-trained heads leave eps~log2 (sharpen at the
normalizer). (3) NORMALIZER SWAP (kgl_blog-swap-*, NORMSWAP=1, load weights + flip variant at
eval): SYMMETRIC COLLAPSE both directions (L1->softmax 1.75->3.18; softmax->L1 1.50->3.23), to
untrained loss. No free lunch: scores are married to their normalizer. TWO GOAT reverse-swap
cells never launched (network flakes at push, GPU cap) - the yat_b crossing both directions is
the complete story; GOAT would only confirm. OPEN if wanted: gve/gne swap cells; the (q.k)^2m
higher-power sharpening dial.

### B-geometry. the-geometry-of-attention  (DRAFT, 2026-07-19)
Title: "The Geometry of Attention Is a Choice of Kernel" (slug the-geometry-of-attention; retitled from "The Kernel Is the Choice of Gravity", user wants attention/geometry/kernel in the title). The
partition view of attention: a head is a function over ALL of query space, and the
top-1 weight carves that space into per-key territories whose SHAPE is fixed by the
score law, not by training. Derivations (exact, in-post): softmax/bilinear owner
boundaries are hyperplanes through the origin -> territories are convex cones
("wedges of sky"); q -> tq never changes the softmax winner (t = inverse
temperature, the scale gauge next to the BxC shift gauge); a key strictly inside
the others' convex hull NEVER wins any query (convexity one-liner, converse by
separating hyperplane); the Yat kernel closes curved pockets around keys
(weighted-Voronoi family, Aurenhammer 1987), owns-its-own-ground bound
kappa(k,k)=(||k||^2+b)^2/eps, and its far field -> (u.k)^2 (unsigned cones, but
the sign rides the 1/t term: antipodal symmetry only wins past radius ~20-100 on
the toy, stated). Experiment: reran yat_attention.py with GEOMETRY=1 (exports
trained q/k per head on the fixed window; bundles kgl_blog-attngeom-qk softmax@3e-4
+ yat_b@3e-3, kgl_blog-attngeom-goat goat_v@3e-2, seed 0; analysis
export_attention_geometry_viz.py, assets public/the-geometry-of-attention/).
Measured: scale test softmax changes 0 of 15,240 winners (theorem to the digit)
vs kernel 12.1%/15.1%/23.3% at t=0.5/2/4 (goat 16.7/24.2); census occupancy
softmax 27->72 distinct winners vs kernel 31->54 (per-head 7..120), the INVERSION
(sharper rows entropy 0.49 yet MORE distinct winners than the diffuse 0.72 kernel,
whose maps grow horizontal annexation runs); hull LP: 0 of 3,072 keys interior in
48-d (both models), so disenfranchisement is statistical not absolute in high-d;
winner-kind table: softmax winner = aligned 99.9% (definition), kernel splits
55.9% aligned / 51.8% nearest, both ~51% nearest vs ~3% chance (a fact about the
learned embedding). Five fresh panels (geomattn.js, live 2-D law math + real
exported data): GravityMap (draggable bodies, side-by-side territory maps, zoom),
RayRide (query rides a ray; softmax sharpens, kernel hands off, mass readout),
InteriorPlanet (hull theorem staged, 0.0% readout inside hull), TerritoryCensus
(real ownership scatter per layer/head, scrub 4 training checkpoints),
ScaleTheQuery (real q/k of sharpest/most-diffuse heads, live winner recompute vs
t, softmax strip provably dark). Spends by link: NW smoother, BxC kernel/mass/
entropy/shift-gauge, qk-projections, prototype wells; new-derived: territory
partition, cone-vs-pocket boundaries, scale gauge, hull disenfranchisement,
occupancy census, far-field unsigned cones. REBALANCED 2026-07-20 (user: the
post is about the GEOMETRY, not the experiments): the two experimental
sections collapsed into one "coda" (scale test + census + hull LP, panels
kept, aligned/nearest table cut), intro/description now promise the map not
the contest, and the geometry gained the output-side beat (row = barycentric
coordinates, outputs confined to the value hull, territory map = output map).
Companion
the-geometry-of-attention-jax-flax-nnx (2026-07-19, D): the GEOMETRY=1 export +
f16 reconstruction check (max err 4e-4), scale-test/census/hull-LP code quoted,
5 GIFs + 1 PNG from render_attention_geometry_gifs.py (territory morph, hull
crossing with the exact-0.0% counter, ray ride, real-q/k scale sweep with the
0-vs-25% curves, far-field zoom with the live antipodal-agreement counter;
census checkpoints = a PNG grid, four snapshots are four facts).

## 4. Concept ledger (what is already spent, do not re-explain)

| concept | established in |
| --- | --- |
| pointwise activations wreck manifold geometry (diagonal-Jacobian modulation) | activations-are-bad-for-geometry |
| **opposition is not difference**: max difference of unit vectors is orthogonality (cos 0), not antipodality (cos -1); antiparallel pairs collapse to a 1-D line and waste a dimension | opposite-is-not-different |
| cross-entropy singularity asymmetry (disjoint support ~ orthogonality; KL blows up at the boundary contrastive losses aim for); the overgeneration/hallucination lopsidedness; why InfoNCE needs huge batches (negatives concentrate near cos 0) | not-all-infinities-are-equal |
| contrastive-loss history/taxonomy (pair/triplet/InfoNCE/CLIP/SupCon/SigLIP/align+uniform/cos->0); "which losses know when to stop" | untangling-the-moons |
| the latent codebook: collapse; **regular simplex** as optimal centered code (pairwise cos -1/(C-1)); Welch bound + equiangular tight frame when concepts outnumber dimensions; neural collapse | welch-bound-good-latent-space |
| latent space = lossy finite-dim encoding of a label-similarity kernel; codebook = top eigenmodes, dark knowledge rides the modes below; structured codebook makes better mistakes; simplex is optimal only when classes are strangers | latent-on-the-spectrum |
| three states of information (random/organized/structured, matter analogy); loss plateaus = reorganization before it shows in the loss | three-states-of-information |
| separate-vs-represent tradeoff; the modality gap as a chosen readout, not a bug | modality-gap-complementary (D) |
| direction vs prototype; neuron as a picture | C1 |
| no-training hand-built classifier (place prototypes + one-hot) | C1, C2 |
| Yat kernel is positive-definite / Mercer; RKHS feature map; representer theorem; centers live in input space (so pictures) | C0, C1 |
| self-explaining prediction / convex attribution | C0, C1 |
| **OOD abstention, "knows when it doesn't know"** (bounded kernel) | C1 + companion |
| teach / forget / class-incremental / exact unlearning | C2 |
| attractor field, inverse-square wells, softening length, basins, superposition, order-independence | C2 |
| kernel mean embedding `mu_c`; score algebra | C2 |
| construct the head on learned features; construction-vs-optimization boundary | C3 |
| hand-built features (oriented edges + corner, HOG lineage); 343 named dimensions; the construction ladder | C4 |
| shared-operator / fixed-point network; deep-equilibrium; contraction = single basin in state-space; adaptive depth by residual halting; implicit differentiation; recursion unrolled + shared-vs-per-layer-alpha; depth motivation = propagation/extrapolation (train small, iterate more to solve bigger) via the maze reachability experiment | C5 (draft) |
| convex/conic/affine/linear readout taxonomy; non-identifiability of convex attributions; readout convex-by-construction via normalized kernel coefficients | readout-as-convex-combination |
| weight and data in one RKHS ("direction vs place" at function level); representer theorem as a build recipe; the optimal weight lives in the span of the data | where-does-a-weight-live |
| Mercer decomposition as a price list for roughness; eigenvalue decay = smoothness class; Sobolev vs Gaussian vs bandlimited; sphere harmonics; where the Yat/IMQ gate sits | what-can-a-weight-be |
| Yat FFN as an exact key-value memory in a real transformer: read / attribute / edit / abstain on a trained LM (Geva reading made exact) | mlp-block-is-a-representer-theorem |
| regularization as budget capping on the RKHS norm; effective dimension d_eff = Σ λₖ/(λₖ+λ); eigenvalue decay as the mechanism of generalization; support vector selection by budget; assembly order of representer sum by importance | regularization-is-a-price-list |
| Nadaraya-Watson reading of attention; the affordances of a kernel (mass conservation, basis, geometry) | attention-is-a-kernel |
| random features / FAVOR+; positive vs trig features; linear attention as a different kernel | cheap-attention-is-linear-attention |
| bilinear score; symmetric + antisymmetric split; rank budget; gauge freedom; RoPE as relative position | why-attention-needs-qk-projections |
| hand-built layer-2 vocabulary; redundancy of constructed ANDs; the combinatorial/selection wall (selection IS training) | C4.5 (draft) |
| reliability diagram / ECE; temperature scaling; softmax shift-invariance deletes field magnitude; selective prediction | C6 (draft) |
| distillation as kernel transfer; relational-loss isolation; spectrum + confusion inheritance; T as the kernel-on-the-wire knob | A-capstone (draft) |
| depth=time / skip=Euler dictionary; non-crossing obstruction and the exactness ceiling; Euler-vs-leapfrog energy drift; momentum-net invertibility and (1/mu)^L friction irreversibility | D1 |
| readout-vs-dynamics row split in equilibrium editing; certificate as load-bearing wall; edit evaporation; silenced-vs-erased | C5.5 (draft) |
| Cox partial likelihood + DeepSurv as a Yat vote over prototype patients; survival as a resemblance sum; Nadaraya-Watson survival curve from neighbors | C7 |
| exact closed-form cohort deletion on dense covariates (delta h = -a_u phi_u(x)); "exact and local" vs "provably unchanged"; kernel-max OOD on tabular clinical data | C7 |
| the O(n³) Gram solve is plumbing not the deal; solve-vs-descend equivalence of one kernel (r=0.95); the three walls a solve dies at (memory / scale / composition); random features (Rahimi-Recht) as the historical escape that kept the gradient step by dropping the kernel object | C9 |
| depth-telemetry as an instrument (residual-stream path length + per-sub-update turning angle through depth); pre-norm Transformer residual = forward Euler so D1's dictionary transfers; nGPT as first-order-on-a-sphere | D2 |
| conservation by construction (HNN: field = rotated gradient of a learned scalar, dH/dt = 0 identically, "a loss says please, the architecture says cannot"); the free-field drift diagnosis (errors need a direction to leak; level sets remove it) | D3 |
| symplectic/leapfrog residual block (kick-drift-kick of a learned potential, state (q,p), h=T/L); the h² shadow-Hamiltonian band measured in a trained net; depth-as-resolution extrapolation (fixed T; the integrator not weight-tying is what survives); learned energy as the missing row (undefined vs unmeasured); marble-on-terrain dictionary table | D3 |
| reversible backprop by block inversion (custom_vjp with endpoint-only residuals, recompute-vs-store); O(1) activation memory measured flat to depth 512; the (1/mu)^L noise-budget napkin L* = ln(1/eps)/ln(1/mu) locating gradient death; XLA memory_analysis + fresh-process peaks as the honest instruments; dissipation-destroys-information (friction breaks time-reversal); conservation = reversibility = constant-memory training, one property three coats | D4 |
| error-controlled inference rendering (step doubling on a trained flow, no retraining); depth as a per-input expenditure with a tolerance dial; the tol^(-1/3) cost law (integrator order in the compute bill); effort measures flow stiffness, NOT classification difficulty (measured negative, contrast with C5's convergence-based halting); train-resolution calibration (the fully-resolved flow is a slightly different function than the training render) | D5 |
| no-softmax attention (kappa/sum kappa as literal Nadaraya-Watson); the row-mass channel (softmax's shift gauge deletes level, the kernel keeps it); boundedness-for-bounded-inputs is not boundedness-under-training (kappa(q,q)=||q||^4/eps); exp-as-soft-argmax vs polynomial ratios (measured diffuseness gap, entropy 0.74 vs 0.48); kernelize BETWEEN the roles (directionality lives in f_Q != f_K) | BxC |
| attention as a partition of query space (per-key territories); softmax territories = convex cones through the origin, kernel territories = closed pockets (weighted Voronoi); the QUERY-SCALE gauge (q -> tq preserves softmax ranking, t = inverse temperature; measured 0 of 15,240 winner changes) vs length-as-address under the kernel (12-23% re-elections); hull disenfranchisement (interior key never wins under bilinear scores; every key owns its ground under the kernel via (||k||^2+b)^2/eps); occupancy census + the sharp-rows/many-winners vs diffuse-rows/few-winners inversion; kernel far field -> (u.k)^2 unsigned cones with 1/t sign correction | the-geometry-of-attention (D) |

Reuse these only by reference (link the prior post), never by re-derivation.

---

## 5. Open threads / candidate next posts

**Redundancy watch (2026-07-16 audit).** The ledger discipline is holding: shared concepts are mostly reused by link, not re-derived (C4 credits C3's ladder and the shared 83.2/85.7 numbers by reference; not-all-infinities uses orthogonality as an *analogy* for the KL singularity, a distinct argument). Two live items to fix rather than grow: (a) `welch-bound-good-latent-space` re-states the opposition-collapses-a-dimension / orthogonality-is-the-real-target claim that `opposite-is-not-different` owns, with no cross-link (a one-clause credit closes it; done 2026-07-16). (b) The negation/possessive title families are near saturation (see the title-pattern watch in §1). No true duplicate post exists. The root cause of Arc A re-derivation was that the concept ledger had no Arc A ownership rows; those are now added (see §4).

**Active build queue (opened 2026-07-16; CLOSED 2026-07-16 night: all four published).** Four posts committed, sequenced. Kaggle launches work from this workspace (auth = access_token; kgl.py in the skill dir). Status:
- **D3 ~done:** run v1 complete (bundle `kgl_blog-hamiltonian-v1`); explainer `a-network-that-conserves-energy` fully drafted and REWRITTEN to the style guide (question-driven, marble-on-terrain dictionary table, h² surprise promoted); 4 live panels (PendulumRace, LearnedEnergyMap, DepthLedger, FinerTime + `hamnet.js`), vizcheck 0/0; companion + NNX check (`hamiltonian_nnx_check.py`, ran green on Kaggle) + 4 GIFs rendered. Headline numbers: pendulum drift 9.6% plain vs 0.2% HNN; acc parity (spirals 98.8 vs 92.9±5.4, plain ahead, stated straight); 4L extrapolation spirals 68.0% plain vs 92.4% ham; energy band shrinks x16-22 when h shrinks x4 (the h² signature). KNOWN FIX IN FLIGHT: v1 exported a single-class viz sample (unshuffled Xte[:400]); v2 rerun (`blog-hamiltonian-v2`) with balanced sample is running; then re-export (`export_hamiltonian_viz.py` points at v2), re-render 2 GIFs, final screenshot pass, publish.
- **D4 validated:** `scripts/reversible_memory.py` smoke green (`kgl_blog-revmem-smoke2`): XLA temp memory 13.4->44.8 MB standard vs FLAT 3.2 MB reversible; grad cosine 1.000000 with rel err tracking (1/mu)^L; trains to identical acc. Full run queued behind v2 (GPU cap 2).
- **D5 scripted:** `scripts/adaptive_depth.py` (step-doubling adaptive renderer on the trained D3-style leapfrog net; fidelity/effort-histogram/difficulty-correlation/acc-vs-mean-steps). Smoke running on CPU kernel (`blog-adaptive-smoke2`).
- **BxC scripted:** `scripts/yat_attention.py` (softmax vs Yat-kernel score slot, no softmax needed since kappa>=0: weights = kappa/sum kappa; telemetry: bounded scores + kernel-mass AUROC vs correctness). Needs smoke then full (GPU).
1. **D3 - "A Network That Conserves Energy" (Hamiltonian nets, Greydanus 2019).** Hidden state (q,p); block = symplectic/leapfrog step of a learned scalar energy, so energy is conserved by construction. Existence proof: matches a plain ResNet on moons/rings/spirals while its learned energy stays flat through depth and the plain net's drifts, plus stability when depth is pushed past training (fixed T, h=T/L). Physics anchor: pendulum, HNN energy drift ~0 vs a plain MLP field. Experiment script written: `scripts/hamiltonian_net.py` (compiles; awaiting Kaggle run -> `public/hamiltonian-net/{physics,networks,results}.json`). Then explainer (4 fresh viz: pendulum energy-drift race, the (q,p) phase portrait through depth, energy-trace-through-depth plain-vs-ham, depth-extrapolation stability) + companion (GIFs: rollout energy holding vs leaking, phase-space area preserved, energy flat through depth).
2. **D4 - "Backprop Without the Memory" (reversible nets, O(1)-memory training).** D1's exact invertibility (rewind by dividing by mu) lets you recompute activations backward instead of storing them: peak memory flat in depth vs linear, at matched accuracy; reconstruction error IS D1's (1/mu)^L float-noise budget, now load-bearing.
3. **D5 - "Depth on Demand" (adaptive-step / learned computation time).** Ties C5's residual-halting to Arc D: an adaptive-step integrator spends more steps on hard inputs; steps-to-tolerance correlates with difficulty, accuracy holds while average depth drops.
4. **BxC - "Attention Is a Compatibility Kernel" (Yat attention).** `why-attention-needs-qk-projections` closes on exactly `s_ij = kappa(f_Q(x_i), f_K(x_j))`: kernelize the compatibility BETWEEN roles (asymmetry kept via f_Q != f_K). Yat kernel in the score slot: bounded scores, prototypes in token space, heads that abstain; SLAY (arXiv 2602.04915, Yat kernel + positive random features) makes it linear-time. Bridges Arc B and Arc C. Existence-proof framing (see the yat-program memory), not a benchmark contest.

Audited against the catalog (re-audited 2026-07-02); these are genuinely uncovered.

Near-term obligations (unblock existing posts):
1. **C5 companion** (`your-network-is-a-fixed-point` JAX + GIFs): no longer gates publish (C5 shipped 2026-07-08); still the top companion debt. GIF slate: residual collapsing to equilibrium; scattered starts contracting (~0.66/step, Banach); boundary freezing with the plateau at depth 6; settle-time map/histogram; **implicit differentiation** (the `custom_vjp` adjoint solved by the same iteration, the concept the post never shows); the 27x27 flood-fill climbing to 99.5% + the r=0.98 scatter.
2. **what-an-mlp-knows companion** (train the Yat trunk on arithmetic painting in Flax NNX): reproduces the paper's knockout/naming numbers inside the repo, re-drives KnockoutBars with real JSON, and yields 5-6 natural GIFs (footprints emerging, knockout bars from a real run, prototype drift, slot surgery).
3. **Finish modality-gap-complementary** (a live post links to it) and unpark simo2 when the SimO2/AFCL paper settles (then add `scripts/jax-simo2/` with the b-sweep bifurcation run).

New-post candidates, in rough order of appeal:
4. **Adversarial robustness: "a bounded neuron is hard to fool."** Locality means an attacker must push the input out of the true basin and into a wrong one, a large move, where a direction-neuron flips on an imperceptible nudge. Unifies with OOD (attack = climbing out of a basin; OOD = sitting between basins), ties straight back to the AttractorField. **Risk:** apparent robustness can be gradient masking. Must verify with FGSM + PGD on a Yat net vs a matched ReLU net (accuracy-vs-perturbation curves) before committing. Strongest candidate.
5. **Yat attention.** `why-attention-needs-qk-projections` closes on "the real kernel version of attention" and now anticipates it explicitly. Replace exp(q.k) with the Yat kernel in the score slot: bounded scores, prototypes in token space, abstaining heads; the SLAY paper (arXiv 2602.04915, Yat kernel + positive random features) is the payoff citation and makes it linear-time. Bridges Arc B and Arc C into one story.
6. **Few-shot / one-shot: learn a class from a single picture.** Place one prototype. Safe, definitely works, a striking extension of editability; pairs naturally with C3's "features make placement better." Good fallback if robustness does not hold.
7. ~~Depth by construction~~ LIVE as C4.5 (`depth-by-construction`, 2026-07-04).
8. ~~Calibration, deeply~~ LIVE as C6 (`calibration-of-a-bounded-net`, 2026-07-04). Its own open thread: does a magnitude-aware head (or T fitted inside training) fix the bent reliability curve, and does the story hold at K > 50?
9. ~~The fixed point that edits itself~~ DONE as draft C5.5 (`edit-a-fixed-point`, verified; publishes after C5). Its open thread: true erasure of trained knowledge from the dynamics = retraining by another name.
10. ~~Distillation as kernel transfer~~ LIVE as Arc A capstone (`distillation-is-kernel-transfer`, 2026-07-04).
11. **Arc D next beats** (from D1's close): Hamiltonian networks that conserve a learned energy (Greydanus); adaptive-step depth as learned computation time (ties to C5's residual halting); the memory-free backprop that exact invertibility buys.
12. ~~D2 candidate: "Transformers With a Velocity Ledger"~~ LIVE as D2 (`transformers-with-a-velocity-ledger`, 2026-07-09); related-work map moved into the D2 beat. The Momentum-nGPT synthesis hook was tested and answered: ngpt+ledger is the worst of the four on best-val (1.456), momentum on the sphere hurts. Remaining open thread from D2: does the shorter, straighter residual-stream path buy anything at scale (trainability at large L, pruning depth, early exit)?

---

## 6. Where things live

- Posts: `src/content/blog/<slug>.mdx`
- Series registry: `src/data/series.ts` (slugs in reading order + a reader-facing
  `description` per arc and `status: 'ongoing'` while an arc is open; it feeds the
  series-first home page, `/series/<id>` landing pages, series-aware prev/next,
  and `/map`, which also extracts the cross-reference graph from post bodies at
  build time, so plain `/blog/<slug>/` markdown links double as map edges)
- Engine viz components: `src/components/viz/*.astro` (+ shared helpers like `yatedit.js`, `yatedit.worker.js`)
- Viz engine: `src/components/viz/engine/` (`vizkit.js`, `jax.js`, `draw.js`, `nanograd.js`)
- Experiments + renderers: `scripts/*.py`
- Public assets (sprites, JSON, GIFs): `public/` and `public/<slug>/`
- Deploy: merge to `master` -> `.github/workflows/deploy.yml` -> `gh-pages`
