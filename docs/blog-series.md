# Blog series bible

Working reference for the blog at `src/content/blog/`. Records what exists, the
conventions we write to, the story arc of the active (Yat-kernel) series, a
ledger of which concepts are already "spent" so we stop re-explaining them, and
the open threads for future posts.

_Last updated: 2026-07-08 (six new JAX companions published; systemic GIF audit: 38 disguised-chart GIFs converted to static figures, 79 real-process GIFs remain; every live post except what-an-mlp-knows now has a live companion). `D` = draft._

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

## 2. The catalog (45 posts, five arcs)

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

### Arc D: networks as integrators (new, opened 2026-07-04)
The standing move: numerical analysis as an architecture catalog. Each
structure-preserving integrator (symplectic, reversible, adaptive-step,
energy-conserving) is a candidate network, and its conservation law is a
testable prediction about trained hidden states.
| status | explainer | companion |
| --- | --- | --- |
| live | skip-connections-are-half-of-newton | momentum-resnet-jax-flax-nnx |

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
on everyone else, framed honestly as "exact and local", not "provably unchanged");
OOD abstention via kernel-max (7.1 real vs 3.7 stranger). Carries a "research
illustration, not a clinical tool" note. Six fresh viz (PatientPrototypes,
WhoDoYouLookLike, RiskLandscape, SurvivalCurveFromNeighbors, ForgetACohort,
AbstainOnStrangers) + six GIFs. Spends nothing new-derived: references prototype-as-picture
(C1), attractor field (C2), convex attribution (C0/C1), teach/forget (C2), OOD
abstention (C1), calibration (C6), RKHS/representer (C0) by link.

### D1. skip-connections-are-half-of-newton  (LIVE, 2026-07-04; opens Arc D)
A skip connection x + f(x) is forward Euler: depth=time, hidden state=position,
residual branch=vector field; Euler is HALF of Newton, there is no velocity anywhere.
Physics anchor (`scripts/momentum_resnet.py`, pure-math part): Kepler orbit, dt=0.02,
Euler gains **+68.3%** energy over 20 orbits (apoapsis 1.0 -> 3.37) while leapfrog
holds **0.016%**. Network result: no cliff exists in the data; the honest finding is
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

Reuse these only by reference (link the prior post), never by re-derivation.

---

## 5. Open threads / candidate next posts

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
