# Technical blog editorial plan

Date: 2026-08-24

Scope: the 78 current MDX files, treated as 45 editorial topics when an explainer and runnable companion form one unit.

## Consolidation completed — 2026-08-25

The repetition audit reduced the collection from 78 to 67 MDX files. The surviving canonical articles now own the full argument and the retired URLs emit redirect pages with canonical destinations:

- `opposite-is-not-different` → `untangling-the-moons`
- `what-an-mlp-knows` and `your-neuron-is-a-picture` → `what-a-finite-kernel-buys-an-mlp`
- `yat-mlp-fmnist-jax-flax-nnx` → `yat-mlp-jax-flax-nnx`
- `regularization-is-a-price-list` → `what-can-a-weight-be`
- `regularization-is-a-price-list-jax-flax-nnx` → `what-can-a-weight-be-jax-flax-nnx`
- `you-dont-have-to-train-the-features` → `train-the-features`
- `handbuilt-features-jax-flax-nnx` → `train-the-features-jax-flax-nnx`
- `a-risk-model-that-names-its-reasons` → `survival-model-on-trial`
- `a-risk-model-that-names-its-reasons-jax-flax-nnx` → `survival-model-on-trial-jax-flax-nnx`
- `fifteen-ideas` → `mercer-microscope`

The merge preserved the unique experiments: opposition versus orthogonality and simplex feasibility; Fashion-MNIST prototype initialization and arithmetic interventions; fixed-kernel regularization and effective dimension; the zero-gradient hand-built feature pipeline; exact survival-risk terms and model edits; and the CIFAR-100 concept-count, taxonomy, packing, and stability audit. Series order and internal links now point directly to the canonical pages.

## Implementation status

Implemented on 2026-08-24:

- rebuilt the contrastive “opposition” thesis around local derivatives, coupled Gram geometry, and valid claims about SigLIP and cross-entropy;
- replaced the invalid FFT/RKHS companion with positive periodic spectra constructed by design and convergence-audited partial norms;
- corrected fixed-point “certificate” language to local stability audits and initialization-sensitivity tests;
- separated exact continuous Hamiltonian conservation from bounded discrete symplectic energy error;
- corrected convex/additive attribution terminology and the Brownian-kernel RKHS boundary condition;
- scoped the representation-state, latent-spectrum, prototype, feature-construction, activation-rank, and random-feature claims to the objects actually derived or measured;
- adopted affirmative boundary language across companion posts and recorded the house voice in `docs/writing-style.md`;
- promoted Mercer Microscope, Spectral Surgery, Yat Protocol, and A Network Made of Parts from draft status;
- retained four drafts: One Hundred Classes, Fifteen Ideas; Sixteen Patches in Conversation; the modality-gap experiment; and SimO2. Their remaining evidence or theoretical gaps are now stated in frontmatter;
- passed a complete production build after the edits.

The decision tables below remain the rationale for the work. Where a temporary unpublish was recommended, the relevant post was instead repaired before publication state was finalized.

## Editorial objective

Keep the voice bold, curious, and concrete. Make every strong sentence earn its strength from one of three sources:

1. an exact identity or proof;
2. a measured result with the setup named;
3. an interpretation stated as an interpretation.

Honesty does not require defensive prose. The reader needs a clean boundary around a claim, not an apology for making it.

## Decision vocabulary

- **Keep:** technically sound; perform a normal clarity and evidence pass.
- **Revise:** the central story is sound, but scope, terminology, or support needs work.
- **Rewrite:** valuable material remains, but the thesis or argument must be rebuilt.
- **Temporarily unpublish:** a central public claim is currently false or unsupported. Preserve the file and URL, but do not present it as finished until corrected.
- **Hold as draft:** do not publish until a named theoretical or empirical gap is closed.
- **Retire or merge:** the topic duplicates a stronger article and has no independent job.

Nothing in the present corpus needs to be permanently deleted immediately. Quarantine is better than deletion: the experiments and visualizations can often support a more accurate article.

## Immediate publication decisions

### Temporarily unpublish and rebuild

1. **Opposite Is Not Different**
   - Reason: the claim that ordinary cross-entropy reaches a finite simplex stopping point is false for separable, unregularized cross-entropy. The claim that most negative gradients seek cosine `-1` is also inconsistent with several losses discussed in the series.
   - New thesis: **Different Does Not Mean Opposite: What Contrastive Losses Actually Ask Negatives to Do.**
   - Rebuild around the derivative of each loss with respect to similarity. Separate finite-margin losses, softmax-coupled losses, sigmoid losses, and uniformity objectives. Do not infer a global training equilibrium from the sign of a single partial derivative.
   - Exit criterion: every loss has its actual stationary or asymptotic behavior stated, and batch-size claims are tied to a derivation or direct experiment.

2. **What a Weight Can Be, in JAX/Flax NNX**
   - Reason: the FFT of a naively truncated inverse-multiquadric kernel produces negative coefficients, so the plotted values are not valid RKHS eigenvalues. Finite spectral truncation is also presented too strongly as a membership test.
   - New thesis: **Reading an RKHS Price List from a Valid Spectrum.**
   - Replace the kernel with a correctly periodized positive-definite kernel or use a domain/operator whose eigensystem is known. Show convergence as truncation increases. Treat finite partial norms as diagnostics, not proof of finite or infinite norm.
   - Exit criterion: nonnegative eigenvalues up to numerical tolerance, explicit domain and measure, a convergence plot, and an analytic benchmark.

3. **Your Network Is a Fixed Point** and its JAX companion
   - Reason: Jacobian norms at test equilibria are local measurements, not a global contraction certificate. They do not establish uniqueness or convergence from every initialization.
   - New thesis: **When a Shared Operator Settles to a Fixed Point.**
   - Preserve the fixed-point construction, implicit differentiation, maze experiment, and measured iteration counts. Replace global claims with measured basin/stability claims unless a true global Lipschitz bound is derived.
   - Exit criterion: either prove a global bound over a named domain, or consistently call the Jacobian results local stability measurements and map convergence across initializations.

4. **Edit One Operator, Edit Every Depth** and its JAX companion
   - Reason: the editing experiment is valuable, but it inherits the false contraction-certificate language.
   - New thesis: **How a Local Edit Propagates Through an Equilibrium.**
   - Report fixed-point displacement, class effects, solver convergence, and basin changes before and after editing. State exactly which guarantees survive recursion and which become measurements.
   - Exit criterion: no use of “certified” without a uniform bound or formal certificate.

### Urgent corrections while live

1. **Untangling the Moons** and **Organizing Randomness**
   - Rewrite the comparison table and conclusion using the taxonomy developed for the rebuilt contrastive post.
   - Correct the SigLIP discussion: it removes global softmax normalization, not necessarily pairwise computation, and its gradients do not stop at a universal equilibrium cosine.
   - Replace “which losses know when to stop” with “how each loss changes its pressure as similarity moves.”

2. **A Network That Conserves Energy** and companion
   - Separate three statements: the continuous Hamiltonian vector field conserves the learned Hamiltonian; leapfrog is symplectic; the numerical trajectory exhibits bounded energy error and approximately follows a shadow Hamiltonian.
   - Retitle to **A Network Built from Hamiltonian Steps** or **A Network with Bounded Energy Drift** unless exact discrete conservation is actually implemented.
   - Replace “no layer may change” and “held by the architecture” with the precise discrete property.

3. **A Risk Model That Names Its Reasons** companion
   - Change “exact convex attribution” to “exact additive attribution.”
   - Keep “reasons” as the narrative term, but define it operationally: signed prototype contributions to the model score, not causes of the clinical outcome.

4. **What Can a Weight Be?** explainer
   - Define the Brownian-kernel RKHS precisely: absolutely continuous functions with square-integrable derivative and the appropriate boundary condition, rather than all of `H^1`.
   - Rename the plotted cusp if “spike” invites readers to infer a Dirac impulse.
   - Distinguish a finite-resolution price estimate from true RKHS membership.

## Voice standard: rigorous without sounding defensive

### Use affirmative scope

Write the strongest sentence the evidence supports, then stop.

| Defensive construction | Direct, rigorous construction |
| --- | --- |
| “This does not prove the operator is globally contractive.” | “At the observed equilibria, the largest measured Jacobian norm is 0.92. Global contraction requires a uniform bound over the state space.” |
| “We are not claiming attention weights are causal explanations.” | “Attention weights expose routing arithmetic. Causal attribution requires an intervention.” |
| “This is only one small experiment.” | “This result covers three seeds on Fashion-MNIST with the architecture above.” |
| “The metaphor should not be taken too literally.” | “The correspondence is exact for the weighted sum; it ends when later nonlinear layers mix the result.” |
| “Of course, more work is needed.” | “The next unresolved question is whether the effect survives a larger model and a second dataset.” |
| “Surprisingly, the method failed, but that does not mean…” | “The method failed on all five datasets. The failure isolates calibration, not discrimination, as the unresolved part.” |

### Remove these habits

- “to be clear,” “we make no claim,” “of course,” and “it is worth noting”;
- paragraphs that begin with a disclaimer before stating the result;
- repeated caveats after the scope has already been defined;
- “proves,” “guarantees,” “certificate,” “exact,” and “by construction” when the property was measured rather than derived;
- “only,” “merely,” and “just” when they diminish a legitimate scoped result.

### Give every post a claim ledger

Place a compact box after the opening puzzle, written in normal prose rather than legal language:

> **What is exact:** the algebraic identity or architectural invariant.
> **What is measured:** dataset, model, seeds, and principal result.
> **What I think it means:** the interpretation the article will test.
> **What remains open:** the single boundary most likely to change the conclusion.

The box lets the rest of the article speak confidently without repeating qualifications.

### Structure every technical claim in this order

1. **Question:** create the tension.
2. **Picture:** give the reader an object they can imagine.
3. **Identity or experiment:** show how the question is answered.
4. **Result:** state the number or theorem directly.
5. **Boundary:** state once where the result ends.
6. **Consequence:** explain why it matters.

## Complete topic-by-topic plan

### Representation geometry and contrastive learning

| Topic | Decision | Required change |
| --- | --- | --- |
| What Activations Do to Geometry | Revise | Keep the Jacobian row-scaling argument. Replace any universal depth-compounding claim with conditions on gates, rank, and layer composition. Add one counterexample where an activation does not lose rank. |
| Opposite Is Not Different | Temporarily unpublish; rewrite | Rebuild around loss derivatives and attainable configurations. Remove the finite cross-entropy stopping claim and the batch-size causal story unless experimentally isolated. |
| Untangling the Moons / Organizing Randomness | Urgent rewrite | Correct SigLIP, margin, uniformity, and cosine-zero behavior. Make the playground report forces at the current state rather than declare a universal destination. |
| Welch Bound / Auditing Latent Geometry | Keep | Add a visible assumption card for normalized vectors, `n` versus `d`, and when simplex or ETF equality is attainable. |
| Latent on the Spectrum | Revise | Say that a label kernel designs a target codebook geometry; do not imply all learned latent spaces inherit it. Separate classical MDS identities from neural-collapse observations. |
| Three States of Information | Revise | Present the three states as an observed trajectory and diagnostic vocabulary. Replace phase-law language with measurable indicators and show at least one schedule that changes or skips the sequence. |
| Distillation Is a Geometry | Revise lightly | Keep the relational experiment. Describe “kernel transfer” as the tested mechanism in this setup and compare it explicitly with logits, labels, and a random relational target. |
| Modality Gap Is a Choice | Hold as draft | Add a second data-generating process and a failure case. Publish only when the article can distinguish a constructed possibility from an explanation of real multimodal models. |
| SimO2 | Hold; retire if crux remains unresolved | Resolve epsilon collapse mathematically or constrain it architecturally and rerun. If the construction cannot guarantee its advertised geometry, reuse the experiment in a negative-results post instead. |

### Attention

| Topic | Decision | Required change |
| --- | --- | --- |
| What Attention Weights Can Explain / JAX | Keep | Preserve the current routing-versus-causality distinction. Replace any remaining “explanation” shorthand with the exact object: routing weights, value mixture, or intervention. |
| What an MLP Knows, When It Is a Kernel | Revise or merge | Give it one independent job: a taxonomy of inspectable quantities. If it repeats the neuron/readout/kernel-memory posts, turn it into the series map and remove duplicate derivations. |
| Cheap Attention / JAX | Keep | Add the approximation assumptions, positivity mechanism, estimator variance, and actual memory accounting in one compact panel. |
| Why Attention Needs Q and K / JAX | Keep | Make “needs” architectural rather than absolute: separate Q/K projections buy directional role-specific bilinear scores. Include the shared-projection countermodel. |
| Kernel Between the Roles / JAX | Revise lightly | Keep the matched ablation. State that the performance result belongs to the Shakespeare setup and foreground the two assumptions the telemetry falsified. |
| Geometry of Attention / JAX | Keep | Keep exact geometric propositions distinct from trained-head measurements. Define the domain for every convex-hull or winner-region claim. |

### Weights, readouts, and RKHS structure

| Topic | Decision | Required change |
| --- | --- | --- |
| Readout as Convex Combination / JAX | Keep | Preserve the four-regime taxonomy. Audit every use of “convex” for nonnegativity and unit-sum conditions. |
| Where Does a Weight Live? / JAX | Keep | State the optimization assumptions of the representer theorem next to the theorem, once. Keep the geometric story confident afterward. |
| What Can a Weight Be? | Urgent correction | Fix the Brownian RKHS definition, finite-truncation interpretation, and “spike” terminology. |
| What a Weight Can Be, in JAX | Temporarily unpublish; rewrite | Replace the invalid periodic spectrum and add numerical convergence and analytic checks. |
| MLP Block Can Be a Kernel Memory / JAX | Revise lightly | Keep “can be.” Make trained centers and classical fixed-center representer solutions visibly different objects. |
| Regularization Is a Price List / JAX | Keep | Preserve the scoped generalization language. Add a short worked example connecting one regularizer to one spectral penalty. |

### Constructed and editable kernel networks

| Topic | Decision | Required change |
| --- | --- | --- |
| What a Finite Kernel Buys an MLP / JAX | Revise | Put the exact finite feature map and theorem assumptions before broad consequences such as attribution and capacity control. Link every claimed property to its actual condition. |
| Your Neuron Is a Picture / JAX | Revise | Change “every neuron becomes a picture” to the architecture-specific fact that centers live in input space. Show both readable learned centers and noisy/random centers so legibility is measured. |
| Edit a Network by Hand / JAX | Keep | Preserve the corrected sum-based arithmetic. Define the edit guarantee at the affected score/layer and distinguish it from machine unlearning or unchanged downstream behavior. |
| You Only Have to Train the Features / JAX | Revise | Lead with the surprising random-feature baseline. Scope the conclusion to the tested Fashion-MNIST construction and add seed variability. |
| You Do Not Even Have to Train the Features / JAX | Revise | Explain which handcrafted priors do the work. Add a dataset where those priors fail so the article identifies the boundary rather than universalizing the title. |
| How Far Down Can You Build? / JAX | Keep | Keep the negative result central. Tighten the causal diagnosis of why the second constructed layer fails. |
| When 80% Should Mean 80% / JAX | Keep | Preserve the falsified expectation and uncertainty reporting. Define the OOD channels operationally. |
| Risk Model That Names Its Reasons / JAX | Urgent correction | Rename convex attribution to additive attribution. Define “reason” as a model-score contribution and keep clinical/causal interpretation separate. |
| Fixed-Point Network / JAX | Temporarily unpublish; rewrite | Replace global contraction claims with local stability measurements or derive a true uniform bound. |
| Edit One Operator, Edit Every Depth / JAX | Temporarily unpublish; rewrite | Remove false certificate language and audit how edits alter equilibria, basins, and convergence. |
| White-Box Survival Model on Trial / JAX | Keep | Keep the five-dataset negative-result structure. Add confidence intervals and retain the distinction between benchmark performance and clinical evidence. |
| One Kernel Family, Fitted Two Ways / JAX | Keep | Preserve the current title. Make the differing hypothesis classes, optimization, and resource regimes the central comparison. |
| How Many Random Neurons Buy a Trained One? / JAX | Keep | State the measured Monte Carlo scaling as a fitted empirical exponent, then compare with the theoretical rate under named assumptions. |

### Current research drafts

| Topic | Decision | Required change |
| --- | --- | --- |
| Mercer Microscope | Revise, then publish | Call it an empirical spectral microscope. Specify which matrix is diagonalized, what its modes mean, and what Mercer theory does and does not identify. |
| Spectral Surgery | Publish after light revision | Preserve the failure: algebraic axis removal is exact while semantic removal fails. This is a strong example of rigorous, non-defensive negative reporting. |
| How to Interrogate a Kernel Network | Publish after light revision | Make this the methodological standard for the series. Keep measurements and interpretations in separate columns. |
| A Network Made of Parts | Publish after light revision | Keep the three failed predictions. Clarify that patch votes provide exact arithmetic decomposition, not causal attribution. |
| One Hundred Classes, Fifteen Ideas | Hold as draft | Add seeds or independent reruns and define the compression object precisely. Otherwise merge its strongest result into Mercer Microscope. |
| Sixteen Patches in Conversation | Hold as draft | Add baselines, seeds, and a second dataset. Publish when the architectural effect survives more than one run. |

### Depth, dynamics, and language models

| Topic | Decision | Required change |
| --- | --- | --- |
| Your Skip Connection Is Half of Newton / JAX | Keep | Mark Euler identities as exact and orbit/training consequences as tested predictions. Retain the finite-precision reversibility boundary. |
| Transformers With a Velocity Ledger / JAX | Revise lightly | Make the telemetry result primary. Keep broad transformer conclusions tied to the parameter-matched character model. |
| Network That Conserves Energy / JAX | Urgent rewrite | Correct continuous versus discrete conservation and retitle around Hamiltonian structure or bounded drift. |
| Backprop Without the Memory / JAX | Keep | Preserve the exact inverse algebra, measured memory tradeoff, and floating-point failure boundary. |
| Depth on Demand / JAX | Keep | Keep the controller derivation and honest probe accounting. State when step doubling is a reliable error proxy. |
| I Removed Every MLP from Gemma 4 12B | Keep | Preserve intervention, controls, and limitations. Add model/checkpoint/version provenance and keep causal claims limited to the ablation. |

## Merge and retirement decisions

1. **Do not merge explainer/companion pairs by default.** They serve different reading modes. Give each pair reciprocal links and one shared claim ledger so they cannot drift apart.
2. **Consider converting What an MLP Knows into a series map.** Retire it as a standalone argument if it cannot contribute a distinct taxonomy beyond the attention, prototype-neuron, readout, and kernel-memory posts.
3. **Merge One Hundred Classes, Fifteen Ideas into Mercer Microscope** if additional runs do not produce an independent conclusion.
4. **Retire SimO2 as a proposed construction** if epsilon collapse cannot be resolved. Its failure can become an honest post about why repulsive objectives evade geometric guarantees.
5. **Keep negative-result drafts.** Spectral Surgery, Yat Protocol, and A Network Made of Parts strengthen the corpus because the measurements are allowed to defeat the pitch.

## Execution order

### Phase 0: protect readers

- Temporarily unpublish the two mathematically invalid topics and the two fixed-point topic pairs.
- Correct the risk-model metadata.
- Add an editorial note to the contrastive survey if it remains accessible during revision.

### Phase 1: repair knowledge

1. Rebuild the contrastive-loss taxonomy and propagate it into all three affected posts.
2. Rebuild the RKHS spectrum experiment from a valid eigensystem.
3. Reframe the fixed-point experiments around local stability, or derive a genuine global certificate.
4. Correct continuous/discrete energy language and figures.
5. Correct the Brownian RKHS and attribution terminology.

### Phase 2: apply the voice system

- Add the four-line claim ledger to every article.
- Replace disclaimer openings with affirmative scope.
- Audit theorem-strength words globally.
- Let each limitation appear once, at the point where the claim reaches its boundary.
- End on the next scientific question, not a defensive limitations paragraph.

### Phase 3: strengthen empirical work

- Record model, dataset, split, seed count, selection rule, and uncertainty consistently.
- Add counterexamples to architecture-specific stories.
- Ensure every chart can be traced to a script and result artifact.
- Add a “what would change my mind?” test to interpretation-heavy posts.

### Phase 4: publish the strongest drafts

Suggested order:

1. How to Interrogate a Kernel Network
2. Spectral Surgery
3. A Network Made of Parts
4. Mercer Microscope

Keep the remaining drafts private until their named evidence gaps close.

## Quality gate before republishing

A topic returns to publication only when all answers are yes:

- Is every theorem stated with its assumptions and domain?
- Is every empirical conclusion tied to its actual model, data, seed count, and selection procedure?
- Can a reader tell identities, measurements, and interpretations apart without hunting for caveats?
- Do the title and description make no stronger claim than the body supports?
- Does each use of “exact,” “guaranteed,” “certificate,” “conserves,” “convex,” or “by construction” have a derivation immediately available?
- Does the runnable companion implement the same mathematical object described by the explainer?
- Is the article interesting because the result is sharp, rather than because its scope is hidden?
- Are limitations stated once, directly, and without apology?

## Definition of the house voice

The desired voice is **bold about the object, precise about the boundary, and curious about what remains**.

It says:

> Here is the mechanism. Here is what follows exactly. Here is what happened when I ran it. The result reaches this far. Beyond that edge is the next experiment.

That is rigorous writing. It is not defensive writing.
