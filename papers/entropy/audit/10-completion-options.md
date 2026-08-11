# Completion and reframing options

## What is established

- The finite-alphabet support criterion for infinite cross-entropy/KL.
- The coefficient (S(p,q)) for uniform mixture smoothing.
- Exact coordinatewise boundary identities.

## What remains open or unsupported

- A path-independent notion of “singularity severity.”
- Any theorem connecting embedding-distribution support to the batch-label cross-entropy used by CLIP/InfoNCE.
- A causal or optimization result linking cross-entropy to hallucination/overcoverage.
- A batch-size theorem derived from the proposed singularity.
- A defensible novelty comparison.

## Single most valuable missing theorem

A correct theorem in the actual contrastive variables. It would need to specify embeddings, logits, temperature, batch sampling, and objective, then characterize a genuine limiting regime and its gradients. The current intended bridge—orthogonal cosine similarity implies a zero softmax probability—is false, so this cannot be a patch to the existing argument.

## Path A — Minimal defensible revision

- **Required work:** Delete Sections 7.1–7.3 and the final synthesis; correct definitions, arrows, gradient wording, tables, and figures; define a path-relative rate; add standard references.
- **Resulting contribution:** A concise expository note on finite-alphabet support mismatch and uniform-smoothing asymptotics.
- **Strengths:** Mathematically clean and achievable quickly.
- **Risks:** Limited novelty; likely better suited to a technical note/blog than a research venue.
- **Claims to change:** Remove all causal ML claims and “undefined distance” rhetoric.
- **Possible title:** *Support Mismatch and Regularization Rates for Finite-Alphabet Cross-Entropy*.
- **Abstract emphasis:** Exact support criterion, path-specific coefficient, and limitations.

## Path B — Stronger complete paper

- **Required work:** Prove a general path theorem (q_x^{(\varepsilon)}\sim c_x\varepsilon^{a_x}); characterize first/second-order terms; analyze constrained and logit gradients; develop a correct contrastive-loss theorem; run controlled temperature/batch experiments; add a serious literature review.
- **Resulting contribution:** A mathematically substantive theory of boundary regularization and contrastive logit geometry.
- **Strengths:** Could connect the clean finite-alphabet result to real objectives without conflation.
- **Risks:** The current singularity mechanism may disappear entirely; this is a new paper-sized project.
- **Claims to change:** Replace orthogonality/support claims with statements proved in logit space.
- **Possible title:** *Boundary Regularization and Logit Geometry in Cross-Entropy Objectives*.
- **Abstract emphasis:** General smoothing exponents plus verified consequences for specified objectives.

## Path C — Alternative reframing

- **Required work:** Turn the false bridge into a negative result: show rigorously why embedding-support singularity does not explain finite-temperature InfoNCE; catalog object mismatches and give counterexamples; compare alternative explanations.
- **Resulting contribution:** A conceptual correction/negative-results note preventing a common category error.
- **Strengths:** The counterexample is crisp, relevant, and more novel than the current elementary taxonomy.
- **Risks:** Requires evidence that the misconception is present in the literature, not only in this draft.
- **Claims to change:** The applied claims become propositions ruling out the singularity explanation.
- **Possible title:** *Orthogonality Is Not a Softmax Singularity: A Note on Contrastive Loss Geometry*.
- **Abstract emphasis:** Finite-logit positivity, bounded logit gradients, and the distinction between embedding measures and batch-label probabilities.
