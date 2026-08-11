# Referee Report — *Beyond Binary Finiteness: The Geometry of Cross-Entropy Divergence*

**Auditor tier:** Deep · **Date:** 2026-08-11 · **Source:** `papers/entropy/main.tex`

## 1. Paper summary

The paper studies infinite cross-entropy for probability vectors on a finite alphabet. It defines a violating support set and mass, proves the standard support criterion for infinite cross-entropy/KL, derives the leading divergence coefficient under uniform mixture smoothing, and records a coordinatewise boundary asymmetry. It then argues that the same singularity provides a unifying explanation for modality gaps, hallucination, and large batch requirements in contrastive learning.

## 2. Main contribution as actually supported

The strongest supported result is Theorem 6: for fixed finite-alphabet probability vectors and uniform mixture smoothing,

\[
H(p,q^{(\varepsilon)})=S(p,q)(-\log\varepsilon)+S(p,q)\log n+C_{p,q}+O(\varepsilon).
\]

The support criterion, coordinate identities, and KL corollary are also correct after small convention/wording repairs. This yields a coherent short note about how a particular regularization path resolves support mismatch. It does not yield the paper's claimed explanations of contemporary ML behavior.

## 3. Overall mathematical assessment

**Headline verdict: the finite-alphabet core is verified, but the main applied bridge is contradicted by explicit counterexamples.** The paper changes mathematical objects when moving from Sections 2–5 to Section 7: the distributions of image/text embeddings are not the target and predicted class distributions in CLIP's batch softmax. With finite cosine logits and positive temperature, every softmax probability is positive; exact orthogonality therefore has finite cross-entropy. The claimed singularity does not occur there. The hallucination inference likewise does not follow from a coordinate summand, and the softmax gradient (q-p) directly penalizes false coordinates. The batch-size argument assumes uniform weights from one near-zero similarity and plots an unnormalized exponential rather than the stated weight. Because these errors invalidate the advertised unifying contribution, the paper is not ready for submission in its current form.

## 4. Major mathematical concerns

### Critical — the modality-gap bridge changes objects and is false (ISS-001)

At lines 590–596, Theorem 4 is applied first to continuous embedding distributions and then to the CLIP loss without any identification theorem. A two-pair counterexample suffices: take orthonormal unit vectors, make matched embeddings identical and mismatched embeddings orthogonal. Each row has logits ((1/\tau,0)), so the loss is (\log(1+e^{-1/\tau})<\infty) for every finite positive temperature. Thus orthogonal separation is reachable with finite loss. **Verdict: Contradicted by a counterexample · High confidence.**

### Critical — the InfoNCE batch-size mechanism does not follow (ISS-003)

At lines 637–705, orthogonality is again called the singular boundary. But (w_k\approx1/N) follows only when all relevant logits are comparable, not from (\operatorname{sim}_k\approx0) alone. With one positive similarity 1, 32,767 orthogonal negatives, and (\tau=.07), each negative weight is (6.12\times10^{-7}), whereas (1/N=3.05\times10^{-5}). If all logits really are equal, total negative mass is order one, so “each is (O(1/N))” still does not establish that a large N is required. **Verdict: Contradicted by a counterexample · High confidence.**

### Major — coordinatewise zero cost does not imply overcoverage/hallucination bias (ISS-002)

At lines 611–634, a local summand identity is promoted to a global optimization bias. On the simplex, increasing a false coordinate displaces other mass. Under softmax logits, (\partial H/\partial z_i=q_i-p_i); if (p_i=0<q_i), gradient descent reduces that coordinate. The introduction's own statement that (q=p) is the unique minimizer also conflicts with “no constraint on precision.” **Verdict: Unsupported, with the stated gradient-direction claim Likely false · High confidence.**

### Major — Proposition 9 confuses ambient, constrained, and logit gradients (ISS-004)

The formula (-p_i/q_i) is the ambient derivative in direct q-coordinates. It is not intrinsically “the gradient under the simplex constraint”; tangent gradients depend on a chart/metric, and softmax-logit gradients are bounded. The theorem survives only after narrowing its statement. **Verdict: Verified under additional assumptions · High confidence.**

### Major — the figures contradict the mathematics (ISS-005, ISS-008)

Figure 1 cannot depict the stated smoothing family: at (\varepsilon=1), every curve must equal (\log n), yet the curves have different endpoints; the S=0 curve is not generally constant. Figure 4 draws p and q in the simplex interior, where both have full support, while labeling them disjoint and KL-infinite. **Verdict: Contradicted by exact identities · High confidence.**

### Major — central terminology is mathematically wrong (ISS-007)

The paper defines (D_{KL}=+\infty) and then calls it “undefined”; calls KL a distance metric despite asymmetry and failure of the triangle inequality; and calls disjoint support “independence” rather than mutual singularity. These are not cosmetic because the title, abstract, and geometry depend on them. **Verdict: Contradicted by the paper's own definitions · High confidence.**

### Major — novelty is not established (ISS-014)

No citation is supplied for the standard absolute-continuity criterion, the taxonomy, or the smoothing asymptotic. The theorem is elementary, and the paper does not compare it to prior information-theory treatments. **Verdict: Unsupported · High confidence; exact priority Unable to verify · Medium confidence.**

## 5. Theorem-level findings

| ID | Number | Type | Statement | Assumptions | Dependencies | Used by | Proof status | Main issue | Severity | Confidence |
|---|---|---|---|---|---|---|---|---|---|---|
| THM-001 | Theorem 4 | theorem | H is infinite iff V is nonempty | finite X; extended-real convention | DEF-001, DEF-003 | THM-003, applied discussion | Verified | Proof arrows reversed; definition sign typo | Minor | High |
| THM-002 | Theorem 6 | theorem | Uniform-smoothing asymptotic with coefficient S | fixed p,q; finite X; fixed log base; uniform path | DEF-001, DEF-003, DEF-004 | abstract, Figure 1 | Verified under additional assumptions | Taylor remainder underexplained; figure invalid | Major (presentation/evidence) | High |
| PROP-001 | Proposition 7 | proposition | Two coordinatewise boundary identities | extended-real convention | DEF-001 | hallucination discussion | Verified | Global interpretation does not follow | Major downstream | High |
| THM-003 | Theorem 8 | theorem | KL is infinite iff V is nonempty | finite X; extended-real KL | THM-001, DEF-002 | geometry | Verified under additional assumptions | “Undefined” conflicts with +infinity | Major framing | High |
| PROP-002 | Proposition 9 | proposition | q-coordinate derivative diverges | direct ambient q-coordinates | DEF-002 | Figure 3 | Verified under additional assumptions | Not a simplex/logit gradient as stated | Major | High |

The five discussion remarks are also registered. REM-001 and REM-004 are **Contradicted by a counterexample · High confidence**; REM-002 and REM-005 are **Unsupported · High confidence**; REM-003 is **Likely false · High confidence**.

## 6. Structural findings

The paper is visually polished and the formal core is easy to follow, but organization hides the categorical break between theorem and application. Section 3 should be the center, with a precise path-dependent rate definition and a generalized smoothing theorem. Section 4 should distinguish coordinate summands from feasible simplex/logit directions. Section 5 should separate the narrow nonnegative-vector orthogonality identity from metric and independence language. Section 7 should be removed in the minimal revision; its disclaimers do not cure claims contradicted by the actual objective. The alternatives table needs boundary differentiability and domain hypotheses corrected.

## 7. Novelty and literature positioning

The cited CLIP, SimCLR, CPC, modality-gap, GAN, and optimal-transport sources exist and support the limited contextual statements for which they are cited. They do not support the proposed singularity mechanism. Directly relevant omitted work studies contrastive gradient reduction, mini-batch versus asymptotic contrastive objectives, and temperature-dependent modality-gap geometry. A July 2026 preprint on InfoNCE and modality gap postdates this draft but should be included in a current revision. The finite-alphabet support condition is standard; the priority and significance of isolating S as the uniform-smoothing coefficient remain unestablished.

## 8. Missing work

The single most valuable missing result is a theorem stated in the actual contrastive variables: embeddings, finite-temperature logits, batch sampling, and objective. It must identify a genuine limiting regime and derive its gradients. The current intended lemma—orthogonality implies zero probability—is false. For the finite-alphabet contribution, a general path theorem (q_x^{(\varepsilon)}\sim c_x\varepsilon^{a_x}) would be valuable and would show that the coefficient is (\sum_{x\in V}a_xp_x), making the uniform result a corollary.

## 9. Reframing options

- **Path A — minimal defensible revision:** Remove the applied discussion; correct definitions, gradient wording, tables, and figures; present a modest finite-alphabet expository note.
- **Path B — stronger complete paper:** Generalize smoothing paths and build a correct theorem/experiment suite in contrastive logit space.
- **Path C — alternative reframing:** Turn the exact orthogonal-softmax counterexample into a negative-results note titled along the lines of *Orthogonality Is Not a Softmax Singularity*.

## 10. Prioritized revision checklist

**Blocking**

- Remove or replace the modality-gap inference at lines 590–596.
- Remove or rederive the singularity-driven InfoNCE batch-size mechanism at lines 637–705.
- Remove the hallucination/overcoverage causal claim unless a model-specific constrained result is proved.
- Decide whether the paper is an expository finite-alphabet note, a new contrastive-theory paper, or a negative-results correction.

**High**

- Rewrite Proposition 9 to distinguish ambient q, simplex tangent, and softmax-logit gradients.
- Replace Figures 1, 4, and 5 with mathematically valid plots/diagrams.
- Replace “undefined,” “distance metric,” and “independence” with correct extended-real/divergence/mutual-singularity terminology.
- Add a novelty-focused literature review.

**Medium**

- Define the rate tier as path-relative and specify its normalization.
- Correct the impossible (|V|=n) disjoint-support table row.
- Correct JSD boundary differentiability and state MMD/Wasserstein assumptions.
- State the log base and dependence of the (O(\varepsilon)) constant.

**Low**

- Correct the sign in Definition 1.
- Swap the implication labels in Theorem 4's proof.
- Cite standard absolute-continuity terminology and distinguish cross-entropy from a divergence/metric.

## 11. Final scores

| Dimension | Score | Justification |
|---|:---:|---|
| Correctness | 4/10 | The finite-alphabet results survive, but two central applied mechanisms are explicitly false. |
| Proof completeness | 7/10 | Formal proofs are short and mostly complete; the main failures are unjustified theorem transfers rather than missing algebra. |
| Definition quality | 5/10 | V and S are clear, but the rate tier, boundary convention, and “undefined” terminology need repair. |
| Assumption quality | 4/10 | Core finite assumptions are visible; applied domain, parameterization, and smoothing-path assumptions are missing. |
| Internal consistency | 3/10 | Unique minimization at q=p conflicts with “no constraint on precision”; figures contradict their captions/theorem. |
| Novelty | 3/10 | The core is elementary and no prior-art comparison establishes novelty. |
| Significance | 4/10 | A clean smoothing coefficient is useful, but the high-impact ML interpretation does not survive. |
| Exposition | 6/10 | The prose and layout are polished, though rhetoric repeatedly outruns the mathematics. |
| Reproducibility | 5/10 | LaTeX is self-contained, but figures are hand-coded without explicit underlying distributions/parameters and no experiments support the applied claims. |
| Publication readiness | 2/10 | Critical issues require reframing, not local edits. |

## 12. Recommendation

**Not yet ready for submission**

The issue register contains two unrepaired Critical errors that invalidate the central applied synthesis, plus several Major errors in gradients, figures, terminology, and novelty positioning. The valid finite-alphabet core can be preserved, but the current title/abstract/conclusion promise substantially more than the paper proves.

---

**Verification coverage:** by hand: every definition, theorem, proposition, remark, proof, claim, dependency, and figure claim · symbolically: smoothing Taylor term, softmax gradients, JSD boundary derivative · numerically: 2,100 smoothing probes, orthogonal-InfoNCE and weight examples, figure endpoint examples · solver: not used · formally: exact hand formalization of the two-pair counterexample; no proof assistant used · not independently verified: exhaustive novelty priority, empirical hallucination causality, and full experimental modality-gap/batch-size behavior.
