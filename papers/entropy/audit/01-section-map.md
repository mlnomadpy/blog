# Section map and structural review

## Abstract and Introduction (lines 71–109)

- **Purpose:** State the finite-alphabet singularity results and motivate an applied ML interpretation.
- **Inputs:** Standard cross-entropy/KL facts.
- **Outputs:** Claims of a three-tier classification and a geometry connecting support singularity to orthogonality.
- **Concern:** The abstract mixes verified finite-alphabet results with unsupported geometry. “Undefined,” “distance,” and “independence” are used incorrectly. Novelty is not situated.
- **Recommendation:** Lead with the exact smoothing theorem and clearly label all applied material as conjectural—or remove it.

## 2. Preliminaries (lines 113–146)

- **Purpose:** Define cross-entropy, KL, support, violating set, and mass.
- **Concern:** The sign in the (p\log 0) convention is wrong. The logarithm base is unspecified.
- **Recommendation:** State extended-real conventions precisely and fix a log base.

## 3. Taxonomy of Singularities (lines 150–282)

- **Purpose:** Prove the support criterion and uniform-smoothing asymptotic.
- **Outputs:** THM-001 and THM-002, the strongest valid content.
- **Concern:** “Rate tier” is undefined without a path/normalization. The theorem is right, but Figure 1 and one table row are impossible under the stated setup. The theorem's proof should explicitly invoke a Taylor expansion on the finitely many coordinates with (p_x>0,q_x>0).
- **Recommendation:** Make this the mathematical center. Define the path-relative rate and replace the figure with curves generated from explicit distributions.

## 4. Asymmetry (lines 286–387)

- **Purpose:** Record two coordinatewise boundary identities.
- **Output:** PROP-001, verified.
- **Concern:** The prose turns a summand identity into “adding falsehood is free” and “no constraint on precision,” contradicting the simplex constraint and strict propriety.
- **Recommendation:** End at the exact coordinatewise statement, then separately analyze feasible simplex/logit perturbations.

## 5. KL and Geometry (lines 391–540)

- **Purpose:** Transfer the support criterion to KL and introduce the orthogonality analogy.
- **Outputs:** THM-003; PROP-002.
- **Concern:** PROP-002 confuses ambient q-coordinate derivatives with constrained or logit gradients. KL is called undefined/a distance; disjointness is called independence. Figure 4 draws full-support distributions while claiming disjoint support.
- **Recommendation:** Separate extended-real KL, simplex differential geometry, and the narrow algebraic fact (p\cdot q=0\iff\operatorname{supp}p\cap\operatorname{supp}q=\varnothing) for nonnegative vectors.

## 6. Finite-at-boundary alternatives (lines 544–572)

- **Purpose:** Brief survey.
- **Concern:** JSD is not differentiable on every boundary face; general MMD/Wasserstein claims need kernel/metric/moment hypotheses.
- **Recommendation:** Keep the table explicitly finite-alphabet, add citations for JSD/MMD/TV, and distinguish value finiteness from boundary differentiability.

## 7. Discussion (lines 576–706)

- **Purpose:** Apply the singularity lens to modality gap, hallucination, and batch size.
- **Concern:** This section contains both Critical findings. It changes mathematical objects without a linking theorem, and its InfoNCE calculation does not imply the stated conclusion. Calling the material “hypothesis” does not cure direct contradictions.
- **Recommendation:** Remove the section in a minimal revision. In a stronger revision, replace it with a formally specified contrastive objective and prove statements in logit/embedding variables.

## Proposed revised outline

1. **Introduction:** finite-alphabet support mismatch and path-relative regularization severity; modest novelty claim.
2. **Definitions and conventions:** extended-real cross-entropy/KL, absolute continuity, fixed log base.
3. **Support criterion:** THM-001 with corrected arrows.
4. **General smoothing paths:** prove a theorem for (q_x^{(\varepsilon)}\sim c_x\varepsilon^{a_x}); obtain coefficient (sum_{x\in V}a_xp_x).
5. **Uniform mixture corollary:** THM-002 as the (a_x=1,c_x=1/n) case, followed immediately by a correct figure.
6. **Coordinatewise versus constrained gradients:** PROP-001, ambient gradient, tangent gradient, and softmax gradient (q-p).
7. **Boundary-finite alternatives:** corrected, cited table.
8. **Scope and limitations:** explicitly state that embedding-distribution support and batch-label softmax are different objects.

The applied ML discussion should return only after a valid bridge theorem or targeted experiment exists.
