# Paper summary

**Audit tier:** Deep
**Source:** `papers/entropy/main.tex` and the nine-page compiled `main.pdf`
**Overall verdict:** **Not yet ready for submission**

## Research question

The paper asks how the infinite boundary value of finite-alphabet cross-entropy can be refined beyond a finite/infinite dichotomy, and whether that refinement explains modality gaps, hallucination, and large-batch behavior in modern contrastive learning.

## Claimed contribution

It introduces the violating set (V_{p,q}=operatorname{supp}(p)\setminus\operatorname{supp}(q)), its mass (S(p,q)), a three-tier singularity taxonomy, and an asymptotic formula under uniform mixture smoothing. It then proposes a geometric analogy with orthogonality and a unifying interpretation of several ML phenomena.

## Strongest contribution actually supported

For fixed probability vectors on a finite alphabet, the manuscript correctly proves:

1. (H(p,q)=+\infty) iff (p\not\ll q), equivalently (V_{p,q}\ne\varnothing).
2. Under (q^{(\varepsilon)}=(1-\varepsilon)q+\varepsilon u),

   \[
   H(p,q^{(\varepsilon)})=S(p,q)(-\log\varepsilon)+S(p,q)\log n+C_{p,q}+O(\varepsilon).
   \]
3. The two coordinatewise boundary identities in Proposition 7.
4. The corresponding extended-real KL criterion.

These are elementary but valid after local convention/assumption repairs. The paper does **not** establish the modality-gap, hallucination, or InfoNCE batch-size explanations.

## Headline assessment

The finite-alphabet core is mostly correct, with one genuinely useful smoothing coefficient. The paper's broader contribution fails at the bridge from that core to modern losses. In CLIP/InfoNCE, the (p,q) in cross-entropy are batch-label distributions produced from finite logits; they are not the distributions of image and text embeddings. Finite softmax logits give strictly positive probabilities, so exact cosine orthogonality has finite loss. This directly contradicts the claimed singularity at the configuration contrastive learning seeks. Two further applied claims—overcoverage/hallucination bias and singularity-driven batch-size hunger—also do not follow and are contradicted by the softmax gradient or explicit logit calculations. The most defensible revision is an expository finite-alphabet note unless the applied bridge is rebuilt from a correct theorem.

## Document integrity

- The LaTeX source is complete, self-contained, and compiles after three passes with no remaining warnings.
- All references and citations resolve; the structure extractor found 14 theorem-like objects and no undefined references.
- The PDF is legible and contains all nine pages.
- Figure 4 is mathematically inconsistent with its caption: points drawn in the simplex interior cannot have disjoint support.
- Figure 5 is visibly clipped and plots an unnormalized exponential, not the stated InfoNCE weight.
- Figure 1 is visually clean but cannot arise from the theorem's smoothing family because its curves do not meet at (\varepsilon=1).
