# Mathematical audit of the blog post

Audited file: `src/content/blog/not-all-infinities-are-equal.mdx`

Date: 2026-08-11

Tier: deep targeted audit. The finite-alphabet calculations, their interpretation, the
contrastive-learning transfer, and the three interactive figures were checked directly.
Literature checks used primary papers. No empirical hallucination experiment was run.

## Overall verdict

**Not yet ready / withdraw pending rewrite · High confidence.**

The finite-alphabet support criterion and the uniform-mixture asymptotic are correct. The
headline causal thesis is unsupported, and the contrastive-learning mechanism is contradicted
by the exact InfoNCE/softmax formula. The post also contains an internal contradiction: it first
states the correct support-inclusion criterion and later says finiteness requires the interior of
the simplex.

## Claim ledger

| Location | Claim | Verdict | Confidence | Reason |
|---|---|---|---|---|
| Lines 67--81 | On a finite alphabet, cross-entropy is infinite iff some coordinate has \(p_i>0,q_i=0\). | **Verified** | High | This is exactly the support-inclusion/absolute-continuity criterion. |
| Lines 83--89 | Under uniform smoothing, the coefficient of \(-\log\varepsilon\) is the violating mass \(S(p,q)\). | **Verified** | High | Direct coordinatewise expansion. The conclusion is specific to this path. |
| Lines 89--93 | Violating mass intrinsically grades one infinity as “more wrong” than another. | **Verified under additional assumptions** | High | The comparison is valid only when the same uniform-smoothing parameterization is used. Anisotropic paths produce coefficient \(\sum_i a_ip_i\), so \(S\) is not an intrinsic rate of the boundary pair alone. |
| Lines 95--110 | A target-zero coordinate contributes zero directly to \(H(p,q)\). | **Verified** | High | The summand \(-p_i\log q_i\) is zero when \(p_i=0\). |
| Lines 112--116 | This coordinatewise fact creates an unambiguous overcoverage bias that explains hallucination. | **Unsupported** | High | On the simplex, log loss is strictly proper and uniquely minimized by \(q=p\). For softmax logits, the gradient on a false coordinate is \(q_i>0\), which penalizes it. No theorem or experiment connects the coordinatewise decomposition to hallucination. |
| Lines 123--126 | \(D_{\rm KL}(p\|q)=H(p,q)-H(p)\) has the same support singularity. | **Verified** | High | Shannon entropy is finite on a finite alphabet. |
| Line 128 | Disjoint-support distributions are the probabilistic analogue of orthogonal vectors. | **Plausible but not fully verified** | Medium | Nonnegative vectors with zero inner product have disjoint support, but this analogy does not transport KL singularities to cosine similarities between embeddings. |
| Line 132 | Cross-entropy and KL stay finite only for distributions in the simplex interior. | **Contradicted by a counterexample** | High | Let \(p=q=(1,0)\). Both are boundary points, yet \(H(p,q)=D_{\rm KL}(p\|q)=0\). The correct condition is \(\operatorname{supp}(p)\subseteq\operatorname{supp}(q)\). |
| Line 134 | InfoNCE, SimCLR, CLIP, and SupCon aim for orthogonal negatives and reach for a probability-support boundary where their loss diverges. | **Contradicted by a counterexample** | High | Their rowwise softmax is over pair indices and has full support for all finite similarities and positive temperature. Cosine orthogonality \(s=0\) is not categorical probability zero. |
| Lines 138--140 | The per-anchor contrastive loss diverges and its gradient blows up at orthogonal negative embeddings. | **Contradicted by a counterexample** | High | If \(s_{ii}=1\) and every negative has \(s_{ij}=0\), then \(\ell_i=\log(1+(N-1)e^{-1/\tau})<\infty\), and \(\partial\ell_i/\partial s_{ij}=Q_{ij}/\tau\in(0,1/\tau)\). |
| Lines 140--142 | A support singularity explains the CLIP/SigLIP modality gap. | **Unsupported** | High | The proposed singularity is absent. Existing work instead studies initialization cones, temperature, mismatched pairs, and gradient-flow dynamics. |
| Line 144 | Near-orthogonal negatives lie in a singularity-induced flat region, forcing enormous batches. | **Unsupported** | High | A negative's softmax weight can be small, but no support singularity occurs. Literature analyzes coverage/collision trade-offs, negative sampling, batch estimators, and hyperspherical uniformity; the post supplies no derivation isolating its mechanism. |
| Lines 148--152 | The three-item “proven core” includes contrastive objectives reaching the disjoint-support boundary. | **Contradicted by a counterexample** | High | Item 3 repeats the invalid identification of embedding orthogonality with zero pair-label probability. |
| Line 154 | Temperature scaling slows the gradient “near the singularity.” | **Unsupported** | High | Gradients contain a \(1/\tau\) factor while probabilities also depend on \(\tau\); monotonic slowing is not established, and the alleged orthogonality singularity does not exist. |
| Lines 154--156 | SigLIP's bias avoids the cross-entropy support boundary, validating the proposed mechanism. | **Unsupported** | High | SigLIP removes global softmax normalization and works at small batches, but this does not identify a support singularity as the reason. |
| Line 158 | Every softmax-cross-entropy training run points toward the singularity. | **Likely false** | High | With full-support targets, including label smoothing, the optimum is an interior softmax distribution. Even for one-hot labels, parameter-space dynamics are not a universal gradient flow toward every support boundary. |

## Exact counterexample to the load-bearing contrastive claim

For a row with positive similarity \(s_+\), negatives \(s_1,\ldots,s_{N-1}\), and
temperature \(\tau>0\),

\[
  \ell
  =-\log\frac{e^{s_+/\tau}}
    {e^{s_+/\tau}+\sum_j e^{s_j/\tau}}
  =\log\left(1+\sum_j e^{(s_j-s_+)/\tau}\right).
\]

At a perfectly aligned positive and orthogonal negatives, \(s_+=1\) and \(s_j=0\),

\[
  \ell=\log\left(1+(N-1)e^{-1/\tau}\right)<\infty.
\]

Moreover,

\[
  \frac{\partial\ell}{\partial s_j}=\frac{Q_j}{\tau},
  \qquad
  \frac{\partial\ell}{\partial s_+}=\frac{Q_+-1}{\tau},
\]

so every similarity derivative has magnitude below \(1/\tau\). With cosine similarities in
\([-1,1]\), the entire row loss is bounded at fixed positive temperature. Therefore the claimed
orthogonality singularity and gradient blow-up do not exist.

## Figure audit

1. **GrowthRates:** **Incomplete · High confidence.** The component plots the leading
   asymptotic expression with an arbitrary constant \(C=0.5\) as if it were an actual
   cross-entropy curve over the full range \(\varepsilon\in[10^{-3},1]\). The \(S=0\) curve is
   forced to be flat although a genuine uniformly smoothed cross-entropy generally changes by
   \(O(\varepsilon)\). The readout says “slope = \(S\) per decade,” but one decade changes
   \(-\log\varepsilon\) by \(\log 10\), so the change is \(S\log 10\), not \(S\).

2. **AsymmetryBars:** **Verified under additional assumptions · High confidence.** Its numeric
   cross-entropies are computed correctly, but “adding falsehood is free” describes only the
   isolated zero-weight summand. Under normalization, assigning false mass \(\delta\) by scaling
   the true-support probabilities by \(1-\delta\) raises cross-entropy exactly by
   \(-\log(1-\delta)\).

3. **VectorVsProbability:** **Mathematically computed but conceptually misleading · High
   confidence.** The displayed KL calculation is correct for its chosen categorical path. No
   theorem identifies orthogonal embedding vectors with disjoint supports of the softmax
   distribution over pair labels, so the juxtaposition cannot support the subsequent contrastive
   conclusion.

## Literature comparison

- SimCLR reports empirical benefits from larger batches; it does not attribute them to a support
  singularity: <https://proceedings.mlr.press/v119/chen20j.html>
- Wang and Isola characterize contrastive learning through alignment and hyperspherical
  uniformity: <https://proceedings.mlr.press/v119/wang20k.html>
- Ash et al. derive a collision--coverage trade-off for the number of negatives:
  <https://proceedings.mlr.press/v151/ash22a.html>
- Koromilas et al. connect InfoNCE minimizers to hyperspherical-energy minimization:
  <https://proceedings.mlr.press/v235/koromilas24a.html>
- Liang et al. attribute the modality gap to initialization and contrastive optimization, with a
  temperature-dependent distance: <https://arxiv.org/abs/2203.02053>
- SigLIP replaces global softmax normalization with a pairwise sigmoid objective and observes that
  batch-size benefits diminish; this is not evidence for the post's singularity mechanism:
  <https://arxiv.org/abs/2303.15343>
- Hallucination literature identifies mechanisms such as exposure bias and training/inference
  mismatch; the post cites no causal evidence for its cross-entropy-asymmetry account:
  <https://aclanthology.org/2022.findings-acl.58/>

## Recommendation

Unpublish the current post pending a ground-up rewrite. A defensible replacement could be a
short expository article titled “Three Boundaries People Confuse in Cross-Entropy” covering:

1. support mismatch in probability space;
2. finite logits versus boundary probabilities;
3. finite-temperature contrastive softmax versus the low-temperature limit.

Remove the hallucination, modality-gap, and large-batch causal claims unless new, direct evidence
is supplied. The current post cannot be repaired by adding caveats because its central narrative
depends on the false orthogonality-singularity identification.
