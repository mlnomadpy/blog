# Literature and citation audit

## Scope and verdicts

- **Citation existence:** **Verified · High confidence.** All seven bibliography entries exist.
- **Contextual support:** **Verified under additional assumptions · High confidence.** The cited works support that CLIP/SimCLR/CPC use contrastive objectives, CLIP used a batch of 32,768, JSD appears in the original GAN analysis, and Villani is an appropriate optimal-transport reference.
- **Novelty positioning:** **Unsupported · High confidence.** No source is cited for the core support criterion, the proposed tier taxonomy, or the smoothing-rate coefficient.
- **Literature completeness:** **Incomplete · High confidence.** The discussion omits direct work on contrastive-loss geometry/batch effects and does not compare the elementary finite-alphabet theorem to standard relative-entropy/absolute-continuity treatments.

## Cited-source checks

1. **Radford et al. (CLIP).** The paper states that CLIP trains a symmetric cross-entropy over the (N\times N) similarity scores and used minibatch size 32,768. This supports the descriptive facts, but it also makes the manuscript's object mismatch clear: the softmax distribution is over batch pair labels, not over the support of image/text embedding distributions. [CLIP paper](https://arxiv.org/pdf/2103.00020)

2. **Liang et al. (modality gap).** This paper reports separated modality regions, an initialization cone effect, and a temperature-dependent contrastive-loss landscape. It does not identify the gap with a KL/cross-entropy support singularity. Its loss-landscape evidence is a closer and materially different account than the manuscript's claimed theorem transfer. [Mind the Gap](https://arxiv.org/abs/2203.02053)

3. **Zhai et al. (SigLIP).** The paper explicitly replaces global softmax normalization with a pairwise sigmoid loss and reports that benefits from increasing batch size diminish, with roughly 32k sufficient in their study. This weakens any general claim that a cross-entropy singularity inherently demands enormous batches. [SigLIP, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Zhai_Sigmoid_Loss_for_Language_Image_Pre-Training_ICCV_2023_paper.html)

4. **van den Oord et al. (CPC/InfoNCE).** The citation correctly identifies the probabilistic contrastive loss and negative sampling lineage. It does not support the manuscript's orthogonality-as-singularity or batch-size derivation. [CPC](https://arxiv.org/abs/1807.03748)

5. **Chen et al. (SimCLR).** SimCLR reports empirical benefit from larger batches and more training steps. That empirical fact does not establish the proposed singularity mechanism. [SimCLR](https://proceedings.mlr.press/v119/chen20j.html)

6. **Goodfellow et al. (GAN).** Appropriate support for the historical statement that the original GAN analysis involves Jensen-Shannon divergence. [Generative Adversarial Nets](https://papers.nips.cc/paper_files/paper/2014/hash/f033ed80deb0234979a61f95710dbe25-Abstract.html)

7. **Villani (optimal transport).** Appropriate general reference for Wasserstein theory, but the manuscript should state the metric/moment hypotheses when leaving the finite-alphabet setting. [Optimal Transport: Old and New](https://link.springer.com/book/10.1007/978-3-540-71050-9)

## Closest omitted or subsequent work

- Koromilas et al. analyze mini-batch and asymptotic contrastive objectives and identify hyperspherical-energy structure. This is directly relevant to any batch-size/geometry positioning and should be discussed. [ICML 2024 paper](https://proceedings.mlr.press/v235/koromilas24a.html)
- Huang et al. explicitly analyze InfoNCE gradient reduction. This is closer to the claimed “flatness” topic than support singularity and should be compared carefully. [ICML 2023 paper](https://proceedings.mlr.press/v202/huang23c)
- A July 2026 preprint, posted after the manuscript's February 2026 date, studies modality gap formation under InfoNCE and reports a low-temperature mechanism. It cannot be treated as an omission from the original draft, but a current revision should engage with it. [Mager et al. 2026](https://arxiv.org/abs/2607.10698)

## Novelty assessment

The condition (D_{\mathrm{KL}}(p\|q)<\infty\iff p\ll q) on a finite alphabet is standard. The uniform-mixture expansion is a short coordinatewise Taylor calculation. The manuscript may still have expository value in emphasizing the coefficient (S(p,q)), but “taxonomy” and “precise measure of severity” require either:

- a literature comparison showing the invariant has not already been isolated;
- a general theorem for nonuniform smoothing paths, continuous spaces, or rates beyond first order; or
- an explicit expository framing rather than a research novelty claim.

No exhaustive novelty search was possible from the manuscript's terminology alone; therefore the exact priority of the (S(p,q)) coefficient remains **Unable to verify from available material · Medium confidence**.

## Post-revision novelty decision (2026-08-11)

- **Research novelty:** **Unsupported · High confidence.** The revised manuscript is mathematically cleaner, but its individual ingredients do not clear a research-contribution threshold. The support criterion is the standard absolute-continuity condition for relative entropy; strict propriety of log loss and the softmax identities are standard; and the contrastive row loss is the established multiclass N-pair/InfoNCE log-sum-exp objective.
- **Pathwise expansion:** **Verified but insufficiently novel · High confidence.** Under the stated power-law hypotheses, the coefficient \(\sum_{x\in V} a_xp(x)\) follows immediately by applying \(-\log(c_x\varepsilon^{a_x}(1+O(\varepsilon^r)))\) coordinatewise. A targeted search did not locate this exact finite-alphabet packaging, but absence of an exact phrase match is not evidence of priority, and the derivation is too elementary to support a standalone novelty claim without a substantially deeper extension.
- **Finite-temperature bounds:** **Known/overlapping · High confidence.** Sohn's multiclass N-pair loss already uses the same rowwise log-sum-exp form. Luthra, Mishra, and Galanti prove bounded softmax masses for similarities in \([-1,1]\) at positive temperature and derive finite gradient bounds. The manuscript's scalar loss bounds and orthogonal-negative substitution are direct specializations.
- **Editorial recommendation:** **Remove as a claimed research paper · High confidence.** Retain only as an explicitly expository note if useful. Do not describe its theorems as original.

Primary comparison sources:

- K. Sohn, “Improved Deep Metric Learning with Multi-class N-pair Loss Objective,” NeurIPS 2016: <https://proceedings.neurips.cc/paper/2016/hash/6b180037abbebea991d8b1232f8a8ca9-Abstract.html>
- T. Chen et al., “A Simple Framework for Contrastive Learning of Visual Representations,” ICML 2020: <https://proceedings.mlr.press/v119/chen20j.html>
- A. Luthra, P. Mishra, and T. Galanti, “On the Alignment Between Supervised and Self-Supervised Contrastive Learning,” ICLR 2026, especially Lemma 4 and the gradient bounds: <https://openreview.net/pdf?id=JkitQScjuL>
- E. Y. Ovcharov, “Existence and Uniqueness of Proper Scoring Rules,” JMLR 2015: <https://www.jmlr.org/papers/v16/ovcharov15a.html>
