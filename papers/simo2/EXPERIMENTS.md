# SimO₂ — Experiment Plan & Cost (grounded in `neoyat/.../simo_contrastive/main.tex`)

**What SimO₂ is.** A hyperparameter-free, anchor-free contrastive loss
`λ₊·L_intra + λ₋·L_inter` on the biased ⵟ-kernel
`ⵟ_{b,ε}(x,y)=(⟨x,y⟩+b)²/(ε+‖x−y‖²)`. No temperature, margin, queue, predictor,
stop-grad, or decorrelation term. The intra/inter masks abstract *the source of
supervision*, so the **same loss** is supervised, self-supervised, or multimodal —
"The Law of Universal Representation."

**What's already proven (don't re-do).** Closed-form equilibrium: regular **simplex
ETF** for `C ≤ d+1, b ≥ 1/(C−1)`; **Welch-bound tight frame** for `C > d`.
**Benign-landscape** proof for a Gram-*non*convex loss (the headline theory).
**Parameter-free certified radius** `r_emb=√(C/(2(C−1))) > √2/2`, `r_in=r_emb/L`.
Theory-verification experiments exist (`verify_certified_radius`, `_effective_rank`,
`_collapse_gap`, `_convexity`, `_strict_saddle`, `_bifurcation`, `_gradient_bound`).

**The empirical gap (the real problem).** The existing CIFAR-10 run **collapses**:
`erank≈1.2, mean_off≈+0.97` (centroids near-identical = the cross-class-collapse
*saddle*), linear probe 34%. Almost certainly because **ε=0.01 is in the small-ε
regime where the benign-landscape transfer theorem does NOT apply** (it needs large ε).
Escaping this is the gate for every empirical claim.

---

## Evaluation protocol — the ⵟ-probe (NOT linear)

Embeddings live in ⵟ-space, and the certified radius is a **ⵟ-nearest-centroid**
statement, so the eval must be in ⵟ-geometry for the *certified-margin → probe-error*
chain to hold.
- **Primary metric — ⵟ-nearest-centroid** (`argmax_a ⵟ(z, μ_a)`): the exact classifier
  the certificate governs. Headline number.
- **Strong metric — ⵟ-kernel SVM/logistic** in the ⵟ-RKHS (`K_ij=ⵟ(z_i,z_j)`).
- **Fairness columns — linear + RBF probes for EVERY method** (SimO₂ and all baselines),
  so "SimO₂ wins under its native probe" is honest, not rigged.
- **Geometry**: effective rank (→ `C−1`?), simplex/Welch deviation, neural-collapse
  NC1–NC4, certified-radius histogram. **Bound**: certified ⵟ-margin → RKHS margin
  generalization bound → ⵟ-NC test error.

---

## Exp 0 — Collapse gate (ε-regime). DO FIRST. Everything is blocked on this.
Sweep `ε ∈ {0.01, 0.05, 0.1, 0.5, 1, 5, 10}` (× `b ∈ {1}`, 2 seeds), CIFAR-10,
ResNet-18, ~50-epoch diagnostic. Report **ⵟ-probe** + erank + simplex-dev.
**Gate:** some ε makes erank climb toward `C−1=9` and the ⵟ-probe jump. *If yes →
proceed. If collapse never breaks → that is the paper's crux; solve before benchmarks.*

## Regime matrix (same loss, swap the mask)

**R1 — Supervised (CIFAR-10/100). Theory's home + certified-robustness win.**
- E1ˢ Small-batch behaviour vs **SupCon, CE, ArcFace**.
- E2 Geometry vs batch (does it reach/hold the simplex?).
- E3 Bound vs reality (certified ⵟ-margin → ⵟ-NC error).
- E4 **Certified robustness**: adversarial/perturbation robustness of the ⵟ-NC
  classifier; show `r_in` predicts a robust margin SupCon/CE lack.

**R2 — Self-supervised (CIFAR-10). The small-batch cliff (primary empirical leg).**
- E1ˢˢ Batch sweep `{16,32,64,128,256,512,1024}` vs **SimCLR, MoCo-v2, BYOL**.
  *Expected:* baselines fall off a cliff as batch ↓ (negatives starve); SimO₂ flat
  (repulsion by construction). **The main figure.**
- Welch-frame geometry check; ⵟ-probe vs batch.

**R3 — Multimodal (CLIP/SigLIP re-align). Cheap novel vignette. [not CIFAR — separate budget].**
- Re-align a pretrained SigLIP with SimO₂; **steer the modality gap** with `b`; measure
  gap geometry + zero-shot/retrieval before/after.

## Ablations
A1 designable knob (`b`,`ε` → equilibrium geometry, closed-form vs measured).
A2 probe fairness (linear/RBF/ⵟ for all). A3 which-kernel-matters. A4 encoder/dataset
scaling (ResNet-18→50; CIFAR-100; STL-10).

## Baselines / datasets / seeds
- Supervised: SupCon, CE, ArcFace. SSL: SimCLR, MoCo-v2, BYOL.
- Shared ResNet-18 backbone, identical aug/optim; Muon (2D) + AdamW (rest).
- CIFAR-10 primary; CIFAR-100 / STL-10 for A4. 3 seeds (2 for lean first-pass).

---

## Compute & cost — CIFAR-10, single L4 (spot $0.27/hr, on-demand $0.85/hr)

Per-run anchor (ResNet-18, CIFAR, L4, incl. periodic ⵟ-probe + geometry eval):
- supervised 100-epoch run ≈ **0.4 GPU-hr**; SSL 200-epoch run ≈ **0.8 GPU-hr**
  (SSL needs more epochs). Diagnostic 50-epoch ≈ 0.2 GPU-hr.

| block | runs (lean) | GPU-hr (lean) | runs (full) | GPU-hr (full) |
|---|---|---|---|---|
| Exp 0 ε-gate (7ε×2s, 50ep) | 14 | 2.8 | 14 | 2.8 |
| R1 sup cliff (4 methods) | 4×3b×2s | 9.6 | 4×7b×3s | 33.6 |
| R2 SSL cliff (4 methods) | 4×3b×2s | 19.2 | 4×7b×3s | 67.2 |
| E2/E3 geometry+bound (eval) | — | 2 | — | 4 |
| E4 robustness (eval) | — | 3 | — | 6 |
| ablations A1–A4 | — | 5 | — | 12 |
| **Total** | | **≈ 42 GPU-hr** | | **≈ 126 GPU-hr** |

**Cost on CIFAR-10:**
- **Lean first-pass (3 batches, 2 seeds): ≈ 42 GPU-hr → ~$11 spot / ~$36 on-demand.**
- **Full camera-ready (7 batches, 3 seeds): ≈ 126 GPU-hr → ~$34 spot / ~$107 on-demand.**
- Wall-clock on one L4: ~2 days (lean) / ~5 days (full); parallelize across 2–4 spot
  L4s to compress to <1–2 days for a few extra dollars.

Add-ons (not CIFAR-10): CIFAR-100 ≈ same per-run cost (same image size); STL-10 ~1.5×;
CLIP/SigLIP re-align (R3) ≈ a handful of GPU-hr (fine-tune, not pretrain); metric
learning (CUB/Cars) separate, ~10 GPU-hr.

**Bottom line: the entire CIFAR-10 program is ~$11–34 on spot.** The "small-compute
regime is the thesis" is literally true — the whole paper's CIFAR-10 evidence costs
less than a dinner.

## Risks / kill-switches
1. **Collapse never breaks (Exp 0)** → crux to solve before any benchmark; the
   universality claim is hollow until one dataset escapes to the simplex/frame.
2. **Vacuous bound** → drop the certified-guarantee headline; keep certified-robustness
   + small-batch empirics.
3. **Large-batch regime** → concede openly; claim is "small-batch + a certificate."
4. **Probe rigging perception** → A2 (all probes for all methods) mitigates.

## Execution order
1. Wire `yat_probe()` (ⵟ-NC + ⵟ-kernel-SVM) into the eval. 2. **Exp 0 ε-gate** (gate
decision). 3. If pass → R2 SSL cliff (main figure) + R1 sup certified-robustness. 4.
Geometry/bound/ablations on those checkpoints. 5. R3 multimodal vignette. 6. Scale (A4).
