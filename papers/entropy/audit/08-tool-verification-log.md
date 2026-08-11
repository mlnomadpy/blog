# Tool verification log

Tool results below are evidence. Passing computations are not proofs of universal claims; a mathematically valid counterexample can refute one.

## 1. Structural extraction

- **Object:** Entire LaTeX source.
- **Tool:** `extract_structure.py` from the math-paper-audit skill.
- **Input:** `papers/entropy/main.tex`.
- **Output:** 14 theorem-like objects; 30 labels; 7 distinct refs; 7 citation keys; no undefined references; 23 unused labels.
- **Interpretation:** The registry covers every declared theorem/definition/proposition/remark. Unused figure/equation labels are not themselves errors.
- **Limitations:** Regex-based structure extraction does not establish correctness or discover all informal claims.
- **Reproducibility:** Python 3.14.6; command recorded in shell history for this audit.

## 2. LaTeX compilation

- **Object:** Document integrity.
- **Tool:** pdfTeX 1.40.29, three passes.
- **Input:** `main.tex`, output directed to `tmp/pdfs/entropy-build/`.
- **Output:** Third pass had no warnings, undefined references, overfull boxes, or errors.
- **Interpretation:** Source compiles cleanly.
- **Limitations:** Compilation does not validate mathematical content or figure semantics.

## 3. PDF rendering and visual inspection

- **Object:** All nine PDF pages; focused inspection of pages 1, 3–6, 8–9.
- **Tool:** Poppler `pdftoppm` at 110 dpi plus visual inspection.
- **Output:** Text and formulas legible. Figure 4 places the purported disjoint distributions inside the simplex. Figure 5 clips the red curve at its y-limit. Figure 1 curves end at different values at epsilon 1.
- **Interpretation:** ISS-005, ISS-008, and ISS-013 are visible in the rendered artifact.
- **Limitations:** Visual inspection does not establish algebraic correctness; source formulas were separately checked.

## 4. Symbolic expansion for THM-002

- **Object:** A nonviolating coordinate in the uniform-smoothing asymptotic.
- **Tool:** SymPy 1.14.0.
- **Expression:** `-p*log((1-eps)*q+eps/n)` with positive symbols (p,q,\varepsilon,n).
- **Output:**

  \[
  -p\log q-\varepsilon p\frac{1-nq}{nq}+O(\varepsilon^2).
  \]
- **Interpretation:** For each fixed coordinate with (p>0,q>0), the paper's (O(\varepsilon)) remainder is correct. Finite summation proves the stated remainder after assumptions are explicit.
- **Limitations:** SymPy simplification is not the proof; positivity and fixed-(q) assumptions are essential.
- **Reproducibility:** `python3 papers/entropy/audit/probe_entropy.py`.

## 5. Numerical smoothing probes

- **Object:** THM-002 remainder scaling.
- **Tool:** NumPy 2.4.4.
- **Input:** Seed 20260811; 2,100 trials; dimensions 2–8; Dirichlet p/q with randomly zeroed q coordinates; epsilon in (10^{-3},10^{-5},10^{-7}).
- **Output:** All values were finite after smoothing and the largest observed `abs(remainder)/epsilon` was 5143.79959631.
- **Interpretation:** No counterexample to the fixed-(p,q) (O(\varepsilon)) claim was found. The large constant occurs when positive q coordinates are very small and illustrates that the bound is not uniform over p,q.
- **Limitations:** Numerical agreement cannot prove the theorem; floating-point cancellation matters at very small epsilon.

## 6. Softmax-gradient symbolic check

- **Object:** Claims that false coordinates are unpenalized and all optimizations see unbounded boundary gradients.
- **Tool:** SymPy 1.14.0.
- **Input:** Binary one-hot cross-entropy with (q_1=e^{z_1}/(e^{z_1}+e^{z_2})).
- **Output:** (\partial L/\partial z_1=q_1-1), (\partial L/\partial z_2=q_2).
- **Interpretation:** A zero-target coordinate with positive predicted probability receives a positive logit gradient and is reduced by gradient descent. Logit gradients remain bounded.
- **Limitations:** Binary calculation is illustrative; the general identity (\nabla_z H=q-p) follows analytically from the softmax Jacobian.

## 7. Exact orthogonal-InfoNCE counterexample

- **Object:** REM-001 and REM-004 claim that orthogonality is a cross-entropy singularity.
- **Method:** Hand derivation, evaluated numerically.
- **Input:** Two orthonormal unit vectors; matched image/text embeddings identical; mismatched pairs orthogonal. Row logits are ((1/\tau,0)).
- **Output:** Per-row loss (\log(1+e^{-1/\tau})), equal to 0.3132617 at tau 1, (6.2487\times10^{-7}) at tau .07, and (3.72\times10^{-44}) at tau .01.
- **Interpretation:** Exact orthogonality is reachable with finite loss for every positive finite temperature. This refutes the claimed singularity and “cannot reach” statement.
- **Limitations:** It refutes the universal mechanism; it does not characterize every global InfoNCE optimum.

## 8. InfoNCE weight check

- **Object:** (w_k\approx1/N) for an orthogonal negative and the inferred large-batch mechanism.
- **Input:** One positive similarity 1, 32,767 negative similarities 0, tau .07.
- **Output:** Each negative weight (6.12337\times10^{-7}); (1/N=3.05176\times10^{-5}); total negative mass 0.02006445.
- **Interpretation:** A negative being orthogonal does not imply uniform weights. The positive logit and temperature control the denominator.
- **Limitations:** One configuration refutes the universal implication but does not establish a complete batch-size theory.

## 9. Figure 1 endpoint and S=0 checks

- **Object:** Figure 1.
- **Method:** Exact identity plus numerical example.
- **Input/output:** At epsilon 1, (q^{(1)}=u), so (H(p,u)=\log n) for every p,q. For p=q=(.9,.1), S=0 but H changes from 0.32508297 at epsilon 0 to 0.69314718 at epsilon 1.
- **Interpretation:** The plotted differing endpoints and constant green curve are not valid theorem instances.
- **Limitations:** This does not affect the local epsilon-to-zero asymptotic.

## 10. JSD boundary derivative

- **Object:** “JSD differentiable: Yes” in the alternatives table.
- **Tool:** SymPy 1.14.0.
- **Output:** For p>0, the coordinate derivative with respect to q is (\tfrac12\log(2q/(p+q))\to-\infty) as q tends to zero from above.
- **Interpretation:** JSD is finite at the boundary but not differentiable on every boundary face.
- **Limitations:** This is a one-coordinate directional calculation; it is sufficient to refute global boundary differentiability.

## 11. Formal/solver coverage

- **Formal proof assistant:** Not used; no Lean/Coq environment was required for the elementary finite-sum core.
- **SMT/SAT solver:** Not used.
- **Manual formalization performed:** The load-bearing CLIP bridge was reduced to the exact two-pair softmax expression above. This is a direct counterexample, not a proof-assistant certificate.
