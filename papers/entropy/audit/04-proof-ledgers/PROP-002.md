# PROP-002 — Proposition 9 — boundary gradient

**Statement:** “Under the simplex constraint,” the gradient with respect to q_i is (-p_i/q_i) and diverges as q_i tends to zero with p_i>0.

**Verdict:** Verified under additional assumptions
**Confidence:** High

## Step ledger

| Step | Claim | Justification given | Assumptions used | Validity | Issue |
|---|---|---|---|---|---|
| 1 | Differentiate (-\sum p_x\log q_x) in q_i | Elementary derivative | Independent ambient q_i>0 | Valid | This is an ambient partial derivative, not yet simplex-constrained |
| 2 | Magnitude diverges for p_i>0,q_i to zero | p_i/q_i to infinity | Direct q coordinates | Valid | Parameterization-dependent optimization conclusion |
| 3 | Any optimization faces unbounded gradients | Figure caption | None supplied | Invalid | Softmax-logit gradient is q-p and bounded |

## Findings

- **Missing steps:** Define a tangent metric/coordinate chart if the simplex-constrained gradient is intended.
- **Hidden assumptions:** Direct probability-coordinate optimization; no softmax/logit parameterization.
- **Exact claim proved?** Only the ambient-coordinate derivative, not the parenthetical constrained-gradient claim or “any optimization” conclusion.
- **Repair:** Rename as the ambient gradient. Add (\nabla_zH=q-p) for softmax logits and distinguish the two.
- **Downstream consequences:** Figure 3's probability-coordinate curve is fine; its universal optimization caption is false.
