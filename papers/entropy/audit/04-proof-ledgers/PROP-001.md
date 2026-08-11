# PROP-001 — Proposition 7 — directional coordinate asymmetry

**Statement:** If q_x=0<p_x, then (-p_x\log q_x=+\infty); if p_x=0<q_x, then the summand is zero.

**Verdict:** Verified
**Confidence:** High

## Step ledger

| Step | Claim | Justification given | Assumptions used | Validity | Issue |
|---|---|---|---|---|---|
| 1 | Missing positive-p support gives an infinite summand | log 0=-infinity | Extended-real convention | Valid | DEF-001 states the signed convention incorrectly |
| 2 | A zero-p coordinate contributes zero | 0 times finite log q is zero | q_x>0 | Valid | None |

## Findings

- **Missing steps:** None.
- **Hidden assumptions:** This is a coordinate summand statement, not a feasible perturbation of q on the simplex.
- **Exact claim proved?** Yes.
- **Repair:** Keep the proposition but delete “adding falsehood is free,” “no constraint on precision,” and hallucination conclusions unless a global theorem is added.
- **Downstream consequences:** REM-002/REM-003 do not follow. In softmax logits, a false coordinate has gradient q_i>0 and is pushed down.
