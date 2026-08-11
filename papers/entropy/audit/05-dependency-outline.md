# Dependency outline

## Main finite-alphabet chain

`ASM-001 + ASM-002 + ASM-003 → DEF-001 + DEF-003 → THM-001`

`THM-001 + DEF-002 → THM-003`

`ASM-005 + DEF-001 + DEF-003 + DEF-004 → THM-002`

## Claimed applied chain

`THM-001 ⇢ REM-001 → closing synthesis` fails because the theorem's probability vectors are replaced by continuous embedding distributions and then conflated with CLIP's batch softmax.

`PROP-001 ⇢ REM-002 → REM-003 → closing synthesis` fails because a coordinate summand is not a feasible simplex perturbation; the softmax gradient has the opposite effect on zero-target positive-prediction coordinates.

`ASM-009 → REM-004 → closing synthesis` fails because ASM-009 is false: finite orthogonal cosine logits do not produce zero softmax probability.

## Bottlenecks and graph diagnostics

- **Mathematical bottleneck:** DEF-004 is incomplete, but THM-002 can stand independently once uniform smoothing is made explicit.
- **Applied bottleneck:** a missing bridge theorem identifying an embedding-space configuration with a zero-probability event in the actual loss. No such theorem is stated, and the obvious identification is false.
- **Cycles:** none.
- **Forward dependencies:** none in the core.
- **Orphans:** THM-002 is not used to derive the applied claims; the discussion mainly invokes THM-001 and PROP-001.
- **Unsupported foundational node:** ASM-009 drives the batch-size discussion and is contradicted.
