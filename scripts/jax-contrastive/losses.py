"""The six contrastive losses, as pure ``jax.numpy`` functions.

Every loss takes ``z`` (an ``(N, 2)`` array of 2D embeddings), integer
``labels`` ``(N,)``, a PRNG ``key`` (used only by the samplers), and the loss's
scalar hyperparameter. It returns a single scalar — the same quantity the
interactive visualisation reports — so that ``jax.grad`` of it reproduces the
visualisation's update step exactly.

Pairwise masks (`same`, `eye`, `triu`) depend only on the labels, not on the
positions, so they are built once on the host and closed over. That keeps the
loss bodies free of Python loops and lets ``jax.jit`` compile a single fused
kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

# ── the unit-sphere constraint ──────────────────────────────────────────
def l2normalize(z: jnp.ndarray) -> jnp.ndarray:
    """Project every row of ``z`` onto the unit circle."""
    return z / (jnp.linalg.norm(z, axis=1, keepdims=True) + 1e-9)


# ── static label masks ──────────────────────────────────────────────────
def make_masks(labels: np.ndarray) -> dict[str, jnp.ndarray]:
    """Build the boolean pair masks once, on the host."""
    same = labels[:, None] == labels[None, :]
    eye = np.eye(len(labels), dtype=bool)
    triu = np.triu(np.ones_like(same), k=1)  # i < j, each unordered pair once
    return {
        "same": jnp.asarray(same),
        "eye": jnp.asarray(eye),
        "triu": jnp.asarray(triu),
        "num_pairs": float(triu.sum()),
    }


# ── 1. Pair contrastive (Hadsell, Chopra & LeCun, 2006) ─────────────────
def loss_pair(z, labels, key, margin, *, m):
    """Pull same-class pairs together; push different-class pairs to a margin.

    Euclidean (not on the sphere). Same-class pairs pay ``||z_i - z_j||^2``;
    different-class pairs pay ``max(0, margin - ||z_i - z_j||)^2`` and then go
    silent once they are far enough apart.
    """
    n = z.shape[0]
    diff = z[:, None, :] - z[None, :, :]
    d2 = jnp.sum(diff ** 2, axis=-1)
    d = jnp.sqrt(d2 + 1e-9)
    upper = m["triu"] > 0
    pos = jnp.where(m["same"] & upper, d2, 0.0)
    neg_active = (~m["same"]) & upper & (d < margin)
    neg = jnp.where(neg_active, (margin - d) ** 2, 0.0)
    # Normalize by N (number of points), matching the visualisation's update
    # scale — not by the number of pairs.
    return (jnp.sum(pos) + jnp.sum(neg)) / n


# ── 2. Triplet (FaceNet; Schroff, Kalenichenko & Philbin, 2015) ─────────
def loss_triplet(z, labels, key, margin, *, m):
    """For each anchor, sample one positive and one negative; require the
    negative to be at least ``margin`` farther than the positive.

    Only *violating* triplets contribute — the loss is the mean hinge over the
    anchors whose triplet is unsatisfied.
    """
    kp, kn = jax.random.split(key)
    pos_mask = m["same"] & (~m["eye"])
    neg_mask = ~m["same"]
    pidx = jax.random.categorical(kp, jnp.where(pos_mask, 0.0, -1e9), axis=1)
    nidx = jax.random.categorical(kn, jnp.where(neg_mask, 0.0, -1e9), axis=1)
    d_ap = jnp.sum((z - z[pidx]) ** 2, axis=1)
    d_an = jnp.sum((z - z[nidx]) ** 2, axis=1)
    hinge = jnp.clip(d_ap - d_an + margin, a_min=0.0)
    violators = jnp.sum(hinge > 0)
    return jnp.sum(hinge) / jnp.maximum(violators, 1.0)


# ── 3. InfoNCE / NT-Xent (van den Oord 2018; Chen et al. 2020) ──────────
def loss_infonce(z, labels, key, tau, *, m):
    """Softmax over cosine similarities. Every other point is a negative; one
    positive per anchor is sampled. The numerator is the positive, the
    denominator is everyone else.
    """
    sim = (z @ z.T) / tau
    logits = jnp.where(m["eye"], -1e9, sim)        # mask self
    log_z = jax.nn.logsumexp(logits, axis=1)        # over k != i
    pos_mask = m["same"] & (~m["eye"])
    pidx = jax.random.categorical(key, jnp.where(pos_mask, 0.0, -1e9), axis=1)
    pos_sim = jnp.take_along_axis(sim, pidx[:, None], axis=1)[:, 0]
    return jnp.mean(log_z - pos_sim)


# ── 4. Supervised Contrastive (Khosla et al., 2020) ─────────────────────
def loss_supcon(z, labels, key, tau, *, m):
    """InfoNCE averaged over *all* same-class positives — no sampling. Pulls
    each point toward its class centroid on the sphere.
    """
    sim = (z @ z.T) / tau
    logits = jnp.where(m["eye"], -1e9, sim)
    log_p = logits - jax.nn.logsumexp(logits, axis=1, keepdims=True)
    pos = (m["same"] & (~m["eye"])).astype(z.dtype)
    p_count = jnp.sum(pos, axis=1)
    per_anchor = -jnp.sum(pos * log_p, axis=1) / jnp.maximum(p_count, 1.0)
    return jnp.mean(per_anchor)


# ── 5. SigLIP (Zhai et al., 2023) ───────────────────────────────────────
def loss_siglip(z, labels, key, target, *, m):
    """Pairwise sigmoid with a learnable-style bias. ``target`` is the cosine
    at which different-class pairs stop being pushed apart — set it near 0 and
    negatives equilibrate at orthogonality, not opposition.
    """
    n = z.shape[0]
    t, b = 10.0, -10.0 * target
    sim = z @ z.T                                   # cosine; z is on the sphere
    sign = jnp.where(m["same"], 1.0, -1.0)          # y_ij = +1 same / -1 diff
    per_pair = jax.nn.softplus(-sign * (t * sim + b))  # log(1 + e^{-y(t·s+b)})
    return jnp.sum(jnp.where(m["triu"] > 0, per_pair, 0.0)) / n


# ── 6. Cosine→0 (the orthogonality objective) ──────────────────────────
def loss_orthog(z, labels, key, _unused, *, m):
    """Pull same-class pairs to cosine 1; push different-class pairs to cosine
    0 — orthogonality, not opposition.
    """
    n = z.shape[0]
    c = z @ z.T
    per_pair = jnp.where(m["same"], 1.0 - c, c ** 2)
    return jnp.sum(jnp.where(m["triu"] > 0, per_pair, 0.0)) / n


# ── registry ────────────────────────────────────────────────────────────
@dataclass
class Cfg:
    fn: Callable
    on_sphere: bool
    lr: float
    param: float
    steps: int
    dataset: str
    title: str
    needs_key: bool = False
    sub: str = ""
    n: int = 60
    seed: int | None = None  # per-loss seed override; falls back to the CLI seed


LOSSES: dict[str, Cfg] = {
    "pair": Cfg(loss_pair, False, 0.4, 1.2, 160, "random",
                "Pair contrastive (Hadsell 2006)", sub="||Δ||²₊ + max(0, m−||Δ||)²₋"),
    "triplet": Cfg(loss_triplet, False, 0.4, 0.3, 220, "random",
                   "Triplet (FaceNet 2015)", needs_key=True,
                   sub="max(0, d²ₐₚ − d²ₐₙ + m)"),
    "infonce": Cfg(loss_infonce, True, 0.25, 0.2, 280, "random",
                   "InfoNCE / NT-Xent (2018)", needs_key=True,
                   sub="−log e^{cos/τ} / Σ e^{cos/τ}"),
    "supcon": Cfg(loss_supcon, True, 0.25, 0.2, 280, "random-4",
                  "SupCon (2020)", sub="multi-positive InfoNCE"),
    "siglip": Cfg(loss_siglip, True, 0.03, -0.05, 320, "random",
                  "SigLIP (2023)", sub="softplus(−y(t·cos + b))"),
    # Two classes: cosine→0 lands them on perpendicular axes. (Four classes
    # can't be mutually orthogonal in 2D — that ceiling is the prior post's
    # topic; here we want a clean organization.) Seed 8 avoids the symmetric
    # stall the loss falls into from a perfectly balanced random start.
    "orthog": Cfg(loss_orthog, True, 0.08, 0.0, 320, "random",
                  "Cosine→0 (orthogonality)", sub="(1 − cos)₊ + cos²₋",
                  n=40, seed=8),
}
