"""Reproducible probes for the entropy-paper audit.

These calculations can refute universal claims and check algebra, but passing
numeric tests do not prove theorems.  See 08-tool-verification-log.md.
"""

from __future__ import annotations

import math
import platform

import numpy as np
import sympy as sp


def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    terms = np.where(p == 0.0, 0.0, -p * np.log(q))
    return float(np.sum(terms))


def smoothing_probe() -> None:
    rng = np.random.default_rng(20260811)
    worst_scaled_remainder = 0.0
    trials = 0
    for n in range(2, 9):
        for _ in range(100):
            p = rng.dirichlet(np.ones(n))
            q = rng.dirichlet(np.ones(n))
            zero_mask = rng.random(n) < 0.35
            if np.all(zero_mask):
                zero_mask[rng.integers(n)] = False
            q[zero_mask] = 0.0
            q /= q.sum()
            violating = (p > 0.0) & (q == 0.0)
            s = float(p[violating].sum())
            c = float(np.sum(-p[~violating] * np.log(q[~violating])))
            for eps in (1e-3, 1e-5, 1e-7):
                q_eps = (1.0 - eps) * q + eps / n
                asymptotic = s * (-math.log(eps) + math.log(n)) + c
                remainder = abs(cross_entropy(p, q_eps) - asymptotic)
                worst_scaled_remainder = max(worst_scaled_remainder, remainder / eps)
                trials += 1
    print(f"smoothing_trials={trials}")
    print(f"max_abs_remainder_over_eps={worst_scaled_remainder:.12g}")


def symbolic_checks() -> None:
    p, q, eps, n = sp.symbols("p q eps n", positive=True)
    term = -p * sp.log((1 - eps) * q + eps / n)
    print("nonviolating_term_series=", sp.series(term, eps, 0, 3))

    z1, z2 = sp.symbols("z1 z2", real=True)
    q1 = sp.exp(z1) / (sp.exp(z1) + sp.exp(z2))
    loss = -sp.log(q1)
    print("d_onehot_ce_dz1=", sp.simplify(sp.diff(loss, z1)))
    print("d_onehot_ce_dz2=", sp.simplify(sp.diff(loss, z2)))

    p0, q0 = sp.symbols("p0 q0", positive=True)
    m0 = (p0 + q0) / 2
    js_coordinate = sp.Rational(1, 2) * p0 * sp.log(p0 / m0) + sp.Rational(1, 2) * q0 * sp.log(q0 / m0)
    js_derivative = sp.simplify(sp.diff(js_coordinate, q0))
    print("d_js_coordinate_dq=", js_derivative)
    print("d_js_limit_q_to_0=", sp.limit(js_derivative, q0, 0, dir="+"))


def info_nce_counterexamples() -> None:
    # Two exactly orthogonal unit embeddings, with matched pairs identical.
    # Each row's logits are [1/tau, 0/tau], so the loss is finite.
    for tau in (1.0, 0.07, 0.01):
        loss = math.log1p(math.exp(-1.0 / tau))
        print(f"orthogonal_negative_tau={tau:g}, loss={loss:.17g}")

    # An orthogonal negative does not have weight 1/N unless all logits match.
    tau = 0.07
    n_batch = 32768
    denom = math.exp(1.0 / tau) + (n_batch - 1)
    negative_weight = 1.0 / denom
    total_negative_weight = (n_batch - 1) / denom
    print(f"clip_like_negative_weight={negative_weight:.12g}")
    print(f"one_over_N={1 / n_batch:.12g}")
    print(f"total_negative_weight={total_negative_weight:.12g}")


def smoothing_figure_endpoint_check() -> None:
    # At eps=1, q^(eps)=uniform for every q, hence H(p,u)=log(n) for every p.
    for n in (2, 3, 10):
        print(f"eps_one_cross_entropy_n={n}: {math.log(n):.12g}")
    p = np.array([0.9, 0.1])
    q = np.array([0.9, 0.1])  # S(p,q)=0
    print(f"S_zero_H_at_eps_0={cross_entropy(p, q):.12g}")
    print(f"S_zero_H_at_eps_1={cross_entropy(p, np.full(2, 0.5)):.12g}")

    # If p and q are probability distributions with disjoint support on n
    # symbols, q has at least one support point, so |V|=|supp(p)| <= n-1.
    for alphabet_size in range(2, 7):
        max_violating = alphabet_size - 1
        print(f"disjoint_n={alphabet_size}, max_possible_abs_V={max_violating}")


if __name__ == "__main__":
    print("python=", platform.python_version())
    print("numpy=", np.__version__)
    print("sympy=", sp.__version__)
    symbolic_checks()
    smoothing_probe()
    info_nce_counterexamples()
    smoothing_figure_endpoint_check()
