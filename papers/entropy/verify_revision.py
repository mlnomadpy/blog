"""Numerical checks for the revised entropy manuscript.

The universal results are proved in main.tex.  These checks guard the plotted
formulas, the softmax-gradient identity, and the finite-temperature bounds.
"""

from __future__ import annotations

import math

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    shifted = z - np.max(z)
    exp = np.exp(shifted)
    return exp / exp.sum()


def check_uniform_figure() -> None:
    eps_values = np.geomspace(1e-6, 1.0, 200)
    for severity in (0.0, 0.1, 0.5, 1.0):
        values = (
            -(1.0 - severity) * np.log(0.5 - 0.25 * eps_values)
            - severity * np.log(0.25 * eps_values)
        )
        assert np.isfinite(values).all()
        assert abs(values[-1] - math.log(4.0)) < 1e-12


def check_softmax_gradient() -> None:
    rng = np.random.default_rng(20260811)
    step = 1e-6
    worst = 0.0
    for _ in range(500):
        n = int(rng.integers(2, 10))
        p = rng.dirichlet(np.ones(n))
        z = rng.normal(size=n)
        q = softmax(z)
        analytic = q - p
        numeric = np.empty(n)
        for i in range(n):
            plus = z.copy()
            minus = z.copy()
            plus[i] += step
            minus[i] -= step
            loss_plus = -float(np.dot(p, np.log(softmax(plus))))
            loss_minus = -float(np.dot(p, np.log(softmax(minus))))
            numeric[i] = (loss_plus - loss_minus) / (2.0 * step)
        worst = max(worst, float(np.max(np.abs(analytic - numeric))))
    assert worst < 1e-8, worst
    print(f"max_softmax_gradient_error={worst:.3e}")


def check_contrastive_bounds() -> None:
    rng = np.random.default_rng(20260811)
    trials = 10_000
    for _ in range(trials):
        n = int(rng.integers(2, 128))
        tau = float(10 ** rng.uniform(-1.5, 0.5))
        similarities = rng.uniform(-1.0, 1.0, size=n)
        match = int(rng.integers(n))
        logits = similarities / tau
        probabilities = softmax(logits)
        loss = -math.log(float(probabilities[match]))
        lower = math.log1p((n - 1) * math.exp(-2.0 / tau))
        upper = math.log1p((n - 1) * math.exp(2.0 / tau))
        assert lower - 1e-12 <= loss <= upper + 1e-12
        gradient = (probabilities - np.eye(1, n, match).ravel()) / tau
        assert np.max(np.abs(gradient)) <= 1.0 / tau + 1e-12
    print(f"contrastive_bound_trials={trials}")


def check_orthogonal_formula() -> None:
    for n in (2, 32, 1024, 32768):
        for tau in (0.01, 0.07, 0.5, 1.0):
            exact = math.log1p((n - 1) * math.exp(-1.0 / tau))
            logits = np.zeros(n)
            logits[0] = 1.0 / tau
            direct = -math.log(float(softmax(logits)[0]))
            assert abs(exact - direct) < 2e-12


if __name__ == "__main__":
    check_uniform_figure()
    check_softmax_gradient()
    check_contrastive_bounds()
    check_orthogonal_formula()
    print("revision_checks=PASS")
