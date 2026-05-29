"""Toy 2D datasets and the nearest-centroid accuracy metric.

These mirror the JavaScript generators in
``src/components/viz/LossExplorer.astro`` so the JAX results line up with the
interactive visualisations in the "Untangling the Moons" post. Everything here
is plain NumPy — the data is generated once on the host and handed to JAX as a
static array.
"""

from __future__ import annotations

import numpy as np

# Class colours, matching the blog palette (LossExplorer.astro `classPalette`).
PALETTE = ["#b3661b", "#4a7fb3", "#3a8f5e", "#9a4f9c"]


def make_random(n: int = 60, k: int = 2, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Uniform points in [-1.5, 1.5]^2 with balanced, shuffled class labels.

    There is *no* spatial signal: the labels are assigned ``i % k`` and then
    shuffled, so position tells you nothing about class. The loss has to impose
    all of the geometry itself — that is the whole point of "organizing
    randomness."
    """
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-1.5, 1.5, size=(n, 2)).astype(np.float32)
    labels = np.array([i % k for i in range(n)], dtype=np.int32)
    rng.shuffle(labels)
    return pts, labels


def make_random4(n: int = 60, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Four-class version of :func:`make_random`."""
    return make_random(n=n, k=4, seed=seed)


def make_moons(n: int = 60, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Two interleaving half-circles with Gaussian noise (the classic dataset)."""
    rng = np.random.default_rng(seed)
    half = n // 2
    pts = np.zeros((n, 2), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int32)
    for i in range(half):
        t = np.pi * (i / max(1, half - 1))
        pts[i] = [np.cos(t) + rng.normal() * 0.07, np.sin(t) + rng.normal() * 0.07]
        labels[i] = 0
    for i in range(n - half):
        t = np.pi * (i / max(1, n - half - 1))
        pts[half + i] = [
            1.0 - np.cos(t) + rng.normal() * 0.07,
            0.4 - np.sin(t) + rng.normal() * 0.07,
        ]
        labels[half + i] = 1
    return pts, labels


def get_dataset(name: str, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch by name with explicit point count and seed."""
    if name == "random":
        return make_random(n, 2, seed)
    if name == "random-4":
        return make_random4(n, seed)
    if name == "moons":
        return make_moons(n, seed)
    raise ValueError(f"unknown dataset {name!r}")


DATASETS = ["random", "random-4", "moons"]


def accuracy(z: np.ndarray, labels: np.ndarray) -> float:
    """Nearest-centroid classification accuracy in the embedding space.

    Compute each class centroid, assign every point to its closest centroid,
    and report the fraction that lands on the right class. This is the same
    cheap linear-separability proxy the visualisations use.
    """
    z = np.asarray(z)
    classes = np.unique(labels)
    centroids = np.stack([z[labels == c].mean(axis=0) for c in classes])
    # (N, C) squared distances from each point to each centroid.
    d2 = ((z[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
    pred = classes[d2.argmin(axis=1)]
    return float((pred == labels).mean())
