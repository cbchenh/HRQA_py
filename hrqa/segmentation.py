"""State-space segmentation: turn continuous multivariate signals into
categorical integer sequences.

Two segmenters share the same fit/transform interface so that bin centers can
be computed once on a reference cohort and reused across all subjects/windows
(critical for cross-subject comparability).

  - HASSegmenter    — Hyperoctree Aggregate Segmentation (recursive 2^d split,
                       capacity-driven). Mirrors HAS-5.m + SymbG-18.m.
  - KMeansSegmenter — sklearn KMeans with explicit K. Centers/labels are
                       sortable by an external permutation (LabelOpt).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# Hyperoctree Aggregate Segmentation
# ---------------------------------------------------------------------------

def _has_recursive(X: np.ndarray, cap: int, ub: np.ndarray, lb: np.ndarray,
                   depth: int, out_ub: list, out_lb: list) -> None:
    """Recursive 2^d hyperoctree split — populates out_ub / out_lb in-place."""
    n, d = X.shape
    if n > cap and depth > 0:
        cb = 0.5 * (ub + lb)
        # Each dimension has 2 choices: lower half or upper half. Iterate over
        # all 2^d sub-octants. Use bitmask k from 0..2^d-1.
        for k in range(1 << d):
            new_ub = np.empty(d)
            new_lb = np.empty(d)
            mask = np.ones(n, dtype=bool)
            for di in range(d):
                upper_half = bool((k >> di) & 1)
                if upper_half:
                    new_ub[di] = ub[di]
                    new_lb[di] = cb[di]
                else:
                    new_ub[di] = cb[di]
                    new_lb[di] = lb[di]
                mask &= (X[:, di] < new_ub[di]) & (X[:, di] >= new_lb[di])
            sub = X[mask]
            _has_recursive(sub, cap, new_ub, new_lb, depth - 1, out_ub, out_lb)
    else:
        out_ub.append(ub.copy())
        out_lb.append(lb.copy())


@dataclass
class HASSegmenter:
    """Hyperoctree Aggregate Segmenter (recursive capacity-driven).

    Parameters
    ----------
    capacity : int
        Maximum samples allowed in a bin before further subdivision.
    max_depth : int, default 100
        Recursion depth limit (matches MATLAB ``set(0,'RecursionLimit',100)``).
    eps : float, default 1e-6
        Small relative padding so that max(X) lies strictly inside the
        outer bound (matches MATLAB ``+0.000001*(max-min)``).
    """
    capacity: int
    max_depth: int = 100
    eps: float = 1e-6

    # Fitted state
    Ub: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    Lb: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    centers: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    K: Optional[int] = field(default=None, init=False, repr=False)
    label_perm: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def fit(self, X: np.ndarray) -> "HASSegmenter":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")
        rng = X.max(axis=0) - X.min(axis=0)
        ub = X.max(axis=0) + self.eps * rng
        lb = X.min(axis=0) - self.eps * rng
        out_ub: list = []
        out_lb: list = []
        _has_recursive(X, self.capacity, ub, lb, self.max_depth, out_ub, out_lb)
        self.Ub = np.vstack(out_ub)
        self.Lb = np.vstack(out_lb)
        self.centers = 0.5 * (self.Ub + self.Lb)
        self.K = self.Ub.shape[0]
        self.label_perm = None
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map each row of X to a 1-based bin label (vectorized SymbG)."""
        if self.Ub is None:
            raise RuntimeError("Call fit(...) first.")
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        K = self.K
        # Build (K, n) boolean mask of which point lies in which bin.
        # For each bin idx: all dims must satisfy lb <= x < ub.
        # Vectorize by broadcasting (n, 1, d) vs (1, K, d).
        diff_ub = X[:, None, :] < self.Ub[None, :, :]
        diff_lb = X[:, None, :] >= self.Lb[None, :, :]
        in_bin = (diff_ub & diff_lb).all(axis=2)  # (n, K)
        # Each row should hit exactly one bin (HAS partitions space). Take argmax.
        # If a row doesn't fall in any bin (shouldn't happen on training data),
        # fall back to the nearest center.
        any_hit = in_bin.any(axis=1)
        labels = np.empty(n, dtype=np.int64)
        labels[any_hit] = in_bin[any_hit].argmax(axis=1) + 1  # 1-based
        if not any_hit.all():
            # Fallback: nearest center for out-of-range points
            miss = ~any_hit
            d = cdist(X[miss], self.centers)
            labels[miss] = d.argmin(axis=1) + 1
        if self.label_perm is not None:
            labels = self.label_perm[labels - 1]  # apply permutation
        return labels

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def set_label_permutation(self, perm: np.ndarray) -> None:
        """Apply a fixed permutation so transform(...) returns canonical labels.

        ``perm`` is 1-based, length K: original label i -> perm[i-1].
        """
        perm = np.asarray(perm, dtype=np.int64).ravel()
        if perm.shape[0] != self.K:
            raise ValueError(f"perm length {perm.shape[0]} != K={self.K}")
        self.label_perm = perm


# ---------------------------------------------------------------------------
# KMeans-based segmentation
# ---------------------------------------------------------------------------

@dataclass
class KMeansSegmenter:
    """KMeans state-space segmenter.

    Parameters
    ----------
    K : int
        Number of clusters / states.
    random_state : int, default 0
    n_init : int, default 10
    """
    K: int
    random_state: int = 0
    n_init: int = 10

    centers: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    label_perm: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _km: Optional[KMeans] = field(default=None, init=False, repr=False)

    def fit(self, X: np.ndarray) -> "KMeansSegmenter":
        X = np.asarray(X, dtype=np.float64)
        km = KMeans(n_clusters=self.K, random_state=self.random_state,
                    n_init=self.n_init)
        km.fit(X)
        self._km = km
        self.centers = km.cluster_centers_
        self.label_perm = None
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.centers is None:
            raise RuntimeError("Call fit(...) first.")
        X = np.asarray(X, dtype=np.float64)
        # Use cdist directly (faster than predict for small batches and avoids
        # sklearn's overhead). predict is fine too — keep it simple & fast.
        labels = self._km.predict(X) + 1  # 1-based
        if self.label_perm is not None:
            labels = self.label_perm[labels - 1]
        return labels

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def set_label_permutation(self, perm: np.ndarray) -> None:
        perm = np.asarray(perm, dtype=np.int64).ravel()
        if perm.shape[0] != self.K:
            raise ValueError(f"perm length {perm.shape[0]} != K={self.K}")
        self.label_perm = perm


def make_segmenter(method: str = "kmeans", **kwargs):
    """Factory: ``make_segmenter('kmeans', K=8)`` or
    ``make_segmenter('has', capacity=500)``.
    """
    method = method.lower()
    if method == "kmeans":
        return KMeansSegmenter(**kwargs)
    if method in ("has", "hyperoctree"):
        return HASSegmenter(**kwargs)
    raise ValueError(f"Unknown segmenter method: {method!r}")
