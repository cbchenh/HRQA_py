"""Utility helpers: combinations, transition trees, default IFS scalar."""
from __future__ import annotations

import numpy as np


def default_alpha(K: int) -> float:
    """MATLAB default: a = 0.9999 * sin(pi/K) / (1 + sin(pi/K))."""
    s = np.sin(np.pi / K)
    return 0.9999 * s / (1.0 + s)


def combn(K: int, r: int) -> np.ndarray:
    """All K^r combinations of integers in [1, K], lexicographic order.

    Mirrors MATLAB combn.m. Returns shape (K**r, r) of dtype int64.
    """
    if r == 0:
        return np.zeros((1, 0), dtype=np.int64)
    grids = np.meshgrid(*[np.arange(1, K + 1)] * r, indexing="ij")
    out = np.stack([g.ravel() for g in grids], axis=1).astype(np.int64)
    return out


def sliding_windows(s: np.ndarray, r: int) -> np.ndarray:
    """Build (N - r + 1, r) matrix of consecutive r-grams from a 1D sequence.

    Vectorized via stride tricks — replaces MATLAB's
        for i = 1:r; LC(:,i) = s(i:end-r+i); end
    and avoids the Signal Processing Toolbox's phaseSpaceReconstruction.
    """
    s = np.ascontiguousarray(s).ravel()
    n = s.shape[0]
    if r > n:
        raise ValueError(f"r={r} larger than sequence length {n}")
    return np.lib.stride_tricks.sliding_window_view(s, r).copy()


def transition_tree(S: np.ndarray, n: int = 1):
    """Mirrors TransitionTree.m — returns (TTree, Tidx, TTreeS, TTreeSfq).

    n = 0 -> single-state frequencies; n >= 1 -> (n+1)-step transitions.
    """
    S = np.asarray(S).ravel().astype(np.int64)
    L = S.shape[0]
    if n < 0 or not np.isfinite(n):
        raise ValueError("n must be a non-negative integer")
    if L < n + 1:
        raise ValueError("sequence too short for given n")
    if n == 0:
        M = S.reshape(-1, 1)
    else:
        M = sliding_windows(S, n + 1)
    TTree, inv = np.unique(M, axis=0, return_inverse=True)
    Tidx = TTree.shape[0]
    TTreeS = inv + 1  # 1-based to match MATLAB
    counts = np.bincount(inv, minlength=Tidx)
    TTreeSfq = np.column_stack([np.arange(1, Tidx + 1), counts])
    return TTree, Tidx, TTreeS, TTreeSfq
