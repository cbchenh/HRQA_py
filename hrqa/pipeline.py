"""Top-level pipeline functions.

Two entry points (per user spec):
    A) hrqa_from_signal   — raw multivariate timeseries -> categorical -> HRQA
    B) hrqa_from_sequence — categorical integer sequence -> HRQA

Plus a sliding-window batch wrapper:
    rhrqa_windowed        — windowed RHRQA over a long categorical sequence
                            with optional joblib parallelism

Both top-level functions return a clean dict (or DataFrame). Tight inner-loop
work always uses NumPy arrays.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .hrqa_core import rhrqa, fhrqa, METRIC_NAMES, N_METRICS
from .ifs import ifs_address
from .segmentation import HASSegmenter, KMeansSegmenter, make_segmenter
from .utils import default_alpha, combn

Segmenter = Union[HASSegmenter, KMeansSegmenter]


# ---------------------------------------------------------------------------
# A) Raw signal entry point
# ---------------------------------------------------------------------------

def hrqa_from_signal(X: np.ndarray,
                     segmenter: Optional[Segmenter] = None,
                     *,
                     # Used only if segmenter is None:
                     method: str = "kmeans",
                     K: Optional[int] = None,
                     capacity: Optional[int] = None,
                     # HRQA params:
                     r: int = 1,
                     a: Optional[float] = None,
                     B: int = 20,
                     label_perm: Optional[np.ndarray] = None,
                     **segmenter_kwargs,
                     ) -> dict:
    """Raw multivariate signal -> HRQA features.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Continuous multivariate time series.
    segmenter : fitted HASSegmenter or KMeansSegmenter, optional
        If provided, used as-is (transform only — no refit). This is the
        typical multi-subject workflow: fit ONCE on a representative pool,
        then pass the same segmenter to every subject so all data shares
        identical state-space partitioning.
        If None, a new segmenter is fit on X using ``method``.
    method : {'kmeans', 'has'}
        Segmenter type when ``segmenter`` is None.
    K : int
        Number of clusters (required for KMeans when segmenter is None).
    capacity : int
        Cell capacity (required for HAS when segmenter is None).
    r : int, default 1
        HRQA local-cluster order.
    label_perm : ndarray, optional
        1-based permutation of length K. If provided, applied to the
        segmenter so that emitted labels follow the canonical order.

    Returns
    -------
    dict with keys: 'S' (categorical sequence), 'segmenter', 'IdxM',
        'HRR', 'HMean', 'HVar', 'HSkew', 'HKurt', 'HEnt', 'HGini'.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if segmenter is None:
        if method.lower() == "kmeans":
            if K is None:
                raise ValueError("K is required for KMeans segmenter")
            segmenter = KMeansSegmenter(K=K, **segmenter_kwargs)
        elif method.lower() in ("has", "hyperoctree"):
            if capacity is None:
                capacity = max(2, X.shape[0] // 4)
            segmenter = HASSegmenter(capacity=capacity, **segmenter_kwargs)
        else:
            raise ValueError(f"Unknown method: {method!r}")
        segmenter.fit(X)

    if label_perm is not None:
        segmenter.set_label_permutation(np.asarray(label_perm, dtype=np.int64))

    S = segmenter.transform(X)
    K_eff = segmenter.K
    out = rhrqa(S, K=K_eff, r=r, a=a, B=B)
    out["S"] = S
    out["segmenter"] = segmenter
    return out


# ---------------------------------------------------------------------------
# B) Categorical sequence entry point
# ---------------------------------------------------------------------------

def hrqa_from_sequence(S,
                       K: Optional[int] = None,
                       r: int = 1,
                       a: Optional[float] = None,
                       B: int = 20,
                       Cv: Optional[np.ndarray] = None) -> dict:
    """Categorical sequence -> HRQA features. Thin wrapper around rhrqa()."""
    return rhrqa(S, K=K, r=r, a=a, B=B, Cv=Cv)


# ---------------------------------------------------------------------------
# Sliding-window RHRQA (batch)
# ---------------------------------------------------------------------------

def _rhrqa_window_one(S: np.ndarray, K: int, r: int, a: float, B: int,
                      Cv_full: Optional[np.ndarray],
                      start: int, win_size: int) -> np.ndarray:
    """Run RHRQA on one window. Returns flat (N_METRICS * K**r,) array."""
    end = start + win_size
    seg = S[start:end]
    Cv_seg = Cv_full[start:end] if Cv_full is not None else None
    if Cv_seg is not None:
        # Cv_seg was computed from the FULL sequence — its values reflect
        # history *before* the window. Rebuild a window-local Cv to match
        # MATLAB's per-window pipeline. (Set Cv=None to recompute fresh.)
        Cv_seg = None
    arr = rhrqa(seg, K=K, r=r, a=a, B=B, Cv=Cv_seg, return_array=True)
    flat = arr.ravel()
    flat[np.isnan(flat)] = 0.0
    return flat


def rhrqa_windowed(S,
                   K: Optional[int] = None,
                   *,
                   win_size: int = 200,
                   step_size: int = 100,
                   r: int = 1,
                   a: Optional[float] = None,
                   B: int = 20,
                   n_jobs: int = 1,
                   as_dataframe: bool = False,
                   ) -> Union[dict, pd.DataFrame]:
    """Sliding-window RHRQA over a long categorical sequence.

    Mirrors demo_c-3.m. With n_jobs > 1, windows are processed in parallel.

    Parameters
    ----------
    S : array-like of int
    win_size, step_size : int
    r : int, default 1
    n_jobs : int, default 1
        joblib worker count (-1 for all cores). Note: for very small windows
        the joblib overhead can dominate; benchmark on your data.
    as_dataframe : bool, default False
        If True, return a pandas DataFrame with WinID/StartIdx/EndIdx + named
        feature columns (matching demo_c-3.m output). Otherwise return a dict
        of NumPy arrays (faster).

    Returns
    -------
    dict or DataFrame
    """
    S = np.asarray(S, dtype=np.int64).ravel()
    N = S.size
    if K is None:
        K = int(S.max())
    if a is None:
        a = default_alpha(K)
    if win_size > N:
        raise ValueError(f"win_size={win_size} > N={N}")

    starts = np.arange(0, N - win_size + 1, step_size, dtype=np.int64)
    n_win = starts.size
    Kr = K ** r
    n_feats = N_METRICS * Kr  # 10 metrics: HRR..HIQR (paper Eqs. 10-19)

    # joblib overhead is ~1-5 ms per chunk; only worth it for large workloads.
    # Heuristic: parallelize when total work > ~50 ms estimated. Each window is
    # O(N_w * K^r) for the moments + O(K^r * <cluster_size>^2) for pdists, so
    # we approximate cost as n_win * win_size.
    parallelize = (n_jobs != 1) and (n_win * win_size > 200_000)
    if not parallelize:
        results = np.empty((n_win, n_feats), dtype=np.float64)
        for w, s0 in enumerate(starts):
            results[w] = _rhrqa_window_one(S, K, r, a, B, None, int(s0), win_size)
    else:
        # Chunk windows so each worker amortizes joblib overhead.
        eff_jobs = n_jobs if n_jobs > 0 else None
        n_chunks = max(1, (eff_jobs or 8) * 2)
        chunk_size = max(1, n_win // n_chunks)

        def _run_chunk(starts_chunk):
            local = np.empty((len(starts_chunk), n_feats), dtype=np.float64)
            for i, s0 in enumerate(starts_chunk):
                local[i] = _rhrqa_window_one(S, K, r, a, B, None,
                                              int(s0), win_size)
            return local

        chunks = [starts[i:i + chunk_size] for i in range(0, n_win, chunk_size)]
        results_list = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_run_chunk)(c) for c in chunks
        )
        results = np.vstack(results_list)

    end_idx = starts + win_size - 1

    if not as_dataframe:
        return dict(
            WinID=np.arange(1, n_win + 1),
            StartIdx=starts + 1,  # MATLAB 1-based for CSV parity
            EndIdx=end_idx + 1,
            features=results,
            K=K, r=r, win_size=win_size, step_size=step_size,
            IdxM=combn(K, r) if r > 0 else np.zeros((1, 0), dtype=np.int64),
        )

    metrics = list(METRIC_NAMES)
    feat_names = [f"{m}_{i+1}" for m in metrics for i in range(Kr)]
    df = pd.DataFrame(results, columns=feat_names)
    df.insert(0, "EndIdx", end_idx + 1)
    df.insert(0, "StartIdx", starts + 1)
    df.insert(0, "WinID", np.arange(1, n_win + 1))
    return df
