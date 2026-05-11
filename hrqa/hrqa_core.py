"""Core Regularized HRQA on a categorical integer sequence.

Vectorized port of RHRQA-13.m. Key speed-ups vs MATLAB:
  - r-grams via stride tricks (no Python/MATLAB loop)
  - group-by unique r-gram via np.unique(..., return_inverse=True), instead of
    O(U·N) ismember(...) per cluster
  - pdist in C (scipy)
  - histogram via np.histogram (C-level)
  - feature output is a contiguous (10, K**r) ndarray for tight inner loops

Output convention — ten quantifications, matching paper Eqs. (10)-(19):
    HRR, HMean, HVar, HSkew, HKurt, HEnt, HGini, HJSD, HMedian, HIQR
where IdxM enumerates all K**r r-grams in lexicographic order.

Implementation notes for the three new metrics:
  - HMedian (Eq. 18) and HIQR (Eq. 19) are quantile-based and computed
    directly from the per-state pairwise distance set D_k. They are well
    defined whenever the cluster has at least two points (otherwise NaN).
  - HJSD (Eq. 17) requires a *global* reference histogram q_b estimated by
    pooling within-state pairwise distances across all states. For JSD to
    be a valid divergence, p_{k,b} and q_b must share the SAME bin support;
    we therefore do a second pass per cluster, rebinning each D_k onto B
    equal-width bins spanning [min, max] of the pooled distances. HEnt and
    HGini continue to use per-state bins (preserving the existing
    RHRQA-13.m behavior) — the p_{k,b} used internally for HJSD is a
    separate computation from the one used by HEnt/HGini.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.spatial.distance import pdist

from .ifs import ifs_address
from .utils import combn, default_alpha, sliding_windows

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False


# ---------------------------------------------------------------------------
# Pass 1: per-cluster pdist + moments + per-state histogram + quantiles
# Returns 11 scalars (mean, var, skew, kurt, ent, gini, median, q25, q75,
# dmin, dmax). dmin/dmax are the per-cluster pairwise-distance min/max,
# tracked so the outer caller can build the global JSD support.
# ---------------------------------------------------------------------------

if _HAS_NUMBA:

    @njit(cache=True, fastmath=True)
    def _pdist_stats_quantiles(pts: np.ndarray, scale: float, B: int):
        """Single-pass pdist + 4 moments + per-state histogram + quantiles.

        Adds median, q25, q75 (via np.sort + linear interpolation, matching
        numpy's default quantile method) and the per-cluster min/max of D.
        The hot-path numerics (mean/var/skew/kurt and per-state ent/gini)
        are unchanged from the original RHRQA-13.m port.
        """
        n = pts.shape[0]
        if n < 2:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0)
        m_pairs = n * (n - 1) // 2
        D = np.empty(m_pairs, dtype=np.float64)
        idx = 0
        for i in range(n - 1):
            xi = pts[i, 0]
            yi = pts[i, 1]
            for j in range(i + 1, n):
                dx = xi - pts[j, 0]
                dy = yi - pts[j, 1]
                D[idx] = scale * np.sqrt(dx * dx + dy * dy)
                idx += 1
        # First pass: mean and min/max
        s = 0.0
        dmin = D[0]
        dmax = D[0]
        for k in range(m_pairs):
            v = D[k]
            s += v
            if v < dmin:
                dmin = v
            elif v > dmax:
                dmax = v
        mean = s / m_pairs
        # Second pass: central moments
        s2 = 0.0
        s3 = 0.0
        s4 = 0.0
        for k in range(m_pairs):
            d = D[k] - mean
            d2 = d * d
            s2 += d2
            s3 += d2 * d
            s4 += d2 * d2
        var = s2 / m_pairs
        if var > 0.0:
            std3 = var ** 1.5
            skew = (s3 / m_pairs) / std3
            kurt = (s4 / m_pairs) / (var * var)
        else:
            skew = 0.0
            kurt = 0.0
        # Per-state histogram on B equal-width bins over [dmin, dmax]
        ent = 0.0
        gini = 0.0
        if dmax > dmin:
            counts = np.zeros(B, dtype=np.int64)
            inv_w = B / (dmax - dmin)
            for k in range(m_pairs):
                bi = int((D[k] - dmin) * inv_w)
                if bi == B:
                    bi = B - 1
                counts[bi] += 1
            total = 0
            for k in range(B):
                total += counts[k]
            if total > 0:
                for k in range(B):
                    if counts[k] > 0:
                        p = counts[k] / total
                        ent -= p * np.log2(p)
                        gini += p * p
                gini = 1.0 - gini
        # Quantiles: sort D and linearly interpolate at positions
        #   h = alpha * (m_pairs - 1)  (numpy default "linear" method)
        D_sorted = np.sort(D)
        last = m_pairs - 1

        # median (alpha = 0.5)
        h = 0.5 * last
        lo = int(h)
        frac = h - lo
        if lo >= last:
            median = D_sorted[last]
        else:
            median = D_sorted[lo] + frac * (D_sorted[lo + 1] - D_sorted[lo])

        # q25 (alpha = 0.25)
        h = 0.25 * last
        lo = int(h)
        frac = h - lo
        if lo >= last:
            q25 = D_sorted[last]
        else:
            q25 = D_sorted[lo] + frac * (D_sorted[lo + 1] - D_sorted[lo])

        # q75 (alpha = 0.75)
        h = 0.75 * last
        lo = int(h)
        frac = h - lo
        if lo >= last:
            q75 = D_sorted[last]
        else:
            q75 = D_sorted[lo] + frac * (D_sorted[lo + 1] - D_sorted[lo])

        return (mean, var, skew, kurt, ent, gini,
                median, q25, q75, dmin, dmax)

    @njit(cache=True, fastmath=True)
    def _pdist_global_hist(pts: np.ndarray, scale: float,
                           lo: float, hi: float, B: int) -> np.ndarray:
        """Histogram of pdist(pts)*scale on B equal-width bins over [lo, hi].

        Second-pass kernel for HJSD: the global support [lo, hi] is computed
        across all clusters in pass 1, then each cluster's distances are
        re-binned here onto that common support. Right edge is inclusive in
        the last bin (matches np.histogram behavior).
        """
        n = pts.shape[0]
        counts = np.zeros(B, dtype=np.int64)
        if n < 2 or hi <= lo:
            return counts
        inv_w = B / (hi - lo)
        for i in range(n - 1):
            xi = pts[i, 0]
            yi = pts[i, 1]
            for j in range(i + 1, n):
                dx = xi - pts[j, 0]
                dy = yi - pts[j, 1]
                v = scale * np.sqrt(dx * dx + dy * dy)
                bi = int((v - lo) * inv_w)
                if bi < 0:
                    bi = 0
                elif bi >= B:
                    bi = B - 1
                counts[bi] += 1
        return counts

else:  # pragma: no cover

    def _pdist_stats_quantiles(pts, scale, B):
        if pts.shape[0] < 2:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0)
        D = scale * pdist(pts)
        m = D.mean()
        d = D - m
        v = (d * d).mean()
        if v > 0:
            std3 = v ** 1.5
            skew = (d * d * d).mean() / std3
            kurt = (d * d * d * d).mean() / (v * v)
        else:
            skew = 0.0
            kurt = 0.0
        hist, _ = np.histogram(D, bins=B)
        p = hist[hist > 0].astype(np.float64)
        if p.size:
            p /= p.sum()
            ent = float(-(p * np.log2(p)).sum())
            gini = float(1.0 - (p * p).sum())
        else:
            ent = 0.0
            gini = 0.0
        median = float(np.median(D))
        q25 = float(np.quantile(D, 0.25))
        q75 = float(np.quantile(D, 0.75))
        dmin = float(D.min())
        dmax = float(D.max())
        return (float(m), float(v), float(skew), float(kurt), ent, gini,
                median, q25, q75, dmin, dmax)

    def _pdist_global_hist(pts, scale, lo, hi, B):
        if pts.shape[0] < 2 or hi <= lo:
            return np.zeros(B, dtype=np.int64)
        D = scale * pdist(pts)
        edges = np.linspace(lo, hi, B + 1)
        counts, _ = np.histogram(D, bins=edges)
        return counts.astype(np.int64)


# Tuple of metric names in canonical (output-row) order. Keep this in sync
# with rhrqa()'s dict construction and return_array stacking, and with the
# windowed pipeline's feat_names generation.
METRIC_NAMES: Tuple[str, ...] = (
    "HRR", "HMean", "HVar", "HSkew", "HKurt",
    "HEnt", "HGini", "HJSD", "HMedian", "HIQR",
)
N_METRICS = len(METRIC_NAMES)  # 10


# JSD numerical-stability constant (matches the "epsilon > 0" in Eq. 17)
_JSD_EPS = 1e-12


def _compute_jsd(p_counts: np.ndarray, q: np.ndarray) -> float:
    """JSD(p_k || q) on a common B-bin support, base-2 log (bits).

    p_counts is integer counts; q is a probability vector. Returns 0 when
    p_counts has zero total (degenerate cluster).
    """
    p_sum = p_counts.sum()
    if p_sum == 0:
        return 0.0
    p = p_counts.astype(np.float64) / p_sum
    m = 0.5 * (p + q)
    # Eq. (17): both terms summed over bins; epsilon stabilizes empty bins
    term1 = 0.5 * np.sum(p * np.log2((p + _JSD_EPS) / (m + _JSD_EPS)))
    term2 = 0.5 * np.sum(q * np.log2((q + _JSD_EPS) / (m + _JSD_EPS)))
    return float(term1 + term2)


def _ravel_rgram(LC: np.ndarray, K: int) -> np.ndarray:
    """Encode each r-gram (row) as a single integer index in [0, K**r).

    Uses base-K positional encoding: i = sum_j (s_j - 1) * K**(r-1-j).
    Replaces unique(LC, 'rows') + ismember loops — O(N) instead of O(N·U).
    """
    LC = np.ascontiguousarray(LC) - 1  # 0-based
    r = LC.shape[1]
    weights = K ** np.arange(r - 1, -1, -1, dtype=np.int64)
    return (LC * weights).sum(axis=1)


def rhrqa(s,
          K: Optional[int] = None,
          r: int = 1,
          a: Optional[float] = None,
          B: int = 20,
          Cv: Optional[np.ndarray] = None,
          return_array: bool = False) -> dict:
    """Regularized Heterogeneous Recurrence Quantification Analysis.

    Parameters
    ----------
    s : array-like of int
        Categorical sequence (1-based labels in [1, K]).
    K : int, optional
        Number of states (default: max(s)).
    r : int, default 1
        Order of local cluster (length of r-grams). Output has K**r rows.
    a : float, optional
        IFS scalar (default: 0.9999*sin(pi/K)/(1+sin(pi/K))).
    B : int, default 20
        Number of histogram bins for entropy / Gini / JSD.
    Cv : ndarray (N, 2), optional
        Pre-computed IFS address (without the origin row). If provided, skip
        recomputation — matches HRQA2-8.m for sliding-window pipelines.
    return_array : bool, default False
        If True, return a single (10, K**r) NumPy array instead of a dict
        (faster — zero dict-construction overhead in tight loops).

    Returns
    -------
    out : dict or ndarray
        If dict: keys 'IdxM' (K**r, r), 'HRR','HMean','HVar','HSkew','HKurt',
        'HEnt','HGini','HJSD','HMedian','HIQR' (each shape (K**r,)).
        If return_array=True: ndarray of shape (10, K**r), rows in the order
        [HRR, HMean, HVar, HSkew, HKurt, HEnt, HGini, HJSD, HMedian, HIQR].
    """
    s = np.asarray(s, dtype=np.int64).ravel()
    if s.size == 0:
        raise ValueError("Input sequence s is empty.")
    if K is None:
        K = int(s.max())
    if K < int(s.max()):
        raise ValueError(f"K={K} < max(s)={int(s.max())}")
    if a is None:
        a = default_alpha(K)
    if r < 0:
        raise ValueError("r must be >= 0")

    # IFS address (without origin row): N x 2
    if Cv is None:
        Cv = ifs_address(s, K=K, a=a, include_origin=False)
    else:
        Cv = np.asarray(Cv, dtype=np.float64)
        if Cv.shape[0] == s.size + 1:
            # Caller passed origin-included version — drop the origin
            Cv = Cv[1:]
        if Cv.shape != (s.size, 2):
            raise ValueError(f"Cv shape {Cv.shape} incompatible with s ({s.size})")

    N = s.size
    scale = (1.0 / a) ** r  # regularization factor (vs plain HRQA)

    # ----- r = 0: global statistics on the entire sequence ------------------
    if r == 0:
        H_bar = np.bincount(s - 1, minlength=K).astype(np.float64)
        HRR_scalar = ((H_bar / N) ** 2).sum()
        D = scale * pdist(Cv)
        HMean = float(D.mean()) if D.size else np.nan
        HVar = float(((D - HMean) ** 2).mean()) if D.size else np.nan
        HSkew = float(((D - HMean) ** 3).mean() / HVar ** 1.5) if HVar > 0 else np.nan
        HKurt = float(((D - HMean) ** 4).mean() / HVar ** 2) if HVar > 0 else np.nan
        if D.size:
            hist, _ = np.histogram(D, bins=B)
            p = hist[hist > 0].astype(np.float64)
            if p.size:
                p /= p.sum()
            HEnt = float(-(p * np.log2(p)).sum()) if p.size else 0.0
            HGini = float(1.0 - (p * p).sum()) if p.size else 0.0
            HMedian = float(np.median(D))
            HIQR = float(np.quantile(D, 0.75) - np.quantile(D, 0.25))
        else:
            HEnt = HGini = 0.0
            HMedian = HIQR = np.nan
        # r=0 has a single cluster (the entire trajectory). The "global
        # reference" q is therefore that cluster's own histogram, so
        # JSD(p_k, q) = JSD(p, p) = 0 by definition.
        HJSD = 0.0

        out = dict(
            IdxM=np.zeros((1, 0), dtype=np.int64),
            HRR=np.array([HRR_scalar]),
            HMean=np.array([HMean]),
            HVar=np.array([HVar]),
            HSkew=np.array([HSkew]),
            HKurt=np.array([HKurt]),
            HEnt=np.array([HEnt]),
            HGini=np.array([HGini]),
            HJSD=np.array([HJSD]),
            HMedian=np.array([HMedian]),
            HIQR=np.array([HIQR]),
        )
        if return_array:
            return np.vstack([out[k] for k in METRIC_NAMES])
        return out

    # ----- r >= 1 -----------------------------------------------------------
    LC = sliding_windows(s, r)              # (N - r + 1, r)
    n_lc = LC.shape[0]
    Kr = K ** r

    # Encode each r-gram as a single integer in [0, Kr).
    keys = _ravel_rgram(LC, K)               # (n_lc,)
    # Sort once so positions of identical keys are contiguous — used to slice
    # per-cluster IFS rows without Python loops over keys.
    order = np.argsort(keys, kind="stable")
    keys_sorted = keys[order]
    # Position in original sequence for HRQA: i + (r-1) (MATLAB shifts by r-1)
    positions = order + (r - 1)              # (n_lc,)

    # Group boundaries
    uniq_keys, group_start = np.unique(keys_sorted, return_index=True)
    group_end = np.concatenate([group_start[1:], [n_lc]])
    H_bars = group_end - group_start         # cardinality of each group

    # Pre-allocate result arrays. NaN-fill the 9 distance-based metrics so
    # missing/degenerate clusters stay NaN by default; only HRR is dense.
    HRR = np.zeros(Kr, dtype=np.float64)
    HMean = np.full(Kr, np.nan, dtype=np.float64)
    HVar = np.full(Kr, np.nan, dtype=np.float64)
    HSkew = np.full(Kr, np.nan, dtype=np.float64)
    HKurt = np.full(Kr, np.nan, dtype=np.float64)
    HEnt = np.full(Kr, np.nan, dtype=np.float64)
    HGini = np.full(Kr, np.nan, dtype=np.float64)
    HJSD = np.full(Kr, np.nan, dtype=np.float64)
    HMedian = np.full(Kr, np.nan, dtype=np.float64)
    HIQR = np.full(Kr, np.nan, dtype=np.float64)

    HRR[uniq_keys] = (H_bars / N) ** 2

    # ---- Pass 1: per-cluster stats + quantiles + per-cluster (dmin, dmax)
    # We keep a small list of (key, idx_set) entries for clusters with
    # H_bar >= 2 so we can re-pdist them in pass 2 for the global histogram.
    # idx_set arrays are O(H_bar) so total storage is O(N) — negligible.
    Cv_c = np.ascontiguousarray(Cv)
    active_clusters: list = []   # (key, idx_set)
    g_min = np.inf
    g_max = -np.inf

    for g_idx in range(uniq_keys.size):
        key = int(uniq_keys[g_idx])
        H_bar = int(H_bars[g_idx])
        if H_bar < 2:
            continue
        idx_set = positions[group_start[g_idx]:group_end[g_idx]]
        pts = Cv_c[idx_set]
        (m, v, sk, kt, en, gi,
         med, q25, q75, dmin_k, dmax_k) = _pdist_stats_quantiles(pts, scale, B)
        HMean[key] = m
        HVar[key] = v
        # Median and IQR are well defined whenever H_bar >= 2 (even with
        # var == 0: median == constant, IQR == 0). Set unconditionally.
        HMedian[key] = med
        HIQR[key] = q75 - q25
        if v > 0.0:
            HSkew[key] = sk
            HKurt[key] = kt
            HEnt[key] = en
            HGini[key] = gi
        # Track support for the global JSD bin scheme
        if dmin_k < g_min:
            g_min = dmin_k
        if dmax_k > g_max:
            g_max = dmax_k
        active_clusters.append((key, idx_set))

    # ---- Pass 2 (HJSD): bin every cluster's distances on the global support
    if active_clusters:
        if g_max > g_min:
            # Get integer counts per cluster on the shared [g_min, g_max] grid
            cluster_counts = []
            for key, idx_set in active_clusters:
                pts = Cv_c[idx_set]
                counts = _pdist_global_hist(pts, scale, g_min, g_max, B)
                cluster_counts.append((key, counts))
            # Global reference q (Eq. 17 — pool counts across all states)
            total = np.zeros(B, dtype=np.int64)
            for _, c in cluster_counts:
                total += c
            tot_sum = total.sum()
            if tot_sum > 0:
                q_ref = total.astype(np.float64) / tot_sum
                for key, counts in cluster_counts:
                    HJSD[key] = _compute_jsd(counts, q_ref)
            else:
                # No within-state pairs anywhere (shouldn't happen since we
                # filtered H_bar < 2, but guard for completeness)
                for key, _ in cluster_counts:
                    HJSD[key] = 0.0
        else:
            # All within-state distances collapse to a single value across
            # ALL clusters -> q has all mass in one bin, p_k likewise -> JSD=0
            for key, _ in active_clusters:
                HJSD[key] = 0.0

    if return_array:
        return np.vstack([HRR, HMean, HVar, HSkew, HKurt, HEnt, HGini,
                          HJSD, HMedian, HIQR])

    return dict(
        IdxM=combn(K, r),
        HRR=HRR, HMean=HMean, HVar=HVar, HSkew=HSkew,
        HKurt=HKurt, HEnt=HEnt, HGini=HGini,
        HJSD=HJSD, HMedian=HMedian, HIQR=HIQR,
    )


def fhrqa(s, K: Optional[int] = None, R: int = 3,
          a: Optional[float] = None, B: int = 20,
          Cv: Optional[np.ndarray] = None) -> np.ndarray:
    """Multi-order regularized HRQA: stack features for r = 1..R.

    Returns a flat 1D feature vector of length 10 * (K + K**2 + ... + K**R).
    Order matches FHRQA-4.m: for each r, [HRR..HIQR] across all r-grams,
    then the next r is appended.
    """
    s = np.asarray(s, dtype=np.int64).ravel()
    if K is None:
        K = int(s.max())
    if a is None:
        a = default_alpha(K)
    # Compute Cv once and reuse across all r values.
    if Cv is None:
        Cv = ifs_address(s, K=K, a=a, include_origin=False)
    chunks = []
    for r in range(1, R + 1):
        arr = rhrqa(s, K=K, r=r, a=a, B=B, Cv=Cv, return_array=True)
        # Row order [HRR, HMean, HVar, HSkew, HKurt, HEnt, HGini,
        #            HJSD, HMedian, HIQR], then flatten metric-by-metric.
        chunks.append(arr.ravel())
    return np.concatenate(chunks)
