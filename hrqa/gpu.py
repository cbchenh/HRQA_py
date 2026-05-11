"""Optional GPU backend (CuPy). Used only where it actually helps:

  1) Batched windowed RHRQA — process many windows simultaneously
  2) Q-learning LabelOpt with many parallel rollouts

For single small-window calls, the CPU path is faster (transfer + launch
overhead dominates). The CPU pipeline auto-falls back when CuPy is missing
or when workload size is below a heuristic threshold.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import cupy as cp  # type: ignore
    _HAS_CUPY = True
except ImportError:  # pragma: no cover
    cp = None
    _HAS_CUPY = False


def gpu_available() -> bool:
    """True if CuPy is importable AND a CUDA device is reachable."""
    if not _HAS_CUPY:
        return False
    try:
        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


def should_use_gpu(n_windows: int, win_size: int,
                   threshold_pts: int = 200_000) -> bool:
    """Heuristic: GPU pays off only above ~200k IFS points across the batch."""
    return gpu_available() and (n_windows * win_size) >= threshold_pts


# ---------------------------------------------------------------------------
# Batched IFS on GPU
# ---------------------------------------------------------------------------

def ifs_address_batched_gpu(S_batch: np.ndarray, K: int, a: float):
    """Compute IFS for many sequences of equal length on GPU.

    Parameters
    ----------
    S_batch : ndarray (B, N) int
        B sequences of length N (1-based labels).
    K, a : IFS params.

    Returns
    -------
    Cv : cupy.ndarray (B, N, 2)

    Note: the recurrence is sequential within each row; we parallelize across
    rows. For B = a few hundred this is a clean win over a CPU loop.
    """
    if not _HAS_CUPY:
        raise RuntimeError("CuPy not available")
    Sg = cp.asarray(S_batch, dtype=cp.int64)
    B, N = Sg.shape
    theta = (2.0 * cp.pi / K) * Sg
    ux = cp.cos(theta)
    uy = cp.sin(theta)
    # Sequential scan along axis=1 — write a tight Python loop on GPU.
    # CuPy doesn't expose lfilter directly; one batched scan is fine here.
    cx = cp.zeros((B, N), dtype=cp.float64)
    cy = cp.zeros((B, N), dtype=cp.float64)
    cx[:, 0] = ux[:, 0]
    cy[:, 0] = uy[:, 0]
    for t in range(1, N):
        cx[:, t] = a * cx[:, t - 1] + ux[:, t]
        cy[:, t] = a * cy[:, t - 1] + uy[:, t]
    return cp.stack([cx, cy], axis=-1)


# ---------------------------------------------------------------------------
# Batched windowed RHRQA on GPU (r=1 specialization — the common case)
# ---------------------------------------------------------------------------

def rhrqa_windowed_gpu(S: np.ndarray,
                       K: int,
                       win_size: int,
                       step_size: int,
                       a: float,
                       B: int = 20,
                       r: int = 1) -> np.ndarray:
    """GPU-accelerated sliding-window RHRQA. r = 1 only for now.

    Returns a (n_windows, 10 * K) NumPy array (transferred back to host).
    The 10 metrics per state follow the canonical row order:
    HRR, HMean, HVar, HSkew, HKurt, HEnt, HGini, HJSD, HMedian, HIQR.
    """
    if not _HAS_CUPY:
        raise RuntimeError("CuPy not available")
    if r != 1:
        raise NotImplementedError("GPU path currently supports r=1 only; "
                                  "use CPU path for higher orders.")
    S = np.asarray(S, dtype=np.int64).ravel()
    N = S.size
    starts = np.arange(0, N - win_size + 1, step_size, dtype=np.int64)
    n_win = starts.size
    # Build the (n_win, win_size) batch of windowed sequences
    S_batch = np.stack([S[s:s + win_size] for s in starts])  # (n_win, win_size)

    # IFS on GPU
    Cv = ifs_address_batched_gpu(S_batch, K, a)  # (n_win, win_size, 2)
    Sg = cp.asarray(S_batch)

    # 10 metrics per state per window (paper Eqs. 10-19):
    # HRR, HMean, HVar, HSkew, HKurt, HEnt, HGini, HJSD, HMedian, HIQR
    N_METRICS_GPU = 10
    n_feats = N_METRICS_GPU * K
    out = cp.zeros((n_win, n_feats), dtype=cp.float64)
    scale = (1.0 / a) ** r

    # Per-window, per-state cache of host-side distance arrays. We need them
    # in pass 2 to build the global (across-state) reference histogram for
    # HJSD, since q must share its bin support with each state's p_k.
    # win_size is bounded so memory is fine; B = 20 makes the bin support cheap.
    eps_jsd = 1e-12

    for w in range(n_win):
        # Per-window collection of (state_col, D_host) for active states
        active: list = []
        g_min = np.inf
        g_max = -np.inf

        for state in range(1, K + 1):
            mask_w = (Sg[w] == state)
            cnt = int(mask_w.sum().get())
            col = state - 1
            # HRR is always defined (counts / win_size)^2
            out[w, 0 * K + col] = (cnt / win_size) ** 2
            if cnt < 2:
                continue
            idx = cp.where(mask_w)[0]
            pts = Cv[w, idx]                       # (cnt, 2)
            diffs = pts[:, None, :] - pts[None, :, :]
            d = cp.sqrt((diffs ** 2).sum(-1))
            tri_idx = cp.triu_indices(cnt, k=1)
            D = scale * d[tri_idx]
            m = D.mean()
            v = ((D - m) ** 2).mean()
            out[w, 1 * K + col] = m
            out[w, 2 * K + col] = v
            v_host = float(v)
            if v_host > 0:
                out[w, 3 * K + col] = ((D - m) ** 3).mean() / v ** 1.5
                out[w, 4 * K + col] = ((D - m) ** 4).mean() / v ** 2
            # Pull D to host for histogram + quantile work (B = 20 is small,
            # transfer is cheap relative to the GPU-side distance build).
            D_host = cp.asnumpy(D)
            # Per-state histogram for HEnt / HGini (Eqs. 15, 16) — keeps the
            # per-state bin support as in the CPU/MATLAB convention.
            hist, _ = np.histogram(D_host, bins=B)
            p = hist[hist > 0].astype(np.float64)
            if p.size and v_host > 0:
                p /= p.sum()
                out[w, 5 * K + col] = -(p * np.log2(p)).sum()
                out[w, 6 * K + col] = 1.0 - (p * p).sum()
            # HMedian (Eq. 18), HIQR (Eq. 19) — always defined when cnt >= 2
            out[w, 8 * K + col] = float(np.median(D_host))
            out[w, 9 * K + col] = float(np.quantile(D_host, 0.75)
                                        - np.quantile(D_host, 0.25))
            # Track per-window global support for HJSD
            d_min = float(D_host.min())
            d_max = float(D_host.max())
            if d_min < g_min:
                g_min = d_min
            if d_max > g_max:
                g_max = d_max
            active.append((col, D_host))

        # ---- Pass 2 (HJSD, Eq. 17): bin every state's D onto the window's
        # global [g_min, g_max] support, sum to get the reference q, then
        # JSD per state. HJSD entries left at 0 (allocated) when degenerate.
        if active and g_max > g_min:
            edges = np.linspace(g_min, g_max, B + 1)
            cluster_counts = []
            total = np.zeros(B, dtype=np.int64)
            for col, D_host in active:
                counts, _ = np.histogram(D_host, bins=edges)
                cluster_counts.append((col, counts))
                total += counts
            tot_sum = total.sum()
            if tot_sum > 0:
                q_ref = total.astype(np.float64) / tot_sum
                for col, counts in cluster_counts:
                    p_sum = counts.sum()
                    if p_sum == 0:
                        continue
                    p_k = counts.astype(np.float64) / p_sum
                    m_kb = 0.5 * (p_k + q_ref)
                    t1 = 0.5 * np.sum(p_k * np.log2((p_k + eps_jsd) / (m_kb + eps_jsd)))
                    t2 = 0.5 * np.sum(q_ref * np.log2((q_ref + eps_jsd) / (m_kb + eps_jsd)))
                    out[w, 7 * K + col] = float(t1 + t2)
        # If g_max == g_min (all distances identical) HJSD = 0 for every
        # state, which is already the initial value.

    return cp.asnumpy(out)
