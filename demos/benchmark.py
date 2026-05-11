"""Benchmark suite: Python vs MATLAB-style reference, single-call + windowed.

Run with:
    python demos/benchmark.py
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from hrqa import rhrqa, rhrqa_windowed
from hrqa.utils import default_alpha, combn
from hrqa.ifs import ifs_address


def rhrqa_reference(s, K, r=1, B=20):
    """Literal MATLAB-style port of RHRQA-13.m, extended with the three
    distribution-shape quantifications HJSD (Eq. 17), HMedian (Eq. 18) and
    HIQR (Eq. 19) so the benchmark times the same workload as the fast core.
    """
    s = np.asarray(s, dtype=np.int64).ravel()
    a = default_alpha(K)
    Cv = ifs_address(s, K, a, include_origin=True)[1:]
    scale = (1.0 / a) ** r
    IdxM = combn(K, r)
    N = s.size
    LC = np.zeros((N - r + 1, r), dtype=np.int64)
    for i in range(r):
        LC[:, i] = s[i:N - r + i + 1]
    UC = np.unique(LC, axis=0)
    N_METRICS = 10
    UC_Res = np.full((len(UC), N_METRICS), np.nan)
    D_per_cluster: list = [None] * len(UC)
    for i in range(len(UC)):
        idx_set = np.where((LC == UC[i]).all(axis=1))[0] + (r - 1)
        H_bar = len(idx_set)
        UC_Res[i, 0] = (H_bar / N) ** 2
        if H_bar > 1:
            LC_D = scale * pdist(Cv[idx_set])
            UC_Res[i, 1] = LC_D.mean()
            UC_Res[i, 2] = ((LC_D - UC_Res[i, 1]) ** 2).sum() / (H_bar * (H_bar - 1) / 2)
            if UC_Res[i, 2] > 0:
                UC_Res[i, 3] = ((LC_D - UC_Res[i, 1]) ** 3).sum() / (H_bar * (H_bar - 1) / 2) / UC_Res[i, 2] ** 1.5
                UC_Res[i, 4] = ((LC_D - UC_Res[i, 1]) ** 4).sum() / (H_bar * (H_bar - 1) / 2) / UC_Res[i, 2] ** 2
                hist, _ = np.histogram(LC_D, bins=B)
                p = hist[hist > 0].astype(float)
                p = p / p.sum()
                UC_Res[i, 5] = -(p * np.log2(p)).sum()
                UC_Res[i, 6] = 1 - (p * p).sum()
            # HMedian (Eq. 18), HIQR (Eq. 19)
            UC_Res[i, 8] = np.median(LC_D)
            UC_Res[i, 9] = np.quantile(LC_D, 0.75) - np.quantile(LC_D, 0.25)
            D_per_cluster[i] = LC_D
    # HJSD (Eq. 17): build global reference q by pooling distances across clusters.
    eps_jsd = 1e-12
    valid = [i for i in range(len(UC)) if D_per_cluster[i] is not None]
    if valid:
        pooled = np.concatenate([D_per_cluster[i] for i in valid])
        g_min, g_max = float(pooled.min()), float(pooled.max())
        if g_max > g_min:
            edges = np.linspace(g_min, g_max, B + 1)
            cluster_counts = {}
            total = np.zeros(B, dtype=np.int64)
            for i in valid:
                c, _ = np.histogram(D_per_cluster[i], bins=edges)
                cluster_counts[i] = c
                total += c
            tot_sum = total.sum()
            if tot_sum > 0:
                q = total.astype(float) / tot_sum
                for i in valid:
                    c = cluster_counts[i]
                    if c.sum() == 0:
                        UC_Res[i, 7] = 0.0
                        continue
                    p_k = c.astype(float) / c.sum()
                    m_kb = 0.5 * (p_k + q)
                    t1 = 0.5 * np.sum(p_k * np.log2((p_k + eps_jsd) / (m_kb + eps_jsd)))
                    t2 = 0.5 * np.sum(q * np.log2((q + eps_jsd) / (m_kb + eps_jsd)))
                    UC_Res[i, 7] = t1 + t2
        else:
            for i in valid:
                UC_Res[i, 7] = 0.0
    M_Res = np.full((len(IdxM), N_METRICS), np.nan)
    for i in range(len(UC)):
        ins = np.where((IdxM == UC[i]).all(axis=1))[0]
        if len(ins):
            M_Res[ins[0]] = UC_Res[i]
    M_Res[np.isnan(M_Res[:, 0]), 0] = 0
    return M_Res


def bench_single():
    print("\n" + "=" * 78)
    print("Single-call RHRQA: Python (Numba) vs MATLAB-style reference")
    print("=" * 78)
    print(f"{'K':>3} {'N':>6} {'r':>2} | {'fast (ms)':>10} {'ref (ms)':>10} {'speedup':>9}")
    print("-" * 78)
    rows = []
    # Warmup JIT
    rhrqa(np.random.randint(1, 5, 200), K=4, r=1)
    rhrqa(np.random.randint(1, 5, 200), K=4, r=2)
    cases = [
        (4, 500, 1), (8, 2000, 1), (8, 2000, 2),
        (8, 5000, 1), (16, 5000, 1), (8, 10000, 1), (16, 20000, 1),
    ]
    for K, N, r in cases:
        np.random.seed(0)
        S = np.random.randint(1, K + 1, size=N)
        t0 = time.perf_counter()
        for _ in range(20):
            rhrqa(S, K=K, r=r)
        t_fast = (time.perf_counter() - t0) / 20
        t0 = time.perf_counter()
        for _ in range(3):
            rhrqa_reference(S, K=K, r=r)
        t_ref = (time.perf_counter() - t0) / 3
        print(f"{K:>3d} {N:>6d} {r:>2d} | {t_fast*1e3:>10.2f} {t_ref*1e3:>10.2f} {t_ref/t_fast:>8.1f}x")
        rows.append(dict(K=K, N=N, r=r, fast_ms=t_fast*1e3, ref_ms=t_ref*1e3,
                         speedup=t_ref/t_fast))
    return pd.DataFrame(rows)


def bench_windowed():
    print("\n" + "=" * 78)
    print("Sliding-window RHRQA")
    print("=" * 78)
    print(f"{'N':>6} {'win':>4} {'step':>4} {'K':>3} {'r':>2} {'n_jobs':>7} | {'time (ms)':>10}")
    print("-" * 78)
    rows = []
    for N, win, step, K, r in [
        (10000, 200, 100, 4, 1),
        (50000, 200, 100, 4, 1),
        (50000, 500, 250, 8, 1),
        (100000, 1000, 500, 8, 2),
    ]:
        np.random.seed(0)
        S = np.random.randint(1, K + 1, size=N)
        for n_jobs in [1, -1]:
            # warmup
            rhrqa_windowed(S, K=K, win_size=win, step_size=step, r=r,
                            n_jobs=n_jobs)
            t0 = time.perf_counter()
            df = rhrqa_windowed(S, K=K, win_size=win, step_size=step, r=r,
                                  n_jobs=n_jobs, as_dataframe=False)
            t = time.perf_counter() - t0
            print(f"{N:>6d} {win:>4d} {step:>4d} {K:>3d} {r:>2d} {str(n_jobs):>7s} |"
                  f" {t*1e3:>10.2f}  ({len(df['WinID'])} windows)")
            rows.append(dict(N=N, win=win, step=step, K=K, r=r,
                             n_jobs=n_jobs, time_ms=t*1e3,
                             n_windows=len(df['WinID'])))
    return pd.DataFrame(rows)


def main():
    df1 = bench_single()
    df2 = bench_windowed()
    print("\n" + "=" * 78)
    print("Summary")
    print("=" * 78)
    print(f"Mean speedup over MATLAB-style reference: {df1['speedup'].mean():.1f}x")
    print(f"Max speedup:  {df1['speedup'].max():.1f}x")
    print(f"Min speedup:  {df1['speedup'].min():.1f}x")


if __name__ == "__main__":
    main()
