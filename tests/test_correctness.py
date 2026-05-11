"""Numerical-equivalence tests against a literal MATLAB-style RHRQA port."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import pdist

from hrqa import (
    rhrqa, hrqa_from_signal, hrqa_from_sequence, rhrqa_windowed,
    KMeansSegmenter, HASSegmenter, optimal_label_order, ifs_address,
)
from hrqa.hrqa_core import METRIC_NAMES, N_METRICS
from hrqa.utils import default_alpha, combn, sliding_windows


# ---------------------------------------------------------------------------
# MATLAB-style reference implementation (used only for testing)
# Returns one row per IdxM, with N_METRICS = 10 columns in canonical order:
# [HRR, HMean, HVar, HSkew, HKurt, HEnt, HGini, HJSD, HMedian, HIQR]
# ---------------------------------------------------------------------------

def rhrqa_reference(s, K, r=1, B=20):
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
    # Pass 1: compute the 7 original metrics + median/IQR, and stash the
    # raw distance arrays so we can build the global JSD reference on the
    # exact same pooled support.
    UC_Res = np.full((len(UC), N_METRICS), np.nan)
    D_per_cluster: list = [None] * len(UC)
    for i in range(len(UC)):
        idx_set = np.where((LC == UC[i]).all(axis=1))[0] + (r - 1)
        H_bar = len(idx_set)
        UC_Res[i, 0] = (H_bar / N) ** 2
        if H_bar > 1:
            D = scale * pdist(Cv[idx_set])
            UC_Res[i, 1] = D.mean()
            UC_Res[i, 2] = ((D - UC_Res[i, 1]) ** 2).sum() / (H_bar * (H_bar - 1) / 2)
            if UC_Res[i, 2] > 0:
                UC_Res[i, 3] = ((D - UC_Res[i, 1]) ** 3).sum() / (H_bar * (H_bar - 1) / 2) / UC_Res[i, 2] ** 1.5
                UC_Res[i, 4] = ((D - UC_Res[i, 1]) ** 4).sum() / (H_bar * (H_bar - 1) / 2) / UC_Res[i, 2] ** 2
                hist, _ = np.histogram(D, bins=B)
                p = hist[hist > 0].astype(float)
                p = p / p.sum()
                UC_Res[i, 5] = -(p * np.log2(p)).sum()
                UC_Res[i, 6] = 1 - (p * p).sum()
            # HMedian (Eq. 18), HIQR (Eq. 19) — well defined for any H_bar >= 2
            UC_Res[i, 8] = np.median(D)
            UC_Res[i, 9] = np.quantile(D, 0.75) - np.quantile(D, 0.25)
            D_per_cluster[i] = D
    # Pass 2: HJSD on shared global support. Pool all D arrays, build edges,
    # rebin each cluster onto those edges, derive q from the total, and
    # compute the Jensen-Shannon divergence per cluster (Eq. 17, base 2).
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
            # All distances identical across all clusters -> JSD = 0
            for i in valid:
                UC_Res[i, 7] = 0.0
    M_Res = np.full((len(IdxM), N_METRICS), np.nan)
    for i in range(len(UC)):
        ins = np.where((IdxM == UC[i]).all(axis=1))[0]
        if len(ins):
            M_Res[ins[0]] = UC_Res[i]
    M_Res[np.isnan(M_Res[:, 0]), 0] = 0
    return M_Res


# ---------------------------------------------------------------------------
# IFS recurrence
# ---------------------------------------------------------------------------

def test_ifs_matches_naive_loop():
    rng = np.random.default_rng(0)
    s = rng.integers(1, 7, size=300)
    K = 6
    a = default_alpha(K)
    Cv_ref = np.zeros((s.size + 1, 2))
    for i in range(s.size):
        Cv_ref[i+1, 0] = a * Cv_ref[i, 0] + np.cos(2*np.pi*s[i]/K)
        Cv_ref[i+1, 1] = a * Cv_ref[i, 1] + np.sin(2*np.pi*s[i]/K)
    Cv = ifs_address(s, K=K)
    np.testing.assert_allclose(Cv, Cv_ref, atol=1e-12)


# ---------------------------------------------------------------------------
# RHRQA equivalence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("K,N,r", [
    (4, 500, 1), (4, 500, 2), (8, 2000, 1), (8, 1000, 2),
    (16, 3000, 1),
])
def test_rhrqa_matches_reference(K, N, r):
    rng = np.random.default_rng(K + N + r)
    S = rng.integers(1, K + 1, size=N)
    out = rhrqa(S, K=K, r=r)
    ref = rhrqa_reference(S, K=K, r=r)
    py = np.column_stack([out[m] for m in METRIC_NAMES])
    py_n = np.nan_to_num(py)
    ref_n = np.nan_to_num(ref)
    np.testing.assert_allclose(py_n, ref_n, atol=1e-10, rtol=1e-9)


def test_rhrqa_r0_returns_scalar():
    rng = np.random.default_rng(0)
    S = rng.integers(1, 5, size=500)
    out = rhrqa(S, K=4, r=0)
    assert out["HRR"].size == 1
    expected = sum((np.sum(S == i) / 500) ** 2 for i in range(1, 5))
    assert np.isclose(out["HRR"][0], expected)


def test_rhrqa_hrr_sums_to_marginal_recurrence():
    rng = np.random.default_rng(0)
    S = rng.integers(1, 6, size=2000)
    K = 5
    out = rhrqa(S, K=K, r=1)
    expected = sum((np.sum(S == i) / 2000) ** 2 for i in range(1, K + 1))
    assert np.isclose(out["HRR"].sum(), expected, atol=1e-12)


def test_rhrqa_with_external_Cv():
    rng = np.random.default_rng(0)
    S = rng.integers(1, 5, size=1000)
    K = 4
    Cv = ifs_address(S, K=K, include_origin=False)
    out_a = rhrqa(S, K=K, r=1)
    out_b = rhrqa(S, K=K, r=1, Cv=Cv)
    for k in METRIC_NAMES:
        np.testing.assert_allclose(np.nan_to_num(out_a[k]),
                                   np.nan_to_num(out_b[k]),
                                   atol=1e-12)


# ---------------------------------------------------------------------------
# Distribution-shape quantifications (Eqs. 17-19): HJSD, HMedian, HIQR
# ---------------------------------------------------------------------------

def test_hmedian_hiqr_well_defined():
    """HMedian/HIQR (Eqs. 18-19) must exist whenever a cluster has >=2
    points (i.e. its distance set D_k is non-empty), and must be consistent
    with the underlying pairwise-distance set: min(D_k) <= median <= max(D_k)
    and IQR >= 0."""
    rng = np.random.default_rng(7)
    S = rng.integers(1, 5, size=1500)
    K = 4
    r = 1
    out = rhrqa(S, K=K, r=r)
    a = default_alpha(K)
    Cv = ifs_address(S, K, a, include_origin=True)[1:]
    scale = (1.0 / a) ** r  # same scaling rhrqa applies internally
    for k in range(1, K + 1):
        idx = np.where(S == k)[0]
        if idx.size < 2:
            continue
        D = scale * pdist(Cv[idx])
        # cluster index in IdxM (r=1) corresponds to row k-1
        med_py = out["HMedian"][k - 1]
        iqr_py = out["HIQR"][k - 1]
        assert np.isfinite(med_py)
        assert np.isfinite(iqr_py)
        assert D.min() - 1e-12 <= med_py <= D.max() + 1e-12
        assert iqr_py >= -1e-12
        # Compare to numpy's reference quantiles directly
        np.testing.assert_allclose(med_py, np.median(D), atol=1e-12)
        np.testing.assert_allclose(
            iqr_py,
            np.quantile(D, 0.75) - np.quantile(D, 0.25),
            atol=1e-12,
        )


def test_hjsd_nonneg_bounded():
    """HJSD (Eq. 17) is a Jensen-Shannon divergence in base-2 nats, so it
    is in [0, 1] for every cluster (up to numerical noise)."""
    rng = np.random.default_rng(11)
    S = rng.integers(1, 6, size=2000)
    K = 5
    out = rhrqa(S, K=K, r=1)
    jsd = out["HJSD"]
    finite = jsd[np.isfinite(jsd)]
    assert finite.size > 0
    assert (finite >= -1e-10).all()
    assert (finite <= 1.0 + 1e-10).all()


def test_hjsd_zero_for_r0():
    """At r=0 the algorithm operates on a single global cluster, so the
    reference histogram q equals the cluster histogram p, making JSD = 0
    by construction (Eq. 17)."""
    rng = np.random.default_rng(13)
    S = rng.integers(1, 5, size=800)
    out = rhrqa(S, K=4, r=0)
    assert out["HJSD"].size == 1
    assert np.isclose(out["HJSD"][0], 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def test_kmeans_segmenter_roundtrip():
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal(0, 1, size=(200, 3)),
        rng.normal(5, 1, size=(200, 3)),
        rng.normal(-5, 1, size=(200, 3)),
    ])
    seg = KMeansSegmenter(K=3).fit(X)
    S = seg.transform(X)
    assert S.min() >= 1 and S.max() <= 3
    assert S.size == X.shape[0]


def test_has_segmenter_partitions_data():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(2000, 3))
    seg = HASSegmenter(capacity=500).fit(X)
    S = seg.transform(X)
    assert S.min() >= 1 and S.max() <= seg.K
    # Every point should be assigned to exactly one cell
    assert set(np.unique(S)).issubset(set(range(1, seg.K + 1)))


def test_label_permutation_applies():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(500, 2))
    seg = KMeansSegmenter(K=4).fit(X)
    S0 = seg.transform(X)
    perm = np.array([3, 1, 4, 2])  # arbitrary
    seg.set_label_permutation(perm)
    S1 = seg.transform(X)
    expected = perm[S0 - 1]
    np.testing.assert_array_equal(S1, expected)


def test_pipeline_A_runs():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(500, 3))
    out = hrqa_from_signal(X, method="kmeans", K=4, r=1)
    assert out["HRR"].size == 4
    assert "S" in out
    assert "segmenter" in out


def test_pipeline_B_runs():
    rng = np.random.default_rng(0)
    S = rng.integers(1, 5, size=500)
    out = hrqa_from_sequence(S, K=4, r=1)
    assert out["HRR"].size == 4


def test_windowed_shape():
    rng = np.random.default_rng(0)
    S = rng.integers(1, 5, size=1000)
    df = rhrqa_windowed(S, K=4, win_size=200, step_size=100,
                          r=1, as_dataframe=True)
    assert "WinID" in df.columns
    assert "HRR_1" in df.columns
    assert "HEnt_4" in df.columns
    # New quantifications must also appear as columns
    assert "HJSD_1" in df.columns
    assert "HMedian_2" in df.columns
    assert "HIQR_4" in df.columns
    # Expected number of windows: floor((1000 - 200) / 100) + 1 = 9
    assert len(df) == 9


# ---------------------------------------------------------------------------
# LabelOpt
# ---------------------------------------------------------------------------

def test_label_opt_brute_force_returns_valid_perm():
    rng = np.random.default_rng(0)
    S = rng.integers(1, 5, size=300)
    perm, reward = optimal_label_order(S, K=4, method="brute_force")
    assert perm.shape == (4,)
    assert sorted(perm.tolist()) == [1, 2, 3, 4]
    assert np.isfinite(reward)


def test_label_opt_q_learning_runs():
    rng = np.random.default_rng(0)
    S = rng.integers(1, 9, size=200)
    perm, reward = optimal_label_order(S, K=8, method="q_learning",
                                        max_episodes=200, patience=500,
                                        seed=0)
    assert sorted(perm.tolist()) == list(range(1, 9))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
