"""Demo B — Categorical sequence pipeline (sliding-window RHRQA).

Mirrors demo_c-3.m: generate (or load) a 4-state Markov sequence, run windowed
RHRQA with win=200, step=100, r=1, save features + visualize.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hrqa import rhrqa_windowed


def make_synthetic_sequence(N: int = 2000, seed: int = 42) -> np.ndarray:
    """Two-regime 4-state Markov chain (matches demo_c-3.m)."""
    rng = np.random.default_rng(seed)
    K = 4
    half = N // 2
    PA = np.array([
        [0.45, 0.35, 0.10, 0.10],
        [0.35, 0.40, 0.15, 0.10],
        [0.15, 0.15, 0.40, 0.30],
        [0.10, 0.10, 0.30, 0.50],
    ])
    PB = np.array([
        [0.10, 0.10, 0.40, 0.40],
        [0.10, 0.15, 0.35, 0.40],
        [0.40, 0.35, 0.15, 0.10],
        [0.40, 0.30, 0.20, 0.10],
    ])
    S = np.zeros(N, dtype=np.int64)
    S[0] = rng.integers(1, K + 1)
    for i in range(1, half):
        S[i] = 1 + np.searchsorted(np.cumsum(PA[S[i-1]-1]), rng.random())
    for i in range(half, N):
        S[i] = 1 + np.searchsorted(np.cumsum(PB[S[i-1]-1]), rng.random())
    return S


def main(out_csv: str = "demo_c_statistics_py.csv",
         win_size: int = 200, step_size: int = 100, r: int = 1):
    here = Path(__file__).resolve().parent

    print("[1] Generating synthetic 4-state Markov sequence")
    S = make_synthetic_sequence(N=2000)
    K = int(S.max())
    print(f"    N = {S.size}, K = {K}")

    # Plot the raw symbolic sequence + state histogram
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(S, ".", markersize=2)
    axes[0].set_title(f"Categorical sequence  (N={S.size}, K={K})")
    axes[0].set_xlabel("sample index")
    axes[0].set_ylabel("state")
    axes[0].set_ylim(0, K + 1)
    axes[1].hist(S, bins=np.arange(1, K + 2) - 0.5, color="#338866", edgecolor="white")
    axes[1].set_title("State histogram")
    axes[1].set_xticks(range(1, K + 1))
    fig.savefig(here / "demo_sequence_input.png", dpi=120)
    plt.close(fig)

    # Sliding-window RHRQA
    print(f"[2] Sliding-window RHRQA  (win={win_size}, step={step_size}, r={r})")
    t0 = time.perf_counter()
    df = rhrqa_windowed(S, K=K, win_size=win_size, step_size=step_size,
                         r=r, n_jobs=1, as_dataframe=True)
    elapsed = time.perf_counter() - t0
    print(f"    {len(df)} windows  in  {elapsed*1e3:.2f} ms")

    out_path = here / out_csv
    df.to_csv(out_path, index=False)
    print(f"[3] Saved features  ->  {out_path}  ({df.shape})")

    # Visualize
    Kr = K ** r
    feature_cols = [c for c in df.columns if "_" in c]
    arr = df[feature_cols].values
    win_centers = (df["StartIdx"].values + df["EndIdx"].values) / 2

    HRR_mat = arr[:, 0:Kr]
    HMean_mat = arr[:, Kr:2 * Kr]
    HEnt_mat = arr[:, 5 * Kr:6 * Kr]
    # New (paper Eqs. 17-19) — appended at the tail of the feature block
    HJSD_mat = arr[:, 7 * Kr:8 * Kr]
    HMedian_mat = arr[:, 8 * Kr:9 * Kr]
    HIQR_mat = arr[:, 9 * Kr:10 * Kr]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    im0 = axes[0].imshow(HRR_mat.T, aspect="auto", cmap="hot", origin="lower")
    axes[0].set_title("HRR per state per window")
    axes[0].set_xlabel("window index")
    axes[0].set_ylabel("state")
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(HEnt_mat.T, aspect="auto", cmap="viridis", origin="lower")
    axes[1].set_title("HEnt per state per window")
    axes[1].set_xlabel("window index")
    axes[1].set_ylabel("state")
    plt.colorbar(im1, ax=axes[1])
    fig.savefig(here / "demo_sequence_heatmaps.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].plot(win_centers, HRR_mat.mean(1), "b-o", markersize=3)
    axes[0].set_title("Mean HRR across all states")
    axes[0].grid(True)
    axes[1].plot(win_centers, HMean_mat.mean(1), "r-o", markersize=3)
    axes[1].set_title("Mean HMean across all states")
    axes[1].grid(True)
    axes[2].plot(win_centers, HEnt_mat.mean(1), "g-o", markersize=3)
    axes[2].set_title("Mean HEnt across all states")
    axes[2].grid(True)
    for ax in axes:
        ax.set_xlabel("window centre (sample)")
    fig.savefig(here / "demo_sequence_aggregate.png", dpi=120)
    plt.close(fig)

    # Showcase the new distributional quantifications (Eqs. 17-19)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].plot(win_centers, np.nanmean(HJSD_mat, axis=1), "m-o", markersize=3)
    axes[0].set_title("Mean HJSD across all states  (Eq. 17)")
    axes[0].grid(True)
    axes[1].plot(win_centers, np.nanmean(HMedian_mat, axis=1), "c-o", markersize=3)
    axes[1].set_title("Mean HMedian across all states  (Eq. 18)")
    axes[1].grid(True)
    axes[2].plot(win_centers, np.nanmean(HIQR_mat, axis=1), "y-o", markersize=3)
    axes[2].set_title("Mean HIQR across all states  (Eq. 19)")
    axes[2].grid(True)
    for ax in axes:
        ax.set_xlabel("window centre (sample)")
    fig.savefig(here / "demo_sequence_new_metrics.png", dpi=120)
    plt.close(fig)

    print("[4] Done.")
    return df


if __name__ == "__main__":
    main()
