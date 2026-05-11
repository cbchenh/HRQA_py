"""Demo C — Optimal label ordering (LabelOpt).

Mirrors demo-26.m. Two cases:
  1) Tiny hand-crafted sequence with K=8 (brute force will fall back to
     q_learning since K=8 -> K! = 40320; we explicitly call q_learning here).
  2) Read demo_seq.csv if available, else generate.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from hrqa import optimal_label_order


def case1():
    print("=" * 60)
    print("Case 1: hand-crafted K=8 sequence")
    print("=" * 60)
    S = np.array([1, 4, 8, 2, 4, 8, 7, 8, 5, 6, 3, 2, 1, 1, 2, 4, 3, 5])
    K = 8
    alpha = 0.99 / (1.0 + np.sin(np.pi / K))  # matches MATLAB demo
    perm, reward = optimal_label_order(
        S, K=K, alpha=alpha, method="q_learning",
        max_episodes=5000, epsilon=0.01, learning_rate=0.05,
        patience=10000, tol=1e-4, show_iter=500, verbose=True,
    )
    print(f"\nOptimal permutation: {perm.tolist()}")
    print(f"Maximum reward (sum-of-distances): {reward:.4f}\n")


def case2():
    print("=" * 60)
    print("Case 2: from demo_seq.csv (or synthetic fallback)")
    print("=" * 60)
    here = Path(__file__).resolve().parent
    name = "demo_seq-27.csv"
    candidates = [
        Path(name),
        here / name,
        here.parent / name,
        here.parent.parent / name,
        Path("/home/user/workspace") / name,
    ]
    p = next((c for c in candidates if c.exists()), candidates[0])
    if p.exists():
        S = np.loadtxt(p, delimiter=",", dtype=np.int64).ravel()
        S = S[S > 0]
    else:
        rng = np.random.default_rng(0)
        S = rng.integers(1, 9, size=2000)
    K = int(S.max())
    alpha = 0.99 / (1.0 + np.sin(np.pi / K))
    perm, reward = optimal_label_order(
        S, K=K, alpha=alpha, method="q_learning",
        max_episodes=15000, epsilon=0.01, learning_rate=0.05,
        patience=5000, tol=1e-4, show_iter=1000, verbose=True,
    )
    print(f"\nOptimal permutation: {perm.tolist()}")
    print(f"Maximum reward: {reward:.4f}")


if __name__ == "__main__":
    case1()
    case2()
