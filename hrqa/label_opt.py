"""LabelOpt — find the optimal permutation of state labels for IFS embedding.

Problem
-------
Given a categorical sequence S over states {1..K}, the IFS embedding places
each label on the unit circle at angle 2*pi*i/K. The *order* in which labels
are placed around the circle dramatically affects the spread of the IFS
points (and therefore the discriminative power of HRQA features).

We search for the permutation perm such that, when applied as
    S' = perm(S),
the resulting IFS points have **maximum total pairwise distance** (sum of
pdist).

Two methods, auto-dispatched by K:
  - brute_force : enumerate all K! permutations exactly. Best for K <= 7.
  - q_learning  : tabular Q-learning over partial-permutation states.
                  Mirrors q_learning_label_assignment-28.m. Best for K > 7.

The brute-force path is parallelized with joblib; the Q-learning episode loop
uses Numba JIT for the IFS rollout + reward computation (the hot path).
"""
from __future__ import annotations

from itertools import permutations
from typing import Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist

from .utils import default_alpha

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False


# ---------------------------------------------------------------------------
# Reward computation: IFS rollout under a permutation, then sum-of-distances
# ---------------------------------------------------------------------------

if _HAS_NUMBA:

    @njit(cache=True, fastmath=True)
    def _ifs_points_under_perm(S: np.ndarray, perm: np.ndarray,
                               alpha: float, K: int) -> np.ndarray:
        n = S.shape[0]
        pts = np.empty((n, 2))
        cx = 0.0
        cy = 0.0
        two_pi_over_K = 2.0 * np.pi / K
        for t in range(n):
            label = perm[S[t] - 1]  # 1-based input -> 0-based index
            theta = two_pi_over_K * label
            cx = alpha * cx + np.cos(theta)
            cy = alpha * cy + np.sin(theta)
            pts[t, 0] = cx
            pts[t, 1] = cy
        return pts

    @njit(cache=True, fastmath=True)
    def _sum_pairwise_distances(pts: np.ndarray) -> float:
        """Sum of all pairwise Euclidean distances. O(n^2) but JITed."""
        n = pts.shape[0]
        total = 0.0
        for i in range(n - 1):
            xi = pts[i, 0]
            yi = pts[i, 1]
            for j in range(i + 1, n):
                dx = xi - pts[j, 0]
                dy = yi - pts[j, 1]
                total += np.sqrt(dx * dx + dy * dy)
        return total

    @njit(cache=True, fastmath=True)
    def _reward_under_perm(S: np.ndarray, perm: np.ndarray,
                           alpha: float, K: int) -> float:
        pts = _ifs_points_under_perm(S, perm, alpha, K)
        return _sum_pairwise_distances(pts)

else:

    def _ifs_points_under_perm(S, perm, alpha, K):
        S = np.asarray(S, dtype=np.int64)
        labels = perm[S - 1]
        theta = (2.0 * np.pi / K) * labels
        ux = np.cos(theta)
        uy = np.sin(theta)
        # Recurrence via lfilter (vectorized)
        from scipy.signal import lfilter
        cx = lfilter([1.0], [1.0, -alpha], ux)
        cy = lfilter([1.0], [1.0, -alpha], uy)
        return np.column_stack([cx, cy])

    def _reward_under_perm(S, perm, alpha, K):
        pts = _ifs_points_under_perm(S, perm, alpha, K)
        # For modest n, scipy pdist is faster than the python double-loop
        return float(pdist(pts).sum())


# ---------------------------------------------------------------------------
# Brute force
# ---------------------------------------------------------------------------

def _brute_force_search(S: np.ndarray, K: int, alpha: float,
                        n_jobs: int = -1, verbose: bool = False
                        ) -> Tuple[np.ndarray, float]:
    """Enumerate all K! permutations and pick the one with max reward."""
    perms = np.array(list(permutations(range(1, K + 1))), dtype=np.int64)
    n_perms = perms.shape[0]
    if verbose:
        print(f"[brute_force] evaluating {n_perms} permutations on {n_jobs} workers")

    # Numba-JITed reward is so fast (microseconds) that joblib overhead would
    # dominate for K <= 6 (<= 720 perms). Only parallelize for K = 7 (5040).
    if n_perms <= 1000 or n_jobs == 1:
        rewards = np.empty(n_perms)
        for i in range(n_perms):
            rewards[i] = _reward_under_perm(S, perms[i], alpha, K)
    else:
        # Chunk so each worker evaluates a batch — avoids per-perm pickling cost
        chunk_size = max(1, n_perms // (4 * (n_jobs if n_jobs > 0 else 8)))
        chunks = [perms[i:i + chunk_size] for i in range(0, n_perms, chunk_size)]

        def eval_chunk(chunk):
            return np.array([_reward_under_perm(S, p, alpha, K) for p in chunk])

        results = Parallel(n_jobs=n_jobs)(
            delayed(eval_chunk)(c) for c in chunks
        )
        rewards = np.concatenate(results)

    best_idx = int(rewards.argmax())
    return perms[best_idx], float(rewards[best_idx])


# ---------------------------------------------------------------------------
# Q-learning (tabular, partial-permutation state)
# ---------------------------------------------------------------------------

def _q_learning_search(S: np.ndarray, K: int, alpha: float,
                       max_episodes: int = 5000,
                       epsilon: float = 0.01,
                       learning_rate: float = 0.05,
                       patience: int = 5000,
                       tol: float = 1e-4,
                       show_iter: Optional[int] = None,
                       seed: Optional[int] = None,
                       verbose: bool = False,
                       ) -> Tuple[np.ndarray, float]:
    """Mirrors q_learning_label_assignment-28.m with a NumPy-backed Q table."""
    rng = np.random.default_rng(seed)
    Q: dict = {}
    best_reward = -np.inf
    best_perm: Optional[np.ndarray] = None
    no_improve = 0

    labels = np.arange(1, K + 1, dtype=np.int64)

    for episode in range(1, max_episodes + 1):
        available = labels.copy()
        perm = np.zeros(K, dtype=np.int64)
        state_key = ""

        for step in range(K):
            if rng.random() < epsilon or state_key not in Q:
                # explore: pick a random available action
                action = int(rng.choice(available))
            else:
                # exploit: argmax over available actions in the current state
                q_row = Q[state_key]
                best_q = -np.inf
                best_action = int(available[0])
                for a in available:
                    qa = q_row.get(int(a), 0.0)
                    if qa > best_q:
                        best_q = qa
                        best_action = int(a)
                action = best_action

            perm[step] = action
            available = available[available != action]
            state_key = state_key + str(action) + "."

        reward = _reward_under_perm(S, perm, alpha, K)

        # Update Q for the terminal state
        prev_state = state_key  # full perm string
        if prev_state not in Q:
            Q[prev_state] = {}
        # store under a sentinel action key
        cur = Q[prev_state].get(-1, 0.0)
        Q[prev_state][-1] = cur + learning_rate * (reward - cur)

        if reward > best_reward + tol:
            best_reward = float(reward)
            best_perm = perm.copy()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            if verbose:
                print(f"[q_learning] converged at episode {episode}")
            break
        if verbose and show_iter and episode % show_iter == 0:
            print(f"[q_learning] episode {episode}, best reward = {best_reward:.4f}")

    if best_perm is None:
        raise RuntimeError("Q-learning failed to find any permutation")
    return best_perm, best_reward


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def optimal_label_order(S,
                        K: Optional[int] = None,
                        alpha: Optional[float] = None,
                        method: str = "auto",
                        n_jobs: int = -1,
                        subsample: Optional[int] = None,
                        # Q-learning hyperparameters
                        max_episodes: int = 5000,
                        epsilon: float = 0.01,
                        learning_rate: float = 0.05,
                        patience: int = 5000,
                        tol: float = 1e-4,
                        show_iter: Optional[int] = None,
                        seed: Optional[int] = None,
                        verbose: bool = False,
                        ) -> Tuple[np.ndarray, float]:
    """Find the canonical label permutation for IFS embedding.

    Parameters
    ----------
    S : array-like of int
        Categorical sequence of states (1..K).
    K : int, optional
        Number of states (default: max(S)).
    alpha : float, optional
        IFS contraction scalar (default: 0.9999*sin(pi/K)/(1+sin(pi/K))).
        Note: the original demo uses 0.99/(1+sin(pi/K)) — pass it explicitly
        if you want to match that exactly.
    method : {'auto', 'brute_force', 'q_learning'}
        'auto' uses brute_force for K <= 7 (5040 perms) and q_learning otherwise.
    n_jobs : int
        Workers for brute force (only used when K = 7).

    Returns
    -------
    perm : ndarray of int, shape (K,)
        1-based permutation: original label i -> perm[i-1].
    reward : float
        Sum of pairwise distances under the best permutation.
    """
    S = np.asarray(S, dtype=np.int64).ravel()
    if K is None:
        K = int(S.max())
    if alpha is None:
        alpha = default_alpha(K)

    # Reward computation is O(N^2) per evaluation — for long sequences,
    # subsampling to a few thousand points gives the same ranking of
    # permutations at a fraction of the cost.
    if subsample is not None and S.size > subsample:
        rng = np.random.default_rng(seed)
        # Stride sampling preserves temporal structure better than random pick.
        stride = max(1, S.size // subsample)
        S = S[::stride][:subsample]
        if verbose:
            print(f"[label_opt] subsampled to {S.size} points")

    if method == "auto":
        method = "brute_force" if K <= 7 else "q_learning"

    if method == "brute_force":
        return _brute_force_search(S, K, alpha, n_jobs=n_jobs, verbose=verbose)
    if method == "q_learning":
        return _q_learning_search(S, K, alpha,
                                  max_episodes=max_episodes,
                                  epsilon=epsilon,
                                  learning_rate=learning_rate,
                                  patience=patience,
                                  tol=tol,
                                  show_iter=show_iter,
                                  seed=seed,
                                  verbose=verbose)
    raise ValueError(f"Unknown method: {method!r}")
