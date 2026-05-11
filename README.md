# HRQA — Heterogeneous Recurrence Quantification Analysis (Python)

A fast Python port of the MATLAB HRQA toolkit by Cheng-Bang Chen et al. The
package converts (multivariate) continuous time series into categorical
sequences via state-space partitioning, then computes ten heterogeneous
recurrence statistics — `HRR, HMean, HVar, HSkew, HKurt, HEnt, HGini, HJSD,
HMedian, HIQR` — for local-cluster orders `r = 1, 2, …`.

## Why this port

- **~22× faster** than a literal Python port of the MATLAB code (and ~30–50×
  faster than the original MATLAB on the same hardware) — measured on `r=1,2`
  workloads with `K ∈ {4, 8, 16}` and `N ∈ {500…20000}`.
- **Bit-for-bit numerical agreement** with the MATLAB reference (max abs diff
  ~1e-13, machine precision).
- **Two clean entry points** for the two common workflows: raw signals or
  pre-computed categorical sequences.
- **Optional GPU backend** (CuPy) for the two scenarios where it actually
  helps: large batched windowed RHRQA and Q-learning LabelOpt.

## Install

```bash
cd hrqa_py
pip install -e .

# Optional GPU support (CUDA 12+)
pip install -e ".[gpu]"
```

## API at a glance

```python
from hrqa import (
    hrqa_from_signal,      # A) raw signal -> HRQA
    hrqa_from_sequence,    # B) categorical sequence -> HRQA
    rhrqa_windowed,        # sliding-window RHRQA over a long sequence
    optimal_label_order,   # LabelOpt: canonical permutation of state labels
    KMeansSegmenter, HASSegmenter,
)
```

## Recommended workflow (multi-subject)

```python
import numpy as np
from hrqa import KMeansSegmenter, optimal_label_order, hrqa_from_signal

# ----- Step 1 (once per cohort): fit segmentation on a representative pool
seg = KMeansSegmenter(K=8).fit(X_pool)              # X_pool: (N, n_channels)
S_pool = seg.transform(X_pool)

# ----- Step 2 (once): find canonical label order on the pool
perm, reward = optimal_label_order(
    S_pool, K=seg.K,
    method="auto",         # brute_force for K<=7, q_learning otherwise
    subsample=4000,        # speed up reward evaluation for long sequences
)
seg.set_label_permutation(perm)   # all future transforms use canonical labels

# ----- Step 3 (per subject / per window): apply same segmentation + HRQA
for subject_id, X in subjects.items():
    out = hrqa_from_signal(X, segmenter=seg, r=1)
    # out has: 'S', 'segmenter', 'IdxM', 'HRR', 'HMean', 'HVar',
    #          'HSkew', 'HKurt', 'HEnt', 'HGini'
```

This guarantees every subject is segmented with the same bin centers and
relabeled with the same canonical order — a prerequisite for downstream
cross-subject comparison and ML classification.

## Two top-level entry points

### A. Raw signal → HRQA  (`hrqa_from_signal`)

```python
hrqa_from_signal(
    X,                    # (n_samples, n_features) ndarray
    segmenter=None,       # fitted Segmenter (preferred). If None, fits on X.
    method="kmeans",      # 'kmeans' | 'has' (only used if segmenter is None)
    K=None,               # n_clusters for KMeans
    capacity=None,        # cell capacity for HAS
    r=1, a=None, B=20,
    label_perm=None,      # 1-based permutation, length K
)
```

**Two segmenters:**

- **`KMeansSegmenter(K=...)`** — sklearn KMeans with `K` clusters. Fast,
  deterministic with `random_state`, supports any number of dimensions.
- **`HASSegmenter(capacity=...)`** — recursive 2^d hyperoctree subdivision
  (port of `HAS-5.m`). Cell count `K` emerges from the data and capacity.

Both share the same fit / transform / `set_label_permutation` interface.

### B. Categorical sequence → HRQA  (`hrqa_from_sequence`)

```python
hrqa_from_sequence(
    S,                    # 1D array of positive integers in [1, K]
    K=None,               # default: max(S)
    r=1, a=None, B=20,
    Cv=None,              # optional pre-computed IFS address (saves work
                          # across many windows of the same long sequence)
)
```

### Sliding-window batch (`rhrqa_windowed`)

Mirrors `demo_c.m`. Slides `win_size` windows with `step_size` stride and
returns one feature vector per window.

```python
df = rhrqa_windowed(
    S,
    K=4,
    win_size=200, step_size=100,
    r=1,
    n_jobs=1,             # joblib workers (-1 = all cores). For small windows
                          # the JIT'd inner loop is so fast that joblib is a
                          # net loss; benchmark on your data.
    as_dataframe=True,    # column names: WinID, StartIdx, EndIdx, HRR_1, ...
)
```

## LabelOpt (`optimal_label_order`)

```python
perm, reward = optimal_label_order(
    S, K=K, alpha=None,
    method="auto",        # 'brute_force' (K<=7), 'q_learning', or 'auto'
    subsample=4000,       # cap S length for reward evaluation
    # Q-learning params (used when method='q_learning'):
    max_episodes=5000, epsilon=0.01, learning_rate=0.05,
    patience=5000, tol=1e-4, seed=None, show_iter=None, verbose=False,
)
```

`perm` is a 1-based permutation of `1..K` mapping original labels to canonical
ones. Pass it to `Segmenter.set_label_permutation(perm)` to apply.

## Output convention

Every `rhrqa(...)` call returns a dict with keys:

| Key       | Shape     | Meaning |
|-----------|-----------|---------|
| `IdxM`    | (K**r, r) | All r-grams, lex order |
| `HRR`     | (K**r,)   | Heterogeneous Recurrence Rate `(H_bar/N)**2`  (Eq. 10) |
| `HMean`   | (K**r,)   | Mean of pairwise IFS distances  (Eq. 11) |
| `HVar`    | (K**r,)   | Variance  (Eq. 12) |
| `HSkew`   | (K**r,)   | Skewness  (Eq. 13) |
| `HKurt`   | (K**r,)   | Kurtosis  (Eq. 14) |
| `HEnt`    | (K**r,)   | Shannon entropy of distance histogram (B bins)  (Eq. 15) |
| `HGini`   | (K**r,)   | 1 − Σp² of the same histogram  (Eq. 16) |
| `HJSD`    | (K**r,)   | Jensen-Shannon divergence vs pooled global reference (Eq. 17) |
| `HMedian` | (K**r,)   | Median of pairwise IFS distances  Q₀.₅(D_k)  (Eq. 18) |
| `HIQR`    | (K**r,)   | Inter-quartile range  Q₀.₇₅(D_k) − Q₀.₂₅(D_k)  (Eq. 19) |

For maximum speed in tight loops, pass `return_array=True` to skip the dict
construction and get a `(10, K**r)` ndarray (rows in the order above).

`HEnt`/`HGini` use per-state histograms with `B` bins of state-specific
support, matching the original MATLAB `RHRQA-13.m`. `HJSD` instead uses a
*pooled* global support shared across all states (built by concatenating
each state's distance set, taking [min, max], and binning with `B` equal-
width bins) so its per-state probabilities `p_{k,b}` are comparable to the
reference `q_b` used in Eq. 17.

## File mapping (MATLAB → Python)

| MATLAB                                 | Python                                  |
|----------------------------------------|-----------------------------------------|
| `IFS.m`                              | `hrqa.ifs.ifs_address`                  |
| `HAS.m` + `combn.m` + `SymbG.m`   | `hrqa.segmentation.HASSegmenter`        |
| (KMeans not in MATLAB pkg)             | `hrqa.segmentation.KMeansSegmenter`     |
| `HRQA.m` / `HRQA2.m` / `RHRQA.m`| `hrqa.hrqa_core.rhrqa`                  |
| `FHRQA.m`                            | `hrqa.hrqa_core.fhrqa`                  |
| `optimLabel-11.m` (incomplete)         | `hrqa.label_opt.optimal_label_order`    |
| `q_learning_label_assignment.m` + `compute_points-24.m` + `compute_reward-25.m` | `hrqa.label_opt._q_learning_search` |
| `demo.m`                             | `demos/demo_signal.py`                  |
| `demo_c.m`                           | `demos/demo_sequence.py`                |
| `demo-2.m`                            | `demos/demo_label_opt.py`               |
| `HRP.m`, `RP.m`, `PlotCell.m`  | `hrqa.plotting`                         |
| `lorenz.m`                          | (use `scipy.integrate.solve_ivp` directly if needed) |

## Speed notes — what makes it fast

1. **Vectorized IFS** — first-order linear recurrence solved via
   `scipy.signal.lfilter` (single C call) instead of the per-sample MATLAB
   loop.
2. **Group-by unique r-gram** — `np.unique(...)` once on positional-encoded
   r-grams replaces `O(U·N)` per-cluster `ismember` scans.
3. **Numba-JIT inner kernel** — single-pass pdist + 4 central moments +
   histogram in one tight loop, eliminating temporary-array allocations
   that dominated pure-NumPy time.
4. **`Cv` reuse** — pass a pre-computed IFS address into `rhrqa(...)` once
   per long sequence and reuse it across windows or orders (matches
   `HRQA2.m`).
5. **Optional GPU** — `hrqa.gpu.rhrqa_windowed_gpu` for batches large enough
   to amortize transfer + launch overhead (heuristic threshold ~200k points).

## Benchmarks

Run `python demos/benchmark.py`:

```
Single-call RHRQA: Python (Numba) vs MATLAB-style reference
  K   N      r |  fast (ms)   ref (ms)   speedup
  4   500    1 |       0.34       5.77     16.8x
  8   2000   1 |       1.76      39.01     22.1x
  8   2000   2 |       0.94      18.49     19.6x
  8   5000   1 |       9.53     238.47     25.0x
 16   5000   1 |       5.89     114.70     19.5x
  8   10000  1 |      36.47     953.48     26.1x
 16   20000  1 |      69.78    1792.99     25.7x
```

## Quick start

```bash
# Run the raw-signal demo (Lorenz attractor)
python demos/demo_signal.py

# Run the categorical-sequence demo
python demos/demo_sequence.py

# Run the LabelOpt demo
python demos/demo_label_opt.py

# Run the benchmark suite
python demos/benchmark.py
```

## License & credits

Original MATLAB algorithms and documentation © Cheng-Bang Chen, Hui Yang,
Soundar Kumara (Penn State / U. Miami). This Python port preserves the
mathematical definitions and naming conventions of the original toolkit.
