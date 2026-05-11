"""
HRQA — Heterogeneous Recurrence Quantification Analysis (Python)
=================================================================

Fast Python port of the MATLAB HRQA toolkit by Cheng-Bang Chen et al.

Two top-level entry points:
    A) hrqa_from_signal   — raw multivariate time series -> categorical -> HRQA
    B) hrqa_from_sequence — categorical integer sequence -> HRQA directly

Plus a side function:
    optimal_label_order   — find canonical label permutation for K states

See README.md for the full workflow.
"""

from .ifs import ifs_address
from .segmentation import HASSegmenter, KMeansSegmenter, make_segmenter
from .hrqa_core import rhrqa, fhrqa
from .label_opt import optimal_label_order
from .pipeline import (
    hrqa_from_signal,
    hrqa_from_sequence,
    rhrqa_windowed,
)
from .plotting import plot_hrp, plot_cells, plot_ifs

__all__ = [
    "ifs_address",
    "HASSegmenter",
    "KMeansSegmenter",
    "make_segmenter",
    "rhrqa",
    "fhrqa",
    "optimal_label_order",
    "hrqa_from_signal",
    "hrqa_from_sequence",
    "rhrqa_windowed",
    "plot_hrp",
    "plot_cells",
    "plot_ifs",
]

__version__ = "0.1.0"
