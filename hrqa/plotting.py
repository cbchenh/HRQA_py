"""Plotting helpers — matplotlib only, no MATLAB toolbox dependencies."""
from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.patches import Rectangle

from .ifs import ifs_address
from .utils import default_alpha


def plot_hrp(S, k: Optional[int] = None, ax=None):
    """Heterogeneous Recurrence Plot: scatter (i, j) where S[i]==S[j]."""
    S = np.asarray(S, dtype=np.int64).ravel()
    if k is None:
        k = int(S.max())
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    n = S.size
    # For each state, mark all (i, j) pairs sharing that state.
    cmap = plt.get_cmap("jet", k)
    for state in range(1, k + 1):
        idx = np.where(S == state)[0]
        if idx.size < 2:
            continue
        ii, jj = np.meshgrid(idx, idx, indexing="ij")
        ax.scatter(ii.ravel(), jj.ravel(), s=2,
                   color=cmap(state - 1), label=f"s={state}")
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.set_xlabel("i")
    ax.set_ylabel("j")
    return ax


def plot_cells(Ub: np.ndarray, Lb: np.ndarray, ax=None):
    """Draw the boundaries of segmented cells (2D rectangles or 3D wireframes)."""
    Ub = np.asarray(Ub)
    Lb = np.asarray(Lb)
    if Ub.shape != Lb.shape:
        raise ValueError("Ub and Lb shapes differ")
    n_cells, d = Ub.shape

    if d == 2:
        if ax is None:
            _, ax = plt.subplots()
        for i in range(n_cells):
            ax.add_patch(Rectangle(Lb[i], *(Ub[i] - Lb[i]),
                                   fill=False, linewidth=2))
        ax.set_xlim(Lb[:, 0].min(), Ub[:, 0].max())
        ax.set_ylim(Lb[:, 1].min(), Ub[:, 1].max())
        ax.set_aspect("equal")
        return ax

    if d >= 3:
        if d > 3:
            print("[plot_cells] only first 3 dims shown")
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        for i in range(n_cells):
            x0, y0, z0 = Lb[i, :3]
            x1, y1, z1 = Ub[i, :3]
            corners = np.array([
                [x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1],
                [x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1],
            ])
            edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                     (4, 5), (5, 6), (6, 7), (7, 4),
                     (0, 4), (1, 5), (2, 6), (3, 7)]
            segs = [[corners[a], corners[b]] for a, b in edges]
            ax.add_collection3d(Line3DCollection(segs, colors="k", linewidths=1))
        ax.set_xlim(Lb[:, 0].min(), Ub[:, 0].max())
        ax.set_ylim(Lb[:, 1].min(), Ub[:, 1].max())
        ax.set_zlim(Lb[:, 2].min(), Ub[:, 2].max())
        return ax


def plot_ifs(S, K: Optional[int] = None, a: Optional[float] = None, ax=None):
    """Scatter plot of the IFS address points colored by state."""
    S = np.asarray(S, dtype=np.int64).ravel()
    if K is None:
        K = int(S.max())
    if a is None:
        a = default_alpha(K)
    Cv = ifs_address(S, K=K, a=a, include_origin=False)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.get_cmap("jet", K)
    ax.scatter(Cv[:, 0], Cv[:, 1], c=S - 1, cmap=cmap, s=4)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.set_title("IFS address")
    return ax
