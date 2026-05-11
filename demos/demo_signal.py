"""Demo A — Raw multivariate signal pipeline.

Mirrors demo-2.m: read Lorenz CSV, segment via HAS, compute RHRQA r=1, save
features to CSV.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hrqa import HASSegmenter, hrqa_from_signal, plot_cells, plot_ifs, plot_hrp
from hrqa.hrqa_core import METRIC_NAMES


def _find_csv(name: str) -> Path:
    """Search standard locations for the demo CSV."""
    here = Path(__file__).resolve().parent
    candidates = [
        Path(name),                       # cwd or absolute
        here / name,                      # demos/
        here.parent / name,               # hrqa_py/
        here.parent.parent / name,        # parent (e.g. matlab/HRQA/)
        Path("/home/user/workspace") / name,
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(
        f"Could not find '{name}'. Tried:\n  " + "\n  ".join(str(c) for c in candidates)
        + f"\nPass an explicit path:  python demos/demo_signal.py /path/to/{name}"
    )


def main(csv_path: str = "Lorenz-23.csv", out_csv: str = "demo_statistics_py.csv"):
    here = Path(__file__).resolve().parent
    csv = _find_csv(csv_path)

    print(f"[1] Reading {csv}")
    data = np.loadtxt(csv, delimiter=",")
    print(f"    shape = {data.shape}")

    # Plot raw channels
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), constrained_layout=True)
    for i, lbl in enumerate("XYZ"):
        axes[i].plot(data[:, i])
        axes[i].set_title(lbl)
        axes[i].set_xlabel("Index")
        axes[i].set_ylabel("Value")
    fig.savefig(here / "demo_signal_channels.png", dpi=120)
    plt.close(fig)

    # Segmentation: HAS with capacity = N/4 (matches demo-2.m)
    capacity = data.shape[0] // 4
    print(f"[2] HAS segmentation (capacity = {capacity})")
    seg = HASSegmenter(capacity=capacity).fit(data)
    print(f"    K = {seg.K} cells")

    # IFS scatter on first 3 dims, with cells overlaid (3D)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    S = seg.transform(data)
    cmap = plt.get_cmap("jet", seg.K)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=S - 1, cmap=cmap, s=2)
    plot_cells(seg.Ub, seg.Lb, ax=ax)
    ax.set_title("Trajectory + HAS cells")
    ax.view_init(elev=25, azim=55)
    fig.savefig(here / "demo_signal_trajectory.png", dpi=120)
    plt.close(fig)

    # RHRQA r=1
    print("[3] RHRQA r=1")
    t0 = time.perf_counter()
    out = hrqa_from_signal(data, segmenter=seg, r=1)
    elapsed = time.perf_counter() - t0
    print(f"    elapsed = {elapsed*1e3:.2f} ms")

    # Build CSV row matching demo-2.m's column convention (now 10 metrics:
    # HRR, HMean, HVar, HSkew, HKurt, HEnt, HGini, HJSD, HMedian, HIQR)
    metrics = list(METRIC_NAMES)
    Kr = out["HRR"].size
    cols = [f"{m}_{i+1}" for m in metrics for i in range(Kr)]
    row = np.concatenate([out[m] for m in metrics])
    row = np.nan_to_num(row, nan=0.0)
    df = pd.DataFrame([row], columns=cols)
    out_path = here / out_csv
    df.to_csv(out_path, index=False)
    print(f"[4] Saved features  ->  {out_path}  ({df.shape})")

    # IFS plot + HRP
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_ifs(out["S"], K=seg.K, ax=ax)
    fig.savefig(here / "demo_signal_ifs.png", dpi=120)
    plt.close(fig)

    print("[5] Done.")
    return df


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "Lorenz-23.csv"
    main(csv_path=arg)
