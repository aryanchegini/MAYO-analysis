#!/usr/bin/env python3
"""
H3 Analysis: plot chi-squared vs ko - m and per-scale heatmaps.

Run with:  python3 analysis/analyse_h3.py
Requires:  matplotlib, numpy, pandas (or csv)
Input:     results/h3/h3_data.csv, results/h3/h3_counts.csv
Output:    results/h3/h3_chi2_vs_gap.pdf, results/h3/h3_heatmap_gap*.pdf
"""

import os
import csv
from collections import defaultdict
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "serif"


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "h3")


def load_data(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "ko_minus_m": int(row["ko_minus_m"]),
                "kn":         int(row["kn"]),
                "position":   int(row["position"]),
                "chi_squared": float(row["chi_squared"]),
            })
    return rows


def load_counts(path):
    """Returns dict: (ko_minus_m, position, value) -> count."""
    counts = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["ko_minus_m"]), int(row["position"]), int(row["value"]))
            counts[key] = int(row["count"])
    return counts


def aggregate_by_gap(rows):
    """Returns dict: gap -> list of chi_squared values (one per coordinate)."""
    by_gap = defaultdict(list)
    for r in rows:
        by_gap[r["ko_minus_m"]].append(r["chi_squared"])
    return by_gap


def plot_chi2_vs_gap(by_gap, out_dir):
    gaps = sorted(by_gap.keys())
    means = [np.mean(by_gap[g]) for g in gaps]
    stds  = [np.std(by_gap[g])  for g in gaps]
    ses   = [s / math.sqrt(len(by_gap[g])) for s, g in zip(stds, gaps)]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(gaps, means, yerr=ses, fmt="o-", color="steelblue",
                capsize=4, linewidth=1.5, markersize=6, label="Mean χ² (±SE)")
    ax.axhline(15, color="crimson", linestyle="--", linewidth=1.2,
               label="Uniform expectation (χ²=15)")

    ax.set_xlabel("ko − m")
    ax.set_ylabel("Mean χ² per coordinate (15 d.o.f.)")
    ax.set_title("Per-coordinate chi-squared vs signing slack (ko − m)")
    ax.set_xticks(gaps)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    path = os.path.join(out_dir, "h3_chi2_vs_gap.pdf")
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_heatmap(counts_dict, gap, kn, n_samples, out_dir):
    """
    Heatmap of Pearson residuals for one gap value.

    Cell (v, pos) = (observed_count - expected) / sqrt(expected), where
    expected = n_samples / 16.  Values are in units of std devs under null;
    +/-2 is the natural signal boundary.
    """
    expected = n_samples / 16.0
    data = np.zeros((16, kn))
    for pos in range(kn):
        for v in range(16):
            observed = counts_dict.get((gap, pos, v), 0)
            data[v, pos] = (observed - expected) / math.sqrt(expected)

    fig_width = max(8, kn // 6)
    fig, ax = plt.subplots(figsize=(fig_width, 5))

    vmax = max(2.0, float(np.abs(data).max()))
    im = ax.imshow(data, aspect="auto", cmap="coolwarm",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Pearson residual (std devs from uniform)")

    ax.set_xlabel("Signature coordinate position")
    ax.set_ylabel("GF(16) value (0–15)")
    ax.set_yticks(range(16))
    ax.set_title(f"Signature distribution: ko − m = {gap}  (kn={kn})")
    fig.tight_layout()

    path = os.path.join(out_dir, f"h3_heatmap_gap{gap}.pdf")
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    data_path  = os.path.join(RESULTS_DIR, "h3_data.csv")
    count_path = os.path.join(RESULTS_DIR, "h3_counts.csv")

    if not os.path.exists(data_path):
        print(f"No data found at {data_path}. Run h3_experiment.py first.")
        return

    rows   = load_data(data_path)
    by_gap = aggregate_by_gap(rows)

    print("Summary:")
    for gap in sorted(by_gap.keys()):
        vals = by_gap[gap]
        print(f"  gap={gap}: n_coords={len(vals)}, "
              f"mean_chi2={np.mean(vals):.2f}, std={np.std(vals):.2f}")

    plot_chi2_vs_gap(by_gap, RESULTS_DIR)

    if os.path.exists(count_path):
        counts_dict = load_counts(count_path)
        kn_by_gap = {}
        for r in rows:
            kn_by_gap[r["ko_minus_m"]] = r["kn"]

        # Infer n_samples from the counts (sum over all values at position 0).
        n_samples_by_gap = {}
        for gap in sorted(by_gap.keys()):
            total = sum(counts_dict.get((gap, 0, v), 0) for v in range(16))
            n_samples_by_gap[gap] = total

        for gap in sorted(by_gap.keys()):
            kn = kn_by_gap[gap]
            n_samples = n_samples_by_gap[gap]
            plot_heatmap(counts_dict, gap, kn, n_samples, RESULTS_DIR)
    else:
        print(f"No counts file at {count_path}, skipping heatmaps.")

    print("\nDone.")


if __name__ == "__main__":
    main()
