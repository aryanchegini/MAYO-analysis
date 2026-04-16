#!/usr/bin/env python3
"""
Distribution Analysis: statistical and exhaustive sampling of MAYO signature components.

Run with:  sage -python old_analysis/distribution_analysis.py
Requires:  SageMath, MAYO-sage, matplotlib, numpy
Output:    results/distribution/

Experiment 1 (statistical): generate 1000 signatures under toy parameters and
record the per-position GF(16) frequency distribution of the signature vector.
Output: a heatmap showing deviation from uniform across all 90 positions.

Experiment 2 (exact): brute-force all GF(16)^{k*n} inputs for micro parameters
conditioned on P*(x) == t for a fixed target t.  Output: a 2x2 grid of bar
charts (one per position).
"""

import sys
import os
from itertools import product
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np

from sage.all import block_matrix, matrix, vector

sys.path.append(os.path.abspath('MAYO-sage'))

try:
    from sagelib.utilities import decode_vec, decode_matrices
    from sagelib.mayo import Mayo, F16, R, z
except ImportError as e:
    sys.exit("Error loading preprocessed sage files. Run `make pyfiles` first. " + str(e))


def evaluate_P_star(s_vec, epk, mayo_ins):
    """
    Evaluate the MAYO public map P* on signature vector s_vec.

    s_vec is a vector of length k*n over GF(16), treated as k blocks of
    length n.  Returns the m-vector P*(s_vec) over GF(16).
    """
    p1 = decode_matrices(epk[:mayo_ins.P1_bytes], mayo_ins.m, mayo_ins.n - mayo_ins.o, mayo_ins.n - mayo_ins.o, triangular=True)
    p2 = decode_matrices(epk[mayo_ins.P1_bytes:mayo_ins.P1_bytes + mayo_ins.P2_bytes], mayo_ins.m, mayo_ins.n - mayo_ins.o, mayo_ins.o, triangular=False)
    p3 = decode_matrices(epk[mayo_ins.P1_bytes + mayo_ins.P2_bytes:mayo_ins.P1_bytes + mayo_ins.P2_bytes + mayo_ins.P3_bytes], mayo_ins.m, mayo_ins.o, mayo_ins.o, triangular=True)

    s_split = [s_vec[i * mayo_ins.n:(i + 1) * mayo_ins.n] for i in range(mayo_ins.k)]
    p = [
        block_matrix([[p1[a], p2[a]], [matrix(F16, mayo_ins.o, mayo_ins.n - mayo_ins.o), p3[a]]])
        for a in range(mayo_ins.m)
    ]
    sip = [[s_split[i] * p[a] for a in range(mayo_ins.m)] for i in range(mayo_ins.k)]

    ell = 0
    y = vector(F16, mayo_ins.m)
    for i in range(mayo_ins.k):
        for j in range(mayo_ins.k - 1, i - 1, -1):
            u = vector(F16, mayo_ins.m)
            for a in range(mayo_ins.m):
                if i == j:
                    u[a] = sip[i][a] * s_split[j]
                else:
                    u[a] = sip[i][a] * s_split[j] + sip[j][a] * s_split[i]
            u = mayo_ins.fx(list(u))
            y = y + vector(z**ell * u)
            ell += 1
    return y


def run_experiment_1():
    """
    Statistical sampling: generate 1000 signatures under reduced parameter set and
    record the per-position GF(16) frequency distribution.

    Returns a dict mapping position index -> {gf16_int_value: percentage}.
    """
    print("Running Experiment 1: Statistical Sampling (Toy Parameters)...")
    toy_params = {
        "name": "mayo_toy",
        "q": 16, "m": 16, "n": 18, "o": 4, "k": 5,
        "sk_salt_bytes": 24, "pk_bytes": 16, "digest_bytes": 32,
        "f": R.irreducible_element(16, algorithm="random")
    }
    mayo_toy = Mayo(toy_params)
    csk, _ = mayo_toy.compact_key_gen()
    esk = mayo_toy.expand_sk(csk)

    msg = b"StatSampleTest"
    n_samples = 10000
    frequencies = defaultdict(Counter)

    for _ in range(n_samples):
        # sign() varies the internal salt each call, producing distinct signatures.
        sig_bytes = mayo_toy.sign(msg, esk)
        s_vec = decode_vec(sig_bytes, mayo_toy.n * mayo_toy.k)
        for i, coeff in enumerate(s_vec):
            frequencies[i][coeff.to_integer()] += 1

    n_vars = mayo_toy.n * mayo_toy.k
    percentages = {}
    for i in range(n_vars):
        total = sum(frequencies[i].values())
        if total > 0:
            percentages[i] = {k: (v / total) * 100 for k, v in frequencies[i].items()}
        else:
            percentages[i] = {}
    return percentages


def run_experiment_2():
    """
    Exhaustive brute-force: enumerate all vectors in GF(16)^{k*n} for micro
    parameters and record the per-position frequency distribution conditioned
    on P*(x) == t for a fixed random target t.

    Returns a dict mapping position index -> {gf16_int_value: percentage}.
    """
    print("Running Experiment 2: Exhaustive Brute-Force (Micro Parameters)...")
    micro_params = {
        "name": "mayo_micro",
        "q": 16, "m": 2, "n": 2, "o": 1, "k": 2,
        "sk_salt_bytes": 16, "pk_bytes": 16, "digest_bytes": 16,
        "f": R.irreducible_element(2, algorithm="random")
    }
    mayo_micro = Mayo(micro_params)
    _, cpk = mayo_micro.compact_key_gen()
    epk = mayo_micro.expand_pk(cpk)

    t = vector(F16, [F16.random_element() for _ in range(mayo_micro.m)])

    frequencies = defaultdict(Counter)
    for coeffs in product(F16, repeat=mayo_micro.n * mayo_micro.k):
        x_vec = vector(F16, coeffs)
        if evaluate_P_star(x_vec, epk, mayo_micro) == t:
            for i, coeff in enumerate(x_vec):
                frequencies[i][coeff.to_integer()] += 1

    n_vars = mayo_micro.n * mayo_micro.k
    percentages = {}
    for i in range(n_vars):
        total = sum(frequencies[i].values())
        if total > 0:
            percentages[i] = {k: (v / total) * 100 for k, v in frequencies[i].items()}
        else:
            percentages[i] = {}
    return percentages


def plot_heatmap(dist, title, out_dir):
    """
    Save a heatmap of GF(16) element frequencies across all signature positions.

    Each cell shows the deviation from the uniform baseline (6.25%), using a
    diverging red-blue colormap centred at zero.  This gives an immediate
    visual summary of where non-uniformity occurs across all positions at once,
    which is unachievable with individual per-position bar charts.

    Rows: GF(16) integer value (0-15)
    Columns: signature position index (sorted)
    Colour: frequency(%) - 6.25%
    """
    positions = sorted(dist.keys())
    n_vars = len(positions)

    # Build deviation matrix: shape (16, n_vars).
    data = np.zeros((16, n_vars))
    for col, i in enumerate(positions):
        for row in range(16):
            data[row, col] = dist[i].get(row, 0) - 6.25

    fig_width = max(8, n_vars // 5)
    fig, ax = plt.subplots(figsize=(fig_width, 5))

    im = ax.imshow(data, aspect='auto', cmap='RdBu_r', vmin=-6.25, vmax=6.25,
                   interpolation='nearest')
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Deviation from uniform (%)")

    ax.set_xlabel("Signature position index")
    ax.set_ylabel("GF(16) element (integer repr.)")
    ax.set_yticks(range(16))
    ax.set_title(title)
    fig.tight_layout()

    safe_title = title.replace(" ", "_").lower()
    path = os.path.join(out_dir, f"{safe_title}_heatmap.pdf")
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap to {path}")


def plot_grid(dist, title, out_dir):
    """
    Save a single figure with one bar chart per signature position.

    Suitable when the number of positions is small (e.g. experiment 2 with 4
    variables).  All subplots share the same y-axis scale and the uniform
    baseline (6.25%) is marked on each panel.
    """
    positions = sorted(dist.keys())
    n_vars = len(positions)
    ncols = min(n_vars, 4)
    nrows = (n_vars + ncols - 1) // ncols
    x_vals = list(range(16))
    safe_title = title.replace(" ", "_").lower()

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
                             squeeze=False)

    all_y = [dist[i].get(v, 0) for i in positions for v in x_vals]
    y_max = max(max(all_y, default=0), 6.5) * 1.15

    for idx, i in enumerate(positions):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        y = [dist[i].get(v, 0) for v in x_vals]
        ax.bar(x_vals, y, color='royalblue', edgecolor='black', linewidth=0.5)
        ax.axhline(6.25, color='red', linestyle='--', linewidth=1.2, label='Uniform (6.25%)')
        ax.set_title(f"Position {i}", fontsize=10)
        ax.set_xlabel("GF(16) value", fontsize=8)
        ax.set_ylabel("Freq (%)", fontsize=8)
        ax.set_xticks(x_vals)
        ax.set_xticklabels(x_vals, fontsize=6)
        ax.set_ylim(0, y_max)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        if idx == 0:
            ax.legend(fontsize=8)

    for idx in range(n_vars, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{safe_title}_grid.pdf")
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved grid plot to {path}")


if __name__ == '__main__':
    out_dir = os.path.join("results", "distribution")
    os.makedirs(out_dir, exist_ok=True)

    d1 = run_experiment_1()
    plot_heatmap(d1, "MAYO Sample Distribution", out_dir)

    d2 = run_experiment_2()
    plot_grid(d2, "MAYO Exact Distribution", out_dir)
