#!/usr/bin/env python3
"""
H3 Experiment: per-coordinate chi-squared uniformity test on MAYO signatures.

Tests whether signature coordinate distributions converge to uniform over F_16
as ko - m increases, using the real MAYO signing algorithm.

Run with:  sage -python analysis/h3_experiment.py
Requires:  SageMath, MAYO-sage
Output:    results/h3/h3_data.csv, results/h3/h3_counts.csv

Design
------
Fix n=20, m=20, o=4.  Vary k in {5, 6, 7, 8, 9, 10}, giving ko - m in
{0, 4, 8, 12, 16, 20}.  At k=5 there is zero signing slack; at k=10 there
are 20 free degrees of freedom.  For each scale, generate N=50,000 signatures
from a single fixed key pair.  For each coordinate position (0..kn-1), count
how often each GF(16) value appears, then compute the Pearson chi-squared
against uniform (15 d.o.f., expected value 15).

Signing uses mayo.sign() directly, which includes the restart loop and the
specific v_i / o_i sampling procedure -- not a preimage solver.  This is the
only distribution that reflects the paper's security argument.

Sanity check: mean chi-squared at ko - m = 0 should be well above 15 (~100).
If it is close to 15, the signing implementation is not working correctly.
"""

import sys
import os
import csv
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'MAYO-sage'))

try:
    from sagelib.utilities import decode_vec
    from sagelib.mayo import Mayo, R
except ImportError as e:
    sys.exit("Error loading MAYO-sage. Run `make pyfiles` first.\n" + str(e))


H3_SCALES = [
    {"ko_minus_m": 0,  "n": 20, "m": 20, "o": 4, "k": 5},
    {"ko_minus_m": 4,  "n": 20, "m": 20, "o": 4, "k": 6},
    {"ko_minus_m": 8,  "n": 20, "m": 20, "o": 4, "k": 7},
    {"ko_minus_m": 12, "n": 20, "m": 20, "o": 4, "k": 8},
    {"ko_minus_m": 16, "n": 20, "m": 20, "o": 4, "k": 9},
    {"ko_minus_m": 20, "n": 20, "m": 20, "o": 4, "k": 10},
]


N_SAMPLES = int(os.environ.get("H3_N_SAMPLES", 50_000))

DATA_FIELDNAMES  = ["ko_minus_m", "n", "m", "o", "k", "kn", "position", "chi_squared"]
COUNT_FIELDNAMES = ["ko_minus_m", "position", "value", "count"]


def _completed_gaps(csv_path):
    """Return set of ko_minus_m values already written to csv_path."""
    completed = set()
    if not os.path.exists(csv_path):
        return completed
    try:
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    completed.add(int(row["ko_minus_m"]))
                except (KeyError, ValueError):
                    pass
    except Exception:
        pass
    return completed


def run_scale(scale, n_samples):
    gap = scale["ko_minus_m"]
    n, m, o, k = scale["n"], scale["m"], scale["o"], scale["k"]
    kn = k * n

    print(f"  gap={gap}: n={n}, m={m}, o={o}, k={k}, kn={kn}, N={n_samples}")

    params = {
        "name": f"mayo_h3_gap{gap}",
        "q": 16, "n": n, "m": m, "o": o, "k": k,
        "sk_salt_bytes": 16, "pk_bytes": 16, "digest_bytes": 16,
        "f": R.irreducible_element(m),
    }
    mayo = Mayo(params)
    csk, _ = mayo.compact_key_gen()
    esk = mayo.expand_sk(csk)

    # counts[position][gf_int_value] -> count
    counts = defaultdict(Counter)

    for i in range(n_samples):
        msg = i.to_bytes(4, "big")
        sig_bytes = mayo.sign(msg, esk)
        s_vec = decode_vec(sig_bytes, kn)
        for pos, elem in enumerate(s_vec):
            counts[pos][elem.to_integer()] += 1

        if (i + 1) % 1000 == 0:
            print(f"    {i + 1}/{n_samples} signatures done")

    expected = n_samples / 16.0
    data_rows  = []
    count_rows = []

    for pos in range(kn):
        chi2 = sum(
            (counts[pos].get(v, 0) - expected) ** 2 / expected
            for v in range(16)
        )
        data_rows.append({
            "ko_minus_m": gap, "n": n, "m": m, "o": o, "k": k, "kn": kn,
            "position": pos, "chi_squared": round(chi2, 6),
        })
        for v in range(16):
            count_rows.append({
                "ko_minus_m": gap, "position": pos,
                "value": v, "count": counts[pos].get(v, 0),
            })

    return data_rows, count_rows


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "h3")
    os.makedirs(out_dir, exist_ok=True)

    data_path  = os.path.join(out_dir, "h3_data.csv")
    count_path = os.path.join(out_dir, "h3_counts.csv")

    completed = _completed_gaps(data_path)
    if completed:
        print(f"Resuming: gaps {sorted(completed)} already done, skipping.")

    data_write_header  = not os.path.exists(data_path)
    count_write_header = not os.path.exists(count_path)

    with open(data_path, "a", newline="") as df, \
         open(count_path, "a", newline="") as cf:

        data_writer  = csv.DictWriter(df, fieldnames=DATA_FIELDNAMES)
        count_writer = csv.DictWriter(cf, fieldnames=COUNT_FIELDNAMES)

        if data_write_header:
            data_writer.writeheader()
        if count_write_header:
            count_writer.writeheader()

        for scale in H3_SCALES:
            gap = scale["ko_minus_m"]
            if gap in completed:
                continue

            print(f"\nRunning gap={gap}...")
            data_rows, count_rows = run_scale(scale, N_SAMPLES)

            data_writer.writerows(data_rows)
            count_writer.writerows(count_rows)
            df.flush()
            cf.flush()

            chi2_vals = [r["chi_squared"] for r in data_rows]
            mean_chi2 = sum(chi2_vals) / len(chi2_vals)
            print(f"  gap={gap}: mean chi2 = {mean_chi2:.2f}  (expected ~15 if uniform)")

    print("\nDone. Output in results/h3/")


if __name__ == "__main__":
    main()
