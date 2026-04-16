#!/usr/bin/env python3
"""
H2 Experiment: Groebner basis on whipped-map vs random-map MQ instances.

Tests whether MAYO's whipping construction provides any structural shortcut
to an attacker compared with a uniformly random MQ map of equal dimensions.

Run with:   sage -python analysis/h2_experiment.py
Requires:   SageMath
Output:     results/h2/h2_results.csv

Whipped-map structure
---------------------
The m underlying P matrices are genuine UOV maps: each is an n x n
upper-triangular matrix with zero oil-oil block (P3 = 0), matching the MAYO
spec.  The oil dimension o is included in each scale tuple (n, m, o, k).

Planted solutions
-----------------
Each instance is generated with a known preimage x0 such that P*(x0) = t.
Variable fixing uses x0's own coordinates for the fixed positions, so the
reduced m x m system always has at least one solution.  Without this, random
variable fixing frequently produces infeasible systems, and the Groebner basis
measures infeasibility certification rather than preimage-finding, a
different computation that makes the whipped-vs-random comparison meaningless.

Pairing
-------
Each CSV row carries a pair_id column equal to the instance index.  Whipped
and random rows with the same (scale, pair_id) are paired instances at
identical dimensions and can be used directly in Wilcoxon signed-rank tests.

Timeout / checkpoint
--------------------
Each solver call runs in a child process hard-killed after TIMEOUT_SECONDS.
CPU time is recovered via RUSAGE_CHILDREN.  The CSV is opened in append mode
so interrupted runs can be resumed without losing progress.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import csv, resource, time, multiprocessing

from mq_GF16 import (
    generate_whipped_instance, generate_random_instance,
    reduce_with_planted_solution,
)
from experiment_utils import mem_delta_kb, run_with_timeout, load_completed

# Experiment parameters

# Each scale is (n, m, o, k).
# v = n-o (vinegar), o (oil), kn = k*n total variables, m equations.
# After planted-solution variable fixing: m equations in m unknowns.
# Constraint: o < n/2  (more vinegar than oil, required for UOV security).
H2_SCALES = [
    (6,  4, 2, 2),   # scale 1:  kn=12,  v=4, o=2, m=4
    (8,  6, 2, 3),   # scale 2:  kn=24,  v=6, o=2, m=6
    (10, 8, 3, 4),   # scale 3:  kn=40,  v=7, o=3, m=8
    (12, 10, 3, 5),  # scale 4:  kn=60,  v=9, o=3, m=10
]
DEMO_MODE = True

if DEMO_MODE:
    # Scales 1-2: full 50 instances (fast, gives real statistics).
    # Scale 3: 20 instances (enough for a distribution, ~7-33min).
    # Scale 4: 3 instances (calibration timings only, up to ~60min).
    INSTANCES_PER_SCALE      = 50
    SCALE_INSTANCES_OVERRIDE = {1:0, 2:0, 3: 0, 4: 3}
    OUTPUT_DIR               = os.path.join("results", "h2_demo")
else:
    INSTANCES_PER_SCALE      = 50
    SCALE_INSTANCES_OVERRIDE = {}
    OUTPUT_DIR               = os.path.join("results", "h2")

TIMEOUT_SECONDS = 5000
SEED_BASE       = 54321
H2_CSV          = os.path.join(OUTPUT_DIR, "h2_results.csv")

FIELDNAMES = [
    "scale", "n", "m", "o", "k", "kn", "instance", "seed",
    "instance_type", "pair_id",
    "timeout_s",
    "cpu_time_s", "wall_time_s", "memory_kb",
    "degree", "success", "timed_out", "error",
]


def _groebner_worker(instance_type, n, m, o, k, base_seed):
    """
    Generate the instance from base_seed and solve via Groebner basis.

    For 'whipped': uses base_seed for instance generation and planted-solution
    variable fixing.  P matrices have genuine UOV structure (zero oil-oil block).

    For 'random': uses base_seed + 5_000_000 for generation and fixing,
    keeping the random baseline independent from the whipped instance.

    In both cases the reduced m x m system is guaranteed to have at least one
    solution (the planted solution's free coordinates), so the Groebner basis
    measures preimage-finding difficulty rather than infeasibility certification.

    Runs inside a child process; no internal timeout is needed.
    """
    kn = k * n
    if instance_type == "whipped":
        R, xs, equations, _, _, x0_int = generate_whipped_instance(n, m, o, k, base_seed)
        eqs_red, feqs, _               = reduce_with_planted_solution(R, xs, equations, kn, m, x0_int, base_seed)
    else:
        rand_seed                      = base_seed + 5_000_000
        R, xs, equations, _, y0_int    = generate_random_instance(kn, m, rand_seed)
        eqs_red, feqs, _               = reduce_with_planted_solution(R, xs, equations, kn, m, y0_int, rand_seed)

    mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t_cpu0     = time.process_time()
    t_wall0    = time.time()

    try:
        gb = R.ideal(eqs_red + feqs).groebner_basis()

        cpu_time  = time.process_time() - t_cpu0
        wall_time = time.time()         - t_wall0
        mem_kb    = mem_delta_kb(
            mem_before, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        )

        solved  = not (len(gb) == 1 and gb[0] == R.one())
        max_deg = max((f.degree() for f in gb), default=0)

        return dict(
            success=solved, cpu_time_s=cpu_time, wall_time_s=wall_time,
            memory_kb=mem_kb, degree=max_deg, timed_out=False, error="",
        )

    except Exception as exc:
        cpu_time  = time.process_time() - t_cpu0
        wall_time = time.time()         - t_wall0
        return dict(
            success=False, cpu_time_s=cpu_time, wall_time_s=wall_time,
            memory_kb=0, degree=-1, timed_out=False, error=str(exc),
        )


# -- CSV writer --

def _write_row(writer, scale_idx, n, m, o, k, kn, inst, seed, itype, timeout, res):
    writer.writerow({
        "scale":         scale_idx,
        "n":             n,
        "m":             m,
        "o":             o,
        "k":             k,
        "kn":            kn,
        "instance":      inst,
        "seed":          seed,
        "instance_type": itype,
        "pair_id":       inst,   # same inst index => paired whipped/random rows
        "timeout_s":     timeout,
        "cpu_time_s":    res["cpu_time_s"],
        "wall_time_s":   res["wall_time_s"],
        "memory_kb":     res["memory_kb"],
        "degree":        res["degree"],
        "success":       res["success"],
        "timed_out":     res["timed_out"],
        "error":         res.get("error", ""),
    })


# -- Main experiment loop --

def run_h2():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    completed = load_completed(H2_CSV)
    if completed:
        print(f"Resuming -- {len(completed)} rows already recorded, skipping those.")

    write_header = not os.path.exists(H2_CSV) or os.path.getsize(H2_CSV) == 0
    with open(H2_CSV, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for scale_idx, (n, m, o, k) in enumerate(H2_SCALES, start=1):
            kn = k * n
            n_instances = SCALE_INSTANCES_OVERRIDE.get(scale_idx, INSTANCES_PER_SCALE)
            print(f"\n  Scale {scale_idx}: n={n}, m={m}, o={o}, k={k}, kn={kn}  ({n_instances} instances)")

            for inst in range(n_instances):
                base_seed = SEED_BASE + scale_idx * 10_000 + inst
                rand_seed = base_seed + 5_000_000

                # Whipped instance
                if (scale_idx, inst, "whipped") not in completed:
                    res_w = run_with_timeout(
                        _groebner_worker,
                        ("whipped", n, m, o, k, base_seed),
                        TIMEOUT_SECONDS,
                    )
                    _write_row(writer, scale_idx, n, m, o, k, kn, inst, base_seed, "whipped", TIMEOUT_SECONDS, res_w)
                    fh.flush()
                    w_tag = (f"{res_w['cpu_time_s']:.3f}s"
                             f"{'  [TO]' if res_w['timed_out'] else ''}")
                else:
                    w_tag = "skipped"

                # Random baseline instance
                if (scale_idx, inst, "random") not in completed:
                    res_r = run_with_timeout(
                        _groebner_worker,
                        ("random", n, m, o, k, base_seed),
                        TIMEOUT_SECONDS,
                    )
                    _write_row(writer, scale_idx, n, m, o, k, kn, inst, rand_seed, "random", TIMEOUT_SECONDS, res_r)
                    fh.flush()
                    r_tag = (f"{res_r['cpu_time_s']:.3f}s"
                             f"{'  [TO]' if res_r['timed_out'] else ''}")
                else:
                    r_tag = "skipped"

                if w_tag != "skipped" or r_tag != "skipped":
                    print(f"    inst {inst+1:2d}/{INSTANCES_PER_SCALE}"
                          f"  whipped={w_tag}  random={r_tag}")

    print(f"\nResults written to {H2_CSV}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    run_h2()
