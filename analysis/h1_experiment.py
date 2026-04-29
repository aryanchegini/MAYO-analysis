#!/usr/bin/env python3
"""
H1 Experiment: Groebner basis vs CryptoMiniSat on random MQ instances over GF(16).

Run with:   sage -python analysis/h1_experiment.py
Requires:   SageMath, pycryptosat  (pip install pycryptosat inside sage's Python)
Output:     results/h1/h1_results.csv

Runs each solver over each scale in order. Completed (scale, instance, solver) triples are skipped on restart.
"""

import sys, os, csv, json, resource, time, multiprocessing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mq_GF16 import (
    F16_TO_INT,
    _GF16_CONST_MAT, _GF16_MUL_PATTERN,
    generate_mq_coeffs, coeffs_to_sage_polys, validate_solution,
)
from experiment_utils import mem_delta_kb, run_with_timeout, load_completed

# Check if CryptoMiniSat is available
try:
    from pycryptosat import Solver as _CMSSolver
    CMS_AVAILABLE = True
except ImportError:
    CMS_AVAILABLE = False
    print("WARNING: pycryptosat not found - SAT phase will be skipped. Install with:  sage -pip install pycryptosat")


# ---------------------------------------------------------------------------
# Experiment parameters

# Scale table: (scale_idx, n, m)
# Groebner / SAT coverage and instance counts are controlled separately below.
SCALES = [
    (1, 4,  4),
    (2, 5,  5),
    (3, 6,  6),
    (4, 7,  7),
    (5, 8,  8),
    (6, 9,  9),
    (7, 10, 10),
    (8, 11, 11),
    (9, 12, 12),
]

OUTPUT_DIR = os.path.join("results", "h1")
NUM_INSTANCES = 50

# Wall-clock timeout per scale index in seconds. Estimated timings recorded on M1 MacBook Air 8GB Mem
GROEBNER_TIMEOUT_PER_SCALE = {
    1: 5,     # n=4
    2: 10,    # n=5
    3: 20,    # n=6:  groebner ~0.2s
    4: 30,    # n=7:  groebner ~1.4s
    5: 60,    # n=8:  groebner ~10s
    6: 600,   # n=9:  groebner ~50s
    7: 1000,  # n=10: groebner ~280s
    8: 3000,  # n=11: groebner ~1500s
    9: 5000, # n=12: groebner >80 mins (most likely will leave it out and report this)
}
SAT_TIMEOUT_PER_SCALE = {
    1: 10,    # n=4
    2: 30,    # n=5
    3: 200,   # n=6:  SAT ~40s
    4: 1000,  # n=7:  SAT ~300s
    5: 5000,  # n=8:  SAT >80 mins
}

SEED_BASE = 12345
H1_CSV    = os.path.join(OUTPUT_DIR, "h1_results.csv")

# Scale lookup: scale_idx -> (n, m)
SCALE_MAP = {s: (n, m) for s, n, m in SCALES}

FIELDNAMES = [
    "scale", "n", "m", "instance", "seed", "solver",
    "timeout_s",
    "cpu_time_s", "wall_time_s", "memory_kb",
    "degree",
    "success", "timed_out", "error",
    "n_solutions",
    "solution",
    "solution_valid",
]

# ---------------------------------------------------------------------------

def _groebner_worker(n, m, polys_data, target_data):
    """
    Solve via SageMath Groebner basis algorithm (likely Singular F4).

    Field equations x_i^16 - x_i = 0 are appended to restrict solutions to
    GF(16)^n and ensure the ideal is zero-dimensional.

    Runs inside a child process. Parent hard-kills process if wall-clock time is exceeded.
    """
    R, xs, polys, target = coeffs_to_sage_polys(n, m, polys_data, target_data)
    equations = [polys[a] - target[a] for a in range(m)]
    field_eqs = [xs[i]**16 - xs[i] for i in range(n)]

    mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t_cpu0     = time.process_time()
    t_wall0    = time.time()

    try:
        gb = R.ideal(equations + field_eqs).groebner_basis()

        cpu_time  = time.process_time() - t_cpu0
        wall_time = time.time()         - t_wall0
        mem_kb    = mem_delta_kb(
            mem_before, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        )

        # GB = {1} means the ideal is the whole ring -> no solution in GF(16)^n.
        solved  = not (len(gb) == 1 and gb[0] == R.one())
        max_deg = max((f.degree() for f in gb), default=0)

        solution_json  = ""
        solution_valid = ""
        n_solutions    = 0

        if solved:
            variety     = R.ideal(gb).variety()
            n_solutions = len(variety)
            if variety:
                pt             = variety[0]
                vals           = [F16_TO_INT[pt[xs[i]]] for i in range(n)]
                solution_json  = json.dumps(vals)
                solution_valid = validate_solution(m, polys_data, target_data, vals)

        return dict(
            success=solved, cpu_time_s=cpu_time, wall_time_s=wall_time,
            memory_kb=mem_kb, degree=max_deg, timed_out=False, error="",
            n_solutions=n_solutions, solution=solution_json,
            solution_valid=solution_valid,
        )

    except Exception as exc:
        cpu_time  = time.process_time() - t_cpu0
        wall_time = time.time()         - t_wall0
        return dict(
            success=False, cpu_time_s=cpu_time, wall_time_s=wall_time,
            memory_kb=0, degree=-1, timed_out=False, error=str(exc),
            n_solutions="", solution="", solution_valid="",
        )
    

def _sat_worker(n, m, polys_data, target_data):
    """
    Solve via CryptoMiniSat with a GF(16) Boolean encoding.

    Encoding
    --------
    Each GF(16) variable x_i is represented by 4 Boolean variables
    x_i_0..x_i_3 (bits of the polynomial basis over GF(2) w.r.t.
    {1, alpha, alpha^2, alpha^3}).

    Each quadratic monomial x_i * x_j expands into Boolean AND-products of
    bit pairs per _GF16_MUL_PATTERN. Each unique AND-product is assigned one
    Tseitin auxiliary variable, shared across all polynomials (deduplication).

    Multiplication by a known constant c is a linear map over GF(2) (a 4x4
    binary matrix from _GF16_CONST_MAT) and introduces no new AND-products.

    Each of the 4 bit-lanes per polynomial equation becomes one XOR constraint
    added via CryptoMiniSat's native add_xor_clause() interface. CMS handles
    XOR natively via Gaussian elimination.

    Runs inside a child process. Parent hard-kills process if wall-clock time is exceeded.
    """
    from collections import Counter
    from pycryptosat import Solver

    solver       = Solver(threads=1)
    next_var     = [n * 4 + 1]
    product_vars = {}
    mul_cache    = {}

    def new_var():
        v = next_var[0]; next_var[0] += 1; return v

    def x_var(i, k):
        return i * 4 + k + 1

    def get_product(va, vb):
        if va == vb:
            return va
        key = (min(va, vb), max(va, vb))
        if key not in product_vars:
            pv = new_var()
            product_vars[key] = pv
            a, b = key
            solver.add_clause([-pv,a])
            solver.add_clause([-pv,b])
            solver.add_clause([-a,-b,pv])
        return product_vars[key]

    def gf16_mul_var_lists(i, j):
        key = (min(i, j), max(i, j))
        if key not in mul_cache:
            ci, cj = key
            mul_cache[key] = [
                [get_product(x_var(ci, ki), x_var(cj, kj)) for ki, kj in pat]
                for pat in _GF16_MUL_PATTERN
            ]
        return mul_cache[key]

    def const_scale(c, xor_lists):
        rows = _GF16_CONST_MAT[c]
        return [
            [v for src in range(4) if (rows[b] >> src) & 1
               for v in xor_lists[src]]
            for b in range(4)
        ]

    mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t_cpu0     = time.process_time()
    t_wall0    = time.time()

    for a in range(m):
        acc = [[] for _ in range(4)]
        for (i, j), c in polys_data[a].items():
            if c == 0:
                continue
            scaled = const_scale(c, gf16_mul_var_lists(i, j))
            for b in range(4):
                acc[b].extend(scaled[b])

        t = target_data[a]
        for b in range(4):
            cntr  = Counter(acc[b])
            xvars = [v for v, cnt in cntr.items() if cnt % 2 == 1]
            rhs   = bool((t >> b) & 1)
            if not xvars:
                if rhs:
                    solver.add_clause([])
            else:
                solver.add_xor_clause(xvars, rhs)

    try:
        sat, assignment = solver.solve()
    except Exception as exc:
        cpu_time  = time.process_time() - t_cpu0
        wall_time = time.time()         - t_wall0
        return dict(
            success=False, cpu_time_s=cpu_time, wall_time_s=wall_time,
            memory_kb=0, timed_out=False, error=str(exc),
            n_solutions="", solution="", solution_valid="",
        )

    cpu_time  = time.process_time() - t_cpu0
    wall_time = time.time()         - t_wall0
    mem_kb    = mem_delta_kb(
        mem_before, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    )

    solved    = (sat is True)
    timed_out = (sat is None)
    error     = "cms_interrupted" if timed_out else ""

    solution_json  = ""
    solution_valid = ""
    n_solutions    = ""

    if not timed_out:
        if not solved:
            n_solutions = 0
        elif assignment is not None:
            n_solutions = 1
            vals = [
                sum(int(bool(assignment[i * 4 + k + 1])) << k for k in range(4))
                for i in range(n)
            ]
            solution_json  = json.dumps(vals)
            solution_valid = validate_solution(m, polys_data, target_data, vals)

    return dict(
        success=solved, cpu_time_s=cpu_time, wall_time_s=wall_time,
        memory_kb=mem_kb, timed_out=timed_out, error=error,
        n_solutions=n_solutions, solution=solution_json,
        solution_valid=solution_valid,
    )


# CSV writer
def _write_row(writer, scale_idx, n, m, inst, seed, solver_name, timeout, res):
    writer.writerow({
        "scale":          scale_idx,
        "n":              n,
        "m":              m,
        "instance":       inst,
        "seed":           seed,
        "solver":         solver_name,
        "timeout_s":      timeout,
        "cpu_time_s":     res["cpu_time_s"],
        "wall_time_s":    res["wall_time_s"],
        "memory_kb":      res["memory_kb"],
        "degree":         res.get("degree",         ""),
        "success":        res["success"],
        "timed_out":      res["timed_out"],
        "error":          res.get("error",          ""),
        "n_solutions":    res.get("n_solutions",    ""),
        "solution":       res.get("solution",       ""),
        "solution_valid": res.get("solution_valid", ""),
    })


# Main experiment loop
def run_h1():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    completed = load_completed(H1_CSV)
    if completed:
        print(f"Resuming:{len(completed)} rows already recorded, skipping those.")

    # Pre-generate instances so both solvers receive identical problem data.
    all_instances = {}  # scale_idx -> list of (inst, seed, polys_data, target_data)
    for scale_idx, n, m in SCALES:
        instances = []
        for inst in range(NUM_INSTANCES):
            seed = SEED_BASE + scale_idx * 10_000 + inst
            polys_data, target_data = generate_mq_coeffs(n, m, seed)
            instances.append((inst, seed, polys_data, target_data))
        all_instances[scale_idx] = instances

    solvers = [
        ("groebner", _groebner_worker, GROEBNER_TIMEOUT_PER_SCALE),
    ]
    if CMS_AVAILABLE:
        solvers.append(("sat", _sat_worker, SAT_TIMEOUT_PER_SCALE))
    else:
        print("Skipping SAT - pycryptosat not available.")

    write_header = not os.path.exists(H1_CSV) or os.path.getsize(H1_CSV) == 0
    with open(H1_CSV, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for solver_name, worker_fn, timeout_map in solvers:
            print(f"\n=== {solver_name.upper()} ===")
            for scale_idx, n, m in SCALES:
                if scale_idx not in timeout_map:
                    continue
                timeout = timeout_map[scale_idx]
                print(f"\n  Scale {scale_idx}: n={n}, m={m}  ({NUM_INSTANCES} instances)")
                for inst, seed, polys_data, target_data in all_instances[scale_idx]:
                    if (scale_idx, inst, solver_name) in completed:
                        continue
                    res = run_with_timeout(worker_fn, (n, m, polys_data, target_data), timeout)
                    _write_row(writer, scale_idx, n, m, inst, seed, solver_name, timeout, res)
                    fh.flush()
                    tag = f"{res['cpu_time_s']:.3f}s{'  [TO]' if res['timed_out'] else ''}"
                    label = "GB" if solver_name == "groebner" else "SAT"
                    print(f"    inst {inst+1:2d}/{NUM_INSTANCES}  {label}={tag}  seed={seed}")

    print(f"\nResults written to {H1_CSV}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    run_h1()
