#!/usr/bin/env python3
"""
H1 Experiment: Groebner basis vs CryptoMiniSat on random MQ instances over GF(16).

Run with:   sage -python analysis/h1_experiment.py
Requires:   SageMath, pycryptosat  (pip install pycryptosat inside sage's Python)
Output:     results/h1/h1_results.csv

Timeout method
--------------
Each solver call is executed in a child process via multiprocessing.Process.
The parent waits up to `timeout` wall-clock seconds and then hard-kills the
child with p.kill() if it has not finished.

CPU time for each child (including hard-killed timeout rows) is recovered in
the parent via resource.getrusage(RUSAGE_CHILDREN) after the child process is killed.

Checkpoint / resume
-------------------
On startup the script reads any existing rows from H1_CSV and builds a set
of (scale, instance, solver) triples already completed.  Any row already
present is skipped, so the experiment can be safely interrupted and restarted
without losing progress.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import csv, json, resource, time, multiprocessing

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
    print("WARNING: pycryptosat not found -- SAT phase will be skipped. Install with:  sage -pip install pycryptosat")

# Experiment parameters

DEMO_MODE = False

# ---------------------------------------------------------------------------
# Scale table: (scale_idx, n, m)
# Groebner / SAT coverage and instance counts are controlled separately below.
# ---------------------------------------------------------------------------
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

# Full instance count used for all priority scales.
INSTANCES_PER_SCALE = 1 if DEMO_MODE else 50

OUTPUT_DIR = os.path.join("results", "h1_demo" if DEMO_MODE else "h1")

# ---------------------------------------------------------------------------
# Groebner configuration
#   Priority scales (1-6) run first, each with INSTANCES_PER_SCALE instances.
#   Remaining scales run after the SAT priority phase, with the instance
#   counts specified here.
# ---------------------------------------------------------------------------
GROEBNER_PRIORITY_SCALES = list(range(1, 7)) # scales 1-6

GROEBNER_REMAINING_INSTANCES = {  
    7: 50,
    8: 5,
    9: 1,
}

# ---------------------------------------------------------------------------
# SAT configuration
#   Priority scales (1-4) run after the Groebner priority phase, each with
#   INSTANCES_PER_SCALE instances.
# ---------------------------------------------------------------------------
SAT_PRIORITY_SCALES = list(range(1, 5)) # scales 1-4

SAT_REMAINING_INSTANCES = {
    5: 1,
}

# ---------------------------------------------------------------------------
# Remaining phase order
#   After the two priority phases, remaining (solver, scale) pairs are run in
#   the order listed here.  Edit freely to interleave Groebner and SAT scales
#   however you like.  Instance counts come from GROEBNER/SAT_REMAINING_INSTANCES.
# ---------------------------------------------------------------------------
REMAINING_PHASE_ORDER = [
    ("groebner", 7),   # Groebner n=10, 50 instances
    ("groebner", 8),   # Groebner n=11, 10 instances
    ("sat",      5),   # SAT      n=8,  10 instances
    ("groebner", 9),   # Groebner n=12,  5 instances
]

# Wall-clock timeout per scale index in seconds. Estimated timmings recorded on M1 Macbook Air 8GB Mem
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


def _n_instances_for_scale(scale_idx):
    """Max instances to pre-generate for a scale (union of all phases)."""
    counts = []
    if scale_idx in GROEBNER_PRIORITY_SCALES:
        counts.append(INSTANCES_PER_SCALE)
    if scale_idx in GROEBNER_REMAINING_INSTANCES:
        counts.append(GROEBNER_REMAINING_INSTANCES[scale_idx])
    if scale_idx in SAT_PRIORITY_SCALES:
        counts.append(INSTANCES_PER_SCALE)
    if scale_idx in SAT_REMAINING_INSTANCES:
        counts.append(SAT_REMAINING_INSTANCES[scale_idx])
    return max(counts, default=0)

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


def _groebner_worker(n, m, polys_data, target_data):
    """
    Solve via SageMath Groebner basis algorithm (likely Singular F4).

    Field equations x_i^16 - x_i = 0 are appended to restrict solutions to
    GF(16)^n and ensure the ideal is zero-dimensional, which is required for
    Singular's F4 to terminate and for .variety() to work correctly.

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

        # GB = {1} means the ideal is the whole ring => no solution in GF(16)^n.
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
    bit pairs per _GF16_MUL_PATTERN.  Each unique AND-product is assigned one
    Tseitin auxiliary variable, shared across all polynomials (deduplication).

    Multiplication by a known constant c is a linear map over GF(2) (a 4x4
    binary matrix from _GF16_CONST_MAT) and introduces no new AND-products.

    Each of the 4 bit-lanes per polynomial equation becomes one XOR constraint
    added via CryptoMiniSat's native add_xor_clause() interface.  CMS handles
    XOR natively via Gaussian elimination.

    Memory is measured from before clause building through solve() so that
    encoding cost is included in the reported figure.

    Runs inside a child process; no internal timeout is needed.
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


# Main experiment
def run_h1():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    completed = load_completed(H1_CSV)
    if completed:
        print(f"Resuming -- {len(completed)} rows already recorded, skipping those.")

    # Pre-generate all instances so both solvers receive identical problem data.
    # Generate up to the maximum instance count needed across all phases per scale.
    all_instances = {}   # scale_idx -> list of (inst, seed, polys_data, target_data)
    for scale_idx, n, m in SCALES:
        n_inst = _n_instances_for_scale(scale_idx)
        if n_inst == 0:
            continue
        instances = []
        for inst in range(n_inst):
            seed = SEED_BASE + scale_idx * 10_000 + inst
            polys_data, target_data = generate_mq_coeffs(n, m, seed)
            instances.append((inst, seed, polys_data, target_data))
        all_instances[scale_idx] = instances

    def _run_phase(phase_name, solver_name, scale_list, n_inst_map, timeout_map,
                   worker_fn, writer, fh):
        """Run one phase, iterating scale_list in order."""
        if solver_name == "sat" and not CMS_AVAILABLE:
            print(f"\nSkipping {phase_name} -- pycryptosat not available.")
            return
        print(f"\n{phase_name}")
        for scale_idx in scale_list:
            if scale_idx not in all_instances:
                continue
            n, m      = SCALE_MAP[scale_idx]
            n_inst    = n_inst_map(scale_idx)
            instances = all_instances[scale_idx][:n_inst]
            print(f"\n  Scale {scale_idx}: n={n}, m={m}  ({n_inst} instances)")
            timeout = timeout_map[scale_idx]
            for inst, seed, polys_data, target_data in instances:
                if (scale_idx, inst, solver_name) in completed:
                    continue
                res = run_with_timeout(worker_fn, (n, m, polys_data, target_data), timeout)
                _write_row(writer, scale_idx, n, m, inst, seed, solver_name, timeout, res)
                fh.flush()
                tag = (f"{res['cpu_time_s']:.3f}s"
                       f"{'  [TO]' if res['timed_out'] else ''}")
                label = "GB" if solver_name == "groebner" else "SAT"
                print(f"    inst {inst+1:2d}/{n_inst}  {label}={tag}  seed={seed}")

    write_header = not os.path.exists(H1_CSV) or os.path.getsize(H1_CSV) == 0
    with open(H1_CSV, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        # Phase 1: Groebner — priority scales 1-6, full INSTANCES_PER_SCALE each
        _run_phase(
            "Phase 1: Groebner (priority scales 1-6)",
            "groebner",
            GROEBNER_PRIORITY_SCALES,
            lambda _: INSTANCES_PER_SCALE,
            GROEBNER_TIMEOUT_PER_SCALE,
            _groebner_worker, writer, fh,
        )

        # Phase 2: SAT — priority scales 1-4, full INSTANCES_PER_SCALE each
        _run_phase(
            "Phase 2: SAT (priority scales 1-4)",
            "sat",
            SAT_PRIORITY_SCALES,
            lambda _: INSTANCES_PER_SCALE,
            SAT_TIMEOUT_PER_SCALE,
            _sat_worker, writer, fh,
        )

        # Phases 3+: remaining (solver, scale) pairs in user-defined order
        remaining_instance_maps = {
            "groebner": GROEBNER_REMAINING_INSTANCES,
            "sat":      SAT_REMAINING_INSTANCES,
        }
        remaining_timeout_maps = {
            "groebner": GROEBNER_TIMEOUT_PER_SCALE,
            "sat":      SAT_TIMEOUT_PER_SCALE,
        }
        remaining_workers = {
            "groebner": _groebner_worker,
            "sat":      _sat_worker,
        }
        for phase_num, (solver_name, scale_idx) in \
                enumerate(REMAINING_PHASE_ORDER, start=3):
            inst_map  = remaining_instance_maps[solver_name]
            n_inst    = inst_map.get(scale_idx, INSTANCES_PER_SCALE)
            _run_phase(
                f"Phase {phase_num}: {solver_name.upper()} scale {scale_idx} "
                f"({n_inst} instances)",
                solver_name,
                [scale_idx],
                lambda _, n=n_inst: n,
                remaining_timeout_maps[solver_name],
                remaining_workers[solver_name], writer, fh,
            )

    print(f"\nResults written to {H1_CSV}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    run_h1()
