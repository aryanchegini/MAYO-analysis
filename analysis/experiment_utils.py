#!/usr/bin/env python3
"""
Shared experimental infrastructure for the MAYO analysis experiments.

Provides the subprocess timeout runner (with hard-kill and RUSAGE_CHILDREN
CPU time recovery), peak-memory delta helper, and CSV checkpoint loader.
"""

import csv, os, resource, sys, time, multiprocessing, subprocess


def _total_ram_bytes():
    """
    Return total physical RAM in bytes, cross-platform.

    Linux: reads /proc/meminfo.
    macOS: reads hw.memsize via sysctl.
    Fallback: returns 8GB so the cap is still applied even if detection fails.
    """
    try:
        if sys.platform == "darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return int(out.strip())
        else:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 8 * 1024 ** 3


def _default_mem_limit():
    """Cap each child at 60% of total RAM, leaving headroom for OS and parent."""
    return int(_total_ram_bytes() * 0.6)


# Per-child virtual memory cap in bytes.  Prevents the Groebner computation
# from swapping indefinitely, which makes SIGKILL slow to take effect and
# can cause subsequent instances to compete for memory in a cascade.
# Defaults to 60% of total RAM.  Override by setting CHILD_MEM_LIMIT_GB.
_env_gb = os.environ.get("CHILD_MEM_LIMIT_GB")
CHILD_MEM_LIMIT = int(_env_gb) * 1024 ** 3 if _env_gb else _default_mem_limit()


def mem_delta_kb(before, after):
    """
    Convert an ru_maxrss delta to kilobytes.

    On macOS ru_maxrss is in bytes; on Linux it is already in kilobytes.
    """
    delta = after - before
    return delta // 1024 if sys.platform == "darwin" else delta


def _subprocess_entry(worker_fn, args, result_queue):
    # Cap virtual memory before doing any computation.  If the child exceeds
    # this limit it receives SIGSEGV/MemoryError and exits cleanly, which
    # means SIGKILL from the parent always takes effect in milliseconds rather
    # than blocking while the OS flushes dirty pages.
    try:
        resource.setrlimit(resource.RLIMIT_AS, (CHILD_MEM_LIMIT, CHILD_MEM_LIMIT))
    except ValueError:
        pass  # limit exceeds system maximum -- proceed without cap
    result = worker_fn(*args)
    result_queue.put(result)


def run_with_timeout(worker_fn, args, timeout):
    """
    Run worker_fn(*args) in a child process, hard-killing it after timeout
    wall-clock seconds if it has not finished.

    Unlike SIGALRM, p.kill() is delivered by the OS regardless of whether
    the child is sleeping, waiting on I/O, or swapped out.  This prevents
    the failure mode where SIGALRM is delayed by memory pressure or the OS
    scheduler -- a known failure mode where wall_time exceeded 8000s despite
    a nominal 1800s SIGALRM timeout in preliminary runs.

    CPU time is recovered via resource.getrusage(RUSAGE_CHILDREN) after the
    child is killed. A delta snapshot taken before p.start() isolates this
    child's contribution from any prior waited children in this process.
    This gives accurate cpu_time_s even on hard-kill timeout rows.

    The per-child memory cap in _subprocess_entry ensures SIGKILL is always
    fast: without it, a child with hundreds of MB of dirty memory can take
    minutes to reap, blocking the next instance and causing a cascade across
    50-instance runs at the large scales.

    The fallback dict (returned when the queue is empty, i.e. the child was
    killed or crashed) contains only the fields common to all experiments.
    Experiment-specific fields (n_solutions, solution, solution_valid) are
    added by the worker on success; callers should use .get() with a default
    when reading them from a result dict.
    """
    result_queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_subprocess_entry,
        args=(worker_fn, args, result_queue),
    )

    rusage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    t_wall0 = time.time()
    p.start()
    p.join(timeout=timeout)
    # Record wall time before kill overhead.  Under memory pressure,
    # p.join() after SIGKILL can block while the OS reclaims pages --
    # that latency must not appear in the result.
    wall_time = time.time() - t_wall0
    timed_out = p.is_alive()

    if timed_out:
        p.kill()
        p.join(timeout=5)
        if p.is_alive():
            # Still alive after SIGKILL, abandon the process. it will be reaped when the parent exits.
            # The memory cap in _subprocess_entry should eliminate the chance of this happening.
            print(f"WARNING: child {p.pid} did not die after SIGKILL, abandoning.")
            p.close()

    rusage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    child_cpu = (
        (rusage_after.ru_utime + rusage_after.ru_stime)
        - (rusage_before.ru_utime + rusage_before.ru_stime)
    )

    if not result_queue.empty():
        return result_queue.get()

    # Queue is empty: child was killed (timeout) or crashed.
    return dict(
        success=False,
        cpu_time_s=child_cpu,
        wall_time_s=wall_time,
        memory_kb=0,
        degree=-1,
        timed_out=timed_out,
        error="timeout" if timed_out else "subprocess_crashed",
    )


def load_completed(csv_path):
    """
    Return a set of (scale, instance, run_type) triples already recorded in
    csv_path.  Used to skip completed rows on resume after interruption.

    run_type is read from whichever column is present: 'solver' (H1) or
    'instance_type' (H2).
    """
    completed = set()
    if not os.path.exists(csv_path):
        return completed
    try:
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    run_type = row.get("solver") or row.get("instance_type", "")
                    completed.add((
                        int(row["scale"]),
                        int(row["instance"]),
                        run_type,
                    ))
                except (KeyError, ValueError):
                    pass
    except Exception:
        pass
    return completed
