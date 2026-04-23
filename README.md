# MAYO Analysis

Computational experiments analysing the hardness of the multivariate quadratic (MQ)
problem underlying the MAYO post-quantum signature scheme.

## Prerequisites

**SageMath** is required for all experiment scripts.  Download from
https://www.sagemath.org or install via your package manager:

```bash
# Ubuntu / Debian
sudo apt install sagemath

# macOS (Homebrew)
brew install --cask sage
```

Verify the installation:

```bash
sage --version
```

**pycryptosat** is required for the SAT solver phase of H1.  Install it inside
SageMath's Python environment:

```bash
sage -pip install pycryptosat
```

**Python packages** for the analysis / plotting scripts (`analyse_h1_h2.py`, `analyse_h3.py`).
These run under plain Python 3 and do not require SageMath:

```bash
pip install -r requirements.txt
```

If you prefer to keep everything inside SageMath's environment:

```bash
sage -pip install -r requirements.txt
```

## Repository layout

```
analysis/
  mq_GF16.py            GF(16) arithmetic, instance generation, SAT encoding tables
  experiment_utils.py   Subprocess timeout runner, memory cap, CSV checkpoint loader
  h1_experiment.py      H1: Groebner vs CryptoMiniSat on random MQ instances
  h2_experiment.py      H2: Groebner on whipped-map vs random-map instances
  h3_experiment.py      H3: Per-coordinate chi-squared uniformity test on MAYO signatures
  analyse_h1_h2.py      Statistical analysis and plots for H1 and H2 results
  analyse_h3.py         Chi-squared vs signing slack plots and per-scale heatmaps for H3
results/                Output CSVs and plots (created automatically on first run)
MAYO-sage/              Reference MAYO implementation (see note below)
```

### MAYO-sage reference implementation

`h3_experiment.py` imports from `MAYO-sage/sagelib/`.
This directory is not included in the repository.  Clone it alongside the analysis repo:

```bash
git clone https://github.com/PQCMayo/MAYO-sage.git MAYO-sage
```

The H1 and H2 experiments do **not** depend on MAYO-sage.

## Running the experiments

All experiment scripts are run with `sage -python` from the **repository root**.
They write results to `results/h1/`, `results/h2/`, and `results/h3/` respectively
(or to `results/h1_demo/` and `results/h2_demo/` in demo mode — see below).

### Demo run (~1–4 hours)

The H1 and H2 scripts have a `DEMO_MODE = True` flag near the top of the parameters
block.  In demo mode, fewer instances are run and the slowest scales are reduced or
dropped, so you can verify the pipeline and get indicative timings before committing
to an overnight run.

Run H1 and H2 sequentially (H2 starts only if H1 exits cleanly):

```bash
sage -python analysis/h1_experiment.py && sage -python analysis/h2_experiment.py
```

To run in the background and log all output:

```bash
nohup bash -c 'sage -python analysis/h1_experiment.py && sage -python analysis/h2_experiment.py' > run.log 2>&1 &
```

Progress can be monitored live:

```bash
tail -f run.log
```

The CSVs are flushed after every row, so you can also inspect partial results
directly:

```bash
cat results/h1_demo/h1_results.csv
cat results/h2_demo/h2_results.csv
```

### Full overnight run (H1 and H2)

1. Open `analysis/h1_experiment.py` and set `DEMO_MODE = False`.
2. Open `analysis/h2_experiment.py` and set `DEMO_MODE = False`.
3. Run as above.  Full run targets are 50 instances per scale across all scales;
   expect roughly 12–20 hours depending on hardware.

If interrupted, re-running the same command resumes from where it stopped — the
script reads the existing CSV on startup and skips any rows already recorded.

### H3: signature uniformity experiment

H3 requires `MAYO-sage/` to be present (see above).  It fixes n=20, m=20, o=4
and varies k in {5, 6, 7, 8, 9, 10} (giving ko − m in {0, 4, 8, 12, 16, 20}),
generating N=50,000 signatures per scale from a single key pair.

```bash
sage -python analysis/h3_experiment.py
```

Output is written to `results/h3/h3_data.csv` and `results/h3/h3_counts.csv`.
The number of signatures can be overridden:

```bash
H3_N_SAMPLES=10000 sage -python analysis/h3_experiment.py
```

The script checkpoints by gap value, so it can be interrupted and resumed.

### Memory cap

Each child process is capped at 60% of total RAM by default to prevent the OS
from swapping under memory pressure, which would make hard-kills slow and cause
a cascade across 50-instance runs.  Override if needed:

```bash
CHILD_MEM_LIMIT_GB=8 sage -python analysis/h1_experiment.py
```

## Running the analysis

Once the experiment CSVs exist, generate plots and summary statistics.

**H1 and H2:**

```bash
python3 analysis/analyse_h1_h2.py
```

Output is written to `results/h1/plots/` and `results/h2/plots/`, with summary
CSVs at `results/h1/h1_summary.csv` and `results/h2/h2_summary.csv`.

**H3:**

```bash
python3 analysis/analyse_h3.py
```

Output is written to `results/h3/`: a chi-squared vs signing slack plot
(`h3_chi2_vs_gap.pdf`) and per-gap Pearson-residual heatmaps
(`h3_heatmap_gap*.pdf`).

## Correctness check

Before running experiments on a new machine, verify the GF(16) arithmetic and
instance generation:

```bash
sage -python analysis/mq_GF16.py
```

All assertions should pass silently.
