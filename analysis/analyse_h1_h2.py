#!/usr/bin/env python3
"""
Analysis and plotting for H1 and H2 experiments.

Run with:  python3 analysis/analyse_h1_h2.py
Requires:  numpy, scipy, matplotlib, pandas
Output:    results/h1/plots/  (H1 PNG files)
           results/h2/plots/  (H2 PNG files)
           results/h1/h1_summary.csv
           results/h2/h2_summary.csv
           results/h2/h2_mannwhitney.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
from scipy.optimize import curve_fit

RESULTS_DIR = "results"
H1_DIR      = os.path.join(RESULTS_DIR, "h1")
H2_DIR      = os.path.join(RESULTS_DIR, "h2")
H1_PLOTS    = os.path.join(H1_DIR, "plots")
H2_PLOTS    = os.path.join(H2_DIR, "plots")
H1_CSV      = os.path.join(H1_DIR, "h1_results.csv")
H2_CSV      = os.path.join(H2_DIR, "h2_results.csv")

os.makedirs(H1_PLOTS, exist_ok=True)
os.makedirs(H2_PLOTS, exist_ok=True)

# -- Matplotlib style --

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  10,
    "figure.dpi":       150,
})

GROEBNER_COLOR = "#2166ac"   # blue
SAT_COLOR      = "#d6604d"   # red
WHIPPED_COLOR  = "#1b7837"   # green
RANDOM_COLOR   = "#762a83"   # purple


# -- H1: Groebner basis vs SAT scaling --

def load_h1():
    df = pd.read_csv(H1_CSV)
    df["cpu_time_s"]  = pd.to_numeric(df["cpu_time_s"],  errors="coerce")
    df["wall_time_s"] = pd.to_numeric(df["wall_time_s"], errors="coerce")
    df["memory_kb"]   = pd.to_numeric(df["memory_kb"],   errors="coerce")
    df["degree"]      = pd.to_numeric(df["degree"],      errors="coerce")
    df["timed_out"]   = df["timed_out"].astype(str).str.lower().isin(["true", "1"])
    df["success"]     = df["success"].astype(str).str.lower().isin(["true", "1"])
    return df


def h1_summary_table(df):
    """Return a per-scale, per-solver summary table using only non-timeout rows."""
    rows = []
    for solver in ("groebner", "sat"):
        sub = df[(df["solver"] == solver) & (~df["timed_out"])]
        for (scale, n, m), g in sub.groupby(["scale", "n", "m"]):
            times    = g["cpu_time_s"].dropna()
            n_total  = len(df[(df["solver"] == solver) & (df["n"] == n)])
            n_timeout = int(df[(df["solver"] == solver) & (df["n"] == n)]["timed_out"].sum())
            if len(times) == 0:
                continue
            rows.append({
                "scale":    int(scale),
                "n":        int(n),
                "solver":   solver,
                "median_s": np.median(times),
                "mean_s":   np.mean(times),
                "std_s":    np.std(times, ddof=1) if len(times) > 1 else float("nan"),
                "min_s":    np.min(times),
                "max_s":    np.max(times),
                "timeouts": n_timeout,
                "n_inst":   n_total,
            })
    return pd.DataFrame(rows).sort_values(["scale", "solver"])


def _exp_model(n, c, alpha):
    """Exponential model T = c * exp(alpha * n)."""
    return c * np.exp(alpha * n)


def fit_scaling_exponent(df, solver):
    """
    Fit T = c * exp(alpha * n) to median solve times.

    Only non-timeout rows are used -- timeout rows give a lower bound on the
    true solve time and would bias the fit downward.
    Returns (alpha, c, r_squared).
    """
    sub = df[(df["solver"] == solver) & (~df["timed_out"])]
    pts = (sub.groupby("n")["cpu_time_s"]
              .median()
              .reset_index()
              .sort_values("n"))
    pts = pts[pts["cpu_time_s"] > 0]

    if len(pts) < 3:
        return None, None, None

    n_vals = pts["n"].values.astype(float)
    t_vals = pts["cpu_time_s"].values

    try:
        popt, _ = curve_fit(_exp_model, n_vals, t_vals,
                            p0=[t_vals[0], 0.5], maxfev=5000)
        c, alpha = popt
        t_pred   = _exp_model(n_vals, c, alpha)
        ss_res   = np.sum((t_vals - t_pred)**2)
        ss_tot   = np.sum((t_vals - np.mean(t_vals))**2)
        r2       = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return alpha, c, r2
    except Exception:
        return None, None, None


def plot_h1_scaling(df, out_dir=H1_PLOTS):
    """Log-scale CPU time vs n, both solvers, with IQR error bars.

    Timeout rows are excluded from medians and IQR; scales where a solver has
    no non-timeout data are omitted from that solver's line.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    for solver, color, label in [
        ("groebner", GROEBNER_COLOR, "Groebner basis (F4)"),
        ("sat",      SAT_COLOR,      "CryptoMiniSat"),
    ]:
        sub = df[(df["solver"] == solver) & (~df["timed_out"])]
        groups = sub.groupby("n")["cpu_time_s"]

        ns      = sorted(groups.groups.keys())
        data    = [groups.get_group(n).dropna() for n in ns]
        ns      = [n for n, d in zip(ns, data) if len(d) > 0]
        data    = [d for d in data if len(d) > 0]
        if not ns:
            continue

        medians = [np.median(d) for d in data]
        q25     = [np.percentile(d, 25) for d in data]
        q75     = [np.percentile(d, 75) for d in data]
        yerr_lo = np.array(medians) - np.array(q25)
        yerr_hi = np.array(q75)     - np.array(medians)

        ax.errorbar(ns, medians, yerr=[yerr_lo, yerr_hi],
                    marker="o", color=color, label=label,
                    capsize=4, linewidth=1.8, markersize=5)

        alpha, c, r2 = fit_scaling_exponent(df, solver)
        if alpha is not None:
            ax.annotate(f"alpha={alpha:.3f} (R2={r2:.2f})",
                        xy=(ns[-1], medians[-1]),
                        xytext=(-60, 15), textcoords="offset points",
                        color=color, fontsize=9,
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

    ax.set_yscale("log")
    ax.set_xlabel("Number of variables  n  (= m,  square regime)")
    ax.set_ylabel("CPU time (s)  [log scale]")
    ax.set_title("H1 -- Solver scaling over GF(16), square MQ instances")
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = os.path.join(out_dir, "h1_scaling_time.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def plot_h1_boxplots(df, out_dir=H1_PLOTS):
    """Side-by-side box plots of CPU time at each scale.

    Scales where a solver has no data are shown with a single empty box.
    Timeout rows are excluded.
    """
    scales = sorted(df["n"].unique())
    fig, axes = plt.subplots(1, len(scales), figsize=(3 * len(scales), 4),
                             sharey=True)
    if len(scales) == 1:
        axes = [axes]

    for ax, n in zip(axes, scales):
        sub     = df[~df["timed_out"]]
        data_gb  = sub[(sub["solver"] == "groebner") & (sub["n"] == n)]["cpu_time_s"].dropna()
        data_sat = sub[(sub["solver"] == "sat")      & (sub["n"] == n)]["cpu_time_s"].dropna()

        plot_data  = [data_gb]
        plot_labels = ["GB"]
        if len(data_sat) > 0:
            plot_data.append(data_sat)
            plot_labels.append("SAT")

        bp = ax.boxplot(plot_data,
                        labels=plot_labels,
                        patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
        bp["boxes"][0].set_facecolor(GROEBNER_COLOR + "80")
        if len(bp["boxes"]) > 1:
            bp["boxes"][1].set_facecolor(SAT_COLOR + "80")

        ax.set_title(f"n = m = {n}")
        ax.set_xlabel("Solver")

    axes[0].set_ylabel("CPU time (s)")
    fig.suptitle("H1 -- CPU time distribution per scale", y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, "h1_boxplots.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_h1_memory(df, out_dir=H1_PLOTS):
    """Memory usage vs n, both solvers. Timeout rows excluded (memory_kb=0)."""
    fig, ax = plt.subplots(figsize=(6, 4))

    for solver, color, label in [
        ("groebner", GROEBNER_COLOR, "Groebner basis"),
        ("sat",      SAT_COLOR,      "CryptoMiniSat"),
    ]:
        sub    = df[(df["solver"] == solver) & (~df["timed_out"]) & (df["memory_kb"] > 0)]
        groups = sub.groupby("n")["memory_kb"]
        ns     = sorted(groups.groups.keys())
        if not ns:
            continue
        medians = [np.median(groups.get_group(n).dropna()) for n in ns]
        ax.plot(ns, medians, marker="o", color=color, label=label, linewidth=1.8)

    ax.set_yscale("log")
    ax.set_xlabel("Number of variables  n")
    ax.set_ylabel("Peak memory delta (KB)  [log scale]")
    ax.set_title("H1 -- Memory usage scaling")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = os.path.join(out_dir, "h1_memory.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def plot_h1_groebner_degree(df, out_dir=H1_PLOTS):
    """Max Groebner basis degree vs n. Timeout and infeasible rows excluded."""
    fig, ax = plt.subplots(figsize=(6, 4))

    sub    = df[(df["solver"] == "groebner") & (~df["timed_out"]) & (df["degree"] > 0)]
    groups = sub.groupby("n")["degree"]
    ns     = sorted(groups.groups.keys())
    medians = [np.median(groups.get_group(n).replace(-1, np.nan).dropna()) for n in ns]

    ax.plot(ns, medians, marker="s", color=GROEBNER_COLOR, linewidth=1.8)
    ax.set_xlabel("n")
    ax.set_ylabel("Median max degree in Groebner basis")
    ax.set_title("H1 -- Groebner basis: max degree reached")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = os.path.join(out_dir, "h1_groebner_degree.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# -- H2: Whipped vs random structural comparison --

def load_h2():
    df = pd.read_csv(H2_CSV)
    df["cpu_time_s"]  = pd.to_numeric(df["cpu_time_s"],  errors="coerce")
    df["wall_time_s"] = pd.to_numeric(df["wall_time_s"], errors="coerce")
    df["memory_kb"]   = pd.to_numeric(df["memory_kb"],   errors="coerce")
    df["degree"]      = pd.to_numeric(df["degree"],      errors="coerce")
    df["timed_out"]   = df["timed_out"].astype(str).str.lower().isin(["true", "1"])
    df["success"]     = df["success"].astype(str).str.lower().isin(["true", "1"])
    return df


def h2_summary_table(df):
    """Return a per-scale, per-type summary table using only non-timeout rows."""
    rows = []
    for itype in ("whipped", "random"):
        sub = df[(df["instance_type"] == itype) & (~df["timed_out"])]
        for (scale, n, m, o, k), g in sub.groupby(["scale", "n", "m", "o", "k"]):
            times    = g["cpu_time_s"].dropna()
            n_total  = len(df[(df["instance_type"] == itype) & (df["scale"] == scale)])
            n_timeout = int(df[(df["instance_type"] == itype) & (df["scale"] == scale)]["timed_out"].sum())
            if len(times) == 0:
                continue
            rows.append({
                "scale":      int(scale),
                "n":          int(n),
                "m":          int(m),
                "o":          int(o),
                "k":          int(k),
                "type":       itype,
                "median_s":   np.median(times),
                "mean_s":     np.mean(times),
                "std_s":      np.std(times, ddof=1) if len(times) > 1 else float("nan"),
                "timeouts":   n_timeout,
                "n_inst":     n_total,
            })
    return pd.DataFrame(rows).sort_values(["scale", "type"])


def h2_mann_whitney(df):
    """
    Mann-Whitney U test at each scale comparing whipped vs random CPU times.

    Only non-timeout rows are used.  H2 is supported if p-values are not
    significant (alpha = 0.05), indicating no detectable difference between
    whipped and random solving difficulty.
    """
    results = []
    for scale, g in df.groupby("scale"):
        g_ok    = g[~g["timed_out"]]
        w_times = g_ok[g_ok["instance_type"] == "whipped"]["cpu_time_s"].dropna().values
        r_times = g_ok[g_ok["instance_type"] == "random" ]["cpu_time_s"].dropna().values
        if len(w_times) < 5 or len(r_times) < 5:
            continue
        stat, p = stats.mannwhitneyu(w_times, r_times, alternative="two-sided")
        results.append({
            "scale":              int(scale),
            "n_w":                len(w_times),
            "n_r":                len(r_times),
            "U_stat":             stat,
            "p_value":            p,
            "significant_at_0.05": p < 0.05,
        })
    return pd.DataFrame(results)


def plot_h2_boxplots(df, out_dir=H2_PLOTS):
    """Paired box plots: whipped vs random at each scale. Timeout rows excluded."""
    scales     = sorted(df["scale"].unique())
    scale_info = {int(r["scale"]): r for _, r in
                  df[["scale", "n", "m", "o", "k"]].drop_duplicates("scale").iterrows()}

    fig, axes = plt.subplots(1, len(scales), figsize=(3.5 * len(scales), 4),
                             sharey=True)
    if len(scales) == 1:
        axes = [axes]

    for ax, sc in zip(axes, scales):
        info = scale_info[sc]
        sub  = df[(df["scale"] == sc) & (~df["timed_out"])]
        w_t  = sub[sub["instance_type"] == "whipped"]["cpu_time_s"].dropna()
        r_t  = sub[sub["instance_type"] == "random" ]["cpu_time_s"].dropna()

        bp = ax.boxplot([w_t, r_t],
                        labels=["Whipped\nP*", "Random\nQ"],
                        patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
        bp["boxes"][0].set_facecolor(WHIPPED_COLOR + "80")
        if len(bp["boxes"]) > 1:
            bp["boxes"][1].set_facecolor(RANDOM_COLOR + "80")

        ax.set_title(f"n={int(info['n'])}, m={int(info['m'])},\no={int(info['o'])}, k={int(info['k'])}")

    axes[0].set_ylabel("CPU time (s)")
    fig.suptitle("H2 -- Whipped P* vs random Q: solving time per scale", y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, "h2_boxplots.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_h2_ratio(df, out_dir=H2_PLOTS):
    """
    Ratio plot: median(whipped) / median(random) at each scale.
    Ratio ~= 1.0 supports H2 (no structural advantage).
    Only non-timeout rows are used.
    """
    scales     = sorted(df["scale"].unique())
    scale_info = {int(r["scale"]): r for _, r in
                  df[["scale", "m"]].drop_duplicates("scale").iterrows()}

    ratios = []
    m_vals = []
    for sc in scales:
        sub   = df[(df["scale"] == sc) & (~df["timed_out"])]
        w_med = np.median(sub[sub["instance_type"] == "whipped"]["cpu_time_s"].dropna())
        r_med = np.median(sub[sub["instance_type"] == "random" ]["cpu_time_s"].dropna())
        ratios.append(w_med / r_med if r_med > 0 else float("nan"))
        m_vals.append(int(scale_info[sc]["m"]))

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(m_vals, ratios, marker="o", linewidth=2, color=WHIPPED_COLOR)
    ax.axhline(1.0, linestyle="--", color="grey", linewidth=1.2,
               label="Ratio = 1.0  (no difference)")
    ax.set_xlabel("m  (equations in reduced system)")
    ax.set_ylabel("Median(whipped) / Median(random)")
    ax.set_title("H2 -- Relative solving difficulty: whipped vs random")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = os.path.join(out_dir, "h2_ratio.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def plot_h2_scaling(df, out_dir=H2_PLOTS):
    """Log-scale time vs m for whipped and random instances. Timeout rows excluded."""
    fig, ax = plt.subplots(figsize=(6, 4))

    for itype, color, label in [
        ("whipped", WHIPPED_COLOR, "Whipped P*"),
        ("random",  RANDOM_COLOR,  "Random Q"),
    ]:
        sub    = df[(df["instance_type"] == itype) & (~df["timed_out"])]
        groups = sub.groupby("m")["cpu_time_s"]
        ms     = sorted(groups.groups.keys())
        if not ms:
            continue
        medians = [np.median(groups.get_group(m_).dropna()) for m_ in ms]
        q25     = [np.percentile(groups.get_group(m_).dropna(), 25) for m_ in ms]
        q75     = [np.percentile(groups.get_group(m_).dropna(), 75) for m_ in ms]
        yerr_lo = np.array(medians) - np.array(q25)
        yerr_hi = np.array(q75)     - np.array(medians)
        ax.errorbar(ms, medians, yerr=[yerr_lo, yerr_hi],
                    marker="o", color=color, label=label,
                    capsize=4, linewidth=1.8, markersize=5)

    ax.set_yscale("log")
    ax.set_xlabel("m  (equations in reduced system)")
    ax.set_ylabel("CPU time (s)  [log scale]")
    ax.set_title("H2 -- Groebner basis scaling: whipped vs random")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = os.path.join(out_dir, "h2_scaling.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def plot_h2_degree(df, out_dir=H2_PLOTS):
    """Max Groebner degree vs m for whipped and random instances."""
    fig, ax = plt.subplots(figsize=(6, 4))

    for itype, color, label in [
        ("whipped", WHIPPED_COLOR, "Whipped P*"),
        ("random",  RANDOM_COLOR,  "Random Q"),
    ]:
        sub    = df[(df["instance_type"] == itype) & (~df["timed_out"]) & (df["degree"] > 0)]
        groups = sub.groupby("m")["degree"]
        ms     = sorted(groups.groups.keys())
        if not ms:
            continue
        medians = [np.median(groups.get_group(m_).dropna()) for m_ in ms]
        ax.plot(ms, medians, marker="o", color=color, label=label, linewidth=1.8)

    ax.set_xlabel("m  (equations in reduced system)")
    ax.set_ylabel("Median max degree in Groebner basis")
    ax.set_title("H2 -- Groebner degree: whipped vs random")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = os.path.join(out_dir, "h2_degree.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# -- Main --

def main():
    if os.path.exists(H1_CSV):
        print("\nH1 Analysis")
        df1 = load_h1()

        summary1 = h1_summary_table(df1)
        print("\nH1 Summary table:")
        print(summary1.to_string(index=False))

        print("\nScaling exponents  (model: T = c * exp(alpha * n), non-timeout rows only):")
        for solver in ("groebner", "sat"):
            alpha, c, r2 = fit_scaling_exponent(df1, solver)
            if alpha is not None:
                print(f"  {solver:10s}  alpha = {alpha:.4f},  c = {c:.4e},  R2 = {r2:.4f}")
            else:
                print(f"  {solver:10s}  insufficient data for fit")

        plot_h1_scaling(df1)
        plot_h1_boxplots(df1)
        plot_h1_memory(df1)
        plot_h1_groebner_degree(df1)

        summary1.to_csv(os.path.join(H1_DIR, "h1_summary.csv"), index=False)
        print(f"Summary saved to {H1_DIR}/h1_summary.csv")
    else:
        print(f"H1 CSV not found ({H1_CSV}). Run h1_experiment.py first.")

    if os.path.exists(H2_CSV):
        print("\nH2 Analysis")
        df2 = load_h2()

        summary2 = h2_summary_table(df2)
        print("\nH2 Summary table:")
        print(summary2.to_string(index=False))

        mw = h2_mann_whitney(df2)
        print("\nMann-Whitney U tests  (whipped vs random, two-sided, non-timeout rows):")
        print(mw.to_string(index=False))
        if len(mw) > 0:
            interpretation = (
                "SUPPORTS H2  (p >= 0.05: no significant difference)"
                if (mw["p_value"] >= 0.05).all()
                else "DOES NOT SUPPORT H2 at all scales  (some p < 0.05)"
            )
            print(f"\nOverall: {interpretation}")

        plot_h2_boxplots(df2)
        plot_h2_ratio(df2)
        plot_h2_scaling(df2)
        plot_h2_degree(df2)

        summary2.to_csv(os.path.join(H2_DIR, "h2_summary.csv"), index=False)
        mw.to_csv(os.path.join(H2_DIR, "h2_mannwhitney.csv"),   index=False)
        print(f"Summary saved to {H2_DIR}/h2_summary.csv")
        print(f"Mann-Whitney saved to {H2_DIR}/h2_mannwhitney.csv")
    else:
        print(f"H2 CSV not found ({H2_CSV}). Run h2_experiment.py first.")


if __name__ == "__main__":
    main()
