r"""
run_experiment.py — Full Research Pipeline
==========================================
Runs the multi-seed, multi-project, multi-alpha experiment for Part A of:
"Predictive Test Selection in CI with Change-Level Conformal Guarantee"

Usage:
    cd "c:\Users\Bogdan\Desktop\ВУЗ\data maga\Code"
    python src/run_experiment.py

Outputs (in outputs/):
    results_raw.csv     — one row per (project, seed, alpha, method)
    results_table.csv   — mean ± std aggregated per (project, alpha, method)
    figures/            — CMR vs alpha, budget, APFDc plots
"""

import sys
import os
import time
import warnings
import traceback

warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

# --- Path setup (allow running from project root or src/) ---
_this_dir = Path(__file__).resolve().parent
_root_dir = _this_dir.parent
sys.path.insert(0, str(_this_dir))
sys.path.insert(0, str(_root_dir))

from data_builder import load_project, load_patches, load_all_projects
from features import build_features, FEATURE_COLS
from models import chrono_split, train_lgbm, calibrate_isotonic, predict_calibrated, compute_pr_auc
from conformal import (
    compute_conformal_scores,
    compute_global_threshold,
    select_tests_global,
    select_tests_residual_all_commits,
    compute_tau_residual,
    prioritize_cost_aware,
)
from evaluate import compare_all_methods

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Data roots
RTP_ROOT = _root_dir / "rtp-torrent"
CASTOR_ROOT = _root_dir / "CASTOR_replication_package_v2.0"
OUT_DIR = _root_dir / "outputs"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Top-5 projects by strict defective commits (from prior audit)
BEST_PROJECTS = [
    "apache@sling",
    "square@okhttp",
    "l0rdn1kk0n@wicket-bootstrap",
    "CloudifySource@cloudify",
    "SonarSource@sonarqube",          # large — may be slow
]

ALPHAS = [0.05, 0.10, 0.15, 0.20]
N_SEEDS = 20          # number of random seeds for LightGBM
TRAIN_FRAC = 0.60
CALIB_FRAC = 0.20
# test_frac = 1 - TRAIN_FRAC - CALIB_FRAC = 0.20

MIN_DEFECTIVE_CALIB = 10   # skip project if calibration set has too few defective commits
MIN_DEFECTIVE_TEST  = 10   # skip project if test set has too few defective commits

# ---------------------------------------------------------------------------
# Load global job->commit mapping once
# ---------------------------------------------------------------------------

def _load_job_to_commit() -> dict:
    commits_csv = RTP_ROOT / "tr_all_built_commits.csv"
    df_c = pd.read_csv(
        commits_csv,
        usecols=["tr_job_id", "git_commit_id"],
        dtype={"tr_job_id": "int64", "git_commit_id": "str"}
    )
    return dict(zip(df_c["tr_job_id"], df_c["git_commit_id"]))


# ---------------------------------------------------------------------------
# Single-project, single-seed run
# ---------------------------------------------------------------------------

def run_one(project: str, seed: int, job_to_commit: dict) -> list:
    """
    Returns a list of metric dicts for all (alpha, method) combinations.
    Raises on critical errors; caller catches and logs.
    """
    rng = np.random.default_rng(seed)

    # --- 1. Load data ---
    print(f"    [load] {project}", flush=True)
    df_raw = load_project(project, RTP_ROOT, job_to_commit)
    patches = load_patches(project, RTP_ROOT)

    if df_raw is None or len(df_raw) == 0:
        raise ValueError(f"No data loaded for {project}")

    # --- 2. Feature engineering ---
    print(f"    [features] {project}", flush=True)
    df = build_features(df_raw, patches)

    # Basic sanity
    n_def = df[df["label"] == 1]["commit_idx"].nunique()
    print(f"    [info] commits={df['commit_idx'].nunique()}, pairs={len(df)}, defective_commits={n_def}", flush=True)

    if n_def < 30:
        raise ValueError(f"Not enough defective commits: {n_def} < 30")

    # --- 3. Chronological split ---
    train_df, calib_df, test_df = chrono_split(df, TRAIN_FRAC, CALIB_FRAC)

    n_def_calib = calib_df[calib_df["label"] == 1]["commit_idx"].nunique()
    n_def_test  = test_df[test_df["label"] == 1]["commit_idx"].nunique()
    print(f"    [split] train={len(train_df)}, calib={len(calib_df)}, test={len(test_df)}", flush=True)
    print(f"    [split] def_calib={n_def_calib}, def_test={n_def_test}", flush=True)

    if n_def_calib < MIN_DEFECTIVE_CALIB:
        raise ValueError(f"Too few defective commits in calib: {n_def_calib}")
    if n_def_test < MIN_DEFECTIVE_TEST:
        raise ValueError(f"Too few defective commits in test: {n_def_test}")

    # --- 4. Train LightGBM ---
    print(f"    [train] seed={seed}", flush=True)
    model = train_lgbm(train_df, seed=seed)

    # --- 5. Calibrate ---
    print(f"    [calibrate]", flush=True)
    calibrator = calibrate_isotonic(model, calib_df)

    # --- 6. Predict ---
    probs_calib = predict_calibrated(model, calibrator, calib_df)
    probs_test  = predict_calibrated(model, calibrator, test_df)

    pr_auc_calib = compute_pr_auc(calib_df["label"].values, probs_calib)
    pr_auc_test  = compute_pr_auc(test_df["label"].values,  probs_test)
    print(f"    [pr_auc] calib={pr_auc_calib:.4f}, test={pr_auc_test:.4f}", flush=True)

    # --- 7. Conformal + evaluation per alpha ---
    all_rows = []
    for alpha in ALPHAS:
        # Conformal scores on calibration defective commits
        scores = compute_conformal_scores(calib_df, probs_calib)
        lambda_ = compute_global_threshold(scores, alpha)

        # Tau for residual risk
        tau = compute_tau_residual(calib_df, probs_calib, alpha)

        print(f"    [conformal] alpha={alpha:.2f}, lambda={lambda_:.4f}, tau={tau:.4f}", flush=True)

        # Evaluate all methods
        rows = compare_all_methods(
            df_test=test_df,
            probs_test=probs_test,
            threshold_global=lambda_,
            tau_residual=tau,
            alpha=alpha,
            rng=np.random.default_rng(seed + int(alpha * 1000)),
        )

        for row in rows:
            row["project"] = project
            row["seed"] = seed
            row["pr_auc_test"] = pr_auc_test
            row["pr_auc_calib"] = pr_auc_calib
            row["n_def_calib"] = n_def_calib
            row["n_def_test"] = n_def_test
            row["lambda"] = lambda_
            row["tau"] = tau
            all_rows.append(row)

    return all_rows


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_all(projects=None, seeds=None, job_to_commit=None):
    if job_to_commit is None:
        print("Loading job->commit mapping...", flush=True)
        job_to_commit = _load_job_to_commit()
        print(f"Loaded {len(job_to_commit)} mappings.", flush=True)

    target_projects = projects if projects is not None else BEST_PROJECTS
    num_seeds = seeds if seeds is not None else N_SEEDS

    all_results = []
    t0_global = time.time()

    for project in target_projects:
        print(f"\n{'='*60}", flush=True)
        print(f"PROJECT: {project}", flush=True)
        print(f"{'='*60}", flush=True)

        project_rows = []
        for seed in range(num_seeds):
            t0 = time.time()
            print(f"\n  Seed {seed}/{num_seeds-1} ...", flush=True)
            try:
                rows = run_one(project, seed, job_to_commit)
                project_rows.extend(rows)
                print(f"  Seed {seed} OK ({time.time()-t0:.1f}s)", flush=True)
            except Exception as e:
                print(f"  Seed {seed} FAILED: {e}", flush=True)
                traceback.print_exc()
                continue

        if project_rows:
            all_results.extend(project_rows)
            # Save checkpoint after each project
            ckpt = pd.DataFrame(all_results)
            ckpt.to_csv(OUT_DIR / "results_raw.csv", index=False)
            print(f"\n  Checkpoint saved ({len(ckpt)} rows total)", flush=True)
        else:
            print(f"  No results for {project} — skipping.", flush=True)

    print(f"\n\nTotal time: {(time.time()-t0_global)/60:.1f} min", flush=True)
    return all_results


# ---------------------------------------------------------------------------
# Aggregation & table
# ---------------------------------------------------------------------------

def aggregate_results(all_results: list) -> pd.DataFrame:
    """Compute mean ± std per (project, method, alpha)."""
    df = pd.DataFrame(all_results)

    metric_cols = [
        "cmr", "pct_tests_selected", "pct_duration_saved",
        "pct_duration_selected", "ttff", "apfdc", "pr_auc_test",
    ]

    agg_mean = df.groupby(["project", "method", "alpha"])[metric_cols].mean()
    agg_std  = df.groupby(["project", "method", "alpha"])[metric_cols].std()

    agg_mean.columns = [f"{c}_mean" for c in metric_cols]
    agg_std.columns  = [f"{c}_std"  for c in metric_cols]

    table = pd.concat([agg_mean, agg_std], axis=1).reset_index()
    table = table.sort_values(["project", "method", "alpha"])
    return table


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(raw_df: pd.DataFrame):
    """Generate and save publication-quality figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping figures", flush=True)
        return

    methods_order = ["global_conformal", "residual_risk", "fixed_0.5", "random", "recently_failed"]
    colors = {
        "global_conformal": "#1f77b4",
        "residual_risk":    "#ff7f0e",
        "fixed_0.5":        "#2ca02c",
        "random":           "#d62728",
        "recently_failed":  "#9467bd",
    }
    labels = {
        "global_conformal": "Global Conformal (ours)",
        "residual_risk":    "Residual Risk (ours)",
        "fixed_0.5":        "Fixed τ=0.5",
        "random":           "Random",
        "recently_failed":  "Recently Failed",
    }

    projects = raw_df["project"].unique()

    # --- Figure 1: CMR vs alpha (per project) ---
    n_proj = len(projects)
    fig, axes = plt.subplots(1, n_proj, figsize=(5 * n_proj, 4), sharey=False)
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = axes.flatten()

    for ax, proj in zip(axes, projects):
        sub = raw_df[raw_df["project"] == proj]
        for method in methods_order:
            msub = sub[sub["method"] == method]
            if len(msub) == 0:
                continue
            stats = msub.groupby("alpha")["cmr"].agg(["mean", "std"]).reset_index()
            stats["std"] = stats["std"].fillna(0.0)
            ax.plot(stats["alpha"], stats["mean"], marker="o",
                    color=colors[method], label=labels[method])
            ax.fill_between(stats["alpha"],
                            (stats["mean"] - stats["std"]).clip(0),
                            (stats["mean"] + stats["std"]).clip(0, 1),
                            alpha=0.15, color=colors[method])
        # Reference diagonal (CMR = alpha)
        ax.plot([0.05, 0.20], [0.05, 0.20], "k--", linewidth=1, label="CMR = α")
        ax.set_title(proj.split("@")[-1], fontsize=10)
        ax.set_xlabel("Target α")
        ax.set_ylabel("Empirical CMR")
        ax.set_xlim(0.03, 0.22)
        ax.set_ylim(-0.02, 1.0)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("CMR vs α: Conformal Guarantee Verification", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_cmr_vs_alpha.pdf", bbox_inches="tight")
    plt.savefig(FIG_DIR / "fig1_cmr_vs_alpha.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig1_cmr_vs_alpha", flush=True)

    # --- Figure 2: Budget saved (% tests selected) vs alpha ---
    fig, axes = plt.subplots(1, n_proj, figsize=(5 * n_proj, 4), sharey=True)
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = axes.flatten()

    for ax, proj in zip(axes, projects):
        sub = raw_df[raw_df["project"] == proj]
        for method in ["global_conformal", "residual_risk", "fixed_0.5", "random"]:
            msub = sub[sub["method"] == method]
            if len(msub) == 0:
                continue
            stats = msub.groupby("alpha")["pct_tests_selected"].agg(["mean", "std"]).reset_index()
            stats["std"] = stats["std"].fillna(0.0)
            ax.plot(stats["alpha"], stats["mean"] * 100, marker="s",
                    color=colors[method], label=labels[method])
            ax.fill_between(stats["alpha"],
                            ((stats["mean"] - stats["std"]) * 100).clip(0),
                            ((stats["mean"] + stats["std"]) * 100).clip(0, 100),
                            alpha=0.15, color=colors[method])
        ax.set_title(proj.split("@")[-1], fontsize=10)
        ax.set_xlabel("Target α")
        ax.set_ylabel("Tests Selected (%)")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Budget: Fraction of Tests Selected vs α", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_budget_vs_alpha.pdf", bbox_inches="tight")
    plt.savefig(FIG_DIR / "fig2_budget_vs_alpha.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig2_budget_vs_alpha", flush=True)

    # --- Figure 3: APFDc bar chart (alpha=0.10) ---
    sub10 = raw_df[raw_df["alpha"] == 0.10]
    fig, axes = plt.subplots(1, n_proj, figsize=(5 * n_proj, 4), sharey=True)
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = axes.flatten()

    for ax, proj in zip(axes, projects):
        sub = sub10[sub10["project"] == proj]
        means, stds, xlabels = [], [], []
        for method in methods_order:
            msub = sub[sub["method"] == method]
            if len(msub) == 0:
                continue
            means.append(msub["apfdc"].mean())
            s = msub["apfdc"].std()
            stds.append(0.0 if pd.isna(s) else s)
            xlabels.append(labels[method])
        x = np.arange(len(xlabels))
        bars = ax.bar(x, means, yerr=stds, capsize=4,
                      color=[colors[m] for m in methods_order if m in [
                          k for k, v in labels.items() if v in xlabels]])
        ax.set_title(proj.split("@")[-1], fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("APFDc")
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("APFDc Comparison (α=0.10)", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_apfdc_alpha10.pdf", bbox_inches="tight")
    plt.savefig(FIG_DIR / "fig3_apfdc_alpha10.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig3_apfdc_alpha10", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predictive Test Selection Experiment")
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke test (1 project, 1 seed)")
    parser.add_argument("--project", type=str, default=None, help="Project name (e.g. l0rdn1kk0n@wicket-bootstrap)")
    parser.add_argument("--seeds", type=int, default=None, help="Number of seeds")
    args = parser.parse_args()

    if args.smoke:
        target_projects = [args.project or "l0rdn1kk0n@wicket-bootstrap"]
        num_seeds = args.seeds or 1
        print(">>> RUNNING IN SMOKE TEST MODE <<<", flush=True)
    else:
        target_projects = [args.project] if args.project else BEST_PROJECTS
        num_seeds = args.seeds if args.seeds else N_SEEDS

    print("=" * 60, flush=True)
    print("Predictive Test Selection — Part A Experiment", flush=True)
    print(f"Projects : {target_projects}", flush=True)
    print(f"Alphas   : {ALPHAS}", flush=True)
    print(f"Seeds    : {num_seeds}", flush=True)
    print("=" * 60, flush=True)

    # Run experiment
    all_results = run_all(projects=target_projects, seeds=num_seeds)

    if not all_results:
        print("ERROR: No results collected. Check data paths and project names.", flush=True)
        sys.exit(1)

    raw_df = pd.DataFrame(all_results)
    raw_df.to_csv(OUT_DIR / "results_raw.csv", index=False)
    print(f"\nRaw results saved to {OUT_DIR / 'results_raw.csv'} ({len(raw_df)} rows)", flush=True)

    # Aggregate
    table = aggregate_results(all_results)
    table.to_csv(OUT_DIR / "results_table.csv", index=False)
    print(f"Summary table saved to {OUT_DIR / 'results_table.csv'}", flush=True)

    # Print preview
    print("\n--- Results Table Preview (global_conformal) ---", flush=True)
    preview = table[table["method"] == "global_conformal"][
        ["project", "alpha", "cmr_mean", "cmr_std",
         "pct_tests_selected_mean", "pct_duration_saved_mean", "apfdc_mean"]
    ].to_string(index=False)
    print(preview, flush=True)

    # Figures
    print("\nGenerating figures...", flush=True)
    make_figures(raw_df)

    print("\nDONE.", flush=True)
