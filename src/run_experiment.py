r"""
run_experiment.py — Full Research Pipeline (v2)
================================================
Runs the multi-seed, multi-project, multi-alpha experiment for:
"Predictive Test Selection in CI with Change-Level Conformal Guarantee"

Architecture:
  - CIHistorySimulator as the temporal data layer
  - 4-way chronological split (Train → ProbCalib → ConfCalib → Test)
  - Expanded feature set with native NaN handling
  - All baselines: retest-all, top-k%, fixed-0.5, global conformal,
    residual risk, naive residual, recently failed, random matched budget

Usage:
    python src/run_experiment.py
    python src/run_experiment.py --smoke
    python src/run_experiment.py --project apache@sling --seeds 5

Outputs (in outputs/):
    results_raw.csv     — one row per (project, seed, alpha, method)
    results_table.csv   — mean ± std aggregated per (project, alpha, method)
    figures/            — CMR vs alpha, budget, APFDc plots
"""

from __future__ import annotations

import sys
import os
import time
from datetime import datetime
import warnings
import traceback

warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

# --- Path setup ---
_this_dir = Path(__file__).resolve().parent
_root_dir = _this_dir.parent
sys.path.insert(0, str(_this_dir))
sys.path.insert(0, str(_root_dir))

from simulator import CIHistorySimulator
from features import build_features_batch, FEATURE_COLS
from models import chrono_split_4, train_lgbm, calibrate_isotonic, predict_calibrated, compute_pr_auc, compute_brier_score
from conformal import (
    compute_conformal_scores,
    compute_global_threshold,
    compute_tau_residual,
)
from evaluate import compare_all_methods

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RTP_ROOT = _root_dir / "rtp-torrent"
OUT_DIR = _root_dir / "outputs"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

BEST_PROJECTS = [
    "apache@sling",
    "square@okhttp",
    "l0rdn1kk0n@wicket-bootstrap",
    "CloudifySource@cloudify",
    "SonarSource@sonarqube",
]

ALPHAS = [0.05, 0.10, 0.15, 0.20]
N_SEEDS = 20
TOP_K_BUDGETS = [0.10, 0.30, 0.50]

# Split fractions
TRAIN_FRAC = 0.50
PROB_CALIB_FRAC = 0.15
CONF_CALIB_FRAC = 0.15
# test_frac = 1 - 0.50 - 0.15 - 0.15 = 0.20

MIN_DEFECTIVE_CONF_CALIB = 5
MIN_DEFECTIVE_TEST = 5


# ---------------------------------------------------------------------------
# Build features + labels for a project
# ---------------------------------------------------------------------------

def build_project_data(sim: CIHistorySimulator, policy: str = "confirmed_new_fail") -> pd.DataFrame:
    """Build complete feature+label dataset for one project."""
    all_commits = sim.get_all_commits()
    print(f"    [features] building for {len(all_commits)} commits...", flush=True)

    df_features = build_features_batch(sim, all_commits, verbose=True)

    # Pre-fetch ground truth once per commit (O(C) instead of O(N))
    print(f"    [ground_truth] revealing ground truth for {len(all_commits)} commits...", flush=True)
    gt_by_commit = {sha: sim.reveal_ground_truth(sha, policy=policy) for sha in all_commits}

    labels = []
    durations = []
    statuses = []
    parent_statuses = []
    current_statuses = []
    parent_shas = []
    mixed_flags = []
    parent_sources = []

    for row in df_features.itertuples(index=False):
        sha = row.commit
        test_name = row.testName
        gt = gt_by_commit.get(sha)
        if gt and test_name in gt.test_results:
            tr = gt.test_results[test_name]
            labels.append(tr.label)
            durations.append(tr.actual_duration)
            statuses.append(tr.label_status)
            parent_statuses.append(tr.parent_status)
            current_statuses.append(tr.current_status)
            parent_shas.append(tr.parent_sha)
            mixed_flags.append(tr.mixed_outcomes_same_sha)
            parent_sources.append(tr.parent_source)
        else:
            labels.append(0)
            durations.append(0.0)
            statuses.append("missing")
            parent_statuses.append("unknown")
            current_statuses.append("pass")
            parent_shas.append(None)
            mixed_flags.append(False)
            parent_sources.append("none")

    df_features["label"] = labels
    df_features["actual_duration"] = durations
    df_features["label_status"] = statuses
    df_features["label_attribution"] = statuses  # alias
    df_features["parent_status"] = parent_statuses
    df_features["current_status"] = current_statuses
    df_features["parent_sha"] = parent_shas
    df_features["mixed_outcomes_same_sha"] = mixed_flags
    df_features["parent_source"] = parent_sources

    # Print parent_source statistics
    ps_counts = df_features.groupby("commit")["parent_source"].first().value_counts()
    print(f"    [parent_source] {dict(ps_counts)}", flush=True)

    return df_features


# ---------------------------------------------------------------------------
# Single-project, single-seed run
# ---------------------------------------------------------------------------

def run_one(sim: CIHistorySimulator, df: pd.DataFrame, seed: int) -> list:
    """Run one experiment iteration.

    Returns a list of metric dicts for all (alpha, method) combinations.
    """
    rng = np.random.default_rng(seed)
    project = sim.project

    n_def = df[df["label"] == 1]["commit_idx"].nunique()
    print(f"    [info] commits={df['commit_idx'].nunique()}, "
          f"pairs={len(df)}, defective_commits={n_def}", flush=True)

    if n_def < 15:
        raise ValueError(f"Not enough defective commits: {n_def} < 15")

    # --- 4-way split ---
    train_df, prob_calib_df, conf_calib_df, test_df = chrono_split_4(
        df, TRAIN_FRAC, PROB_CALIB_FRAC, CONF_CALIB_FRAC
    )

    n_def_conf = conf_calib_df[conf_calib_df["label"] == 1]["commit_idx"].nunique()
    n_def_test = test_df[test_df["label"] == 1]["commit_idx"].nunique()

    print(f"    [split] train={len(train_df)}, prob_calib={len(prob_calib_df)}, "
          f"conf_calib={len(conf_calib_df)}, test={len(test_df)}", flush=True)
    print(f"    [split] def_conf_calib={n_def_conf}, def_test={n_def_test}", flush=True)

    if n_def_conf < MIN_DEFECTIVE_CONF_CALIB:
        raise ValueError(f"Too few defective in conf_calib: {n_def_conf}")
    if n_def_test < MIN_DEFECTIVE_TEST:
        raise ValueError(f"Too few defective in test: {n_def_test}")

    # --- Train LightGBM ---
    print(f"    [train] seed={seed}", flush=True)
    model = train_lgbm(train_df, FEATURE_COLS, seed=seed)

    # --- Isotonic calibration (on prob_calib set) ---
    print(f"    [calibrate]", flush=True)
    calibrator = calibrate_isotonic(model, prob_calib_df, FEATURE_COLS)

    # --- Predict ---
    probs_conf_calib = predict_calibrated(model, calibrator, conf_calib_df, FEATURE_COLS)
    probs_test = predict_calibrated(model, calibrator, test_df, FEATURE_COLS)

    pr_auc_test = compute_pr_auc(test_df["label"].values, probs_test)
    brier_test = compute_brier_score(test_df["label"].values, probs_test)
    print(f"    [pr_auc] test={pr_auc_test:.4f}  brier={brier_test:.4f}", flush=True)

    # --- Conformal + evaluation per alpha ---
    all_rows: list[dict] = []
    for alpha in ALPHAS:
        # Conformal scores on conf_calib defective commits
        scores = compute_conformal_scores(conf_calib_df, probs_conf_calib)
        lambda_ = compute_global_threshold(scores, alpha)

        # Tau for residual risk (on conf_calib)
        tau = compute_tau_residual(conf_calib_df, probs_conf_calib, alpha)

        print(f"    [conformal] alpha={alpha:.2f}, lambda={lambda_:.4f}, tau={tau:.4f}", flush=True)

        # Evaluate all methods
        rows = compare_all_methods(
            df_test=test_df,
            probs_test=probs_test,
            threshold_global=lambda_,
            tau_residual=tau,
            alpha=alpha,
            rng=np.random.default_rng(seed + int(alpha * 1000)),
            duration_col="actual_duration",
            df_val=conf_calib_df,
            probs_val=probs_conf_calib,
            top_k_budgets=TOP_K_BUDGETS,
        )

        for row in rows:
            row["project"] = project
            row["seed"] = seed
            row["pr_auc_test"] = pr_auc_test
            row["brier_score_test"] = brier_test
            row["n_def_conf_calib"] = n_def_conf
            row["n_def_test"] = n_def_test
            row["lambda"] = lambda_
            row["tau"] = tau
            all_rows.append(row)

    return all_rows


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_all(projects=None, seeds=None, out_dir: Path = OUT_DIR, policy: str = "confirmed_new_fail"):
    target_projects = projects if projects is not None else BEST_PROJECTS
    num_seeds = seeds if seeds is not None else N_SEEDS

    all_results: list[dict] = []
    t0_global = time.time()

    for project in target_projects:
        print(f"\n{'='*60}", flush=True)
        print(f"PROJECT: {project}", flush=True)
        print(f"{'='*60}", flush=True)

        # --- Build simulator and features (once per project) ---
        try:
            print(f"  [simulator] loading {project}...", flush=True)
            sim = CIHistorySimulator(RTP_ROOT, project)
            print(f"  [simulator] {len(sim.get_all_commits())} commits loaded", flush=True)

            print(f"  [data] building features + labels (policy={policy})...", flush=True)
            df = build_project_data(sim, policy=policy)
            print(f"  [data] {len(df)} (commit, test) pairs", flush=True)

            # Exclude unknown_parent from strict training/evaluation
            n_unknown = (df["label_attribution"] == "unknown_parent").sum()
            if n_unknown > 0:
                print(f"  [data] excluding {n_unknown} unknown_parent pairs", flush=True)
                df = df[df["label_attribution"] != "unknown_parent"].copy()

        except Exception as e:
            print(f"  FAILED to load project: {e}", flush=True)
            traceback.print_exc()
            continue

        project_rows: list[dict] = []
        _reliability_saved = False
        for seed in range(num_seeds):
            t0 = time.time()
            print(f"\n  Seed {seed}/{num_seeds-1} ...", flush=True)
            try:
                rows = run_one(sim, df, seed)
                project_rows.extend(rows)
                print(f"  Seed {seed} OK ({time.time()-t0:.1f}s)", flush=True)
                # Generate reliability diagram once per project (seed 0)
                if not _reliability_saved:
                    try:
                        from models import train_lgbm, calibrate_isotonic, predict_calibrated
                        from models import chrono_split_4 as _csp
                        _train, _prob_cal, _conf_cal, _test = _csp(
                            df, TRAIN_FRAC, PROB_CALIB_FRAC, CONF_CALIB_FRAC)
                        _m = train_lgbm(_train, FEATURE_COLS, seed=seed)
                        _iso = calibrate_isotonic(_m, _prob_cal, FEATURE_COLS)
                        _probs = predict_calibrated(_m, _iso, _test, FEATURE_COLS)
                        make_reliability_diagram(
                            _test["label"].values, _probs, project, out_dir / "figures")
                        _reliability_saved = True
                    except Exception as _e:
                        print(f"  Reliability diagram FAILED: {_e}", flush=True)
            except Exception as e:
                print(f"  Seed {seed} FAILED: {e}", flush=True)
                traceback.print_exc()
                continue

        if project_rows:
            all_results.extend(project_rows)
            ckpt = pd.DataFrame(all_results)
            ckpt.to_csv(out_dir / "results_raw.csv", index=False)
            print(f"\n  Checkpoint saved ({len(ckpt)} rows total)", flush=True)
        else:
            print(f"  No results for {project} — skipping.", flush=True)

    print(f"\n\nTotal time: {(time.time()-t0_global)/60:.1f} min", flush=True)
    return all_results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(all_results: list) -> pd.DataFrame:
    """Compute mean ± std per (project, method, alpha).

    Uses ddof=1 for std; does NOT replace NaN with 0.
    """
    df = pd.DataFrame(all_results)

    metric_cols = [
        "cmr", "pct_tests_selected", "pct_duration_saved",
        "pct_duration_selected", "ttff", "apfdc", "pr_auc_test",
    ]
    # Include optional columns if present
    for opt_col in ["brier_score_test", "cmr_ci_low", "cmr_ci_high"]:
        if opt_col in df.columns:
            metric_cols.append(opt_col)

    agg_mean = df.groupby(["project", "method", "alpha"])[metric_cols].mean()
    agg_std  = df.groupby(["project", "method", "alpha"])[metric_cols].std(ddof=1)

    agg_mean.columns = [f"{c}_mean" for c in metric_cols]
    agg_std.columns  = [f"{c}_std"  for c in metric_cols]

    table = pd.concat([agg_mean, agg_std], axis=1).reset_index()
    table = table.sort_values(["project", "method", "alpha"])
    return table


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_reliability_diagram(
    labels: np.ndarray,
    probs: np.ndarray,
    project: str,
    fig_dir: Path,
    n_bins: int = 10,
) -> None:
    """Generate and save a reliability (calibration) diagram.

    Plots fraction of positives vs. mean predicted probability per bin.
    Saved as fig4_reliability_diagram.pdf and .png.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping reliability diagram", flush=True)
        return

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers, frac_pos, mean_pred = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        frac_pos.append(labels[mask].mean())
        mean_pred.append(probs[mask].mean())
        bin_centers.append((lo + hi) / 2)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, "o-", color="#1f77b4", label="Isotonic calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Reliability Diagram — {project.split('@')[-1]}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig4_reliability_diagram.pdf", bbox_inches="tight")
    plt.savefig(fig_dir / "fig4_reliability_diagram.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig4_reliability_diagram", flush=True)


def run_rolling(
    sim: "CIHistorySimulator",
    df: pd.DataFrame,
    seed: int,
    n_splits: int = 3,
    out_dir: Path = None,
) -> list:
    """Run rolling/expanding window evaluation.

    For each split window records: window, n_def_conf, n_def_test, cmr,
    cmr_ci_high, pct_tests_selected, pct_duration_saved, lambda, tau, cmr-alpha.
    """
    from models import rolling_time_splits, train_lgbm, calibrate_isotonic, predict_calibrated, compute_pr_auc

    rng = np.random.default_rng(seed)
    project = sim.project
    splits = rolling_time_splits(
        df,
        n_splits=n_splits,
        min_train_frac=0.25,
        prob_calib_frac=PROB_CALIB_FRAC,
        conf_calib_frac=CONF_CALIB_FRAC,
        test_frac=0.15,
    )
    rows = []
    for win_idx, (train_df, prob_calib_df, conf_calib_df, test_df) in enumerate(splits):
        n_def_conf = conf_calib_df[conf_calib_df["label"] == 1]["commit_idx"].nunique()
        n_def_test = test_df[test_df["label"] == 1]["commit_idx"].nunique()
        if n_def_conf < MIN_DEFECTIVE_CONF_CALIB or n_def_test < MIN_DEFECTIVE_TEST:
            print(f"    [rolling] window {win_idx}: skip (def_conf={n_def_conf}, def_test={n_def_test})", flush=True)
            continue
        try:
            model = train_lgbm(train_df, FEATURE_COLS, seed=seed)
            calibrator = calibrate_isotonic(model, prob_calib_df, FEATURE_COLS)
            probs_conf = predict_calibrated(model, calibrator, conf_calib_df, FEATURE_COLS)
            probs_test = predict_calibrated(model, calibrator, test_df, FEATURE_COLS)

            for alpha in ALPHAS:
                scores = compute_conformal_scores(conf_calib_df, probs_conf)
                lambda_ = compute_global_threshold(scores, alpha)
                tau = compute_tau_residual(conf_calib_df, probs_conf, alpha)

                from conformal import select_tests_global, select_tests_residual_all_commits
                from evaluate import compute_cmr_wilson_ci, compute_budget_saved

                for m_name, sel_df in [
                    ("global_conformal", select_tests_global(test_df, probs_test, lambda_)),
                    ("residual_risk", select_tests_residual_all_commits(test_df, probs_test, tau)),
                ]:
                    mask = test_df.index.isin(sel_df.index)
                    cmr_val, ci_low, ci_high = compute_cmr_wilson_ci(test_df.assign(_sel=mask), mask)
                    budget = compute_budget_saved(test_df, mask)

                    rows.append({
                        "project": project,
                        "seed": seed,
                        "window": win_idx,
                        "method": m_name,
                        "alpha": alpha,
                        "n_def_conf": n_def_conf,
                        "n_def_test": n_def_test,
                        "cmr": cmr_val,
                        "cmr_ci_high": ci_high,
                        "pct_tests_selected": budget["pct_tests_selected"],
                        "pct_duration_saved": budget["pct_duration_saved"],
                        "lambda": lambda_,
                        "tau": tau,
                        "cmr_minus_alpha": cmr_val - alpha,
                    })
        except Exception as e:
            print(f"    [rolling] window {win_idx} FAILED: {e}", flush=True)
    return rows


def make_rolling_figure(rolling_df: pd.DataFrame, fig_dir: Path) -> None:
    """Plot CMR dynamics across rolling windows."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    projects = rolling_df["project"].unique()
    alphas_plot = sorted(rolling_df["alpha"].unique())
    colors_a = {0.05: "#1f77b4", 0.10: "#ff7f0e", 0.15: "#2ca02c", 0.20: "#d62728"}
    methods_plot = ["global_conformal", "residual_risk"] if "method" in rolling_df.columns else [None]
    linestyles = {"global_conformal": "-", "residual_risk": "--"}

    fig, axes = plt.subplots(1, len(projects), figsize=(5 * len(projects), 4), sharey=False)
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = axes.flatten()

    for ax, proj in zip(axes, projects):
        sub = rolling_df[rolling_df["project"] == proj]
        for m in methods_plot:
            msub = sub[sub["method"] == m] if m else sub
            for alpha in alphas_plot:
                asub = msub[msub["alpha"] == alpha]
                if len(asub) == 0:
                    continue
                stats = asub.groupby("window")["cmr"].agg(["mean", "std"]).reset_index()
                stats["std"] = stats["std"].fillna(0.0)
                lbl = f"α={alpha:.2f} ({m})" if m else f"α={alpha:.2f}"
                ax.plot(stats["window"], stats["mean"], "o-",
                        linestyle=linestyles.get(m, "-"),
                        color=colors_a.get(alpha, "grey"), label=lbl)
                ax.fill_between(stats["window"],
                                (stats["mean"] - stats["std"]).clip(0),
                                (stats["mean"] + stats["std"]).clip(0, 1),
                                alpha=0.10, color=colors_a.get(alpha, "grey"))
        for alpha in alphas_plot:
            ax.axhline(alpha, color=colors_a.get(alpha, "grey"),
                       linestyle=":", linewidth=0.8)
        ax.set_title(proj.split("@")[-1], fontsize=10)
        ax.set_xlabel("Rolling window")
        ax.set_ylabel("Empirical CMR")
        ax.set_ylim(-0.02, 1.0)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    fig.suptitle("CMR Dynamics — Rolling Evaluation Windows", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig5_rolling_cmr.pdf", bbox_inches="tight")
    plt.savefig(fig_dir / "fig5_rolling_cmr.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig5_rolling_cmr", flush=True)

def make_figures(raw_df: pd.DataFrame, fig_dir: Path = FIG_DIR):
    """Generate publication-quality figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping figures", flush=True)
        return

    methods_order = [
        "retest_all", "global_conformal", "global_conformal_cost_aware",
        "residual_risk", "residual_risk_cost_aware",
        "naive_residual", "fixed_0.5",
        "top_10pct", "top_30pct", "top_50pct",
        "random_matched", "recently_failed",
    ]
    colors = {
        "retest_all":                  "#000000",
        "global_conformal":            "#1f77b4",
        "global_conformal_cost_aware": "#005580",
        "residual_risk":               "#ff7f0e",
        "residual_risk_cost_aware":    "#cc5500",
        "naive_residual":              "#bcbd22",
        "fixed_0.5":                   "#2ca02c",
        "top_10pct":                   "#17becf",
        "top_30pct":                   "#aec7e8",
        "top_50pct":                   "#c5b0d5",
        "random_matched":              "#d62728",
        "recently_failed":             "#9467bd",
        "threshold_target_recall":     "#8c564b",
    }
    labels = {
        "retest_all":                  "Retest-All",
        "global_conformal":            "Global Conformal",
        "global_conformal_cost_aware": "Global Conformal (cost-aware)",
        "residual_risk":               "Residual Risk",
        "residual_risk_cost_aware":    "Residual Risk (cost-aware)",
        "naive_residual":              "Naive Residual (τ=α)",
        "fixed_0.5":                   "Fixed τ=0.5",
        "top_10pct":                   "Top-10%",
        "top_30pct":                   "Top-30%",
        "top_50pct":                   "Top-50%",
        "random_matched":              "Random (matched budget)",
        "recently_failed":             "Recently Failed",
        "threshold_target_recall":     "Threshold (target recall)",
    }

    projects = raw_df["project"].unique()
    n_proj = len(projects)

    # --- Figure 1: CMR vs alpha ---
    cmr_methods = ["global_conformal", "residual_risk", "naive_residual",
                   "fixed_0.5", "random_matched", "recently_failed"]
    fig, axes = plt.subplots(1, n_proj, figsize=(5 * n_proj, 4), sharey=False)
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = axes.flatten()

    for ax, proj in zip(axes, projects):
        sub = raw_df[raw_df["project"] == proj]
        for method in cmr_methods:
            msub = sub[sub["method"] == method]
            if len(msub) == 0:
                continue
            stats = msub.groupby("alpha")["cmr"].agg(["mean", "std"]).reset_index()
            stats["std"] = stats["std"].fillna(0.0)
            ax.plot(stats["alpha"], stats["mean"], marker="o",
                    color=colors.get(method, "grey"),
                    label=labels.get(method, method))
            ax.fill_between(stats["alpha"],
                            (stats["mean"] - stats["std"]).clip(0),
                            (stats["mean"] + stats["std"]).clip(0, 1),
                            alpha=0.15, color=colors.get(method, "grey"))
        ax.plot([0.05, 0.20], [0.05, 0.20], "k--", linewidth=1, label="CMR = α")
        ax.set_title(proj.split("@")[-1], fontsize=10)
        ax.set_xlabel("Target α")
        ax.set_ylabel("Empirical CMR")
        ax.set_xlim(0.03, 0.22)
        ax.set_ylim(-0.02, 1.0)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    fig.suptitle("CMR vs α: Conformal Guarantee Verification", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig1_cmr_vs_alpha.pdf", bbox_inches="tight")
    plt.savefig(fig_dir / "fig1_cmr_vs_alpha.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig1_cmr_vs_alpha", flush=True)

    # --- Figure 2: Budget vs alpha ---
    budget_methods = ["global_conformal", "residual_risk", "naive_residual",
                      "fixed_0.5", "random_matched"]
    fig, axes = plt.subplots(1, n_proj, figsize=(5 * n_proj, 4), sharey=True)
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = axes.flatten()

    for ax, proj in zip(axes, projects):
        sub = raw_df[raw_df["project"] == proj]
        for method in budget_methods:
            msub = sub[sub["method"] == method]
            if len(msub) == 0:
                continue
            stats = msub.groupby("alpha")["pct_tests_selected"].agg(["mean", "std"]).reset_index()
            stats["std"] = stats["std"].fillna(0.0)
            ax.plot(stats["alpha"], stats["mean"] * 100, marker="s",
                    color=colors.get(method, "grey"),
                    label=labels.get(method, method))
        ax.set_title(proj.split("@")[-1], fontsize=10)
        ax.set_xlabel("Target α")
        ax.set_ylabel("Tests Selected (%)")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Budget: Fraction of Tests Selected vs α", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig2_budget_vs_alpha.pdf", bbox_inches="tight")
    plt.savefig(fig_dir / "fig2_budget_vs_alpha.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig2_budget_vs_alpha", flush=True)

    # --- Figure 3: APFDc bar chart (alpha=0.10) ---
    sub10 = raw_df[raw_df["alpha"] == 0.10]
    fig, axes = plt.subplots(1, n_proj, figsize=(5 * n_proj, 4), sharey=True)
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = axes.flatten()

    bar_methods = [m for m in methods_order if m in sub10["method"].values]
    for ax, proj in zip(axes, projects):
        sub = sub10[sub10["project"] == proj]
        means, stds, xlabels = [], [], []
        for method in bar_methods:
            msub = sub[sub["method"] == method]
            if len(msub) == 0:
                continue
            means.append(msub["apfdc"].mean())
            s = msub["apfdc"].std(ddof=1)
            stds.append(0.0 if pd.isna(s) else s)
            xlabels.append(labels.get(method, method))
        x = np.arange(len(xlabels))
        ax.bar(x, means, yerr=stds, capsize=4,
               color=[colors.get(m, "grey") for m in bar_methods[:len(xlabels)]])
        ax.set_title(proj.split("@")[-1], fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=35, ha="right", fontsize=6)
        ax.set_ylabel("APFDc")
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("APFDc Comparison (α=0.10)", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig3_apfdc_alpha10.pdf", bbox_inches="tight")
    plt.savefig(fig_dir / "fig3_apfdc_alpha10.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig3_apfdc_alpha10", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predictive Test Selection Experiment v2")
    parser.add_argument("--smoke", action="store_true",
                        help="Run fast smoke test (1 project, 1 seed)")
    parser.add_argument("--project", type=str, default=None,
                        help="Project name")
    parser.add_argument("--seeds", type=int, default=None,
                        help="Number of seeds")
    parser.add_argument("--rolling", action="store_true",
                        help="Run rolling window evaluation (saves results_rolling.csv and fig5)")
    parser.add_argument("--rolling-splits", type=int, default=3,
                        help="Number of rolling window splits (default: 3)")
    parser.add_argument("--policy", type=str, default="confirmed_new_fail",
                        choices=["confirmed_new_fail", "strict_all_fail", "any_fail"],
                        help="Label policy for ground-truth assignment (default: confirmed_new_fail)")
    args = parser.parse_args()

    if args.smoke:
        target_projects = [args.project or "l0rdn1kk0n@wicket-bootstrap"]
        num_seeds = args.seeds or 1
        proj_tag = "smoke_" + target_projects[0].split("@")[-1]
        print(">>> RUNNING IN SMOKE TEST MODE <<<", flush=True)
    else:
        target_projects = [args.project] if args.project else BEST_PROJECTS
        num_seeds = args.seeds if args.seeds else N_SEEDS
        proj_tag = target_projects[0].split("@")[-1] if args.project else f"all{len(target_projects)}"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_dir = OUT_DIR / "runs" / f"{timestamp}_{proj_tag}_seeds{num_seeds}"
    run_fig_dir = run_dir / "figures"
    run_fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print("Predictive Test Selection - v2 Experiment", flush=True)
    print(f"Output   : {run_dir}", flush=True)
    print(f"Projects : {target_projects}", flush=True)
    print(f"Alphas   : {ALPHAS}", flush=True)
    print(f"Seeds    : {num_seeds}", flush=True)
    print(f"Policy   : {args.policy}", flush=True)
    print(f"Rolling  : {args.rolling}", flush=True)
    print(f"Split    : {TRAIN_FRAC}/{PROB_CALIB_FRAC}/{CONF_CALIB_FRAC}/"
          f"{1-TRAIN_FRAC-PROB_CALIB_FRAC-CONF_CALIB_FRAC:.2f} "
          f"(train/prob_calib/conf_calib/test)", flush=True)
    print("=" * 60, flush=True)

    all_results = run_all(projects=target_projects, seeds=num_seeds, out_dir=run_dir,
                          policy=args.policy)

    if not all_results:
        print("ERROR: No results collected.", flush=True)
        sys.exit(1)

    raw_df = pd.DataFrame(all_results)
    raw_df.to_csv(run_dir / "results_raw.csv", index=False)
    print(f"\nRaw results saved ({len(raw_df)} rows) -> {run_dir / 'results_raw.csv'}", flush=True)

    table = aggregate_results(all_results)
    table.to_csv(run_dir / "results_table.csv", index=False)
    print(f"Summary table saved -> {run_dir / 'results_table.csv'}", flush=True)

    print("\n--- Results Preview (global_conformal) ---", flush=True)
    preview_cols = ["project", "alpha", "cmr_mean", "cmr_std",
                    "pct_tests_selected_mean", "pct_duration_saved_mean", "apfdc_mean"]
    available_cols = [c for c in preview_cols if c in table.columns]
    preview = table[table["method"] == "global_conformal"][available_cols]
    print(preview.to_string(index=False), flush=True)

    print("\nGenerating figures...", flush=True)
    make_figures(raw_df, fig_dir=run_fig_dir)

    # Reliability diagram (use last loaded sim + test probs if available)
    if "_last_sim" in dir():
        pass  # generated inside run_all when probs available

    # --- Rolling window evaluation ---
    if args.rolling:
        print("\n>>> ROLLING WINDOW EVALUATION <<<", flush=True)
        all_rolling: list[dict] = []
        for project in target_projects:
            try:
                sim = CIHistorySimulator(RTP_ROOT, project)
                df_proj = build_project_data(sim, policy=args.policy)
                n_unknown = (df_proj["label_attribution"] == "unknown_parent").sum()
                if n_unknown > 0:
                    df_proj = df_proj[df_proj["label_attribution"] != "unknown_parent"].copy()
                for seed in range(num_seeds):
                    rrows = run_rolling(sim, df_proj, seed=seed,
                                        n_splits=args.rolling_splits, out_dir=run_dir)
                    all_rolling.extend(rrows)
            except Exception as e:
                print(f"  Rolling FAILED for {project}: {e}", flush=True)
                traceback.print_exc()

        if all_rolling:
            rolling_df = pd.DataFrame(all_rolling)
            rolling_df.to_csv(run_dir / "results_rolling.csv", index=False)
            print(f"Rolling results saved -> {run_dir / 'results_rolling.csv'}", flush=True)
            make_rolling_figure(rolling_df, fig_dir=run_fig_dir)
        else:
            print("No rolling results collected.", flush=True)

    print(f"\nDONE. All outputs in: {run_dir}", flush=True)
