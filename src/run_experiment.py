r"""
run_experiment.py — Full Research Pipeline (v2.1)
=================================================
Runs the multi-seed, multi-project, multi-alpha experiment for:
"Predictive Test Selection in CI with Change-Level Conformal Guarantee"

Architecture:
  - CIHistorySimulator as the temporal data layer
  - Pre-execution candidate test inventory via get_candidate_tests()
  - Three-valued ground-truth labeling (1, 0, None) with strict exclusion of
    ambiguous / unknown / unobserved observations
  - Strict Git first-parent filtering (parent_source == 'git_parent' by default;
    optional --include-fallback-parents for sensitivity analysis)
  - 4-way chronological split (Train -> ProbCalib -> ConfCalib -> Test)
    and expanding-window rolling evaluation (--rolling)
  - All baselines: retest-all, top-k%, fixed-0.5, global conformal,
    residual risk, naive residual, recently failed, random matched budget,
    threshold by target recall

Usage:
    python src/run_experiment.py --seeds 5 --rolling
    python src/run_experiment.py --smoke --rolling
    python src/run_experiment.py --project apache@sling --seeds 5 --include-fallback-parents

Outputs (in outputs/runs/<timestamp>_<tag>_seeds<N>/):
    results_raw.csv     — one row per (project, seed, alpha, method)
    results_table.csv   — mean +- std aggregated per (project, alpha, method)
    results_rolling.csv — per-window rolling evaluation (including skipped windows)
    figures/            — CMR vs alpha, budget, Conditional APFDc, reliability, rolling CMR
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# --- Path setup ---
_this_dir = Path(__file__).resolve().parent
_root_dir = _this_dir.parent
sys.path.insert(0, str(_this_dir))
sys.path.insert(0, str(_root_dir))

from simulator import CIHistorySimulator
from features import build_features_batch, FEATURE_COLS
from models import (
    chrono_split_4,
    train_lgbm,
    calibrate_isotonic,
    calibrate_sigmoid,
    evaluate_and_select_calibrators,
    predict_calibrated,
    predict_raw_and_calibrated,
    compute_pr_auc,
    compute_pr_auc_baseline,
    compute_brier_score,
    compute_brier_baseline,
    compute_brier_skill_score,
)
from conformal import (
    compute_conformal_scores,
    compute_global_threshold,
    compute_tau_residual,
    select_tests_global,
    select_tests_residual_all_commits,
)
from evaluate import (
    _candidate_mask,
    _commit_miss_indicators,
    analyze_miss_causes,
    compare_all_methods,
    compute_budget_saved,
    compute_exact_binomial_ci_counts,
    compute_wilson_ci_counts,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RTP_ROOT = _root_dir / "rtp-torrent"
OUT_DIR = _root_dir / "outputs"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Selected 4 viable RTPTorrent projects (from outputs/dataset_audit.csv under
# git_parent ONLY + confirmed_new_fail; square@okhttp has 0 ConfCalib/Test
# defects and CloudifySource@cloudify has only ConfCalib=5 and 1/3 rolling).
BEST_PROJECTS = [
    "apache@sling",
    "l0rdn1kk0n@wicket-bootstrap",
    "SonarSource@sonarqube",
    "deeplearning4j@deeplearning4j",
]

METHOD_ROLES = {
    "global_conformal": "conformal_marginal_guarantee",
    "global_conformal_cost_aware": "conformal_marginal_guarantee",
    "global_conformal_qtf": "conformal_marginal_guarantee",
    "residual_risk": "heuristic_calibrated_tau",
    "residual_risk_cost_aware": "heuristic_calibrated_tau",
}

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

# Label statuses treated as indeterminate under confirmed_new_fail
INDETERMINATE_STATUSES = {
    "unknown_parent",
    "parent_mixed",
    "ambiguous_current",
    "mixed_current",
    "unobserved_candidate",
    "missing",
}


def min_defective_for_nonzero_threshold(alpha: float) -> int:
    """Minimum number of defective calibration commits n such that
    floor(alpha * (n + 1)) >= 1 (yielding a non-zero conformal threshold).
    """
    return max(1, int(math.ceil(1.0 / alpha - 1e-12)) - 1)


# ---------------------------------------------------------------------------
# Build features + labels for a project
# ---------------------------------------------------------------------------

def build_project_data(
    sim: CIHistorySimulator,
    policy: str = "confirmed_new_fail",
) -> pd.DataFrame:
    """Build complete feature+label dataset for one project by uniting
    pre-execution ``get_candidate_tests(sha)`` with full ``reveal_ground_truth(sha)``.

    Sets ``in_candidate_inventory``:
      - ``True``  for tests returned by ``get_candidate_tests(sha)`` (including
                  unobserved candidates marked ``label=None``, ``label_status='unobserved_candidate'``)
      - ``False`` for tests that ran in CI logs (``reveal_ground_truth``) but were
                  absent from ``get_candidate_tests(sha)`` (never selectable, yet
                  counted in full ground truth during conformal calibration and CMR).
    """
    all_commits = sim.get_all_commits()
    print(f"    [features] building for {len(all_commits)} commits...", flush=True)

    df_features = build_features_batch(sim, all_commits, verbose=True)

    # Pre-fetch ground truth once per commit (O(C) instead of O(N))
    print(f"    [ground_truth] revealing ground truth for {len(all_commits)} commits...", flush=True)
    gt_by_commit = {sha: sim.reveal_ground_truth(sha, policy=policy) for sha in all_commits}

    has_get_parent_source = hasattr(sim, "get_parent_source")
    cand_pairs_by_commit: Dict[str, set] = {}

    if len(df_features) > 0:
        for sha_val, tname_val in zip(df_features["commit"], df_features["testName"]):
            cand_pairs_by_commit.setdefault(sha_val, set()).add(tname_val)

        labels: list = []
        durations: list[float] = []
        statuses: list[str] = []
        parent_statuses: list[str] = []
        current_statuses: list[str] = []
        parent_shas: list[Optional[str]] = []
        ambiguous_flags: list[bool] = []
        parent_sources: list[str] = []

        for row in df_features.itertuples(index=False):
            sha = row.commit
            test_name = row.testName
            gt = gt_by_commit.get(sha)
            if gt is not None and test_name in gt.test_results:
                tr = gt.test_results[test_name]
                labels.append(tr.label)
                durations.append(tr.actual_duration)
                statuses.append(tr.label_status)
                parent_statuses.append(tr.parent_status)
                current_statuses.append(tr.current_status)
                parent_shas.append(tr.parent_sha)
                ambig = getattr(
                    tr,
                    "ambiguous_outcome_same_sha",
                    getattr(tr, "mixed_outcomes_same_sha", False),
                )
                ambiguous_flags.append(bool(ambig))
                parent_sources.append(tr.parent_source)
            else:
                # Candidate test from pre-run inventory not observed in CI logs for this SHA
                labels.append(None)
                durations.append(float("nan"))
                statuses.append("unobserved_candidate")
                parent_statuses.append("unknown")
                current_statuses.append("unobserved")
                parent_shas.append(None)
                ambiguous_flags.append(False)
                p_src = sim.get_parent_source(sha) if has_get_parent_source else "none"
                parent_sources.append(p_src)

        df_features = df_features.copy()
        df_features["in_candidate_inventory"] = True
        df_features["label"] = pd.Series(labels, dtype="object")
        df_features["actual_duration"] = durations
        df_features["label_status"] = statuses
        df_features["label_attribution"] = statuses  # backward-compatible alias
        df_features["parent_status"] = parent_statuses
        df_features["current_status"] = current_statuses
        df_features["parent_sha"] = parent_shas
        df_features["ambiguous_outcome_same_sha"] = ambiguous_flags
        df_features["mixed_outcomes_same_sha"] = ambiguous_flags  # backward-compatible alias
        df_features["parent_source"] = parent_sources

    # Union tests that appeared ONLY in reveal_ground_truth(sha)
    extra_rows: list[dict] = []
    for sha in all_commits:
        gt = gt_by_commit.get(sha)
        if gt is None:
            continue
        seen_cands = cand_pairs_by_commit.get(sha, ())
        c_idx = sim.get_commit_order(sha)
        for test_name, tr in gt.test_results.items():
            if test_name in seen_cands:
                continue
            ambig = bool(getattr(
                tr,
                "ambiguous_outcome_same_sha",
                getattr(tr, "mixed_outcomes_same_sha", False),
            ))
            rec: dict = {
                "commit": sha,
                "commit_idx": c_idx,
                "testName": test_name,
            }
            for col in FEATURE_COLS:
                rec[col] = np.nan
            rec.update({
                "in_candidate_inventory": False,
                "label": tr.label,
                "actual_duration": tr.actual_duration,
                "label_status": tr.label_status,
                "label_attribution": tr.label_status,
                "parent_status": tr.parent_status,
                "current_status": tr.current_status,
                "parent_sha": tr.parent_sha,
                "ambiguous_outcome_same_sha": ambig,
                "mixed_outcomes_same_sha": ambig,
                "parent_source": tr.parent_source,
            })
            extra_rows.append(rec)

    if len(df_features) == 0 and not extra_rows:
        return df_features

    if extra_rows:
        df_extra = pd.DataFrame(extra_rows)
        df_extra["label"] = pd.Series(df_extra["label"].tolist(), dtype="object")
        if len(df_features) > 0:
            df_out = pd.concat([df_features, df_extra], ignore_index=True)
            df_out = df_out.sort_values(["commit_idx", "testName"], kind="mergesort").reset_index(drop=True)
        else:
            df_out = df_extra.sort_values(["commit_idx", "testName"], kind="mergesort").reset_index(drop=True)
    else:
        df_out = df_features

    df_out["in_candidate_inventory"] = df_out["in_candidate_inventory"].astype(bool)

    ps_counts = df_out.groupby("commit")["parent_source"].first().value_counts()
    print(f"    [parent_source] { {str(k): int(v) for k, v in ps_counts.items()} }", flush=True)

    # Candidate inventory completeness statistics (over observed ground-truth runs)
    obs_mask = df_out["current_status"] != "unobserved"
    df_obs = df_out.loc[obs_mask]
    n_obs = len(df_obs)
    out_obs = df_obs[~df_obs["in_candidate_inventory"]]
    n_out = len(out_obs)
    pos_obs = df_obs[df_obs["label"] == 1]
    n_pos = len(pos_obs)
    out_pos = pos_obs[~pos_obs["in_candidate_inventory"]]
    n_out_pos = len(out_pos)
    n_out_pos_commits = int(out_pos["commit"].nunique()) if n_out_pos > 0 else 0
    pct_out = (100.0 * n_out / n_obs) if n_obs > 0 else 0.0
    print(
        f"    [inventory] observed_gt_pairs={n_obs}, outside_candidate_inventory={n_out} "
        f"({pct_out:.2f}%), confirmed_fail_pairs={n_pos}, "
        f"confirmed_fail_outside_inventory={n_out_pos} (commits={n_out_pos_commits})",
        flush=True,
    )

    return df_out


def filter_valid_pairs(
    df: pd.DataFrame,
    include_fallback_parents: bool = False,
    policy: str = "confirmed_new_fail",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """Filter out non-Git parents (unless sensitivity flag set) and indeterminate
    observations (label is None / NaN or unobserved candidates).

    Returns ``(filtered_df, stats)`` where ``stats`` records counts of excluded
    pairs and commits by reason.
    """
    if len(df) == 0:
        return df.copy(), {}

    work = df.copy()
    stats: Dict[str, Dict[str, int]] = {}
    commit_col = "commit" if "commit" in work.columns else "commit_idx"

    # 1. Parent-source filtering (main experiment requires confirmed git_parent)
    if not include_fallback_parents and "parent_source" in work.columns:
        non_git_mask = work["parent_source"] != "git_parent"
        if non_git_mask.any():
            for src_val, grp in work[non_git_mask].groupby("parent_source"):
                key = f"parent_source:{src_val}"
                stats[key] = {
                    "pairs": int(len(grp)),
                    "commits": int(grp[commit_col].nunique()),
                }
            work = work.loc[~non_git_mask].copy()

    # 2. Exclude label is None / NaN (three-valued ground truth) and unobserved candidates
    null_label_mask = work["label"].isna()
    if policy == "confirmed_new_fail" and "label_status" in work.columns:
        indet_mask = null_label_mask | work["label_status"].isin(INDETERMINATE_STATUSES)
    else:
        # Even for sensitivity policies, unobserved candidates have no ground truth
        unobs_mask = (
            work["label_status"].isin({"unobserved_candidate", "missing"})
            if "label_status" in work.columns
            else pd.Series(False, index=work.index)
        )
        indet_mask = null_label_mask | unobs_mask

    if indet_mask.any():
        excluded = work.loc[indet_mask]
        if "label_status" in excluded.columns:
            for status_val, grp in excluded.groupby("label_status"):
                stats[str(status_val)] = {
                    "pairs": int(len(grp)),
                    "commits": int(grp[commit_col].nunique()),
                }
        else:
            stats["label_none"] = {
                "pairs": int(len(excluded)),
                "commits": int(excluded[commit_col].nunique()),
            }
        work = work.loc[~indet_mask].copy()

    # Convert surviving clean labels to integer 0/1
    if len(work) > 0:
        work["label"] = work["label"].astype(int)

    if verbose:
        total_removed_pairs = len(df) - len(work)
        total_removed_commits = df[commit_col].nunique() - (work[commit_col].nunique() if len(work) > 0 else 0)
        print(
            f"  [filter] kept {len(work)}/{len(df)} pairs "
            f"(excluded {total_removed_pairs} pairs, {total_removed_commits} commits completely dropped; "
            f"include_fallback_parents={include_fallback_parents})",
            flush=True,
        )
        for reason, cnts in sorted(stats.items()):
            print(
                f"    - {reason}: {cnts['pairs']} pairs across {cnts['commits']} commits",
                flush=True,
            )
        if "in_candidate_inventory" in work.columns and len(work) > 0:
            outside = work[~work["in_candidate_inventory"].astype(bool)]
            outside_pos = outside[outside["label"] == 1]
            print(
                f"    - [inventory_after_filter] outside_candidate_inventory={len(outside)}/{len(work)} pairs "
                f"({100.0 * len(outside) / len(work):.2f}%), "
                f"confirmed_fail_outside={len(outside_pos)}/{int((work['label'] == 1).sum())}",
                flush=True,
            )

    return work, stats


def _predict_split_probs(
    model,
    calibrator,
    df_split: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict ``(probs_calibrated, probs_raw)`` strictly on
    ``in_candidate_inventory == True`` rows; non-candidate rows receive 0.0.
    """
    n = len(df_split)
    probs_cal = np.zeros(n, dtype=float)
    probs_raw = np.zeros(n, dtype=float)
    if n == 0:
        return probs_cal, probs_raw
    if "in_candidate_inventory" in df_split.columns:
        cand_mask = df_split["in_candidate_inventory"].fillna(False).astype(bool).values
    else:
        cand_mask = np.ones(n, dtype=bool)
    if cand_mask.any():
        raw_sub, cal_sub = predict_raw_and_calibrated(
            model, calibrator, df_split.loc[cand_mask], feature_cols
        )
        probs_raw[cand_mask] = raw_sub
        probs_cal[cand_mask] = cal_sub
    return probs_cal, probs_raw


def _validate_candidate_training_splits(
    train_cand: pd.DataFrame,
    prob_cand: pd.DataFrame,
) -> Tuple[bool, str]:
    """Verify that candidate-only training and probability calibration slices
    are non-empty and contain both classes (0 and 1).
    """
    if len(train_cand) == 0 or train_cand["label"].nunique() < 2:
        return False, "train_cand_single_class_or_empty"
    if len(prob_cand) == 0 or prob_cand["label"].nunique() < 2:
        return False, "prob_cand_single_class_or_empty"
    return True, ""


def _compute_test_inventory_diagnostics(test_df: pd.DataFrame) -> Dict[str, float]:
    """Measure pre-execution candidate inventory gaps on the test split."""
    n_tot = len(test_df)
    if n_tot == 0 or "in_candidate_inventory" not in test_df.columns:
        return {
            "inv_miss_pair_rate_test": 0.0,
            "inv_miss_fail_rate_test": 0.0,
            "inv_miss_commit_rate_test": 0.0,
            "n_inv_miss_only_defective_commits": 0,
        }

    cand = test_df["in_candidate_inventory"].fillna(False).astype(bool)
    non_cand = ~cand
    pair_rate = float(non_cand.sum() / n_tot)

    pos_df = test_df[test_df["label"] == 1]
    n_pos = len(pos_df)
    fail_rate = float((~cand.loc[pos_df.index]).sum() / n_pos) if n_pos > 0 else 0.0

    def_commits = pos_df["commit_idx"].unique()
    n_def_commits = len(def_commits)
    n_uncatchable_commits = 0
    if n_def_commits > 0:
        for cid in def_commits:
            sub = pos_df[pos_df["commit_idx"] == cid]
            if not cand.loc[sub.index].any():
                n_uncatchable_commits += 1
    commit_rate = float(n_uncatchable_commits / n_def_commits) if n_def_commits > 0 else 0.0

    return {
        "inv_miss_pair_rate_test": pair_rate,
        "inv_miss_fail_rate_test": fail_rate,
        "inv_miss_commit_rate_test": commit_rate,
        "n_inv_miss_only_defective_commits": int(n_uncatchable_commits),
    }


# ---------------------------------------------------------------------------
# Single-project, single-seed run
# ---------------------------------------------------------------------------

def run_one(
    sim: CIHistorySimulator,
    df: pd.DataFrame,
    seed: int,
    diag_out: Optional[dict] = None,
    strict_min_defective: int = 0,
    calibration_method: str = "auto",
    miss_records_out: Optional[list] = None,
) -> list:
    """Run one experiment iteration.

    Returns a list of metric dicts for all (alpha, method) combinations.
    If ``diag_out`` dict is provided, stores ``y_true``, ``probs``,
    ``probs_iso``, and ``probs_sigmoid`` on ``in_candidate_inventory == True`` rows
    for reliability diagram generation.
    """
    project = sim.project

    n_def = df[df["label"] == 1]["commit_idx"].nunique()
    print(
        f"    [info] commits={df['commit_idx'].nunique()}, "
        f"pairs={len(df)}, defective_commits={n_def}",
        flush=True,
    )

    if n_def < 15:
        raise ValueError(f"Not enough defective commits: {n_def} < 15")

    # --- 4-way split ---
    train_df, prob_calib_df, conf_calib_df, test_df = chrono_split_4(
        df, TRAIN_FRAC, PROB_CALIB_FRAC, CONF_CALIB_FRAC
    )

    n_def_conf = int(conf_calib_df[conf_calib_df["label"] == 1]["commit_idx"].nunique())
    n_def_test = int(test_df[test_df["label"] == 1]["commit_idx"].nunique())
    n_def_conf_plus_test = n_def_conf + n_def_test

    print(
        f"    [split] train={len(train_df)}, prob_calib={len(prob_calib_df)}, "
        f"conf_calib={len(conf_calib_df)}, test={len(test_df)}",
        flush=True,
    )
    print(
        f"    [split] def_conf_calib={n_def_conf}, def_test={n_def_test}, "
        f"conf+test={n_def_conf_plus_test}",
        flush=True,
    )

    if n_def_conf < MIN_DEFECTIVE_CONF_CALIB:
        raise ValueError(f"Too few defective in conf_calib: {n_def_conf}")
    if n_def_test < MIN_DEFECTIVE_TEST:
        raise ValueError(f"Too few defective in test: {n_def_test}")
    if strict_min_defective > 0 and n_def_conf_plus_test < strict_min_defective:
        raise ValueError(
            f"ConfCalib + Test defective commits ({n_def_conf_plus_test}) "
            f"< strict_min_defective ({strict_min_defective})"
        )

    # Restrict model training, probability calibration, and ML model metrics strictly
    # to in_candidate_inventory == True
    has_inv = "in_candidate_inventory" in df.columns
    if has_inv:
        train_cand = train_df[train_df["in_candidate_inventory"].fillna(False).astype(bool)].copy()
        prob_cand = prob_calib_df[prob_calib_df["in_candidate_inventory"].fillna(False).astype(bool)].copy()
        cand_mask_test = test_df["in_candidate_inventory"].fillna(False).astype(bool).values
        test_cand = test_df.loc[cand_mask_test]
    else:
        train_cand = train_df
        prob_cand = prob_calib_df
        cand_mask_test = np.ones(len(test_df), dtype=bool)
        test_cand = test_df

    split_ok, split_reason = _validate_candidate_training_splits(train_cand, prob_cand)
    if not split_ok:
        raise ValueError(f"Invalid candidate training/prob_calib slice: {split_reason}")

    # --- Train LightGBM ---
    print(f"    [train] seed={seed} (candidates={len(train_cand)}/{len(train_df)})", flush=True)
    model = train_lgbm(train_cand, FEATURE_COLS, seed=seed)

    # --- Probability calibration: Isotonic vs Platt/Sigmoid (evaluated strictly on ProbCalib) ---
    print(
        f"    [calibrate] mode={calibration_method} (candidates={len(prob_cand)}/{len(prob_calib_df)})",
        flush=True,
    )
    train_prior = float(train_cand["label"].mean())
    cal_info = evaluate_and_select_calibrators(model, prob_cand, FEATURE_COLS, train_prior=train_prior)
    if calibration_method == "isotonic":
        active_cal = cal_info["iso"]
        selected_cal_name = "isotonic"
    elif calibration_method == "sigmoid":
        active_cal = cal_info["sigmoid"]
        selected_cal_name = "sigmoid"
    else:
        active_cal = cal_info["selected_calibrator"]
        selected_cal_name = cal_info["selected_name"]

    # --- Predict (forced 0.0 outside candidate inventory) ---
    probs_conf_calib, _ = _predict_split_probs(model, active_cal, conf_calib_df, FEATURE_COLS)
    probs_test, probs_raw_test = _predict_split_probs(model, active_cal, test_df, FEATURE_COLS)
    probs_test_iso, _ = _predict_split_probs(model, cal_info["iso"], test_df, FEATURE_COLS)
    probs_test_sig, _ = _predict_split_probs(model, cal_info["sigmoid"], test_df, FEATURE_COLS)

    # Evaluate ranking/calibration quality strictly on selectable candidates (in_candidate_inventory == True)
    y_test_cand = test_cand["label"].values
    probs_test_cand = probs_test[cand_mask_test]
    probs_raw_test_cand = probs_raw_test[cand_mask_test]
    probs_iso_test_cand = probs_test_iso[cand_mask_test]
    probs_sig_test_cand = probs_test_sig[cand_mask_test]

    pr_auc_test = compute_pr_auc(y_test_cand, probs_test_cand) if len(test_cand) > 0 else float("nan")
    pr_auc_baseline_test = compute_pr_auc_baseline(y_test_cand)
    brier_test = compute_brier_score(y_test_cand, probs_test_cand)
    brier_iso_test = compute_brier_score(y_test_cand, probs_iso_test_cand)
    brier_sig_test = compute_brier_score(y_test_cand, probs_sig_test_cand)
    brier_raw_test = compute_brier_score(y_test_cand, probs_raw_test_cand)
    brier_baseline_test = compute_brier_baseline(train_cand["label"].values, y_test_cand)

    bss_selected_test = compute_brier_skill_score(brier_test, brier_baseline_test)
    bss_iso_test = compute_brier_skill_score(brier_iso_test, brier_baseline_test)
    bss_sig_test = compute_brier_skill_score(brier_sig_test, brier_baseline_test)
    bss_raw_test = compute_brier_skill_score(brier_raw_test, brier_baseline_test)

    if len(probs_test_cand) > 0:
        prob_p50 = float(np.percentile(probs_test_cand, 50))
        prob_p90 = float(np.percentile(probs_test_cand, 90))
        prob_p99 = float(np.percentile(probs_test_cand, 99))
        prob_max = float(np.max(probs_test_cand))
    else:
        prob_p50 = prob_p90 = prob_p99 = prob_max = float("nan")

    if diag_out is not None:
        diag_out["y_true"] = y_test_cand
        diag_out["probs"] = probs_test_cand
        diag_out["probs_iso"] = probs_iso_test_cand
        diag_out["probs_sigmoid"] = probs_sig_test_cand
        diag_out["selected_calibrator"] = selected_cal_name
        diag_out["bss_iso"] = bss_iso_test
        diag_out["bss_sigmoid"] = bss_sig_test

    inv_diag = _compute_test_inventory_diagnostics(test_df)
    print(
        f"    [prob_calib_cv] selected={selected_cal_name}  cv_brier: iso={cal_info['prob_cv_brier_iso']:.6f}, "
        f"sig={cal_info['prob_cv_brier_sigmoid']:.6f}, base={cal_info['prob_cv_brier_baseline']:.6f}",
        flush=True,
    )
    print(
        f"    [pr_auc|cand] test={pr_auc_test:.6f} (baseline={pr_auc_baseline_test:.6f})  "
        f"brier[cand]: sel={brier_test:.6f}, iso={brier_iso_test:.6f}, sig={brier_sig_test:.6f}, "
        f"raw={brier_raw_test:.6f}, base={brier_baseline_test:.6f}  "
        f"BSS: iso={bss_iso_test:+.3f}, sig={bss_sig_test:+.3f}  "
        f"[inv_miss] fails={inv_diag['inv_miss_fail_rate_test']:.2%}, "
        f"commits={inv_diag['n_inv_miss_only_defective_commits']}",
        flush=True,
    )

    scores = compute_conformal_scores(conf_calib_df, probs_conf_calib)
    conf_score_min = float(np.min(scores)) if len(scores) > 0 else float("nan")
    conf_score_median = float(np.median(scores)) if len(scores) > 0 else float("nan")

    # --- Conformal + evaluation per alpha (on full ground truth test_df) ---
    cand_mask_eval = _candidate_mask(test_df)
    all_rows: list[dict] = []
    for alpha in ALPHAS:
        lambda_ = compute_global_threshold(scores, alpha)
        tau = compute_tau_residual(conf_calib_df, probs_conf_calib, alpha)

        min_n = min_defective_for_nonzero_threshold(alpha)
        suff = bool(n_def_conf >= min_n)
        cal_status = "ok" if suff else "insufficient_for_nonzero_threshold"
        print(
            f"    [conformal] alpha={alpha:.2f}, lambda={lambda_:.4f}, tau={tau:.4f} "
            f"(def_conf={n_def_conf}, min_for_nonzero={min_n}, status={cal_status}, "
            f"S_min={conf_score_min:.4f}, prob_p90={prob_p90:.4f})",
            flush=True,
        )

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

        if miss_records_out is not None:
            probs_eval = np.where(cand_mask_eval, np.asarray(probs_test, dtype=float), 0.0)
            sel_g = select_tests_global(test_df, probs_eval, lambda_)
            mask_g = test_df.index.isin(sel_g.index) & cand_mask_eval
            sel_r = select_tests_residual_all_commits(test_df, probs_eval, tau)
            mask_r = test_df.index.isin(sel_r.index) & cand_mask_eval
            for m_label, m_mask in (("global_conformal", mask_g), ("residual_risk", mask_r)):
                _, m_recs = analyze_miss_causes(test_df, m_mask, probs_test=probs_eval)
                for rec in m_recs:
                    rec["project"] = project
                    rec["seed"] = seed
                    rec["alpha"] = alpha
                    rec["method"] = m_label
                    rec["selected_calibrator"] = selected_cal_name
                    miss_records_out.append(rec)

        for row in rows:
            row["project"] = project
            row["seed"] = seed
            row["protocol"] = "holdout_50_15_15_20"
            row["guarantee_type"] = METHOD_ROLES.get(row["method"], "baseline")
            row["calibration_mode"] = calibration_method
            row["selected_calibrator"] = selected_cal_name
            row["pr_auc_test"] = pr_auc_test
            row["pr_auc_baseline_test"] = pr_auc_baseline_test
            row["brier_score_test"] = brier_test
            row["brier_iso_test"] = brier_iso_test
            row["brier_sigmoid_test"] = brier_sig_test
            row["brier_raw_test"] = brier_raw_test
            row["brier_baseline_test"] = brier_baseline_test
            row["bss_selected_test"] = bss_selected_test
            row["bss_iso_test"] = bss_iso_test
            row["bss_sigmoid_test"] = bss_sig_test
            row["bss_raw_test"] = bss_raw_test
            row["prob_cv_brier_iso"] = cal_info["prob_cv_brier_iso"]
            row["prob_cv_brier_sigmoid"] = cal_info["prob_cv_brier_sigmoid"]
            row["prob_cv_brier_baseline"] = cal_info["prob_cv_brier_baseline"]
            row["prob_cv_bss_iso"] = cal_info["prob_cv_bss_iso"]
            row["prob_cv_bss_sigmoid"] = cal_info["prob_cv_bss_sigmoid"]
            row["prob_cand_p50"] = prob_p50
            row["prob_cand_p90"] = prob_p90
            row["prob_cand_p99"] = prob_p99
            row["prob_cand_max"] = prob_max
            row["conf_score_min"] = conf_score_min
            row["conf_score_median"] = conf_score_median
            row["inv_miss_pair_rate_test"] = inv_diag["inv_miss_pair_rate_test"]
            row["inv_miss_fail_rate_test"] = inv_diag["inv_miss_fail_rate_test"]
            row["inv_miss_commit_rate_test"] = inv_diag["inv_miss_commit_rate_test"]
            row["n_inv_miss_only_defective_commits"] = inv_diag["n_inv_miss_only_defective_commits"]
            row["n_def_conf_calib"] = n_def_conf
            row["n_def_test"] = n_def_test
            row["n_def_conf_plus_test"] = n_def_conf_plus_test
            row["meets_150_conf_test"] = bool(n_def_conf_plus_test >= 150)
            row["min_def_conf_nonzero"] = min_n
            row["sufficient_for_nonzero_threshold"] = suff
            row["calibration_status"] = cal_status
            row["lambda"] = lambda_
            row["tau"] = tau
            all_rows.append(row)

    return all_rows


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_all(
    projects=None,
    seeds=None,
    out_dir: Path = OUT_DIR,
    policy: str = "confirmed_new_fail",
    include_fallback_parents: bool = False,
    cache_out: Optional[Dict[str, Tuple[CIHistorySimulator, pd.DataFrame]]] = None,
    strict_min_defective: int = 0,
    calibration_method: str = "auto",
    miss_records_out: Optional[list] = None,
):
    target_projects = projects if projects is not None else BEST_PROJECTS
    num_seeds = seeds if seeds is not None else N_SEEDS

    all_results: list[dict] = []
    miss_records: list[dict] = miss_records_out if miss_records_out is not None else []
    reliability_data: Dict[str, dict] = {}
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
            df_raw = build_project_data(sim, policy=policy)
            print(f"  [data] {len(df_raw)} raw (commit, test) candidate pairs", flush=True)

            df, _ = filter_valid_pairs(
                df_raw,
                include_fallback_parents=include_fallback_parents,
                policy=policy,
                verbose=True,
            )

            if cache_out is not None:
                cache_out[project] = (sim, df)

            if len(df) == 0:
                print(f"  No valid pairs left for {project} — skipping.", flush=True)
                continue

        except Exception as e:
            print(f"  FAILED to load project: {e}", flush=True)
            traceback.print_exc()
            continue

        project_rows: list[dict] = []
        _reliability_saved = False
        for seed in range(num_seeds):
            t0 = time.time()
            print(f"\n  Seed {seed}/{num_seeds-1} ...", flush=True)
            diag: dict = {}
            try:
                rows = run_one(
                    sim,
                    df,
                    seed,
                    diag_out=diag if not _reliability_saved else None,
                    strict_min_defective=strict_min_defective,
                    calibration_method=calibration_method,
                    miss_records_out=miss_records,
                )
                project_rows.extend(rows)
                print(f"  Seed {seed} OK ({time.time()-t0:.1f}s)", flush=True)

                if not _reliability_saved and "y_true" in diag and "probs" in diag:
                    reliability_data[project] = dict(diag)
                    try:
                        make_reliability_diagram(
                            diag["y_true"],
                            diag["probs"],
                            project,
                            out_dir / "figures",
                            probs_iso=diag.get("probs_iso"),
                            probs_sigmoid=diag.get("probs_sigmoid"),
                            bss_iso=diag.get("bss_iso"),
                            bss_sigmoid=diag.get("bss_sigmoid"),
                            selected_calibrator=diag.get("selected_calibrator", "isotonic"),
                        )
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

    # Multi-panel reliability diagram across all projects (overwrites single fig4)
    if len(reliability_data) > 1:
        try:
            make_reliability_diagram_multi(reliability_data, out_dir / "figures")
        except Exception as _e:
            print(f"  Multi-panel reliability diagram FAILED: {_e}", flush=True)

    if miss_records:
        pd.DataFrame(miss_records).to_csv(out_dir / "miss_analysis.csv", index=False)

    print(f"\n\nTotal time: {(time.time()-t0_global)/60:.1f} min", flush=True)
    return all_results


# ---------------------------------------------------------------------------
# Aggregation & specialized summaries
# ---------------------------------------------------------------------------

def aggregate_results(all_results: list) -> pd.DataFrame:
    """Compute mean +- std per (project, method, alpha).

    Note: ``std`` across ``seeds`` measures model-training seed sensitivity
    on the same chronological test split (NOT independent commit sampling variance).
    Commit-level statistical uncertainty is reported in ``cmr_exact_ci_*``,
    ``cmr_upper_95``, ``cmr_ci_*`` (Wilson), and ``cmr_boot_ci_*``.
    Uses ddof=1 for std; does NOT replace NaN with 0.
    """
    df = pd.DataFrame(all_results)

    metric_cols = [
        "cmr", "pct_tests_selected", "pct_duration_saved",
        "pct_duration_selected", "ttff", "apfdc", "pr_auc_test",
    ]
    for opt_col in [
        "ttff_median",
        "ttff_p90",
        "total_test_duration",
        "selected_test_duration",
        "n_obs_pairs",
        "n_selected_obs_pairs",
        "conditional_apfdc",
        "pr_auc_baseline_test",
        "brier_score_test",
        "brier_iso_test",
        "brier_sigmoid_test",
        "brier_raw_test",
        "brier_baseline_test",
        "bss_selected_test",
        "bss_iso_test",
        "bss_sigmoid_test",
        "bss_raw_test",
        "prob_cv_brier_iso",
        "prob_cv_brier_sigmoid",
        "prob_cv_brier_baseline",
        "prob_cv_bss_iso",
        "prob_cv_bss_sigmoid",
        "prob_cand_p50",
        "prob_cand_p90",
        "prob_cand_p99",
        "prob_cand_max",
        "conf_score_min",
        "conf_score_median",
        "inv_miss_pair_rate_test",
        "inv_miss_fail_rate_test",
        "inv_miss_commit_rate_test",
        "n_inv_miss_only_defective_commits",
        "n_defective_commits",
        "n_missed_defective_commits",
        "miss_missing_from_inventory",
        "miss_empty_prefix",
        "miss_no_prior_history",
        "miss_score_below_cutoff",
        "n_def_conf_calib",
        "n_def_test",
        "n_def_conf_plus_test",
        "cmr_ci_low",
        "cmr_ci_high",
        "cmr_exact_ci_low",
        "cmr_exact_ci_high",
        "cmr_upper_95",
        "cmr_boot_ci_low",
        "cmr_boot_ci_high",
    ]:
        if opt_col in df.columns:
            metric_cols.append(opt_col)

    agg_mean = df.groupby(["project", "method", "alpha"])[metric_cols].mean()
    agg_std = df.groupby(["project", "method", "alpha"])[metric_cols].std(ddof=1)

    agg_mean.columns = [f"{c}_mean" for c in metric_cols]
    agg_std.columns = [f"{c}_std" for c in metric_cols]

    table = pd.concat([agg_mean, agg_std], axis=1).reset_index()
    if "guarantee_type" in df.columns:
        role_map = df.groupby("method")["guarantee_type"].first().to_dict()
        table["guarantee_type"] = table["method"].map(role_map).fillna("baseline")
    if "protocol" in df.columns:
        table["protocol"] = str(df["protocol"].iloc[0])
    table = table.sort_values(["project", "method", "alpha"])
    return table


def build_calibration_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize ranking & probability calibration (Raw vs Isotonic vs Sigmoid vs Baseline) per project."""
    if len(raw_df) == 0:
        return pd.DataFrame()

    # Model predictions do not vary across alpha or method for a given (project, seed)
    per_seed = raw_df.groupby(["project", "seed"], as_index=False).first()
    num_cols = [
        c for c in [
            "pr_auc_test", "pr_auc_baseline_test",
            "brier_raw_test", "brier_iso_test", "brier_sigmoid_test",
            "brier_score_test", "brier_baseline_test",
            "bss_raw_test", "bss_iso_test", "bss_sigmoid_test", "bss_selected_test",
            "prob_cv_brier_iso", "prob_cv_brier_sigmoid", "prob_cv_brier_baseline",
            "prob_cv_bss_iso", "prob_cv_bss_sigmoid",
            "prob_cand_p50", "prob_cand_p90", "prob_cand_p99", "prob_cand_max",
            "conf_score_min", "conf_score_median",
        ]
        if c in per_seed.columns
    ]
    summary = per_seed.groupby("project", as_index=False)[num_cols].mean()
    if "selected_calibrator" in per_seed.columns:
        modes = per_seed.groupby("project")["selected_calibrator"].agg(
            lambda s: str(s.mode().iloc[0]) if len(s) > 0 else "unknown"
        ).to_dict()
        summary["selected_calibrator_mode"] = summary["project"].map(modes)
    return summary.sort_values("project")


def build_savings_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize test selection and duration savings per project plus unweighted macro mean
    and actual CI-duration-weighted cross-project aggregate.
    """
    if len(raw_df) == 0:
        return pd.DataFrame()

    rows: list[dict] = []
    # 1. Per-project averages across seeds
    for (proj, method, alpha), grp in raw_df.groupby(["project", "method", "alpha"], sort=True):
        rows.append({
            "row_type": "per_project",
            "project": proj,
            "method": method,
            "guarantee_type": grp["guarantee_type"].iloc[0] if "guarantee_type" in grp.columns else METHOD_ROLES.get(method, "baseline"),
            "alpha": float(alpha),
            "pct_tests_selected": float(grp["pct_tests_selected"].mean()),
            "pct_duration_saved": float(grp["pct_duration_saved"].mean()),
            "cmr": float(grp["cmr"].mean()),
            "total_test_duration": float(grp["total_test_duration"].mean()) if "total_test_duration" in grp.columns else float("nan"),
            "selected_test_duration": float(grp["selected_test_duration"].mean()) if "selected_test_duration" in grp.columns else float("nan"),
        })

    per_proj_df = pd.DataFrame(rows)
    if len(per_proj_df) == 0:
        return per_proj_df

    # 2. Unweighted macro mean across projects
    for (method, alpha), grp in per_proj_df.groupby(["method", "alpha"], sort=True):
        rows.append({
            "row_type": "macro_mean_across_projects",
            "project": "ALL_MACRO",
            "method": method,
            "guarantee_type": grp["guarantee_type"].iloc[0],
            "alpha": float(alpha),
            "pct_tests_selected": float(grp["pct_tests_selected"].mean()),
            "pct_duration_saved": float(grp["pct_duration_saved"].mean()),
            "cmr": float(grp["cmr"].mean()),
            "total_test_duration": float(grp["total_test_duration"].sum()),
            "selected_test_duration": float(grp["selected_test_duration"].sum()),
        })

    # 3. Duration-weighted & pair-weighted cross-project aggregate (per seed, then averaged)
    if {"total_test_duration", "selected_test_duration", "n_obs_pairs", "n_selected_obs_pairs"}.issubset(raw_df.columns):
        seed_agg: list[dict] = []
        for (seed, method, alpha), grp in raw_df.groupby(["seed", "method", "alpha"], sort=True):
            tot_d = float(grp["total_test_duration"].sum())
            sel_d = float(grp["selected_test_duration"].sum())
            tot_p = float(grp["n_obs_pairs"].sum())
            sel_p = float(grp["n_selected_obs_pairs"].sum())
            n_def_tot = float(grp["n_defective_commits"].sum()) if "n_defective_commits" in grp.columns else 0.0
            n_miss_tot = float(grp["n_missed_defective_commits"].sum()) if "n_missed_defective_commits" in grp.columns else 0.0
            seed_agg.append({
                "seed": seed,
                "method": method,
                "guarantee_type": grp["guarantee_type"].iloc[0] if "guarantee_type" in grp.columns else METHOD_ROLES.get(method, "baseline"),
                "alpha": float(alpha),
                "pct_tests_selected": (sel_p / tot_p) if tot_p > 0 else float("nan"),
                "pct_duration_saved": (1.0 - sel_d / tot_d) if tot_d > 0 else float("nan"),
                "cmr": (n_miss_tot / n_def_tot) if n_def_tot > 0 else float("nan"),
                "total_test_duration": tot_d,
                "selected_test_duration": sel_d,
            })
        s_df = pd.DataFrame(seed_agg)
        for (method, alpha), grp in s_df.groupby(["method", "alpha"], sort=True):
            rows.append({
                "row_type": "duration_weighted_overall",
                "project": "ALL_DURATION_WEIGHTED",
                "method": method,
                "guarantee_type": grp["guarantee_type"].iloc[0],
                "alpha": float(alpha),
                "pct_tests_selected": float(grp["pct_tests_selected"].mean()),
                "pct_duration_saved": float(grp["pct_duration_saved"].mean()),
                "cmr": float(grp["cmr"].mean()),
                "total_test_duration": float(grp["total_test_duration"].mean()),
                "selected_test_duration": float(grp["selected_test_duration"].mean()),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _binned_reliability(
    labels: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> Tuple[list, list]:
    """Compute (mean_pred, frac_pos) calibration curve bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    frac_pos, mean_pred = [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs <= hi if i == n_bins - 1 else probs < hi)
        if mask.sum() == 0:
            continue
        frac_pos.append(float(labels[mask].mean()))
        mean_pred.append(float(probs[mask].mean()))
    return mean_pred, frac_pos


def make_reliability_diagram(
    labels: np.ndarray,
    probs: np.ndarray,
    project: str,
    fig_dir: Path,
    n_bins: int = 10,
    probs_iso: Optional[np.ndarray] = None,
    probs_sigmoid: Optional[np.ndarray] = None,
    bss_iso: Optional[float] = None,
    bss_sigmoid: Optional[float] = None,
    selected_calibrator: str = "isotonic",
) -> None:
    """Generate and save a reliability (calibration) diagram comparing Isotonic vs Platt."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping reliability diagram", flush=True)
        return

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

    if probs_iso is not None or probs_sigmoid is not None:
        if probs_iso is not None:
            mp, fp = _binned_reliability(labels, probs_iso, n_bins=n_bins)
            bss_s = f" (BSS={bss_iso:+.2f})" if bss_iso is not None and np.isfinite(bss_iso) else ""
            ax.plot(mp, fp, "o-", color="#1f77b4", label=f"Isotonic{bss_s}")
        if probs_sigmoid is not None:
            mp, fp = _binned_reliability(labels, probs_sigmoid, n_bins=n_bins)
            bss_s = f" (BSS={bss_sigmoid:+.2f})" if bss_sigmoid is not None and np.isfinite(bss_sigmoid) else ""
            ax.plot(mp, fp, "s--", color="#ff7f0e", label=f"Platt/Sigmoid{bss_s}")
    else:
        mp, fp = _binned_reliability(labels, probs, n_bins=n_bins)
        ax.plot(mp, fp, "o-", color="#1f77b4", label=f"Calibrated ({selected_calibrator})")

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    short_name = project.split("@")[-1]
    ax.set_title(f"Reliability Diagram — {short_name} (active: {selected_calibrator})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig4_reliability_diagram.pdf", bbox_inches="tight")
    plt.savefig(fig_dir / "fig4_reliability_diagram.png", dpi=150, bbox_inches="tight")
    plt.savefig(fig_dir / f"fig4_reliability_{short_name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig4_reliability_diagram ({short_name})", flush=True)


def make_reliability_diagram_multi(
    reliability_data: Dict[str, Any],
    fig_dir: Path,
    n_bins: int = 10,
) -> None:
    """Generate a combined multi-panel reliability diagram across all projects."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    projects = list(reliability_data.keys())
    fig, axes = plt.subplots(1, len(projects), figsize=(4.5 * len(projects), 4.2))
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = axes.flatten()

    for ax, proj in zip(axes, projects):
        entry = reliability_data[proj]
        if isinstance(entry, dict):
            labels = entry["y_true"]
            p_iso = entry.get("probs_iso")
            p_sig = entry.get("probs_sigmoid")
            p_act = entry["probs"]
            bss_i = entry.get("bss_iso")
            bss_s = entry.get("bss_sigmoid")
            sel_cal = entry.get("selected_calibrator", "auto")
        else:
            labels, p_act = entry
            p_iso = p_sig = bss_i = bss_s = None
            sel_cal = "isotonic"

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")
        if p_iso is not None or p_sig is not None:
            if p_iso is not None:
                mp, fp = _binned_reliability(labels, p_iso, n_bins=n_bins)
                lbl = f"Isotonic ({bss_i:+.2f})" if bss_i is not None and np.isfinite(bss_i) else "Isotonic"
                ax.plot(mp, fp, "o-", color="#1f77b4", label=lbl)
            if p_sig is not None:
                mp, fp = _binned_reliability(labels, p_sig, n_bins=n_bins)
                lbl = f"Sigmoid ({bss_s:+.2f})" if bss_s is not None and np.isfinite(bss_s) else "Sigmoid"
                ax.plot(mp, fp, "s--", color="#ff7f0e", label=lbl)
        else:
            mp, fp = _binned_reliability(labels, p_act, n_bins=n_bins)
            ax.plot(mp, fp, "o-", color="#1f77b4", label="Calibrated")

        ax.set_title(f"{proj.split('@')[-1]} ({sel_cal})", fontsize=10)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Reliability Diagrams (Seed 0, Isotonic vs Platt/Sigmoid)", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig4_reliability_diagram.pdf", bbox_inches="tight")
    plt.savefig(fig_dir / "fig4_reliability_diagram.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved multi-panel fig4_reliability_diagram", flush=True)


def run_rolling(
    sim: "CIHistorySimulator",
    df: pd.DataFrame,
    seed: int,
    n_splits: int = 3,
    out_dir: Path = None,
    calibration_method: str = "auto",
) -> list:
    """Run rolling/expanding window evaluation.

    Logs both executed windows (status='ok') and skipped windows (status='skipped',
    with explicit skip_reason), exact miss counts (n_missed_test), binomial CIs,
    and calibration threshold sufficiency for each alpha.
    """
    from models import rolling_time_splits, train_lgbm
    from conformal import select_tests_global, select_tests_residual_all_commits

    project = sim.project
    if len(df) == 0:
        return []

    splits = rolling_time_splits(
        df,
        n_splits=n_splits,
        min_train_frac=0.25,
        prob_calib_frac=PROB_CALIB_FRAC,
        conf_calib_frac=CONF_CALIB_FRAC,
        test_frac=0.15,
    )
    rows: list[dict] = []
    for win_idx, (train_df, prob_calib_df, conf_calib_df, test_df) in enumerate(splits):
        n_def_conf = int(conf_calib_df[conf_calib_df["label"] == 1]["commit_idx"].nunique())
        n_def_test = int(test_df[test_df["label"] == 1]["commit_idx"].nunique())

        if "in_candidate_inventory" in train_df.columns:
            train_cand = train_df[train_df["in_candidate_inventory"].fillna(False).astype(bool)].copy()
            prob_cand = prob_calib_df[prob_calib_df["in_candidate_inventory"].fillna(False).astype(bool)].copy()
        else:
            train_cand = train_df
            prob_cand = prob_calib_df

        reasons = []
        if n_def_conf < MIN_DEFECTIVE_CONF_CALIB:
            reasons.append(f"def_conf={n_def_conf}<{MIN_DEFECTIVE_CONF_CALIB}")
        if n_def_test < MIN_DEFECTIVE_TEST:
            reasons.append(f"def_test={n_def_test}<{MIN_DEFECTIVE_TEST}")
        split_ok, split_reason = _validate_candidate_training_splits(train_cand, prob_cand)
        if not split_ok:
            reasons.append(split_reason)

        if reasons:
            skip_reason = "; ".join(reasons)
            print(f"    [rolling] window {win_idx}: skip ({skip_reason})", flush=True)
            for alpha in ALPHAS:
                min_n = min_defective_for_nonzero_threshold(alpha)
                for m_name in ("global_conformal", "residual_risk"):
                    rows.append({
                        "project": project,
                        "seed": seed,
                        "window": win_idx,
                        "method": m_name,
                        "guarantee_type": METHOD_ROLES.get(m_name, "baseline"),
                        "selected_calibrator": "skipped",
                        "alpha": alpha,
                        "protocol": "rolling_expanding",
                        "status": "skipped",
                        "calibration_status": "skipped",
                        "skip_reason": skip_reason,
                        "n_def_conf": n_def_conf,
                        "n_def_test": n_def_test,
                        "n_missed_test": 0,
                        "min_def_conf_nonzero": min_n,
                        "sufficient_for_nonzero_threshold": bool(n_def_conf >= min_n),
                        "cmr": float("nan"),
                        "cmr_ci_low": float("nan"),
                        "cmr_ci_high": float("nan"),
                        "cmr_exact_ci_low": float("nan"),
                        "cmr_exact_ci_high": float("nan"),
                        "cmr_upper_95": float("nan"),
                        "pct_tests_selected": float("nan"),
                        "pct_duration_saved": float("nan"),
                        "lambda": float("nan"),
                        "tau": float("nan"),
                        "cmr_minus_alpha": float("nan"),
                    })
            continue

        try:
            model = train_lgbm(train_cand, FEATURE_COLS, seed=seed)
            cal_info = evaluate_and_select_calibrators(
                model,
                prob_cand,
                FEATURE_COLS,
                train_prior=float(train_cand["label"].mean()),
            )
            if calibration_method == "isotonic":
                calibrator = cal_info["iso"]
                sel_cal_name = "isotonic"
            elif calibration_method == "sigmoid":
                calibrator = cal_info["sigmoid"]
                sel_cal_name = "sigmoid"
            else:
                calibrator = cal_info["selected_calibrator"]
                sel_cal_name = cal_info["selected_name"]

            probs_conf, _ = _predict_split_probs(model, calibrator, conf_calib_df, FEATURE_COLS)
            probs_test, _ = _predict_split_probs(model, calibrator, test_df, FEATURE_COLS)

            for alpha in ALPHAS:
                scores = compute_conformal_scores(conf_calib_df, probs_conf)
                lambda_ = compute_global_threshold(scores, alpha)
                tau = compute_tau_residual(conf_calib_df, probs_conf, alpha)
                min_n = min_defective_for_nonzero_threshold(alpha)
                suff = bool(n_def_conf >= min_n)
                cal_status = "ok" if suff else "insufficient_for_nonzero_threshold"

                for m_name, sel_df in [
                    ("global_conformal", select_tests_global(test_df, probs_test, lambda_)),
                    ("residual_risk", select_tests_residual_all_commits(test_df, probs_test, tau)),
                ]:
                    mask = test_df.index.isin(sel_df.index)
                    indicators = _commit_miss_indicators(test_df, mask)
                    n_def_w = int(len(indicators))
                    n_missed_w = int(indicators.sum())
                    cmr_val, ci_low, ci_high = compute_wilson_ci_counts(n_missed_w, n_def_w)
                    _, exact_lo, exact_hi, upper_95 = compute_exact_binomial_ci_counts(n_missed_w, n_def_w)
                    budget = compute_budget_saved(test_df, mask)

                    rows.append({
                        "project": project,
                        "seed": seed,
                        "window": win_idx,
                        "method": m_name,
                        "guarantee_type": METHOD_ROLES.get(m_name, "baseline"),
                        "selected_calibrator": sel_cal_name,
                        "alpha": alpha,
                        "protocol": "rolling_expanding",
                        "status": "ok",
                        "calibration_status": cal_status,
                        "skip_reason": "",
                        "n_def_conf": n_def_conf,
                        "n_def_test": n_def_w,
                        "n_missed_test": n_missed_w,
                        "min_def_conf_nonzero": min_n,
                        "sufficient_for_nonzero_threshold": suff,
                        "cmr": cmr_val,
                        "cmr_ci_low": ci_low,
                        "cmr_ci_high": ci_high,
                        "cmr_exact_ci_low": exact_lo,
                        "cmr_exact_ci_high": exact_hi,
                        "cmr_upper_95": upper_95,
                        "pct_tests_selected": budget["pct_tests_selected"],
                        "pct_duration_saved": budget["pct_duration_saved"],
                        "lambda": lambda_,
                        "tau": tau,
                        "cmr_minus_alpha": cmr_val - alpha,
                    })
        except Exception as e:
            print(f"    [rolling] window {win_idx} FAILED: {e}", flush=True)
            for alpha in ALPHAS:
                min_n = min_defective_for_nonzero_threshold(alpha)
                for m_name in ("global_conformal", "residual_risk"):
                    rows.append({
                        "project": project,
                        "seed": seed,
                        "window": win_idx,
                        "method": m_name,
                        "guarantee_type": METHOD_ROLES.get(m_name, "baseline"),
                        "selected_calibrator": "failed",
                        "alpha": alpha,
                        "protocol": "rolling_expanding",
                        "status": "failed",
                        "calibration_status": "failed",
                        "skip_reason": str(e),
                        "n_def_conf": n_def_conf,
                        "n_def_test": n_def_test,
                        "n_missed_test": 0,
                        "min_def_conf_nonzero": min_n,
                        "sufficient_for_nonzero_threshold": bool(n_def_conf >= min_n),
                        "cmr": float("nan"),
                        "cmr_ci_low": float("nan"),
                        "cmr_ci_high": float("nan"),
                        "cmr_exact_ci_low": float("nan"),
                        "cmr_exact_ci_high": float("nan"),
                        "cmr_upper_95": float("nan"),
                        "pct_tests_selected": float("nan"),
                        "pct_duration_saved": float("nan"),
                        "lambda": float("nan"),
                        "tau": float("nan"),
                        "cmr_minus_alpha": float("nan"),
                    })

    if rows:
        for alpha in ALPHAS:
            min_n = min_defective_for_nonzero_threshold(alpha)
            win_flags = {
                r["window"]: r["sufficient_for_nonzero_threshold"]
                for r in rows
                if abs(r["alpha"] - alpha) < 1e-12
            }
            n_tot_w = len(win_flags)
            n_insuff_w = sum(1 for ok in win_flags.values() if not ok)
            print(
                f"    [rolling-summary] {project} (seed={seed}) alpha={alpha:.2f}: "
                f"sufficient_for_nonzero_threshold==False in {n_insuff_w}/{n_tot_w} windows "
                f"(min_def_conf={min_n})",
                flush=True,
            )
    return rows


def aggregate_rolling_results(rolling_df: pd.DataFrame) -> pd.DataFrame:
    """Compute pooled commit-weighted CMR (``sum(M_w) / sum(N_w)``) over
    non-overlapping executed rolling test windows, with exact Clopper-Pearson CIs.
    """
    if len(rolling_df) == 0:
        return pd.DataFrame()

    ok_df = rolling_df[rolling_df["status"] == "ok"].copy()
    if len(ok_df) == 0:
        return pd.DataFrame()

    seed_rows: list[dict] = []
    for (proj, seed, method, alpha), grp in ok_df.groupby(
        ["project", "seed", "method", "alpha"], sort=True
    ):
        n_def_sum = int(grp["n_def_test"].sum())
        n_miss_sum = int(grp["n_missed_test"].sum())
        pooled_cmr, ex_lo, ex_hi, up95 = compute_exact_binomial_ci_counts(n_miss_sum, n_def_sum)
        _, w_lo, w_hi = compute_wilson_ci_counts(n_miss_sum, n_def_sum)
        n_insuff = int((~grp["sufficient_for_nonzero_threshold"].astype(bool)).sum())
        seed_rows.append({
            "project": proj,
            "seed": seed,
            "method": method,
            "alpha": alpha,
            "n_windows_ok": int(len(grp)),
            "n_windows_insufficient_threshold": n_insuff,
            "pooled_n_def_test": n_def_sum,
            "pooled_n_missed_test": n_miss_sum,
            "pooled_cmr": pooled_cmr,
            "pooled_cmr_exact_ci_low": ex_lo,
            "pooled_cmr_exact_ci_high": ex_hi,
            "pooled_cmr_upper_95": up95,
            "pooled_cmr_wilson_low": w_lo,
            "pooled_cmr_wilson_high": w_hi,
            "mean_window_cmr": float(grp["cmr"].mean()),
            "pct_tests_selected_mean": float(grp["pct_tests_selected"].mean()),
            "pct_duration_saved_mean": float(grp["pct_duration_saved"].mean()),
        })

    df_seed = pd.DataFrame(seed_rows)
    agg_cols = [
        "n_windows_ok",
        "n_windows_insufficient_threshold",
        "pooled_n_def_test",
        "pooled_n_missed_test",
        "pooled_cmr",
        "pooled_cmr_exact_ci_low",
        "pooled_cmr_exact_ci_high",
        "pooled_cmr_upper_95",
        "pooled_cmr_wilson_low",
        "pooled_cmr_wilson_high",
        "mean_window_cmr",
        "pct_tests_selected_mean",
        "pct_duration_saved_mean",
    ]
    out = df_seed.groupby(["project", "method", "alpha"], as_index=False)[agg_cols].mean()
    out["protocol"] = "rolling_expanding_pooled"
    return out


def _get_git_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_root_dir),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def write_run_manifest(
    run_dir: Path,
    run_id: str,
    target_projects: List[str],
    num_seeds: int,
    policy: str,
    include_fallback_parents: bool,
    rolling: bool,
    rolling_splits: int,
    strict_min_defective: int,
    completed_projects: List[str],
    calibration_method: str = "auto",
) -> Path:
    """Write reproducibility manifest JSON into run_dir."""
    cfg = {
        "projects": list(target_projects),
        "alphas": list(ALPHAS),
        "seeds": int(num_seeds),
        "policy": policy,
        "calibration_method": calibration_method,
        "include_fallback_parents": bool(include_fallback_parents),
        "rolling": bool(rolling),
        "rolling_splits": int(rolling_splits) if rolling else 0,
        "strict_min_defective": int(strict_min_defective),
        "holdout_split": {
            "train": TRAIN_FRAC,
            "prob_calib": PROB_CALIB_FRAC,
            "conf_calib": CONF_CALIB_FRAC,
            "test": round(1.0 - TRAIN_FRAC - PROB_CALIB_FRAC - CONF_CALIB_FRAC, 4),
        },
        "rolling_split": {
            "min_train_frac": 0.25,
            "prob_calib": PROB_CALIB_FRAC,
            "conf_calib": CONF_CALIB_FRAC,
            "test": 0.15,
        },
    }
    cfg_hash = hashlib.sha256(
        json.dumps(cfg, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]

    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _get_git_commit_short(),
        "config_hash": cfg_hash,
        "dataset_root": str(RTP_ROOT),
        "completed_projects": list(completed_projects),
        "config": cfg,
    }
    out_path = run_dir / "run_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_path


def make_rolling_figure(rolling_df: pd.DataFrame, fig_dir: Path) -> None:
    """Plot CMR dynamics across executed rolling windows."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plot_df = rolling_df[rolling_df["status"] == "ok"] if "status" in rolling_df.columns else rolling_df
    if len(plot_df) == 0:
        print("  No executed rolling windows to plot — skipping fig5_rolling_cmr", flush=True)
        return

    projects = plot_df["project"].unique()
    alphas_plot = sorted(plot_df["alpha"].unique())
    colors_a = {0.05: "#1f77b4", 0.10: "#ff7f0e", 0.15: "#2ca02c", 0.20: "#d62728"}
    methods_plot = ["global_conformal", "residual_risk"] if "method" in plot_df.columns else [None]
    linestyles = {"global_conformal": "-", "residual_risk": "--"}

    fig, axes = plt.subplots(1, len(projects), figsize=(5 * len(projects), 4), sharey=False)
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = axes.flatten()

    for ax, proj in zip(axes, projects):
        sub = plot_df[plot_df["project"] == proj]
        for m in methods_plot:
            msub = sub[sub["method"] == m] if m else sub
            for alpha in alphas_plot:
                asub = msub[msub["alpha"] == alpha]
                if len(asub) == 0:
                    continue
                stats = asub.groupby("window")["cmr"].agg(["mean", "std"]).reset_index()
                stats["std"] = stats["std"].fillna(0.0)
                lbl = f"a={alpha:.2f} ({m})" if m else f"a={alpha:.2f}"
                ax.plot(
                    stats["window"],
                    stats["mean"],
                    marker="o",
                    linestyle=linestyles.get(m, "-"),
                    color=colors_a.get(alpha, "grey"),
                    label=lbl,
                )
                ax.fill_between(
                    stats["window"],
                    (stats["mean"] - stats["std"]).clip(0),
                    (stats["mean"] + stats["std"]).clip(0, 1),
                    alpha=0.10,
                    color=colors_a.get(alpha, "grey"),
                )
        for alpha in alphas_plot:
            ax.axhline(alpha, color=colors_a.get(alpha, "grey"), linestyle=":", linewidth=0.8)
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
        "retest_all", "qtf", "latest_fail_cc",
        "global_conformal", "global_conformal_cost_aware", "global_conformal_qtf",
        "residual_risk", "residual_risk_cost_aware",
        "naive_residual", "fixed_0.5",
        "top_10pct", "top_30pct", "top_50pct",
        "random_matched", "recently_failed",
    ]
    colors = {
        "retest_all":                  "#000000",
        "qtf":                         "#7f7f7f",
        "latest_fail_cc":              "#e377c2",
        "global_conformal":            "#1f77b4",
        "global_conformal_cost_aware": "#005580",
        "global_conformal_qtf":        "#4c9ed9",
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
        "qtf":                         "QTF (full suite)",
        "latest_fail_cc":              "LatestFail+CC (full suite)",
        "global_conformal":            "Global Conformal",
        "global_conformal_cost_aware": "Global Conformal (cost-aware)",
        "global_conformal_qtf":        "Global Conformal (QTF)",
        "residual_risk":               "Residual Risk (heuristic)",
        "residual_risk_cost_aware":    "Residual Risk (cost-aware)",
        "naive_residual":              "Naive Residual (tau=alpha)",
        "fixed_0.5":                   "Fixed tau=0.5",
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
    cmr_methods = [
        "global_conformal", "residual_risk", "naive_residual",
        "fixed_0.5", "random_matched", "recently_failed",
    ]
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
            ax.plot(
                stats["alpha"], stats["mean"], marker="o",
                color=colors.get(method, "grey"),
                label=labels.get(method, method),
            )
            ax.fill_between(
                stats["alpha"],
                (stats["mean"] - stats["std"]).clip(0),
                (stats["mean"] + stats["std"]).clip(0, 1),
                alpha=0.15, color=colors.get(method, "grey"),
            )
        ax.plot([0.05, 0.20], [0.05, 0.20], "k--", linewidth=1, label="CMR = alpha")
        ax.set_title(proj.split("@")[-1], fontsize=10)
        ax.set_xlabel("Target alpha")
        ax.set_ylabel("Empirical CMR")
        ax.set_xlim(0.03, 0.22)
        ax.set_ylim(-0.02, 1.0)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    fig.suptitle("CMR vs alpha: Conformal Risk Evaluation", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig1_cmr_vs_alpha.pdf", bbox_inches="tight")
    plt.savefig(fig_dir / "fig1_cmr_vs_alpha.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig1_cmr_vs_alpha", flush=True)

    # --- Figures 2 & 6: Budget (% selected) and Duration Saved (%) vs alpha ---
    budget_methods = [
        "global_conformal", "global_conformal_cost_aware",
        "residual_risk", "naive_residual", "fixed_0.5", "random_matched",
    ]
    for metric_col, ylabel, sup_title, stem in [
        (
            "pct_tests_selected",
            "Tests Selected (%)",
            "Budget: Fraction of Tests Selected vs alpha",
            "fig2_budget_vs_alpha",
        ),
        (
            "pct_duration_saved",
            "CI Duration Saved (%)",
            "Resource Savings: Fraction of CI Time Saved vs alpha",
            "fig6_duration_saved_vs_alpha",
        ),
    ]:
        if metric_col not in raw_df.columns:
            continue
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
                stats = msub.groupby("alpha")[metric_col].agg(["mean", "std"]).reset_index()
                stats["std"] = stats["std"].fillna(0.0)
                ax.plot(
                    stats["alpha"], stats["mean"] * 100, marker="s",
                    color=colors.get(method, "grey"),
                    label=labels.get(method, method),
                )
            ax.set_title(proj.split("@")[-1], fontsize=10)
            ax.set_xlabel("Target alpha")
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, 105)
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)

        fig.suptitle(sup_title, fontsize=12)
        plt.tight_layout()
        plt.savefig(fig_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.savefig(fig_dir / f"{stem}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {stem}", flush=True)

    # --- Figure 3: Conditional APFDc bar chart (alpha=0.10) ---
    sub10 = raw_df[raw_df["alpha"] == 0.10]
    fig, axes = plt.subplots(1, n_proj, figsize=(5 * n_proj, 4), sharey=True)
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = axes.flatten()

    apfdc_col = "conditional_apfdc" if "conditional_apfdc" in sub10.columns else "apfdc"
    bar_methods = [m for m in methods_order if m in sub10["method"].values]
    for ax, proj in zip(axes, projects):
        sub = sub10[sub10["project"] == proj]
        means, stds, xlabels, bar_colors = [], [], [], []
        for method in bar_methods:
            msub = sub[sub["method"] == method]
            if len(msub) == 0:
                continue
            means.append(msub[apfdc_col].mean())
            s = msub[apfdc_col].std(ddof=1)
            stds.append(0.0 if pd.isna(s) else s)
            xlabels.append(labels.get(method, method))
            bar_colors.append(colors.get(method, "grey"))
        x = np.arange(len(xlabels))
        ax.bar(x, means, yerr=stds, capsize=4, color=bar_colors)
        ax.set_title(proj.split("@")[-1], fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=35, ha="right", fontsize=6)
        ax.set_ylabel("Conditional APFDc (selected suite)")
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Conditional APFDc Comparison on Detected Commits (alpha=0.10)", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig3_apfdc_alpha10.pdf", bbox_inches="tight")
    plt.savefig(fig_dir / "fig3_apfdc_alpha10.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig3_apfdc_alpha10", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predictive Test Selection Experiment v2.1")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run fast smoke test (1 project, 1 seed)",
    )
    parser.add_argument(
        "--project", type=str, default=None,
        help="Single project identifier, e.g. apache@sling",
    )
    parser.add_argument(
        "--seeds", type=int, default=None,
        help="Number of random seeds",
    )
    parser.add_argument(
        "--rolling", action="store_true",
        help="Run rolling window evaluation (saves results_rolling.csv and fig5)",
    )
    parser.add_argument(
        "--rolling-splits", type=int, default=3,
        help="Number of rolling window splits (default: 3)",
    )
    parser.add_argument(
        "--policy", type=str, default="confirmed_new_fail",
        choices=["confirmed_new_fail", "strict_all_fail", "any_fail"],
        help="Label policy for ground-truth assignment (default: confirmed_new_fail)",
    )
    parser.add_argument(
        "--calibration", type=str, default="auto",
        choices=["auto", "isotonic", "sigmoid"],
        help="Probability calibration mode (default: auto selects on ProbCalib CV)",
    )
    parser.add_argument(
        "--include-fallback-parents", action="store_true",
        help="Include chronological_fallback parents (sensitivity analysis; default uses git_parent only)",
    )
    parser.add_argument(
        "--strict-min-defective", type=int, default=0,
        help="Require at least N defective commits in ConfCalib + Test combined (e.g. 150)",
    )
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
    run_id = f"{timestamp}_{proj_tag}_seeds{num_seeds}"
    run_dir = OUT_DIR / "runs" / run_id
    run_fig_dir = run_dir / "figures"
    run_fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print("Predictive Test Selection - v2.1 Experiment", flush=True)
    print(f"Run ID          : {run_id}", flush=True)
    print(f"Output          : {run_dir}", flush=True)
    print(f"Projects        : {target_projects}", flush=True)
    print(f"Alphas          : {ALPHAS}", flush=True)
    print(f"Seeds           : {num_seeds}", flush=True)
    print(f"Policy          : {args.policy}", flush=True)
    print(f"Calibration     : {args.calibration}", flush=True)
    print(f"Parent filter   : {'git_parent + fallback' if args.include_fallback_parents else 'git_parent ONLY'}", flush=True)
    print(f"Rolling         : {args.rolling} (splits={args.rolling_splits})", flush=True)
    print(
        f"Split           : {TRAIN_FRAC}/{PROB_CALIB_FRAC}/{CONF_CALIB_FRAC}/"
        f"{1-TRAIN_FRAC-PROB_CALIB_FRAC-CONF_CALIB_FRAC:.2f} "
        f"(train/prob_calib/conf_calib/test)",
        flush=True,
    )
    if args.strict_min_defective > 0:
        print(f"Strict Conf+Test: >= {args.strict_min_defective} defective commits", flush=True)
    print("=" * 60, flush=True)

    project_cache: Dict[str, Tuple[CIHistorySimulator, pd.DataFrame]] = {}
    all_results = run_all(
        projects=target_projects,
        seeds=num_seeds,
        out_dir=run_dir,
        policy=args.policy,
        include_fallback_parents=args.include_fallback_parents,
        cache_out=project_cache,
        strict_min_defective=args.strict_min_defective,
        calibration_method=args.calibration,
    )

    if not all_results:
        print("ERROR: No results collected.", flush=True)
        sys.exit(1)

    raw_df = pd.DataFrame(all_results)
    raw_df["run_id"] = run_id
    raw_df.to_csv(run_dir / "results_raw.csv", index=False)
    print(f"\nRaw results saved ({len(raw_df)} rows) -> {run_dir / 'results_raw.csv'}", flush=True)

    table = aggregate_results(all_results)
    table["run_id"] = run_id
    table.to_csv(run_dir / "results_table.csv", index=False)
    print(f"Summary table saved -> {run_dir / 'results_table.csv'}", flush=True)

    cal_summary = build_calibration_summary(raw_df)
    if len(cal_summary) > 0:
        cal_summary["run_id"] = run_id
        cal_summary.to_csv(run_dir / "calibration_summary.csv", index=False)
        print(f"Calibration summary saved -> {run_dir / 'calibration_summary.csv'}", flush=True)

    savings_summary = build_savings_summary(raw_df)
    if len(savings_summary) > 0:
        savings_summary["run_id"] = run_id
        savings_summary.to_csv(run_dir / "savings_summary.csv", index=False)
        print(f"Savings summary saved -> {run_dir / 'savings_summary.csv'}", flush=True)

    print("\n--- Results Preview (global_conformal) ---", flush=True)
    preview_cols = [
        "project", "alpha", "cmr_mean", "cmr_exact_ci_high_mean", "cmr_upper_95_mean",
        "pct_tests_selected_mean", "pct_duration_saved_mean", "apfdc_mean",
    ]
    available_cols = [c for c in preview_cols if c in table.columns]
    preview = table[table["method"] == "global_conformal"][available_cols]
    print(preview.to_string(index=False), flush=True)

    if len(cal_summary) > 0:
        cal_cols = [
            c for c in [
                "project", "selected_calibrator_mode", "pr_auc_test", "pr_auc_baseline_test",
                "brier_raw_test", "brier_iso_test", "brier_sigmoid_test", "brier_baseline_test",
                "bss_iso_test", "bss_sigmoid_test",
            ]
            if c in cal_summary.columns
        ]
        print("\n--- Calibration Summary (Candidates Only) ---", flush=True)
        print(cal_summary[cal_cols].to_string(index=False), flush=True)

    print("\nGenerating figures...", flush=True)
    make_figures(raw_df, fig_dir=run_fig_dir)

    # --- Rolling window evaluation (reuses cached filtered project data) ---
    if args.rolling:
        print("\n>>> ROLLING WINDOW EVALUATION <<<", flush=True)
        all_rolling: list[dict] = []
        for project in target_projects:
            try:
                if project in project_cache:
                    sim, df_proj = project_cache[project]
                else:
                    sim = CIHistorySimulator(RTP_ROOT, project)
                    df_raw = build_project_data(sim, policy=args.policy)
                    df_proj, _ = filter_valid_pairs(
                        df_raw,
                        include_fallback_parents=args.include_fallback_parents,
                        policy=args.policy,
                        verbose=True,
                    )
                if len(df_proj) == 0:
                    continue
                for seed in range(num_seeds):
                    rrows = run_rolling(
                        sim,
                        df_proj,
                        seed=seed,
                        n_splits=args.rolling_splits,
                        out_dir=run_dir,
                        calibration_method=args.calibration,
                    )
                    all_rolling.extend(rrows)
            except Exception as e:
                print(f"  Rolling FAILED for {project}: {e}", flush=True)
                traceback.print_exc()

        if all_rolling:
            rolling_df = pd.DataFrame(all_rolling)
            rolling_df["run_id"] = run_id
            rolling_df.to_csv(run_dir / "results_rolling.csv", index=False)
            print(f"Rolling results saved -> {run_dir / 'results_rolling.csv'}", flush=True)

            pooled_df = aggregate_rolling_results(rolling_df)
            if len(pooled_df) > 0:
                pooled_df["run_id"] = run_id
                pooled_df.to_csv(run_dir / "results_rolling_pooled.csv", index=False)
                print(f"Pooled rolling summary saved -> {run_dir / 'results_rolling_pooled.csv'}", flush=True)
                print("\n--- Pooled Rolling Summary ---", flush=True)
                print(
                    pooled_df[[
                        "project", "method", "alpha", "n_windows_ok",
                        "pooled_n_def_test", "pooled_n_missed_test",
                        "pooled_cmr", "pooled_cmr_exact_ci_high", "pooled_cmr_upper_95",
                    ]].to_string(index=False),
                    flush=True,
                )

            make_rolling_figure(rolling_df, fig_dir=run_fig_dir)
        else:
            print("No rolling results collected.", flush=True)

    manifest_path = write_run_manifest(
        run_dir=run_dir,
        run_id=run_id,
        target_projects=target_projects,
        num_seeds=num_seeds,
        policy=args.policy,
        include_fallback_parents=args.include_fallback_parents,
        rolling=args.rolling,
        rolling_splits=args.rolling_splits,
        strict_min_defective=args.strict_min_defective,
        completed_projects=sorted(raw_df["project"].unique().tolist()),
        calibration_method=args.calibration,
    )
    print(f"Run manifest saved -> {manifest_path}", flush=True)
    print(f"\nDONE. All outputs in: {run_dir}", flush=True)
