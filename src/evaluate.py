"""
evaluate.py — Metrics & Baseline Comparisons
=============================================
Implements:
  - CMR (Change Miss Rate)
  - Budget saved (% tests, % duration)
  - TTFF (Time-To-First-Failure, normalized)
  - APFDc (Average Percentage of Faults Detected with cost)
  - Baseline selectors: fixed_threshold, random, recently_failed
  - compare_all_methods() aggregator
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


# ---------------------------------------------------------------------------
# Core Metrics
# ---------------------------------------------------------------------------

def compute_cmr(
    df: pd.DataFrame,
    selected_mask: np.ndarray,
) -> float:
    """
    Change Miss Rate = fraction of defective commits where NO selected test
    has label == 1.

    Parameters
    ----------
    df            : test-set DataFrame with columns ['commit_idx', 'label']
    selected_mask : boolean array aligned with df rows (True = selected)

    Returns
    -------
    cmr : float in [0, 1]
    """
    df = df.copy()
    df["_sel"] = selected_mask

    defective_commits = df.groupby("commit_idx")["label"].transform("max")
    defective_df = df[defective_commits == 1]

    if len(defective_df["commit_idx"].unique()) == 0:
        return 0.0

    misses = 0
    for cid, grp in defective_df.groupby("commit_idx"):
        selected_and_failed = grp[grp["_sel"] & (grp["label"] == 1)]
        if len(selected_and_failed) == 0:
            misses += 1

    return misses / len(defective_df["commit_idx"].unique())


def compute_budget_saved(
    df: pd.DataFrame,
    selected_mask: np.ndarray,
    duration_col: str = "duration_mean",
) -> Dict[str, float]:
    """
    Returns:
      pct_tests_selected   : fraction of test-commit pairs selected
      pct_duration_saved   : fraction of total expected duration NOT run
      pct_duration_selected: fraction of total expected duration that IS run
    """
    n_total = len(df)
    n_selected = int(selected_mask.sum())

    pct_tests_selected = n_selected / n_total if n_total > 0 else 0.0

    if duration_col in df.columns:
        dur_vals = df[duration_col].fillna(0).values
        total_dur = dur_vals.sum()
        sel_dur = dur_vals[selected_mask].sum()
        pct_dur_sel = sel_dur / total_dur if total_dur > 0 else 0.0
        pct_dur_saved = 1.0 - pct_dur_sel
    else:
        pct_dur_sel = pct_tests_selected
        pct_dur_saved = 1.0 - pct_tests_selected

    return {
        "pct_tests_selected": pct_tests_selected,
        "pct_duration_selected": pct_dur_sel,
        "pct_duration_saved": pct_dur_saved,
    }


def compute_ttff(
    df: pd.DataFrame,
    order_col: str = "_rank",
) -> float:
    """
    Time-To-First-Failure (TTFF) — per commit, normalized position of the
    first label==1 test in the ordered execution list.

    For each defective commit where at least one failure is selected:
        ttff_c = rank_of_first_failure / total_selected_for_commit
    Returns mean over all such commits. If no commit catches a failure, returns 1.0.

    Parameters
    ----------
    df       : DataFrame with ['commit_idx', 'label', order_col]
               Only selected tests should be in this DataFrame.
    order_col: column giving execution order (1-based rank within commit)
    """
    scores = []
    for cid, grp in df.groupby("commit_idx"):
        failures = grp[grp["label"] == 1]
        if len(failures) == 0:
            continue
        total = len(grp)
        first_rank = failures[order_col].min()
        scores.append(first_rank / total if total > 0 else 1.0)

    return float(np.mean(scores)) if scores else 1.0


def compute_apfdc(
    df: pd.DataFrame,
    order_col: str = "_rank",
    duration_col: str = "duration_mean",
) -> float:
    """
    APFDc — duration-weighted Average Percentage of Faults Detected.

    For each defective commit, computes the area-under-detection-curve
    weighted by test execution cost (duration).

    APFDc_c = 1 - (sum_{failing test i} cost_i * rank_i) / (TF * TC) + 1/(2*TF)
    where TF = number of failing tests, TC = total cost of selected tests.

    Returns mean APFDc over all defective commits where at least one failure selected.
    If none, returns 0.0.
    """
    scores = []
    for cid, grp in df.groupby("commit_idx"):
        failures = grp[grp["label"] == 1]
        if len(failures) == 0:
            continue

        # cost per test (use 1 if duration missing)
        if duration_col in grp.columns:
            costs = grp[duration_col].fillna(1.0).clip(lower=1e-3).values
        else:
            costs = np.ones(len(grp))

        total_cost = costs.sum()
        tf = len(failures)
        if total_cost == 0 or tf == 0:
            continue

        ranks = grp[order_col].values
        fail_mask = (grp["label"] == 1).values

        # sum of cost_i * rank_i for failing tests
        weighted_sum = (costs[fail_mask] * ranks[fail_mask]).sum()

        apfdc_c = 1.0 - weighted_sum / (tf * total_cost) + 1.0 / (2.0 * tf)
        scores.append(np.clip(apfdc_c, 0.0, 1.0))

    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Baseline Selectors
# ---------------------------------------------------------------------------

def _add_rank_within_commit(df: pd.DataFrame, sort_col: str, ascending: bool = False) -> pd.DataFrame:
    """Add '_rank' column = rank within each commit group sorted by sort_col."""
    df = df.copy()
    df["_rank"] = df.groupby("commit_idx")[sort_col].rank(
        method="first", ascending=ascending
    ).astype(int)
    return df


def baseline_fixed_threshold(
    df: pd.DataFrame,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Select tests with p̂ >= threshold (fixed, no conformal calibration)."""
    return probs >= threshold


def baseline_random(
    df: pd.DataFrame,
    pct: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Randomly select pct fraction of tests per commit.
    pct is chosen to match a typical budget (e.g. 50%).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    mask = pd.Series(False, index=df.index)
    for cid, grp in df.groupby("commit_idx"):
        idx = grp.index.tolist()
        k = max(1, int(len(idx) * pct))
        chosen = rng.choice(idx, size=k, replace=False)
        mask.loc[chosen] = True
    return mask.values


def baseline_recently_failed(
    df: pd.DataFrame,
    n_recent: int = 5,
) -> np.ndarray:
    """
    Select tests that failed in any of the last n_recent commits
    (using 'last_fail_dist' feature: 1 means failed on previous commit).

    Falls back to selecting top-50% by fail_rate_N if column missing.
    """
    mask = np.zeros(len(df), dtype=bool)
    if "last_fail_dist" in df.columns:
        # select tests that failed within the last n_recent commits
        mask = (df["last_fail_dist"] <= n_recent) & (df["last_fail_dist"] > 0)
        mask = mask.values
    elif "fail_rate_N" in df.columns:
        threshold = df["fail_rate_N"].median()
        mask = (df["fail_rate_N"] >= threshold).values
    return mask


# ---------------------------------------------------------------------------
# Unified evaluator
# ---------------------------------------------------------------------------

def evaluate_selection(
    df: pd.DataFrame,
    selected_mask: np.ndarray,
    method_name: str,
    alpha: float,
    duration_col: str = "duration_mean",
    prob_col: str = "_prob",
) -> Dict[str, Any]:
    """
    Given a boolean selection mask, compute all metrics for one method.

    Returns a dict with all scalar metrics.
    """
    # Build execution order (by prob desc within commit, for selected rows)
    df_sel = df.copy()
    df_sel["_sel"] = selected_mask
    df_sel["_prob"] = df_sel.get(prob_col, 0.0) if prob_col in df_sel.columns else 0.0

    # Assign rank within each commit for selected tests (by prob desc)
    df_selected = df_sel[df_sel["_sel"]].copy()
    if len(df_selected) > 0 and "_prob" in df_selected.columns:
        df_selected["_rank"] = df_selected.groupby("commit_idx")["_prob"].rank(
            method="first", ascending=False
        ).astype(int)
    else:
        df_selected["_rank"] = 1

    cmr = compute_cmr(df_sel, selected_mask)
    budget = compute_budget_saved(df_sel, selected_mask, duration_col)
    ttff = compute_ttff(df_selected) if len(df_selected) > 0 else 1.0
    apfdc = compute_apfdc(df_selected, duration_col=duration_col) if len(df_selected) > 0 else 0.0

    return {
        "method": method_name,
        "alpha": alpha,
        "cmr": cmr,
        "pct_tests_selected": budget["pct_tests_selected"],
        "pct_duration_selected": budget["pct_duration_selected"],
        "pct_duration_saved": budget["pct_duration_saved"],
        "ttff": ttff,
        "apfdc": apfdc,
        "n_defective_commits": int(
            df[df["label"] == 1].groupby("commit_idx").ngroups
        ),
        "n_total_pairs": len(df),
        "n_selected": int(selected_mask.sum()),
    }


def compare_all_methods(
    df_test: pd.DataFrame,
    probs_test: np.ndarray,
    threshold_global: float,
    tau_residual: float,
    alpha: float,
    rng: Optional[np.random.Generator] = None,
    duration_col: str = "duration_mean",
) -> List[Dict[str, Any]]:
    """
    Evaluate all methods on the test set for a given alpha.

    Parameters
    ----------
    df_test          : test-set DataFrame (commit_idx, label, features...)
    probs_test       : calibrated probabilities for test rows
    threshold_global : lambda from compute_global_threshold
    tau_residual     : tau from compute_tau_residual
    alpha            : risk level
    rng              : random generator for baseline_random

    Returns
    -------
    list of metric dicts, one per method
    """
    from conformal import select_tests_global, select_tests_residual_all_commits

    results = []

    # --- Global conformal ---
    sel_global = select_tests_global(df_test, probs_test, threshold_global)
    mask_global = df_test.index.isin(sel_global.index)
    df_test_with_prob = df_test.copy()
    df_test_with_prob["_prob"] = probs_test
    results.append(
        evaluate_selection(df_test_with_prob, mask_global, "global_conformal", alpha, duration_col)
    )

    # --- Residual risk ---
    sel_residual = select_tests_residual_all_commits(df_test, probs_test, tau_residual)
    mask_residual = df_test.index.isin(sel_residual.index)
    results.append(
        evaluate_selection(df_test_with_prob, mask_residual, "residual_risk", alpha, duration_col)
    )

    # --- Fixed threshold (0.5) ---
    mask_fixed = baseline_fixed_threshold(df_test, probs_test, threshold=0.5)
    results.append(
        evaluate_selection(df_test_with_prob, mask_fixed, "fixed_0.5", alpha, duration_col)
    )

    # --- Random (match global selection rate) ---
    pct_sel = mask_global.mean() if mask_global.mean() > 0 else 0.5
    mask_random = baseline_random(df_test, pct=float(pct_sel), rng=rng)
    results.append(
        evaluate_selection(df_test_with_prob, mask_random, "random", alpha, duration_col)
    )

    # --- Recently failed ---
    mask_recent = baseline_recently_failed(df_test)
    results.append(
        evaluate_selection(df_test_with_prob, mask_recent, "recently_failed", alpha, duration_col)
    )

    return results
