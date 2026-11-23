"""
evaluate.py — Metrics & Baseline Comparisons (v2)
==================================================
Implements:
  - CMR  (Change Miss Rate)
  - Budget saved (% tests, % duration)
  - TTFF (cumulative actual duration to first failure)
  - APFDc (standard cost-cognizant formula, no clip)
  - Baselines: retest-all, top-k%, fixed threshold, random matched budget,
               recently failed, naive residual risk, threshold by target recall
  - Cost-aware prioritization comparison (probability order vs p/duration order)
  - compare_all_methods() aggregator

NOTE: TTFF and APFDc are *conditional* metrics — computed only on
detected defective commits.  They **must** always be reported
alongside CMR (the fraction of defective commits missed entirely).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Core Metrics
# ---------------------------------------------------------------------------

def compute_cmr(
    df: pd.DataFrame,
    selected_mask: np.ndarray,
) -> float:
    """Change Miss Rate = fraction of defective commits where NO selected
    test has ``label == 1``.
    """
    df = df.copy()
    df["_sel"] = selected_mask

    defective_commits = df.groupby("commit_idx")["label"].transform("max")
    defective_df = df[defective_commits == 1]

    unique_defective = defective_df["commit_idx"].unique()
    if len(unique_defective) == 0:
        return 0.0

    misses = 0
    for cid in unique_defective:
        grp = defective_df[defective_df["commit_idx"] == cid]
        selected_and_failed = grp[grp["_sel"] & (grp["label"] == 1)]
        if len(selected_and_failed) == 0:
            misses += 1

    return misses / len(unique_defective)


def compute_cmr_wilson_ci(
    df: pd.DataFrame,
    selected_mask: np.ndarray,
    confidence: float = 0.95,
) -> tuple:
    """Wilson score 95% CI for CMR (miss rate proportion).

    Returns (cmr, ci_low, ci_high) tuple.

    Uses the Wilson score interval which is valid even for extreme
    proportions (cmr ≈ 0 or 1) and small sample sizes.

    Parameters
    ----------
    df : DataFrame with 'commit_idx' and 'label' columns
    selected_mask : boolean mask of selected tests
    confidence : confidence level (default 0.95)

    Returns
    -------
    (cmr, ci_low, ci_high) : all in [0, 1]
    """
    df = df.copy()
    df["_sel"] = selected_mask

    defective_commits = df.groupby("commit_idx")["label"].transform("max")
    defective_df = df[defective_commits == 1]
    unique_defective = defective_df["commit_idx"].unique()
    n = len(unique_defective)

    if n == 0:
        return (0.0, 0.0, 0.0)

    misses = 0
    for cid in unique_defective:
        grp = defective_df[defective_df["commit_idx"] == cid]
        if not (grp["_sel"] & (grp["label"] == 1)).any():
            misses += 1

    p_hat = misses / n

    # Wilson score interval
    from scipy.stats import norm as _norm
    z = _norm.ppf((1 + confidence) / 2)
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p_hat + z2 / (2 * n)) / denom
    spread = z * np.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n)) / denom
    ci_low = float(np.clip(centre - spread, 0.0, 1.0))
    ci_high = float(np.clip(centre + spread, 0.0, 1.0))

    return (float(p_hat), ci_low, ci_high)


def compute_budget_saved(
    df: pd.DataFrame,
    selected_mask: np.ndarray,
    duration_col: str = "actual_duration",
) -> Dict[str, float]:
    """Budget usage: fraction of tests selected and duration saved."""
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
    duration_col: str = "actual_duration",
) -> float:
    """Time-To-First-Failure — cumulative actual duration to first failure.

    For each defective commit where ≥ 1 failure is selected:
      * Tests are executed in ``_rank`` order.
      * ``TTFF_c = Σ actual_duration`` for ranks 1 … r where *r* is the
        rank of the **first** test with ``label == 1``.

    Returns mean TTFF (seconds) over all such commits, or ``NaN`` if
    no failures were detected.

    **Conditional metric** — always report alongside CMR.
    """
    scores: list[float] = []
    for cid, grp in df.groupby("commit_idx"):
        failures = grp[grp["label"] == 1]
        if len(failures) == 0:
            continue
        first_fail_rank = failures["_rank"].min()
        tests_up_to = grp[grp["_rank"] <= first_fail_rank]
        ttff_c = tests_up_to[duration_col].sum() if duration_col in tests_up_to.columns else float(first_fail_rank)
        scores.append(float(ttff_c))

    return float(np.mean(scores)) if scores else float("nan")


def compute_apfdc(
    df: pd.DataFrame,
    duration_col: str = "actual_duration",
) -> float:
    """APFDc — standard cost-cognizant formula.

    ``APFDc = 1 − Σ_{i=1}^{m} (C_i − t_i/2) / (m · T)``

    where:
      * ``C_i`` = cumulative actual duration up to and including the test
        that detected fault *i*
      * ``t_i`` = actual duration of that test
      * ``m``   = number of distinct faults detected (failing tests found)
      * ``T``   = total actual duration of all selected tests for this commit

    Returns mean APFDc over all defective commits where ≥ 1 failure was
    selected.  Does **not** clip to [0, 1].

    Returns ``NaN`` if no failures detected.

    **Conditional metric** — always report alongside CMR.
    """
    scores: list[float] = []
    for cid, grp in df.groupby("commit_idx"):
        failures = grp[grp["label"] == 1]
        if len(failures) == 0:
            continue

        grp_sorted = grp.sort_values("_rank")
        durations = grp_sorted[duration_col].values if duration_col in grp_sorted.columns else np.ones(len(grp_sorted))
        labels = grp_sorted["label"].values

        T = durations.sum()
        if T <= 0:
            continue

        cum_dur = np.cumsum(durations)
        fault_indices = np.where(labels == 1)[0]
        m = len(fault_indices)
        if m == 0:
            continue

        weighted_sum = 0.0
        for fi in fault_indices:
            C_i = cum_dur[fi]
            t_i = durations[fi]
            weighted_sum += (C_i - t_i / 2.0)

        apfdc_c = 1.0 - weighted_sum / (m * T)
        scores.append(apfdc_c)

    return float(np.mean(scores)) if scores else float("nan")


# ---------------------------------------------------------------------------
# Baseline Selectors
# ---------------------------------------------------------------------------

def baseline_retest_all(df: pd.DataFrame) -> np.ndarray:
    """Select all tests (100% budget)."""
    return np.ones(len(df), dtype=bool)


def baseline_fixed_threshold(
    df: pd.DataFrame,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Select tests with p̂ ≥ threshold (no conformal calibration)."""
    return probs >= threshold


def baseline_top_k_pct(
    df: pd.DataFrame,
    probs: np.ndarray,
    k_pct: float,
) -> np.ndarray:
    """Select top k% of tests per commit by predicted probability."""
    mask = np.zeros(len(df), dtype=bool)
    df_work = df.copy()
    df_work["_prob_bk"] = probs
    df_work["_iloc"] = np.arange(len(df))
    for cid, grp in df_work.groupby("commit_idx"):
        n_sel = max(1, int(np.ceil(len(grp) * k_pct)))
        top_idx = grp.nlargest(n_sel, "_prob_bk")["_iloc"].values.astype(int)
        mask[top_idx] = True
    return mask


def baseline_random_matched_budget(
    df: pd.DataFrame,
    reference_mask: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Random selection matching per-commit budget of reference method."""
    if rng is None:
        rng = np.random.default_rng(42)

    mask = np.zeros(len(df), dtype=bool)
    df_work = df.copy()
    df_work["_ref"] = reference_mask
    df_work["_iloc"] = np.arange(len(df))

    for cid, grp in df_work.groupby("commit_idx"):
        k = int(grp["_ref"].sum())
        if k <= 0:
            continue
        idx_pool = grp["_iloc"].values.astype(int)
        k = min(k, len(idx_pool))
        chosen = rng.choice(idx_pool, size=k, replace=False)
        mask[chosen] = True
    return mask


def baseline_random(
    df: pd.DataFrame,
    pct: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Randomly select pct fraction of tests per commit."""
    if rng is None:
        rng = np.random.default_rng(42)
    mask = np.zeros(len(df), dtype=bool)
    df_work = df.copy()
    df_work["_iloc"] = np.arange(len(df))
    for cid, grp in df_work.groupby("commit_idx"):
        idx = grp["_iloc"].values.astype(int)
        k = max(1, int(len(idx) * pct))
        chosen = rng.choice(idx, size=k, replace=False)
        mask[chosen] = True
    return mask


def baseline_recently_failed(
    df: pd.DataFrame,
    n_recent: int = 20,
) -> np.ndarray:
    """Select tests that failed recently in the historical window.

    Prefers fail_rate_20 > 0 (or fail_rate_50 > 0), falling back to
    last_fail_dist > 0 if fail-rate columns are absent.
    """
    mask = np.zeros(len(df), dtype=bool)
    if "fail_rate_20" in df.columns:
        mask = (df["fail_rate_20"] > 0).values
        # If fail_rate_20 selects nothing across the dataset, fallback to fail_rate_50
        if not mask.any() and "fail_rate_50" in df.columns:
            mask = (df["fail_rate_50"] > 0).values
    elif "fail_rate_50" in df.columns:
        mask = (df["fail_rate_50"] > 0).values
    elif "last_fail_dist" in df.columns:
        mask = ((df["last_fail_dist"] <= n_recent) & (df["last_fail_dist"] > 0)).values
    return mask


def baseline_naive_residual(
    df: pd.DataFrame,
    probs: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Naive residual risk: τ = α directly (no conformal calibration)."""
    from conformal import select_tests_residual_all_commits
    sel = select_tests_residual_all_commits(df, probs, tau=alpha)
    return df.index.isin(sel.index)


def baseline_threshold_target_recall(
    df_val: pd.DataFrame,
    probs_val: np.ndarray,
    df_test: pd.DataFrame,
    probs_test: np.ndarray,
    target_recall: float = 0.90,
) -> np.ndarray:
    """Find threshold on validation achieving target change-level recall,
    then apply to test set.
    """
    thresholds = np.sort(np.unique(probs_val))
    best_thr = 0.0
    for thr in thresholds:
        mask = probs_val >= thr
        cmr = compute_cmr(df_val, mask)
        recall = 1.0 - cmr
        if recall >= target_recall:
            best_thr = thr
        else:
            break
    return probs_test >= best_thr


# ---------------------------------------------------------------------------
# Unified evaluator
# ---------------------------------------------------------------------------

def evaluate_selection(
    df: pd.DataFrame,
    selected_mask: np.ndarray,
    method_name: str,
    alpha: float,
    duration_col: str = "actual_duration",
    prob_col: str = "_prob",
    order_by: str = "prob",
    est_duration_col: str = "duration_median_20",
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Compute all metrics for one method and one selection mask.

    Parameters
    ----------
    order_by : str
        "prob" for descending probability ordering,
        "cost_aware" for descending (prob / estimated_duration) ordering,
        "default" for standard test/class name alphabetical ordering (unprioritized),
        "random" for random execution order per commit,
        "recently_failed" for execution order sorted by recency of failure (last_fail_dist).
    """
    df_work = df.copy()
    df_work["_sel"] = selected_mask
    if prob_col not in df_work.columns:
        df_work["_prob"] = 0.0

    df_selected = df_work[df_work["_sel"]].copy()
    if len(df_selected) > 0:
        sort_key = df_selected["testName"].astype(str) if "testName" in df_selected.columns else ""
        df_selected["_sort_name"] = sort_key

        if order_by == "cost_aware":
            # Priority = p / duration_est
            col = est_duration_col if est_duration_col in df_selected.columns else "duration_mean"
            if col in df_selected.columns:
                dur_est = df_selected[col].fillna(1.0).clip(lower=1e-3)
            else:
                dur_est = pd.Series(1.0, index=df_selected.index)
            df_selected["_ordering_key"] = df_selected["_prob"] / dur_est
            df_selected = df_selected.sort_values(
                ["commit_idx", "_ordering_key", "_sort_name"],
                ascending=[True, False, True]
            )
        elif order_by == "default":
            # Standard class/test name alphabetical order, independent of ML _prob
            df_selected = df_selected.sort_values(
                ["commit_idx", "_sort_name"],
                ascending=[True, True]
            )
        elif order_by == "random":
            # Random execution order per commit
            gen = rng if rng is not None else np.random.default_rng(42)
            df_selected["_rand_key"] = gen.random(len(df_selected))
            df_selected = df_selected.sort_values(
                ["commit_idx", "_rand_key"],
                ascending=[True, True]
            )
        elif order_by == "recently_failed":
            # Most recent failure first (last_fail_dist ascending, <=0 placed at the end)
            if "last_fail_dist" in df_selected.columns:
                dist = df_selected["last_fail_dist"].copy()
                dist = dist.apply(lambda x: 1e9 if pd.isna(x) or x <= 0 else float(x))
                df_selected["_dist_key"] = dist
                if "fail_rate_20" in df_selected.columns:
                    df_selected["_fr_key"] = df_selected["fail_rate_20"].fillna(0.0)
                    df_selected = df_selected.sort_values(
                        ["commit_idx", "_dist_key", "_fr_key", "_sort_name"],
                        ascending=[True, True, False, True]
                    )
                else:
                    df_selected = df_selected.sort_values(
                        ["commit_idx", "_dist_key", "_sort_name"],
                        ascending=[True, True, True]
                    )
            elif "fail_rate_20" in df_selected.columns:
                df_selected["_fr_key"] = df_selected["fail_rate_20"].fillna(0.0)
                df_selected = df_selected.sort_values(
                    ["commit_idx", "_fr_key", "_sort_name"],
                    ascending=[True, False, True]
                )
            else:
                df_selected = df_selected.sort_values(
                    ["commit_idx", "_sort_name"],
                    ascending=[True, True]
                )
        else:
            # "prob" or fallback: descending probability
            df_selected["_ordering_key"] = df_selected["_prob"]
            df_selected = df_selected.sort_values(
                ["commit_idx", "_ordering_key", "_sort_name"],
                ascending=[True, False, True]
            )

        df_selected["_rank"] = df_selected.groupby("commit_idx").cumcount() + 1
    else:
        df_selected["_rank"] = 1

    cmr = compute_cmr(df_work, selected_mask)
    cmr_val, cmr_ci_low, cmr_ci_high = compute_cmr_wilson_ci(df_work, selected_mask)
    budget = compute_budget_saved(df_work, selected_mask, duration_col)
    ttff = compute_ttff(df_selected, duration_col) if len(df_selected) > 0 else float("nan")
    apfdc = compute_apfdc(df_selected, duration_col) if len(df_selected) > 0 else float("nan")

    return {
        "method": method_name,
        "alpha": alpha,
        "cmr": cmr,
        "cmr_ci_low": cmr_ci_low,
        "cmr_ci_high": cmr_ci_high,
        "pct_tests_selected": budget["pct_tests_selected"],
        "pct_duration_selected": budget["pct_duration_selected"],
        "pct_duration_saved": budget["pct_duration_saved"],
        "ttff": ttff,
        "apfdc": apfdc,
        "n_defective_commits": int(
            df[df["label"] == 1].groupby("commit_idx").ngroups
            if "commit_idx" in df.columns else 0
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
    duration_col: str = "actual_duration",
    df_val: Optional[pd.DataFrame] = None,
    probs_val: Optional[np.ndarray] = None,
    top_k_budgets: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Evaluate all methods on the test set for a given alpha."""
    from conformal import (
        select_tests_global,
        select_tests_residual_all_commits,
    )

    results: list[Dict[str, Any]] = []
    df_test_p = df_test.copy()
    df_test_p["_prob"] = probs_test

    # 1. Retest-all (standard alphabetical execution order, no ML prioritization)
    mask_all = baseline_retest_all(df_test)
    results.append(evaluate_selection(
        df_test_p, mask_all, "retest_all", alpha, duration_col, order_by="default"
    ))

    # 2. Global conformal (probability ordering)
    sel_global = select_tests_global(df_test, probs_test, threshold_global)
    mask_global = df_test.index.isin(sel_global.index)
    results.append(evaluate_selection(
        df_test_p, mask_global, "global_conformal", alpha, duration_col, order_by="prob"
    ))

    # 3. Global conformal (cost-aware ordering: p / duration)
    results.append(evaluate_selection(
        df_test_p, mask_global, "global_conformal_cost_aware", alpha, duration_col, order_by="cost_aware"
    ))

    # 4. Residual risk conformal (probability ordering)
    sel_res = select_tests_residual_all_commits(df_test, probs_test, tau_residual)
    mask_residual = df_test.index.isin(sel_res.index)
    results.append(evaluate_selection(
        df_test_p, mask_residual, "residual_risk", alpha, duration_col, order_by="prob"
    ))

    # 5. Residual risk conformal (cost-aware ordering)
    results.append(evaluate_selection(
        df_test_p, mask_residual, "residual_risk_cost_aware", alpha, duration_col, order_by="cost_aware"
    ))

    # 6. Naive residual risk (tau = alpha)
    sel_naive = select_tests_residual_all_commits(df_test, probs_test, alpha)
    mask_naive = df_test.index.isin(sel_naive.index)
    results.append(evaluate_selection(
        df_test_p, mask_naive, "naive_residual", alpha, duration_col, order_by="prob"
    ))

    # 7. Fixed threshold 0.5
    mask_fixed = baseline_fixed_threshold(df_test, probs_test, threshold=0.5)
    results.append(evaluate_selection(
        df_test_p, mask_fixed, "fixed_0.5", alpha, duration_col, order_by="prob"
    ))

    # 8. Top-k%
    if top_k_budgets is None:
        top_k_budgets = [0.10, 0.30, 0.50]
    for k_pct in top_k_budgets:
        mask_topk = baseline_top_k_pct(df_test, probs_test, k_pct)
        label = f"top_{int(k_pct*100)}pct"
        results.append(evaluate_selection(
            df_test_p, mask_topk, label, alpha, duration_col, order_by="prob"
        ))

    # 9. Random matched budget (matches global conformal, random execution order)
    mask_rand = baseline_random_matched_budget(df_test, mask_global, rng=rng)
    results.append(evaluate_selection(
        df_test_p, mask_rand, "random_matched", alpha, duration_col, order_by="random", rng=rng
    ))

    # 10. Recently failed (ordered by recency of failure)
    mask_recent = baseline_recently_failed(df_test)
    results.append(evaluate_selection(
        df_test_p, mask_recent, "recently_failed", alpha, duration_col, order_by="recently_failed"
    ))

    # 11. Threshold by target recall (if validation set provided)
    if df_val is not None and probs_val is not None:
        mask_tgt = baseline_threshold_target_recall(
            df_val, probs_val, df_test, probs_test, target_recall=1.0 - alpha
        )
        results.append(evaluate_selection(
            df_test_p, mask_tgt, "threshold_target_recall", alpha, duration_col, order_by="prob"
        ))

    return results
