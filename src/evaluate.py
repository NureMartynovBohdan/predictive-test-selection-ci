"""
evaluate.py — Metrics & Baseline Comparisons (v2.1)
===================================================
Implements:
  - CMR  (Change Miss Rate) with Wilson score 95% CI and moving block bootstrap CI
  - Budget saved (% tests, % duration)
  - TTFF (simulated sequential cumulative actual duration to first failure)
  - Conditional APFDc (standard cost-cognizant formula computed over selected
    tests on detected defective commits, no clip)
  - Baselines: retest-all, top-k%, fixed threshold, random matched budget,
               recently failed, naive residual risk, threshold by target recall
  - Cost-aware prioritization comparison (probability order vs p/duration order)
  - compare_all_methods() aggregator

NOTE:
  1. TTFF measures *simulated sequential execution time* (sum of individual test
     durations in execution order) without modeling parallel CI workers.
  2. TTFF and APFDc are *conditional* metrics — computed only on detected
     defective commits, and APFDc uses the total duration of the *selected*
     test subset as denominator T. They **must** always be reported alongside
     CMR and test budget.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple


def _drop_missing_labels(
    df: pd.DataFrame,
    selected_mask: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """Safely filter out rows with None/NaN labels before binary metric evaluation."""
    if "label" not in df.columns:
        return df, selected_mask
    valid = df["label"].notna().values
    if valid.all():
        return df, selected_mask
    df_valid = df.loc[valid].copy()
    mask_valid = selected_mask[valid] if selected_mask is not None else None
    return df_valid, mask_valid


def _candidate_mask(df: pd.DataFrame) -> np.ndarray:
    """Boolean mask of tests present in the pre-execution candidate inventory.

    Rows with ``in_candidate_inventory == False`` may exist in full ground truth
    for CMR evaluation, but selectors must never select them.
    """
    if "in_candidate_inventory" in df.columns:
        return df["in_candidate_inventory"].fillna(False).astype(bool).values
    return np.ones(len(df), dtype=bool)


# ---------------------------------------------------------------------------
# Core Metrics
# ---------------------------------------------------------------------------

def compute_cmr(
    df: pd.DataFrame,
    selected_mask: np.ndarray,
) -> float:
    """Change Miss Rate = fraction of defective commits where NO selected
    test has ``label == 1``.

    Rows with ``label`` equal to ``None``/``NaN`` are excluded.
    """
    df_valid, mask_valid = _drop_missing_labels(df, selected_mask)
    if len(df_valid) == 0:
        return 0.0

    df_work = df_valid.copy()
    df_work["_sel"] = mask_valid

    defective_commits = df_work.groupby("commit_idx")["label"].transform("max")
    defective_df = df_work[defective_commits == 1]

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


def _commit_miss_indicators(
    df: pd.DataFrame,
    selected_mask: np.ndarray,
) -> np.ndarray:
    """Return chronological binary array of miss indicators (1=missed, 0=detected)
    over defective commits.
    """
    df_valid, mask_valid = _drop_missing_labels(df, selected_mask)
    if len(df_valid) == 0:
        return np.array([], dtype=int)

    df_work = df_valid.copy()
    df_work["_sel"] = mask_valid

    defective_commits = df_work.groupby("commit_idx")["label"].transform("max")
    defective_df = df_work[defective_commits == 1]
    unique_defective = np.sort(defective_df["commit_idx"].unique())

    indicators = np.zeros(len(unique_defective), dtype=int)
    for i, cid in enumerate(unique_defective):
        grp = defective_df[defective_df["commit_idx"] == cid]
        if not (grp["_sel"] & (grp["label"] == 1)).any():
            indicators[i] = 1
    return indicators


def compute_cmr_wilson_ci(
    df: pd.DataFrame,
    selected_mask: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Wilson score confidence interval for CMR (descriptive, assumes i.i.d. commits).

    Returns ``(cmr, ci_low, ci_high)`` tuple, all in ``[0, 1]``.
    """
    indicators = _commit_miss_indicators(df, selected_mask)
    n = len(indicators)
    if n == 0:
        return (0.0, 0.0, 0.0)

    p_hat = float(indicators.mean())

    from scipy.stats import norm as _norm
    z = float(_norm.ppf((1.0 + confidence) / 2.0))
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p_hat + z2 / (2.0 * n)) / denom
    spread = z * np.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n)) / denom
    ci_low = float(np.clip(centre - spread, 0.0, 1.0))
    ci_high = float(np.clip(centre + spread, 0.0, 1.0))

    return (p_hat, ci_low, ci_high)


def compute_cmr_block_bootstrap_ci(
    df: pd.DataFrame,
    selected_mask: np.ndarray,
    block_size: int = 5,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """Moving block bootstrap confidence interval for CMR over chronological
    defective commits, accounting for temporal dependence across adjacent commits.

    Returns ``(cmr, ci_low, ci_high)`` tuple, all in ``[0, 1]``.
    """
    indicators = _commit_miss_indicators(df, selected_mask)
    n = len(indicators)
    if n == 0:
        return (0.0, 0.0, 0.0)

    p_hat = float(indicators.mean())
    if n == 1:
        return (p_hat, p_hat, p_hat)

    gen = rng if rng is not None else np.random.default_rng(42)
    b = max(1, min(block_size, n))
    n_blocks = int(np.ceil(n / b))
    max_start = n - b + 1

    boot_means = np.empty(n_bootstrap, dtype=float)
    for k in range(n_bootstrap):
        starts = gen.integers(0, max_start, size=n_blocks)
        sampled = np.concatenate([indicators[s:s + b] for s in starts])[:n]
        boot_means[k] = sampled.mean()

    alpha_tail = (1.0 - confidence) / 2.0
    ci_low = float(np.clip(np.quantile(boot_means, alpha_tail), 0.0, 1.0))
    ci_high = float(np.clip(np.quantile(boot_means, 1.0 - alpha_tail), 0.0, 1.0))
    return (p_hat, ci_low, ci_high)


def compute_budget_saved(
    df: pd.DataFrame,
    selected_mask: np.ndarray,
    duration_col: str = "actual_duration",
) -> Dict[str, float]:
    """Budget usage: fraction of tests selected and duration saved over the
    observed space (excluding any missing/unobserved rows with NaN labels or NaN duration).
    """
    df_valid, mask_valid = _drop_missing_labels(df, selected_mask)
    if duration_col in df_valid.columns:
        dur_valid = df_valid[duration_col].notna().values
        if not dur_valid.all():
            df_valid = df_valid.loc[dur_valid].copy()
            mask_valid = mask_valid[dur_valid] if mask_valid is not None else None

    if mask_valid is None:
        mask_valid = np.zeros(len(df_valid), dtype=bool)
    else:
        mask_valid = np.asarray(mask_valid, dtype=bool)

    n_total = len(df_valid)
    n_selected = int(mask_valid.sum())
    pct_tests_selected = n_selected / n_total if n_total > 0 else 0.0

    if duration_col in df_valid.columns:
        dur_vals = df_valid[duration_col].fillna(0).values
        total_dur = float(dur_vals.sum())
        sel_dur = float(dur_vals[mask_valid].sum())
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
    """Time-To-First-Failure (simulated sequential execution time in seconds).

    For each defective commit where >= 1 failure is selected:
      * Tests are executed sequentially in ``_rank`` order (no parallel workers).
      * ``TTFF_c = sum(actual_duration)`` for ranks 1 .. r where *r* is the
        rank of the **first** test with ``label == 1``.

    Returns mean TTFF (seconds) over all detected defective commits, or ``NaN``
    if no failures were detected.

    **Conditional metric** — always report alongside CMR.
    """
    df_valid, _ = _drop_missing_labels(df)
    scores: list[float] = []
    for cid, grp in df_valid.groupby("commit_idx"):
        failures = grp[grp["label"] == 1]
        if len(failures) == 0:
            continue
        first_fail_rank = failures["_rank"].min()
        tests_up_to = grp[grp["_rank"] <= first_fail_rank]
        ttff_c = (
            tests_up_to[duration_col].fillna(0.0).sum()
            if duration_col in tests_up_to.columns
            else float(first_fail_rank)
        )
        scores.append(float(ttff_c))

    return float(np.mean(scores)) if scores else float("nan")


def compute_apfdc(
    df: pd.DataFrame,
    duration_col: str = "actual_duration",
) -> float:
    """Conditional APFDc — cost-cognizant fault detection rate within the
    *selected* test subset on detected defective commits.

    ``Conditional APFDc = 1 - sum_{i=1}^{m} (C_i - t_i / 2) / (m * T)``

    where:
      * ``C_i`` = cumulative actual duration up to and including the test
        that detected fault *i*
      * ``t_i`` = actual duration of that test
      * ``m``   = number of distinct faults detected among selected tests
      * ``T``   = total actual duration of all *selected* tests for this commit

    Because ``T`` is computed over the selected subset (not the full suite),
    this metric measures prioritization quality *conditional* on selection and
    detection. It **must** always be interpreted jointly with CMR and budget.
    Does **not** clip to [0, 1]. Returns ``NaN`` if no failures detected.
    """
    df_valid, _ = _drop_missing_labels(df)
    scores: list[float] = []
    for cid, grp in df_valid.groupby("commit_idx"):
        failures = grp[grp["label"] == 1]
        if len(failures) == 0:
            continue

        grp_sorted = grp.sort_values("_rank")
        durations = (
            grp_sorted[duration_col].fillna(0.0).values
            if duration_col in grp_sorted.columns
            else np.ones(len(grp_sorted))
        )
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


def compute_conditional_apfdc(
    df: pd.DataFrame,
    duration_col: str = "actual_duration",
) -> float:
    """Explicit alias for ``compute_apfdc`` emphasizing conditional evaluation
    over the selected test subset on detected defective commits.
    """
    return compute_apfdc(df, duration_col=duration_col)


# ---------------------------------------------------------------------------
# Baseline Selectors
# ---------------------------------------------------------------------------

def baseline_retest_all(df: pd.DataFrame) -> np.ndarray:
    """Full-suite CI reference (100% of executed tests, 0% savings, 0% misses)."""
    return np.ones(len(df), dtype=bool)


def baseline_fixed_threshold(
    df: pd.DataFrame,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Select candidate tests with p_hat >= threshold (no conformal calibration)."""
    cand = _candidate_mask(df)
    eff_probs = np.where(cand, np.asarray(probs, dtype=float), 0.0)
    return (eff_probs >= threshold) & cand


def baseline_top_k_pct(
    df: pd.DataFrame,
    probs: np.ndarray,
    k_pct: float,
) -> np.ndarray:
    """Select top k% of selectable candidate tests per commit by probability."""
    cand = _candidate_mask(df)
    mask = np.zeros(len(df), dtype=bool)
    df_work = df.copy()
    df_work["_prob_bk"] = np.where(cand, np.asarray(probs, dtype=float), -np.inf)
    df_work["_cand"] = cand
    df_work["_iloc"] = np.arange(len(df))
    for cid, grp in df_work.groupby("commit_idx"):
        cand_grp = grp[grp["_cand"]]
        if len(cand_grp) == 0:
            continue
        n_sel = max(1, int(np.ceil(len(cand_grp) * k_pct)))
        n_sel = min(n_sel, len(cand_grp))
        top_idx = cand_grp.nlargest(n_sel, "_prob_bk")["_iloc"].values.astype(int)
        mask[top_idx] = True
    return mask


def baseline_random_matched_budget(
    df: pd.DataFrame,
    reference_mask: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Random selection within candidate inventory matching per-commit budget of reference."""
    if rng is None:
        rng = np.random.default_rng(42)

    cand = _candidate_mask(df)
    mask = np.zeros(len(df), dtype=bool)
    df_work = df.copy()
    df_work["_ref"] = np.asarray(reference_mask, dtype=bool) & cand
    df_work["_cand"] = cand
    df_work["_iloc"] = np.arange(len(df))

    for cid, grp in df_work.groupby("commit_idx"):
        k = int(grp["_ref"].sum())
        if k <= 0:
            continue
        idx_pool = grp.loc[grp["_cand"], "_iloc"].values.astype(int)
        if len(idx_pool) == 0:
            continue
        k = min(k, len(idx_pool))
        chosen = rng.choice(idx_pool, size=k, replace=False)
        mask[chosen] = True
    return mask


def baseline_random(
    df: pd.DataFrame,
    pct: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Randomly select pct fraction of candidate tests per commit."""
    if rng is None:
        rng = np.random.default_rng(42)
    cand = _candidate_mask(df)
    mask = np.zeros(len(df), dtype=bool)
    df_work = df.copy()
    df_work["_cand"] = cand
    df_work["_iloc"] = np.arange(len(df))
    for cid, grp in df_work.groupby("commit_idx"):
        idx = grp.loc[grp["_cand"], "_iloc"].values.astype(int)
        if len(idx) == 0:
            continue
        k = max(1, int(len(idx) * pct))
        k = min(k, len(idx))
        chosen = rng.choice(idx, size=k, replace=False)
        mask[chosen] = True
    return mask


def baseline_recently_failed(
    df: pd.DataFrame,
    n_recent: int = 20,
) -> np.ndarray:
    """Select candidate tests that failed recently in the historical window.

    Prefers fail_rate_20 > 0 (or fail_rate_50 > 0), falling back to
    last_fail_dist > 0 if fail-rate columns are absent.
    """
    cand = _candidate_mask(df)
    mask = np.zeros(len(df), dtype=bool)
    if "fail_rate_20" in df.columns:
        mask = (df["fail_rate_20"].fillna(0.0) > 0).values
        if not (mask & cand).any() and "fail_rate_50" in df.columns:
            mask = (df["fail_rate_50"].fillna(0.0) > 0).values
    elif "fail_rate_50" in df.columns:
        mask = (df["fail_rate_50"].fillna(0.0) > 0).values
    elif "last_fail_dist" in df.columns:
        mask = ((df["last_fail_dist"] <= n_recent) & (df["last_fail_dist"] > 0)).values
    return mask & cand


def baseline_naive_residual(
    df: pd.DataFrame,
    probs: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Naive residual risk: tau = alpha directly (no conformal calibration)."""
    from conformal import select_tests_residual_all_commits
    sel = select_tests_residual_all_commits(df, probs, tau=alpha)
    return df.index.isin(sel.index) & _candidate_mask(df)


def baseline_threshold_target_recall(
    df_val: pd.DataFrame,
    probs_val: np.ndarray,
    df_test: pd.DataFrame,
    probs_test: np.ndarray,
    target_recall: float = 0.90,
) -> np.ndarray:
    """Find threshold on validation candidates achieving target change-level recall,
    then apply to test candidates (never selects non-candidates).
    """
    cand_val = _candidate_mask(df_val)
    cand_test = _candidate_mask(df_test)
    eff_val = np.where(cand_val, np.asarray(probs_val, dtype=float), 0.0)
    eff_test = np.where(cand_test, np.asarray(probs_test, dtype=float), 0.0)

    cand_vals = np.unique(eff_val[cand_val]) if cand_val.any() else np.array([0.0])
    thresholds = np.sort(np.unique(np.concatenate(([0.0], cand_vals))))
    best_thr = 0.0
    for thr in thresholds:
        mask = (eff_val >= thr) & cand_val
        cmr = compute_cmr(df_val, mask)
        recall = 1.0 - cmr
        if recall >= target_recall:
            best_thr = float(thr)
        else:
            break
    return (eff_test >= best_thr) & cand_test


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
            col = est_duration_col if est_duration_col in df_selected.columns else "duration_mean"
            if col in df_selected.columns:
                dur_est = df_selected[col].fillna(1.0).clip(lower=1e-3)
            else:
                dur_est = pd.Series(1.0, index=df_selected.index)
            df_selected["_ordering_key"] = df_selected["_prob"] / dur_est
            df_selected = df_selected.sort_values(
                ["commit_idx", "_ordering_key", "_sort_name"],
                ascending=[True, False, True],
            )
        elif order_by == "default":
            df_selected = df_selected.sort_values(
                ["commit_idx", "_sort_name"],
                ascending=[True, True],
            )
        elif order_by == "random":
            gen = rng if rng is not None else np.random.default_rng(42)
            df_selected["_rand_key"] = gen.random(len(df_selected))
            df_selected = df_selected.sort_values(
                ["commit_idx", "_rand_key"],
                ascending=[True, True],
            )
        elif order_by == "recently_failed":
            if "last_fail_dist" in df_selected.columns:
                dist = df_selected["last_fail_dist"].copy()
                dist = dist.apply(lambda x: 1e9 if pd.isna(x) or x <= 0 else float(x))
                df_selected["_dist_key"] = dist
                if "fail_rate_20" in df_selected.columns:
                    df_selected["_fr_key"] = df_selected["fail_rate_20"].fillna(0.0)
                    df_selected = df_selected.sort_values(
                        ["commit_idx", "_dist_key", "_fr_key", "_sort_name"],
                        ascending=[True, True, False, True],
                    )
                else:
                    df_selected = df_selected.sort_values(
                        ["commit_idx", "_dist_key", "_sort_name"],
                        ascending=[True, True, True],
                    )
            elif "fail_rate_20" in df_selected.columns:
                df_selected["_fr_key"] = df_selected["fail_rate_20"].fillna(0.0)
                df_selected = df_selected.sort_values(
                    ["commit_idx", "_fr_key", "_sort_name"],
                    ascending=[True, False, True],
                )
            else:
                df_selected = df_selected.sort_values(
                    ["commit_idx", "_sort_name"],
                    ascending=[True, True],
                )
        else:
            df_selected["_ordering_key"] = df_selected["_prob"]
            df_selected = df_selected.sort_values(
                ["commit_idx", "_ordering_key", "_sort_name"],
                ascending=[True, False, True],
            )

        df_selected["_rank"] = df_selected.groupby("commit_idx").cumcount() + 1
    else:
        df_selected["_rank"] = 1

    cmr = compute_cmr(df_work, selected_mask)
    _, cmr_ci_low, cmr_ci_high = compute_cmr_wilson_ci(df_work, selected_mask)
    _, cmr_boot_low, cmr_boot_high = compute_cmr_block_bootstrap_ci(
        df_work, selected_mask, block_size=5, n_bootstrap=500, rng=np.random.default_rng(42)
    )
    budget = compute_budget_saved(df_work, selected_mask, duration_col)
    ttff = compute_ttff(df_selected, duration_col) if len(df_selected) > 0 else float("nan")
    cond_apfdc = compute_conditional_apfdc(df_selected, duration_col) if len(df_selected) > 0 else float("nan")

    df_valid_labels, _ = _drop_missing_labels(df_work)

    return {
        "method": method_name,
        "alpha": alpha,
        "cmr": cmr,
        "cmr_ci_low": cmr_ci_low,
        "cmr_ci_high": cmr_ci_high,
        "cmr_boot_ci_low": cmr_boot_low,
        "cmr_boot_ci_high": cmr_boot_high,
        "pct_tests_selected": budget["pct_tests_selected"],
        "pct_duration_selected": budget["pct_duration_selected"],
        "pct_duration_saved": budget["pct_duration_saved"],
        "ttff": ttff,
        "apfdc": cond_apfdc,
        "conditional_apfdc": cond_apfdc,
        "n_defective_commits": int(
            df_valid_labels[df_valid_labels["label"] == 1].groupby("commit_idx").ngroups
            if "commit_idx" in df_valid_labels.columns and len(df_valid_labels) > 0 else 0
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
    cand_mask = _candidate_mask(df_test)
    probs_eval = np.where(cand_mask, np.asarray(probs_test, dtype=float), 0.0)
    df_test_p = df_test.copy()
    df_test_p["_prob"] = probs_eval

    # 1. Retest-all (full CI suite reference without test selection)
    mask_all = baseline_retest_all(df_test)
    results.append(evaluate_selection(
        df_test_p, mask_all, "retest_all", alpha, duration_col, order_by="default"
    ))

    # 2. Global conformal (probability ordering; strictly within candidate inventory)
    sel_global = select_tests_global(df_test, probs_eval, threshold_global)
    mask_global = df_test.index.isin(sel_global.index) & cand_mask
    results.append(evaluate_selection(
        df_test_p, mask_global, "global_conformal", alpha, duration_col, order_by="prob"
    ))

    # 3. Global conformal (cost-aware ordering: p / duration)
    results.append(evaluate_selection(
        df_test_p, mask_global, "global_conformal_cost_aware", alpha, duration_col, order_by="cost_aware"
    ))

    # 4. Residual risk conformal (probability ordering; strictly within candidate inventory)
    sel_res = select_tests_residual_all_commits(df_test, probs_eval, tau_residual)
    mask_residual = df_test.index.isin(sel_res.index) & cand_mask
    results.append(evaluate_selection(
        df_test_p, mask_residual, "residual_risk", alpha, duration_col, order_by="prob"
    ))

    # 5. Residual risk conformal (cost-aware ordering)
    results.append(evaluate_selection(
        df_test_p, mask_residual, "residual_risk_cost_aware", alpha, duration_col, order_by="cost_aware"
    ))

    # 6. Naive residual risk (tau = alpha)
    sel_naive = select_tests_residual_all_commits(df_test, probs_eval, alpha)
    mask_naive = df_test.index.isin(sel_naive.index) & cand_mask
    results.append(evaluate_selection(
        df_test_p, mask_naive, "naive_residual", alpha, duration_col, order_by="prob"
    ))

    # 7. Fixed threshold 0.5
    mask_fixed = baseline_fixed_threshold(df_test, probs_eval, threshold=0.5) & cand_mask
    results.append(evaluate_selection(
        df_test_p, mask_fixed, "fixed_0.5", alpha, duration_col, order_by="prob"
    ))

    # 8. Top-k%
    if top_k_budgets is None:
        top_k_budgets = [0.10, 0.30, 0.50]
    for k_pct in top_k_budgets:
        mask_topk = baseline_top_k_pct(df_test, probs_eval, k_pct) & cand_mask
        label = f"top_{int(k_pct*100)}pct"
        results.append(evaluate_selection(
            df_test_p, mask_topk, label, alpha, duration_col, order_by="prob"
        ))

    # 9. Random matched budget (matches global conformal, random execution order)
    mask_rand = baseline_random_matched_budget(df_test, mask_global, rng=rng) & cand_mask
    results.append(evaluate_selection(
        df_test_p, mask_rand, "random_matched", alpha, duration_col, order_by="random", rng=rng
    ))

    # 10. Recently failed (ordered by recency of failure)
    mask_recent = baseline_recently_failed(df_test) & cand_mask
    results.append(evaluate_selection(
        df_test_p, mask_recent, "recently_failed", alpha, duration_col, order_by="recently_failed"
    ))

    # 11. Threshold by target recall (if validation set provided)
    if df_val is not None and probs_val is not None:
        mask_tgt = baseline_threshold_target_recall(
            df_val, probs_val, df_test, probs_eval, target_recall=1.0 - alpha
        ) & cand_mask
        results.append(evaluate_selection(
            df_test_p, mask_tgt, "threshold_target_recall", alpha, duration_col, order_by="prob"
        ))

    return results
