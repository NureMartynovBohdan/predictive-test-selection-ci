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


def compute_exact_binomial_ci_counts(
    n_missed: int,
    n_defective: int,
    confidence: float = 0.95,
) -> Tuple[float, float, float, float]:
    """Clopper-Pearson two-sided exact binomial CI and one-sided 95% upper bound.

    At ``n_missed == 0``, the one-sided 95% upper bound equals ``1 - 0.05**(1/n)``.
    Returns ``(cmr, exact_low, exact_high, one_sided_upper)`` in ``[0, 1]``.
    """
    if n_defective <= 0:
        return (0.0, 0.0, 0.0, 0.0)

    k = int(n_missed)
    n = int(n_defective)
    p_hat = k / n

    from scipy.stats import beta as _beta
    alpha_c = 1.0 - confidence
    ci_low = 0.0 if k == 0 else float(_beta.ppf(alpha_c / 2.0, k, n - k + 1))
    ci_high = 1.0 if k >= n else float(_beta.ppf(1.0 - alpha_c / 2.0, k + 1, n - k))

    if k == 0:
        upper_1s = float(1.0 - alpha_c ** (1.0 / n))
    elif k >= n:
        upper_1s = 1.0
    else:
        upper_1s = float(_beta.ppf(confidence, k + 1, n - k))

    return (
        float(p_hat),
        float(np.clip(ci_low, 0.0, 1.0)),
        float(np.clip(ci_high, 0.0, 1.0)),
        float(np.clip(upper_1s, 0.0, 1.0)),
    )


def compute_wilson_ci_counts(
    n_missed: int,
    n_defective: int,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Wilson score interval from aggregated integer counts."""
    if n_defective <= 0:
        return (0.0, 0.0, 0.0)
    k = int(n_missed)
    n = int(n_defective)
    p_hat = k / n
    from scipy.stats import norm as _norm
    z = float(_norm.ppf((1.0 + confidence) / 2.0))
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p_hat + z2 / (2.0 * n)) / denom
    spread = z * np.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n)) / denom
    return (
        float(p_hat),
        float(np.clip(centre - spread, 0.0, 1.0)),
        float(np.clip(centre + spread, 0.0, 1.0)),
    )


def compute_cmr_exact_ci(
    df: pd.DataFrame,
    selected_mask: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float, float, float]:
    """Clopper-Pearson exact binomial CI + one-sided upper bound for CMR.

    Returns ``(cmr, exact_low, exact_high, one_sided_upper)``.
    """
    indicators = _commit_miss_indicators(df, selected_mask)
    return compute_exact_binomial_ci_counts(
        int(indicators.sum()), len(indicators), confidence=confidence
    )


def compute_cmr_wilson_ci(
    df: pd.DataFrame,
    selected_mask: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Wilson score confidence interval for CMR (descriptive, assumes i.i.d. commits).

    Returns ``(cmr, ci_low, ci_high)`` tuple, all in ``[0, 1]``.
    """
    indicators = _commit_miss_indicators(df, selected_mask)
    return compute_wilson_ci_counts(
        int(indicators.sum()), len(indicators), confidence=confidence
    )


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


def _filtered_observed_df_and_mask(
    df: pd.DataFrame,
    selected_mask: np.ndarray,
    duration_col: str = "actual_duration",
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Filter out rows with NaN labels or NaN durations consistently with budget computation."""
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
    return df_valid, mask_valid


def compute_budget_saved(
    df: pd.DataFrame,
    selected_mask: np.ndarray,
    duration_col: str = "actual_duration",
) -> Dict[str, float]:
    """Budget usage: fraction of tests selected and duration saved over the
    observed space (excluding any missing/unobserved rows with NaN labels or NaN duration).
    """
    df_valid, mask_valid = _filtered_observed_df_and_mask(df, selected_mask, duration_col)

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


def compute_ttff_stats(
    df: pd.DataFrame,
    duration_col: str = "actual_duration",
) -> Dict[str, float]:
    """Compute mean, median, and 90th percentile of Time-To-First-Failure (TTFF)
    over detected defective commits (simulated sequential execution in ``_rank`` order).

    Returns ``{"mean": ..., "median": ..., "p90": ...}``, with ``NaN`` if none detected.
    """
    df_valid, _ = _drop_missing_labels(df)
    scores: list[float] = []
    for _, grp in df_valid.groupby("commit_idx"):
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

    if not scores:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan")}
    arr = np.asarray(scores, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
    }


def compute_ttff(
    df: pd.DataFrame,
    duration_col: str = "actual_duration",
) -> float:
    """Mean Time-To-First-Failure (simulated sequential execution time in seconds).

    **Conditional metric** — always report alongside CMR.
    """
    return compute_ttff_stats(df, duration_col=duration_col)["mean"]


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
# Unified evaluator & miss-cause diagnostics
# ---------------------------------------------------------------------------

def _code_coupling_score(df: pd.DataFrame) -> pd.Series:
    """Compute heuristic code-coupling proximity score for LatestFail+CC ordering."""
    score = pd.Series(0.0, index=df.index, dtype=float)
    weights = (
        ("test_file_changed", 4.0),
        ("same_package_changed", 2.0),
        ("same_module_changed", 1.0),
        ("path_max_overlap", 2.0),
        ("co_fail_rate", 2.0),
    )
    for col, w in weights:
        if col in df.columns:
            score = score + w * df[col].fillna(0.0).astype(float)
    if "path_min_dist" in df.columns:
        score = score + (1.0 - df["path_min_dist"].fillna(1.0).astype(float).clip(0.0, 1.0))
    return score


def analyze_miss_causes(
    df: pd.DataFrame,
    selected_mask: np.ndarray,
    probs_test: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    """Break down every missed defective commit into mutually exclusive causes:
      - ``missing_from_inventory``: no failing test was in ``in_candidate_inventory``
      - ``empty_prefix``: candidate failure existed, but 0 tests were selected for the commit
      - ``no_prior_history_low_score``: candidate failure existed with no run history and scored below cutoff
      - ``score_below_cutoff``: candidate failure had prior run history yet scored below cutoff

    Returns ``(summary_counts, per_commit_records)``.
    """
    df_valid, mask_valid = _drop_missing_labels(df, selected_mask)
    if len(df_valid) == 0:
        return {
            "miss_missing_from_inventory": 0,
            "miss_empty_prefix": 0,
            "miss_no_prior_history": 0,
            "miss_score_below_cutoff": 0,
            "miss_total": 0,
        }, []

    work = df_valid.copy()
    work["_sel"] = np.asarray(mask_valid, dtype=bool)
    cand = _candidate_mask(work)
    work["_cand"] = cand
    if probs_test is not None and len(probs_test) == len(df):
        valid_pos = df["label"].notna().values if "label" in df.columns else np.ones(len(df), dtype=bool)
        probs_arr = np.asarray(probs_test, dtype=float)[valid_pos]
        work["_prob_eval"] = np.where(cand, probs_arr, 0.0)
    elif "_prob" in work.columns:
        work["_prob_eval"] = np.where(cand, work["_prob"].fillna(0.0).astype(float), 0.0)
    else:
        work["_prob_eval"] = 0.0

    counts = {
        "miss_missing_from_inventory": 0,
        "miss_empty_prefix": 0,
        "miss_no_prior_history": 0,
        "miss_score_below_cutoff": 0,
        "miss_total": 0,
    }
    records: list[Dict[str, Any]] = []

    def_groups = work.groupby("commit_idx", sort=True)
    for cid, grp in def_groups:
        fail_rows = grp[grp["label"] == 1]
        if len(fail_rows) == 0:
            continue
        if (grp["_sel"] & (grp["label"] == 1)).any():
            continue  # detected

        counts["miss_total"] += 1
        cand_rows = grp[grp["_cand"]].copy()
        cand_fail = fail_rows[fail_rows["_cand"]]
        noncand_fail = fail_rows[~fail_rows["_cand"]]
        n_sel_commit = int(grp["_sel"].sum())

        if len(cand_fail) == 0:
            cause = "missing_from_inventory"
            counts["miss_missing_from_inventory"] += 1
            max_prob = 0.0
            best_rank = np.nan
            has_hist = False
        else:
            if "_sort_name" not in cand_rows.columns:
                cand_rows["_sort_name"] = cand_rows["testName"].astype(str) if "testName" in cand_rows.columns else ""
            cand_sorted = cand_rows.sort_values(["_prob_eval", "_sort_name"], ascending=[False, True]).reset_index(drop=True)
            cand_sorted["_cand_rank"] = np.arange(1, len(cand_sorted) + 1)
            fail_ranks = cand_sorted.loc[cand_sorted["label"] == 1, "_cand_rank"]
            best_rank = int(fail_ranks.min()) if len(fail_ranks) > 0 else np.nan
            max_prob = float(cand_fail["_prob_eval"].max())
            has_hist = bool(
                "run_count_total" in cand_fail.columns
                and (cand_fail["run_count_total"].fillna(0.0) > 0).any()
            )

            if n_sel_commit == 0:
                cause = "empty_prefix"
                counts["miss_empty_prefix"] += 1
            elif "run_count_total" in cand_fail.columns and not has_hist:
                cause = "no_prior_history_low_score"
                counts["miss_no_prior_history"] += 1
            else:
                cause = "score_below_cutoff"
                counts["miss_score_below_cutoff"] += 1

        commit_sha = str(grp["commit"].iloc[0]) if "commit" in grp.columns else str(cid)
        records.append({
            "commit": commit_sha,
            "commit_idx": int(cid),
            "miss_cause": cause,
            "n_selected_commit": n_sel_commit,
            "n_total_cand_commit": int(len(cand_rows)),
            "n_cand_fail": int(len(cand_fail)),
            "n_noncand_fail": int(len(noncand_fail)),
            "max_cand_fail_prob": max_prob,
            "best_cand_fail_rank": best_rank,
            "has_history_on_cand_fail": has_hist,
        })

    return counts, records


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
        "qtf" for ascending historical duration (duration_median_20),
        "latest_fail_cc" for failure recency + code coupling score,
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
        elif order_by == "qtf":
            col = est_duration_col if est_duration_col in df_selected.columns else "duration_mean"
            dur_qtf = (
                df_selected[col].astype(float)
                if col in df_selected.columns
                else pd.Series(np.nan, index=df_selected.index)
            )
            df_selected["_qtf_key"] = dur_qtf.fillna(np.inf)
            df_selected = df_selected.sort_values(
                ["commit_idx", "_qtf_key", "_sort_name"],
                ascending=[True, True, True],
            )
        elif order_by == "latest_fail_cc":
            if "last_fail_dist" in df_selected.columns:
                dist = df_selected["last_fail_dist"].apply(
                    lambda x: 1e9 if pd.isna(x) or x <= 0 else float(x)
                )
            else:
                dist = pd.Series(1e9, index=df_selected.index)
            df_selected["_dist_key"] = dist
            df_selected["_fr_key"] = (
                df_selected["fail_rate_20"].fillna(0.0).astype(float)
                if "fail_rate_20" in df_selected.columns
                else 0.0
            )
            df_selected["_cc_key"] = _code_coupling_score(df_selected)
            df_selected = df_selected.sort_values(
                ["commit_idx", "_dist_key", "_cc_key", "_fr_key", "_sort_name"],
                ascending=[True, True, False, False, True],
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

    indicators = _commit_miss_indicators(df_work, selected_mask)
    n_def_commits = int(len(indicators))
    n_missed_commits = int(indicators.sum())
    cmr = n_missed_commits / n_def_commits if n_def_commits > 0 else 0.0
    _, cmr_ci_low, cmr_ci_high = compute_wilson_ci_counts(n_missed_commits, n_def_commits)
    _, cmr_exact_low, cmr_exact_high, cmr_upper_95 = compute_exact_binomial_ci_counts(
        n_missed_commits, n_def_commits
    )
    _, cmr_boot_low, cmr_boot_high = compute_cmr_block_bootstrap_ci(
        df_work, selected_mask, block_size=5, n_bootstrap=500, rng=np.random.default_rng(42)
    )
    budget = compute_budget_saved(df_work, selected_mask, duration_col)
    df_obs_b, mask_obs_b = _filtered_observed_df_and_mask(df_work, selected_mask, duration_col)
    if duration_col in df_obs_b.columns:
        dur_vals_b = df_obs_b[duration_col].fillna(0.0).values
        total_dur_obs = float(dur_vals_b.sum())
        sel_dur_obs = float(dur_vals_b[mask_obs_b].sum())
    else:
        total_dur_obs = float(len(df_obs_b))
        sel_dur_obs = float(mask_obs_b.sum())

    ttff_stats = (
        compute_ttff_stats(df_selected, duration_col)
        if len(df_selected) > 0
        else {"mean": float("nan"), "median": float("nan"), "p90": float("nan")}
    )
    cond_apfdc = compute_conditional_apfdc(df_selected, duration_col) if len(df_selected) > 0 else float("nan")

    miss_summary, _ = analyze_miss_causes(df_work, selected_mask, probs_test=df_work["_prob"].values)

    return {
        "method": method_name,
        "alpha": alpha,
        "cmr": cmr,
        "cmr_ci_low": cmr_ci_low,
        "cmr_ci_high": cmr_ci_high,
        "cmr_exact_ci_low": cmr_exact_low,
        "cmr_exact_ci_high": cmr_exact_high,
        "cmr_upper_95": cmr_upper_95,
        "cmr_boot_ci_low": cmr_boot_low,
        "cmr_boot_ci_high": cmr_boot_high,
        "pct_tests_selected": budget["pct_tests_selected"],
        "pct_duration_selected": budget["pct_duration_selected"],
        "pct_duration_saved": budget["pct_duration_saved"],
        "total_test_duration": total_dur_obs,
        "selected_test_duration": sel_dur_obs,
        "n_obs_pairs": int(len(df_obs_b)),
        "n_selected_obs_pairs": int(mask_obs_b.sum()),
        "ttff": ttff_stats["mean"],
        "ttff_median": ttff_stats["median"],
        "ttff_p90": ttff_stats["p90"],
        "apfdc": cond_apfdc,
        "conditional_apfdc": cond_apfdc,
        "n_defective_commits": n_def_commits,
        "n_missed_defective_commits": n_missed_commits,
        "miss_missing_from_inventory": miss_summary["miss_missing_from_inventory"],
        "miss_empty_prefix": miss_summary["miss_empty_prefix"],
        "miss_no_prior_history": miss_summary["miss_no_prior_history"],
        "miss_score_below_cutoff": miss_summary["miss_score_below_cutoff"],
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

    # 1. Retest-all (full CI suite references without test selection)
    mask_all = baseline_retest_all(df_test)
    results.append(evaluate_selection(
        df_test_p, mask_all, "retest_all", alpha, duration_col, order_by="default"
    ))
    results.append(evaluate_selection(
        df_test_p, mask_all, "qtf", alpha, duration_col, order_by="qtf"
    ))
    results.append(evaluate_selection(
        df_test_p, mask_all, "latest_fail_cc", alpha, duration_col, order_by="latest_fail_cc"
    ))

    # 2. Global conformal (probability ordering; primary marginal conformal method)
    sel_global = select_tests_global(df_test, probs_eval, threshold_global)
    mask_global = df_test.index.isin(sel_global.index) & cand_mask
    results.append(evaluate_selection(
        df_test_p, mask_global, "global_conformal", alpha, duration_col, order_by="prob"
    ))

    # 3. Global conformal (cost-aware p/duration & QTF orderings)
    results.append(evaluate_selection(
        df_test_p, mask_global, "global_conformal_cost_aware", alpha, duration_col, order_by="cost_aware"
    ))
    results.append(evaluate_selection(
        df_test_p, mask_global, "global_conformal_qtf", alpha, duration_col, order_by="qtf"
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
