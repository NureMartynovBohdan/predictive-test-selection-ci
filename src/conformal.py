"""
conformal.py — Change-Level Conformal Prediction (v2)
=====================================================
Implements:
  1. Global conformal threshold (λ) with CMR guarantee ≤ α
  2. Residual-risk adaptive rule with proper conformal τ calibration
  3. Deterministic tie-breaking for reproducibility
  4. Cost-aware ordering (p̂ / duration)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple


# ---------------------------------------------------------------------------
# Deterministic tie-breaking
# ---------------------------------------------------------------------------

def _effective_candidate_mask(df: pd.DataFrame) -> np.ndarray:
    """Boolean mask of tests present in the pre-execution candidate inventory.

    If ``in_candidate_inventory`` is absent (legacy callers / unit tests),
    all rows are treated as selectable candidates.
    """
    if "in_candidate_inventory" in df.columns:
        return df["in_candidate_inventory"].fillna(False).astype(bool).values
    return np.ones(len(df), dtype=bool)


def _deterministic_sort(
    df: pd.DataFrame,
    probs: np.ndarray,
) -> pd.DataFrame:
    """Sort by ``(-prob, test_name)`` for deterministic ordering, placing any
    non-candidate rows (``in_candidate_inventory == False``) strictly after
    selectable candidates.

    Used in **both** calibration and test-time selection so that
    ties (common after isotonic regression) are always broken
    identically.
    """
    df = df.copy()
    cand_mask = _effective_candidate_mask(df)
    df["_prob"] = np.where(cand_mask, np.asarray(probs, dtype=float), 0.0)
    df["_cand_sort"] = cand_mask.astype(int)
    df["_sort_key"] = df["testName"].astype(str) if "testName" in df.columns else ""
    df = df.sort_values(
        ["_cand_sort", "_prob", "_sort_key"],
        ascending=[False, False, True],
    )
    return df


# ---------------------------------------------------------------------------
# 1.  Global Conformal Threshold
# ---------------------------------------------------------------------------

def compute_conformal_scores(
    df: pd.DataFrame,
    probs: np.ndarray,
) -> np.ndarray:
    """Non-conformity scores for defective commits in calibration set.

    For each defective commit *c*:
        ``S_c = max_{i ∈ Failed(c)} p̂_{c,i}``

    Tests outside ``in_candidate_inventory`` have probability forced to 0.0
    (unselectable), so a commit failing solely on non-candidates receives
    ``S_c = 0.0`` rather than being dropped.

    Returns 1-D array of S_c values, one per defective commit.
    """
    df = df.copy()
    cand_mask = _effective_candidate_mask(df)
    df["_prob"] = np.where(cand_mask, np.asarray(probs, dtype=float), 0.0)

    scores = []
    for commit_idx, grp in df.groupby("commit_idx", sort=True):
        failed = grp[grp["label"] == 1]
        if len(failed) == 0:
            continue
        scores.append(failed["_prob"].max())

    return np.array(scores, dtype=float)


def compute_global_threshold(scores: np.ndarray, alpha: float) -> float:
    """Conformal threshold λ with ⌊α(n+1)⌋ correction.

    Parameters
    ----------
    scores : conformal scores from defective calibration commits
    alpha  : allowed miss rate (must satisfy 0 < alpha < 1)

    Returns
    -------
    lambda_ : float threshold

    Raises
    ------
    ValueError
        If alpha is not in (0, 1), scores is empty, or scores contain
        non-finite values.
    """
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        raise ValueError("scores is empty; cannot compute conformal threshold")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores contains NaN or Inf values")

    scores_sorted = np.sort(scores)
    n = len(scores_sorted)
    k = int(np.floor(alpha * (n + 1)))

    if k < 1:
        return 0.0
    if k > n:
        return float(scores_sorted[-1])

    return float(scores_sorted[k - 1])


def select_tests_global(
    df: pd.DataFrame,
    probs: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Select candidate tests with p̂ ≥ threshold (never selects non-candidates)."""
    df = df.copy()
    cand_mask = _effective_candidate_mask(df)
    df["_prob"] = np.where(cand_mask, np.asarray(probs, dtype=float), 0.0)
    return df[(df["_prob"] >= threshold) & cand_mask].copy()


# ---------------------------------------------------------------------------
# 2.  Residual-Risk Adaptive Rule
# ---------------------------------------------------------------------------

def _residual_risk(probs_sorted_desc: np.ndarray, j: int) -> float:
    """R(j) = 1 − ∏_{i>j} (1 − p̂_{(i)})

    Probability that at least one of the remaining (not-yet-selected)
    tests would fail.  *j* is the number of tests already selected
    (0-based prefix length).
    """
    remaining = probs_sorted_desc[j:]
    if len(remaining) == 0:
        return 0.0
    remaining_clipped = np.clip(remaining, 0.0, 1.0 - 1e-12)
    log_prod = np.sum(np.log1p(-remaining_clipped))
    return float(1.0 - np.exp(log_prod))


def compute_tau_residual(
    df: pd.DataFrame,
    probs: np.ndarray,
    alpha: float,
) -> float:
    """Calibrate τ via proper conformal quantile with (n+1) correction.

    For each defective commit in the calibration set:
      1. Sort tests by ``(-prob, test_name)`` (deterministic tie-breaking).
      2. Find ``j*`` = minimal prefix that includes at least one label==1
         test.
      3. Record ``R(j*−1)`` = residual risk *before* the ``j*``-th test
         was added (the risk level that *triggered* including this test).

    Collect R values ``R_1 … R_n`` from *n* defective commits.
    Sort ascending, then pick ``τ = R_{(k)}`` where ``k = ⌊α·(n+1)⌋``.

    If k < 1  → τ = 0.0  (run all tests for every commit).
    If k > n  → τ = R_{(n)}  (most permissive).

    Parameters
    ----------
    df    : calibration DataFrame with 'commit_idx', 'label', 'testName'
    probs : predicted probabilities aligned with df rows
    alpha : allowed miss rate (must satisfy 0 < alpha < 1)

    Raises
    ------
    ValueError
        If alpha is not in (0, 1), probs is empty, or probs contain
        non-finite values.
    """
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    probs = np.asarray(probs, dtype=float)
    if probs.size == 0:
        raise ValueError("probs is empty; cannot calibrate tau")
    if not np.all(np.isfinite(probs)):
        raise ValueError("probs contains NaN or Inf values")
    df = df.copy()
    cand_mask = _effective_candidate_mask(df)
    df["_prob"] = np.where(cand_mask, probs, 0.0)

    defective_commits = (
        df.groupby("commit_idx")
        .filter(lambda g: g["label"].sum() > 0)["commit_idx"]
        .unique()
    )

    if len(defective_commits) == 0:
        return 0.0

    residual_risks: list[float] = []

    for cid in sorted(defective_commits):
        grp = df[df["commit_idx"] == cid].copy()
        grp = _deterministic_sort(grp, grp["_prob"].values)
        ps = grp["_prob"].values
        labels = grp["label"].values

        # Find j*: minimal prefix capturing at least one label==1
        j_star = None
        for j in range(1, len(ps) + 1):
            if labels[:j].sum() > 0:
                j_star = j
                break

        if j_star is None:
            continue

        # R(j*-1) = residual risk BEFORE adding the j*-th test
        r_before = _residual_risk(ps, j_star - 1)
        residual_risks.append(r_before)

    if not residual_risks:
        return 0.0

    residual_risks_sorted = np.sort(residual_risks)
    n = len(residual_risks_sorted)
    k = int(np.floor(alpha * (n + 1)))

    if k < 1:
        return 0.0
    if k > n:
        return float(residual_risks_sorted[-1])

    return float(residual_risks_sorted[k - 1])


def select_tests_residual(
    df: pd.DataFrame,
    probs: np.ndarray,
    tau: float,
) -> pd.DataFrame:
    """Greedy residual-risk selection for a *single* commit.

    Critical fix: check ``R(0) < τ`` **before** adding the first test.
    If the commit looks safe (all probabilities very low), select 0 tests.

    Tests are sorted by ``(-prob, test_name)`` for deterministic ordering.
    """
    df = _deterministic_sort(df, probs)
    ps = df["_prob"].values

    # Check if residual risk is already below tau before selecting anything
    r0 = _residual_risk(ps, 0)
    if r0 < tau:
        selected = df.iloc[:0].copy()
        selected["_rank"] = []
        return selected

    cutoff = len(ps)
    for j in range(1, len(ps) + 1):
        r = _residual_risk(ps, j)
        if r < tau:
            cutoff = j
            break

    selected = df.iloc[:cutoff].copy()
    selected["_rank"] = np.arange(1, cutoff + 1)
    if "in_candidate_inventory" in selected.columns:
        cand_sel = selected["in_candidate_inventory"].fillna(False).astype(bool)
        selected = selected.loc[cand_sel].copy()
        selected["_rank"] = np.arange(1, len(selected) + 1)
    return selected


def select_tests_residual_all_commits(
    df: pd.DataFrame,
    probs: np.ndarray,
    tau: float,
) -> pd.DataFrame:
    """Apply residual-risk selection per commit."""
    df = df.copy()
    cand_mask = _effective_candidate_mask(df)
    df["_prob"] = np.where(cand_mask, probs, 0.0)
    parts = []
    for cid, grp in df.groupby("commit_idx", sort=True):
        sel = select_tests_residual(grp, grp["_prob"].values, tau)
        parts.append(sel)
    if parts:
        return pd.concat(parts)
    return df.iloc[:0].copy()


# ---------------------------------------------------------------------------
# 3.  Cost-Aware Ordering
# ---------------------------------------------------------------------------

def prioritize_cost_aware(
    selected_df: pd.DataFrame,
    duration_col: str = "duration_median_20",
    min_duration: float = 1e-3,
) -> pd.DataFrame:
    """Sort by p̂ / max(duration, ε) descending (APFDc-optimal order)."""
    df = selected_df.copy()
    # Support duration_median_20 or duration_mean
    col = duration_col if duration_col in df.columns else ("duration_mean" if "duration_mean" in df.columns else None)
    if col is not None:
        dur = df[col].fillna(1.0).clip(lower=min_duration)
    else:
        dur = pd.Series(1.0, index=df.index)

    df["_priority"] = df["_prob"] / dur
    df["_sort_key"] = df["testName"].astype(str) if "testName" in df.columns else ""
    return df.sort_values(["_priority", "_sort_key"], ascending=[False, True]).reset_index(drop=True)
