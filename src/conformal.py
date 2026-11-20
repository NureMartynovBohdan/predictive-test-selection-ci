"""
conformal.py — Change-Level Conformal Prediction
=================================================
Implements:
  1. Global conformal threshold (lambda) with CMR guarantee <= alpha
  2. Residual-risk adaptive rule with tau calibration
  3. Cost-aware ordering (p̂ / duration)
"""

import numpy as np
import pandas as pd
from typing import Tuple


# ---------------------------------------------------------------------------
# 1.  Global Conformal Threshold
# ---------------------------------------------------------------------------

def compute_conformal_scores(
    df: pd.DataFrame,
    probs: np.ndarray,
) -> np.ndarray:
    """
    For each defective commit in the calibration set compute:
        S_c = max_{i in Failed(c)} p̂_{c,i}

    Parameters
    ----------
    df : DataFrame with columns ['commit_idx', 'label'] (calibration rows)
    probs : calibrated probabilities aligned with df rows

    Returns
    -------
    scores : 1-D array of S_c values, one per defective commit in df
    """
    df = df.copy()
    df["_prob"] = probs

    scores = []
    for commit_idx, grp in df.groupby("commit_idx", sort=True):
        failed = grp[grp["label"] == 1]
        if len(failed) == 0:
            continue  # not defective — skip
        scores.append(failed["_prob"].max())

    return np.array(scores, dtype=float)


def compute_global_threshold(scores: np.ndarray, alpha: float) -> float:
    """
    Given conformal scores S_{(1)} <= ... <= S_{(n)} and risk level alpha,
    compute lambda = S_{(k)} where k = floor(alpha * (n + 1)).

    If k < 1, return the minimum possible threshold (0.0) so all tests pass.

    Parameters
    ----------
    scores : conformal scores from defective calibration commits
    alpha  : allowed miss rate (e.g. 0.05, 0.10, 0.15, 0.20)

    Returns
    -------
    lambda_ : float threshold
    """
    scores_sorted = np.sort(scores)
    n = len(scores_sorted)
    k = int(np.floor(alpha * (n + 1)))

    if k < 1:
        return 0.0  # run everything
    if k > n:
        return float(scores_sorted[-1])

    return float(scores_sorted[k - 1])  # 0-indexed: k-th smallest = index k-1


def select_tests_global(
    df: pd.DataFrame,
    probs: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """
    Select tests whose calibrated probability >= threshold.

    Returns
    -------
    DataFrame (subset of df) with extra column '_prob'
    """
    df = df.copy()
    df["_prob"] = probs
    return df[df["_prob"] >= threshold].copy()


# ---------------------------------------------------------------------------
# 2.  Residual-Risk Adaptive Rule
# ---------------------------------------------------------------------------

def _residual_risk(probs_sorted_desc: np.ndarray, j: int) -> float:
    """
    R(j) = 1 - prod_{i > j} (1 - p̂_{(i)})
    probability that at least one of the remaining (not-yet-selected) tests
    is a failure.

    j = number of tests already selected (0-based prefix).
    """
    remaining = probs_sorted_desc[j:]
    if len(remaining) == 0:
        return 0.0
    # clip to avoid log(0)
    remaining_clipped = np.clip(remaining, 0.0, 1.0 - 1e-12)
    log_prod = np.sum(np.log1p(-remaining_clipped))
    return float(1.0 - np.exp(log_prod))


def compute_tau_residual(
    df: pd.DataFrame,
    probs: np.ndarray,
    alpha: float,
) -> float:
    """
    Calibrate tau such that empirical CMR on calibration defective commits <= alpha.

    Strategy: binary search over tau in [0, 1].
    For each defective commit we simulate the greedy selection and check
    whether at least one label==1 test is captured.

    Parameters
    ----------
    df    : calibration DataFrame with 'commit_idx', 'label'
    probs : calibrated probabilities aligned with df
    alpha : target miss rate

    Returns
    -------
    tau : float
    """
    df = df.copy()
    df["_prob"] = probs

    defective_commits = df.groupby("commit_idx").filter(
        lambda g: g["label"].sum() > 0
    )["commit_idx"].unique()

    if len(defective_commits) == 0:
        return 0.0

    def empirical_cmr(tau: float) -> float:
        misses = 0
        for cid in defective_commits:
            grp = df[df["commit_idx"] == cid].copy()
            ps = grp["_prob"].values
            labels = grp["label"].values
            order = np.argsort(-ps)  # descending
            ps_sorted = ps[order]
            labels_sorted = labels[order]
            # greedy selection
            selected_labels = []
            for j in range(len(ps_sorted)):
                selected_labels.append(labels_sorted[j])
                r = _residual_risk(ps_sorted, j + 1)
                if r < tau:
                    break
            if sum(selected_labels) == 0:
                misses += 1
        return misses / len(defective_commits)

    # Binary search: find the LARGEST tau such that CMR <= alpha
    lo, hi = 0.0, 1.0
    best_tau = 0.0
    for _ in range(40):
        mid = (lo + hi) / 2.0
        cmr = empirical_cmr(mid)
        if cmr <= alpha:
            best_tau = mid
            lo = mid
        else:
            hi = mid

    return best_tau


def select_tests_residual(
    df: pd.DataFrame,
    probs: np.ndarray,
    tau: float,
) -> pd.DataFrame:
    """
    For a single commit: greedily add tests (sorted desc by p̂)
    until residual risk R(j) < tau.

    df should contain rows of ONE commit only (or use per-commit loop in caller).

    Returns DataFrame with selected rows + '_prob' and '_rank' columns.
    """
    df = df.copy()
    df["_prob"] = probs
    df = df.sort_values("_prob", ascending=False)
    ps = df["_prob"].values

    cutoff = len(ps)  # default: select all
    for j in range(len(ps)):
        r = _residual_risk(ps, j + 1)
        if r < tau:
            cutoff = j + 1
            break

    selected = df.iloc[:cutoff].copy()
    selected["_rank"] = np.arange(1, cutoff + 1)
    return selected


def select_tests_residual_all_commits(
    df: pd.DataFrame,
    probs: np.ndarray,
    tau: float,
) -> pd.DataFrame:
    """
    Apply residual-risk selection to all commits in df.

    Returns concatenated DataFrame of selected rows (preserving original index).
    """
    df = df.copy()
    df["_prob"] = probs
    parts = []
    for cid, grp in df.groupby("commit_idx", sort=True):
        sel = select_tests_residual(grp, grp["_prob"].values, tau)
        parts.append(sel)
    if parts:
        return pd.concat(parts)
    return df.iloc[:0].copy()


# ---------------------------------------------------------------------------
# 3.  Cost-Aware Ordering (p̂ / duration)
# ---------------------------------------------------------------------------

def prioritize_cost_aware(
    selected_df: pd.DataFrame,
    duration_col: str = "duration_mean",
    min_duration: float = 1e-3,
) -> pd.DataFrame:
    """
    Sort selected tests by p̂_i / max(duration, min_duration) descending.
    This maximises expected bug detections per unit time (APFDc-optimal order).

    Parameters
    ----------
    selected_df  : DataFrame with '_prob' and duration_col columns
    duration_col : column containing expected test duration (seconds)
    min_duration : floor to avoid division by zero

    Returns
    -------
    DataFrame sorted by cost-aware priority, with '_priority' column added.
    """
    df = selected_df.copy()
    dur = df[duration_col].clip(lower=min_duration) if duration_col in df.columns else pd.Series(1.0, index=df.index)
    df["_priority"] = df["_prob"] / dur
    return df.sort_values("_priority", ascending=False).reset_index(drop=True)
