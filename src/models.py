"""
src/models.py — LightGBM training, calibration, chronological splits (v2)
==========================================================================
Split strategy (strict chronological — no data leakage):

  4-way split:   Train (50%)  →  ProbCalib (15%)  →  ConfCalib (15%)  →  Test (20%)

  - **ProbCalib**: isotonic regression probability calibration
  - **ConfCalib**: conformal score / τ computation
  - These are **never** the same set.

Random seeds are supported for multi-seed experiments (10–30 runs).
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[WARN] lightgbm not installed; model training unavailable.")


# ---------------------------------------------------------------------------
# 4-way chronological split (primary)
# ---------------------------------------------------------------------------

def chrono_split_4(
    df: pd.DataFrame,
    train_frac: float = 0.50,
    prob_calib_frac: float = 0.15,
    conf_calib_frac: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by ``commit_idx`` into 4 non-overlapping chronological blocks.

    Returns ``(train, prob_calib, conf_calib, test)``.

    All rows with the same ``commit_idx`` go entirely into ONE split.
    ``ProbCalib`` is used for isotonic regression; ``ConfCalib`` for
    conformal thresholds.  They must **never** overlap.
    """
    commit_indices = sorted(df["commit_idx"].unique())
    n = len(commit_indices)

    n_train = int(np.floor(n * train_frac))
    n_prob  = int(np.floor(n * (train_frac + prob_calib_frac)))
    n_conf  = int(np.floor(n * (train_frac + prob_calib_frac + conf_calib_frac)))

    train_commits = set(commit_indices[:n_train])
    prob_commits  = set(commit_indices[n_train:n_prob])
    conf_commits  = set(commit_indices[n_prob:n_conf])
    test_commits  = set(commit_indices[n_conf:])

    train = df[df["commit_idx"].isin(train_commits)].copy()
    prob  = df[df["commit_idx"].isin(prob_commits)].copy()
    conf  = df[df["commit_idx"].isin(conf_commits)].copy()
    test  = df[df["commit_idx"].isin(test_commits)].copy()

    return train, prob, conf, test


# ---------------------------------------------------------------------------
# Legacy 3-way split (deprecated)
# ---------------------------------------------------------------------------

def chrono_split(
    df: pd.DataFrame,
    train_frac: float = 0.60,
    calib_frac: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """**DEPRECATED** — use ``chrono_split_4`` instead.

    Split by ``commit_idx`` into (train, calib, test).
    """
    warnings.warn(
        "chrono_split is deprecated; use chrono_split_4 for proper "
        "separation of isotonic and conformal calibration.",
        DeprecationWarning,
        stacklevel=2,
    )
    commit_indices = sorted(df["commit_idx"].unique())
    n = len(commit_indices)
    n_train = int(np.floor(n * train_frac))
    n_calib = int(np.floor(n * (train_frac + calib_frac)))

    train = df[df["commit_idx"].isin(set(commit_indices[:n_train]))].copy()
    calib = df[df["commit_idx"].isin(set(commit_indices[n_train:n_calib]))].copy()
    test  = df[df["commit_idx"].isin(set(commit_indices[n_calib:]))].copy()
    return train, calib, test


# ---------------------------------------------------------------------------
# Rolling / expanding splits
# ---------------------------------------------------------------------------

def rolling_time_splits(
    df: pd.DataFrame,
    n_splits: int = 3,
    min_train_frac: float = 0.25,
    prob_calib_frac: float = 0.15,
    conf_calib_frac: float = 0.15,
    test_frac: float = 0.15,
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Generate multiple chronological splits using expanding windows
    with bounded, non-overlapping test windows.
    """
    commit_indices = sorted(df["commit_idx"].unique())
    n = len(commit_indices)
    needed = prob_calib_frac + conf_calib_frac + test_frac
    if min_train_frac + needed > 1.0 + 1e-9:
        raise ValueError("Fractions exceed 1.0")

    splits = []
    available_train = 1.0 - needed
    step = (available_train - min_train_frac) / max(n_splits - 1, 1) if n_splits > 1 else 0.0

    for i in range(n_splits):
        t_frac = min_train_frac + step * i
        i_train = int(n * t_frac)
        i_prob = int(n * (t_frac + prob_calib_frac))
        i_conf = int(n * (t_frac + prob_calib_frac + conf_calib_frac))
        i_test = n if i == n_splits - 1 else int(n * (t_frac + needed))

        train_c = set(commit_indices[:i_train])
        prob_c = set(commit_indices[i_train:i_prob])
        conf_c = set(commit_indices[i_prob:i_conf])
        test_c = set(commit_indices[i_conf:i_test])

        splits.append((
            df[df["commit_idx"].isin(train_c)].copy(),
            df[df["commit_idx"].isin(prob_c)].copy(),
            df[df["commit_idx"].isin(conf_c)].copy(),
            df[df["commit_idx"].isin(test_c)].copy(),
        ))

    return splits


# ---------------------------------------------------------------------------
# LightGBM training
# ---------------------------------------------------------------------------

def _filter_candidate_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict training / probability calibration data strictly to tests in
    ``in_candidate_inventory == True`` when that column is present.
    """
    if "in_candidate_inventory" in df.columns:
        return df[df["in_candidate_inventory"].fillna(False).astype(bool)]
    return df


def train_lgbm(
    train: pd.DataFrame,
    feature_cols: List[str],
    seed: int = 42,
    n_estimators: int = 400,
    num_leaves: int = 63,
    learning_rate: float = 0.05,
    min_child_samples: int = 10,
) -> "lgb.LGBMClassifier":
    """Train a LightGBM classifier strictly on ``in_candidate_inventory == True`` rows.

    **Does NOT use** ``.fillna(0)`` — LightGBM handles ``NaN`` natively.
    """
    if not HAS_LGB:
        raise RuntimeError("lightgbm not installed")

    train_use = _filter_candidate_inventory(train)
    X = train_use[feature_cols]  # NO fillna
    y = train_use["label"]

    pos = int(y.sum())
    neg = int(len(y) - pos)
    spw = neg / max(pos, 1)

    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        min_child_samples=min_child_samples,
        scale_pos_weight=spw,
        n_jobs=-1,
        random_state=seed,
        verbosity=-1,
    )
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# Isotonic calibration
# ---------------------------------------------------------------------------

def calibrate_isotonic(
    model,
    calib: pd.DataFrame,
    feature_cols: List[str],
) -> IsotonicRegression:
    """Fit IsotonicRegression strictly on ``in_candidate_inventory == True`` rows."""
    calib_use = _filter_candidate_inventory(calib)
    X_cal = calib_use[feature_cols]  # NO fillna
    y_cal = calib_use["label"]

    raw_probs = model.predict_proba(X_cal)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_probs, y_cal)
    return iso


def predict_raw_and_calibrated(
    model,
    iso: IsotonicRegression,
    df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(raw_probs, calibrated_probs)`` for all rows in df."""
    if len(df) == 0:
        empty = np.array([], dtype=float)
        return empty, empty
    X = df[feature_cols]  # NO fillna
    raw = np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    cal = np.asarray(iso.predict(raw), dtype=float)
    return raw, cal


def predict_calibrated(
    model,
    iso: IsotonicRegression,
    df: pd.DataFrame,
    feature_cols: List[str],
) -> np.ndarray:
    """Return calibrated probabilities for all rows in df."""
    _, cal = predict_raw_and_calibrated(model, iso, df, feature_cols)
    return cal


# ---------------------------------------------------------------------------
# PR-AUC & Brier evaluation (including baseline references)
# ---------------------------------------------------------------------------

def compute_pr_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Average-precision score (area under precision-recall curve)."""
    from sklearn.metrics import average_precision_score
    if labels.sum() == 0:
        return 0.0
    return float(average_precision_score(labels, probs))


def compute_pr_auc_baseline(labels: np.ndarray) -> float:
    """Random/no-skill PR-AUC baseline = positive class prevalence in labels."""
    arr = np.asarray(labels, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr == 1))


def compute_brier_score(labels: np.ndarray, probs: np.ndarray) -> float:
    """Brier score = mean squared error of probability forecasts.

    BS = (1/n) Σ (p̂_i − y_i)²

    Lower is better (0 = perfect, 0.25 = no-skill uninformative model).
    Returns NaN if labels array is empty.
    """
    labels = np.asarray(labels, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if len(labels) == 0:
        return float("nan")
    return float(np.mean((probs - labels) ** 2))


def compute_brier_baseline(train_labels: np.ndarray, test_labels: np.ndarray) -> float:
    """Constant-prevalence Brier baseline: predict the empirical positive rate
    from ``train_labels`` on every sample in ``test_labels``.
    """
    tr = np.asarray(train_labels, dtype=float)
    te = np.asarray(test_labels, dtype=float)
    if tr.size == 0 or te.size == 0:
        return float("nan")
    prior = float(np.mean(tr == 1))
    const_preds = np.full(te.shape, prior, dtype=float)
    return compute_brier_score(te, const_preds)
