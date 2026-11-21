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
    min_train_frac: float = 0.30,
    prob_calib_frac: float = 0.15,
    conf_calib_frac: float = 0.15,
    test_frac: float = 0.20,
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Generate multiple chronological splits using expanding windows.

    Each split uses a progressively larger training set while keeping
    ``prob_calib``, ``conf_calib`` and ``test`` fractions constant.
    """
    commit_indices = sorted(df["commit_idx"].unique())
    n = len(commit_indices)
    needed = prob_calib_frac + conf_calib_frac + test_frac
    if min_train_frac + needed > 1.0:
        raise ValueError("Fractions exceed 1.0")

    splits = []
    available_train = 1.0 - needed  # max fraction for training
    step = (available_train - min_train_frac) / max(n_splits - 1, 1) if n_splits > 1 else 0

    for i in range(n_splits):
        t_frac = min_train_frac + step * i
        result = chrono_split_4(df, t_frac, prob_calib_frac, conf_calib_frac)
        splits.append(result)

    return splits


# ---------------------------------------------------------------------------
# LightGBM training
# ---------------------------------------------------------------------------

def train_lgbm(
    train: pd.DataFrame,
    feature_cols: List[str],
    seed: int = 42,
    n_estimators: int = 400,
    num_leaves: int = 63,
    learning_rate: float = 0.05,
    min_child_samples: int = 10,
) -> "lgb.LGBMClassifier":
    """Train a LightGBM classifier.

    **Does NOT use** ``.fillna(0)`` — LightGBM handles ``NaN`` natively.
    """
    if not HAS_LGB:
        raise RuntimeError("lightgbm not installed")

    X = train[feature_cols]  # NO fillna
    y = train["label"]

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
    """Fit IsotonicRegression on calibration-set raw probabilities."""
    X_cal = calib[feature_cols]  # NO fillna
    y_cal = calib["label"]

    raw_probs = model.predict_proba(X_cal)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_probs, y_cal)
    return iso


def predict_calibrated(
    model,
    iso: IsotonicRegression,
    df: pd.DataFrame,
    feature_cols: List[str],
) -> np.ndarray:
    """Return calibrated probabilities for all rows in df."""
    X = df[feature_cols]  # NO fillna
    raw = model.predict_proba(X)[:, 1]
    return iso.predict(raw)


# ---------------------------------------------------------------------------
# PR-AUC evaluation
# ---------------------------------------------------------------------------

def compute_pr_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Average-precision score (area under precision-recall curve)."""
    from sklearn.metrics import average_precision_score
    if labels.sum() == 0:
        return 0.0
    return float(average_precision_score(labels, probs))
