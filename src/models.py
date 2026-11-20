"""
src/models.py
=============
LightGBM training, isotonic calibration, and chronological split utilities.

Split strategy (strict chronological — no data leakage):
  60% train | 20% calibration | 20% test

The 20% calibration set is used for:
  1. Isotonic regression probability calibration
  2. Computing conformal scores (S_c) on defective commits

Random seeds are supported for multi-seed experiments (10–30 runs).
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[WARN] lightgbm not installed; model training unavailable.")

from features import FEATURE_COLS


# ---------------------------------------------------------------------------
# Chronological split
# ---------------------------------------------------------------------------

def chrono_split(
    df: pd.DataFrame,
    train_frac: float = 0.60,
    calib_frac: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by commit_idx (chronological). Returns (train, calib, test).
    All rows with the same commit_idx go entirely into ONE split.
    """
    commit_indices = sorted(df["commit_idx"].unique())
    n = len(commit_indices)
    n_train = int(np.floor(n * train_frac))
    n_calib = int(np.floor(n * (train_frac + calib_frac)))

    train_commits = set(commit_indices[:n_train])
    calib_commits = set(commit_indices[n_train:n_calib])
    test_commits  = set(commit_indices[n_calib:])

    train = df[df["commit_idx"].isin(train_commits)].copy()
    calib = df[df["commit_idx"].isin(calib_commits)].copy()
    test  = df[df["commit_idx"].isin(test_commits)].copy()

    return train, calib, test


# ---------------------------------------------------------------------------
# LightGBM training
# ---------------------------------------------------------------------------

def train_lgbm(
    train: pd.DataFrame,
    seed: int = 42,
    n_estimators: int = 400,
    num_leaves: int = 63,
    learning_rate: float = 0.05,
    min_child_samples: int = 10,
) -> "lgb.LGBMClassifier":
    """
    Train a LightGBM classifier on the training split.
    Uses scale_pos_weight to handle extreme class imbalance.
    """
    if not HAS_LGB:
        raise RuntimeError("lightgbm not installed")

    X = train[FEATURE_COLS].fillna(0)
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
) -> IsotonicRegression:
    """
    Fit an IsotonicRegression on calibration-set raw probabilities.
    Returns the fitted isotonic regressor.
    """
    X_cal = calib[FEATURE_COLS].fillna(0)
    y_cal = calib["label"]

    raw_probs = model.predict_proba(X_cal)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_probs, y_cal)
    return iso


def predict_calibrated(
    model,
    iso: IsotonicRegression,
    df: pd.DataFrame,
) -> np.ndarray:
    """
    Return calibrated probabilities for all rows in df.
    """
    X = df[FEATURE_COLS].fillna(0)
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


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data_builder import load_project, load_patches
    from features import build_features, FEATURE_COLS
    import pandas as _pd

    BASE = Path(__file__).parent.parent
    RTP = BASE / "rtp-torrent"
    proj = "apache@sling"

    df_c = _pd.read_csv(RTP / "tr_all_built_commits.csv", dtype={"tr_job_id": "int64"})
    j2c = dict(zip(df_c["tr_job_id"], df_c["git_commit_id"]))
    df = load_project(proj, RTP, j2c)
    patches = load_patches(proj, RTP)

    print("Building features …")
    feat = build_features(df, patches)

    train, calib, test = chrono_split(feat)
    print(f"train={len(train)} calib={len(calib)} test={len(test)}")
    print(f"Positive rate: train={train['label'].mean():.4f} "
          f"calib={calib['label'].mean():.4f} test={test['label'].mean():.4f}")

    print("Training LightGBM …")
    model = train_lgbm(train)

    print("Calibrating …")
    iso = calibrate_isotonic(model, calib)

    probs_test = predict_calibrated(model, iso, test)
    pr_auc = compute_pr_auc(test["label"].values, probs_test)
    print(f"PR-AUC on test set: {pr_auc:.4f}")
