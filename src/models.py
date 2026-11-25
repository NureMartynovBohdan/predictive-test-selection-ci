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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

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
# Probability calibration (Isotonic vs Platt / Sigmoid) & ProbCalib selection
# ---------------------------------------------------------------------------

class PlattCalibrator:
    """Platt scaling: 1D logistic regression on clipped raw log-odds.

    Especially effective when base gradient-boosted trees are trained with
    ``scale_pos_weight``, which shifts raw margins upward while keeping rank
    ordering informative.
    """

    def __init__(self, eps: float = 1e-6, C: float = 1.0):
        self.eps = float(eps)
        self.C = float(C)
        self._lr = LogisticRegression(C=self.C, solver="lbfgs")
        self._const_prob: Optional[float] = None

    def _to_logit(self, raw_probs: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(raw_probs, dtype=float), self.eps, 1.0 - self.eps)
        return np.log(p / (1.0 - p)).reshape(-1, 1)

    def fit(self, raw_probs: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        """Fit logistic map logit(raw_probs) -> P(y=1)."""
        y_arr = np.asarray(y, dtype=int)
        if len(y_arr) == 0:
            raise ValueError("Cannot fit PlattCalibrator on empty data")
        if np.unique(y_arr).size < 2:
            self._const_prob = float(np.mean(y_arr == 1))
            return self
        self._const_prob = None
        self._lr.fit(self._to_logit(raw_probs), y_arr)
        return self

    def predict(self, raw_probs: np.ndarray) -> np.ndarray:
        """Predict calibrated probabilities from 1D raw model probabilities."""
        raw_arr = np.asarray(raw_probs, dtype=float)
        if len(raw_arr) == 0:
            return np.array([], dtype=float)
        if self._const_prob is not None:
            return np.full(len(raw_arr), self._const_prob, dtype=float)
        return np.asarray(self._lr.predict_proba(self._to_logit(raw_arr))[:, 1], dtype=float)


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


def calibrate_sigmoid(
    model,
    calib: pd.DataFrame,
    feature_cols: List[str],
    eps: float = 1e-6,
    C: float = 1.0,
) -> PlattCalibrator:
    """Fit Platt (sigmoid) calibration strictly on ``in_candidate_inventory == True`` rows."""
    calib_use = _filter_candidate_inventory(calib)
    X_cal = calib_use[feature_cols]  # NO fillna
    y_cal = calib_use["label"]

    raw_probs = model.predict_proba(X_cal)[:, 1]
    return PlattCalibrator(eps=eps, C=C).fit(raw_probs, y_cal)


def evaluate_and_select_calibrators(
    model,
    prob_calib: pd.DataFrame,
    feature_cols: List[str],
    train_prior: float,
    n_splits: int = 3,
) -> Dict[str, Any]:
    """Fit both Isotonic and Platt/Sigmoid calibrators on ``ProbCalib`` candidates
    and compare them via deterministic commit-grouped out-of-fold CV on ``ProbCalib``
    alone (zero leakage from ``ConfCalib`` or ``Test``).
    """
    calib_use = _filter_candidate_inventory(prob_calib)
    X_cal = calib_use[feature_cols]  # NO fillna
    y_cal = np.asarray(calib_use["label"].astype(int).values, dtype=int)
    raw_probs = np.asarray(model.predict_proba(X_cal)[:, 1], dtype=float)

    iso_full = IsotonicRegression(out_of_bounds="clip").fit(raw_probs, y_cal)
    sig_full = PlattCalibrator().fit(raw_probs, y_cal)

    commits = (
        np.asarray(calib_use["commit_idx"].values)
        if "commit_idx" in calib_use.columns
        else np.arange(len(calib_use))
    )
    unique_c = np.sort(np.unique(commits))
    pos_commits = [cid for cid in unique_c if (y_cal[commits == cid] == 1).any()]
    neg_commits = [cid for cid in unique_c if not (y_cal[commits == cid] == 1).any()]
    n_pos_commits = len(pos_commits)

    k_eff = min(int(n_splits), n_pos_commits)
    cv_brier_base = compute_brier_score(y_cal, np.full(len(y_cal), float(train_prior), dtype=float))
    cv_brier_iso = float("nan")
    cv_brier_sig = float("nan")
    cv_bss_iso = float("nan")
    cv_bss_sig = float("nan")
    selected_name = "sigmoid"

    if k_eff >= 2 and len(unique_c) >= k_eff:
        fold_map: Dict[Any, int] = {}
        for idx, cid in enumerate(pos_commits):
            fold_map[cid] = idx % k_eff
        for idx, cid in enumerate(neg_commits):
            fold_map[cid] = idx % k_eff
        fold_ids = np.array([fold_map[cid] for cid in commits], dtype=int)

        oof_iso = np.full(len(y_cal), np.nan, dtype=float)
        oof_sig = np.full(len(y_cal), np.nan, dtype=float)
        valid_cv = True

        for f in range(k_eff):
            val_mask = fold_ids == f
            tr_mask = ~val_mask
            if not tr_mask.any() or not val_mask.any() or np.unique(y_cal[tr_mask]).size < 2:
                valid_cv = False
                break
            iso_f = IsotonicRegression(out_of_bounds="clip").fit(raw_probs[tr_mask], y_cal[tr_mask])
            sig_f = PlattCalibrator().fit(raw_probs[tr_mask], y_cal[tr_mask])
            oof_iso[val_mask] = iso_f.predict(raw_probs[val_mask])
            oof_sig[val_mask] = sig_f.predict(raw_probs[val_mask])

        if valid_cv and not (np.isnan(oof_iso).any() or np.isnan(oof_sig).any()):
            cv_brier_iso = compute_brier_score(y_cal, oof_iso)
            cv_brier_sig = compute_brier_score(y_cal, oof_sig)
            cv_bss_iso = compute_brier_skill_score(cv_brier_iso, cv_brier_base)
            cv_bss_sig = compute_brier_skill_score(cv_brier_sig, cv_brier_base)
            selected_name = "sigmoid" if cv_brier_sig <= cv_brier_iso else "isotonic"

    return {
        "iso": iso_full,
        "sigmoid": sig_full,
        "selected_name": selected_name,
        "selected_calibrator": sig_full if selected_name == "sigmoid" else iso_full,
        "prob_cv_brier_iso": cv_brier_iso,
        "prob_cv_brier_sigmoid": cv_brier_sig,
        "prob_cv_brier_baseline": cv_brier_base,
        "prob_cv_bss_iso": cv_bss_iso,
        "prob_cv_bss_sigmoid": cv_bss_sig,
        "n_pos_commits_prob": int(n_pos_commits),
    }


def predict_raw_and_calibrated(
    model,
    calibrator,
    df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(raw_probs, calibrated_probs)`` for all rows in df."""
    if len(df) == 0:
        empty = np.array([], dtype=float)
        return empty, empty
    X = df[feature_cols]  # NO fillna
    raw = np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    cal = np.asarray(calibrator.predict(raw), dtype=float)
    return raw, cal


def predict_calibrated(
    model,
    calibrator,
    df: pd.DataFrame,
    feature_cols: List[str],
) -> np.ndarray:
    """Return calibrated probabilities for all rows in df."""
    _, cal = predict_raw_and_calibrated(model, calibrator, df, feature_cols)
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


def compute_brier_skill_score(brier_model: float, brier_baseline: float) -> float:
    """Brier Skill Score: ``BSS = 1 - BS_model / BS_baseline``.

    ``BSS > 0`` improves over constant prevalence, ``0`` matches baseline,
    and ``< 0`` is worse than constant baseline.
    """
    bm = float(brier_model)
    bb = float(brier_baseline)
    if not np.isfinite(bm) or not np.isfinite(bb) or bb <= 0.0:
        return float("nan")
    return float(1.0 - (bm / bb))
