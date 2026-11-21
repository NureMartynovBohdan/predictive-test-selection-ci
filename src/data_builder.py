"""
src/data_builder.py — Legacy compatibility layer & dataset loader
==================================================================
Delegates to CIHistorySimulator (the temporal data layer).

Provides backward-compatible functions for legacy scripts while ensuring
that all underlying data extraction follows strict temporal ordering
and anti-leakage invariants.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from simulator import CIHistorySimulator, ParentStatus, TestRun
from features import build_features_batch

logger = logging.getLogger(__name__)

TOP_PROJECTS = [
    "apache@sling",
    "square@okhttp",
    "SonarSource@sonarqube",
    "l0rdn1kk0n@wicket-bootstrap",
    "CloudifySource@cloudify",
    "deeplearning4j@deeplearning4j",
    "jOOQ@jOOQ",
    "facebook@buck",
]

BEST_PROJECTS = [
    "apache@sling",
    "square@okhttp",
    "SonarSource@sonarqube",
    "l0rdn1kk0n@wicket-bootstrap",
    "CloudifySource@cloudify",
]


def load_project(
    proj: str,
    rtp_dir: str | Path,
    job_to_commit: Optional[dict] = None,
    label_policy: str = "confirmed_new_fail",
) -> Optional[pd.DataFrame]:
    """Load project data via CIHistorySimulator.

    Returns a DataFrame with columns:
        commit, testName, commit_idx, label, actual_duration,
        current_status, parent_status, label_status,
        mixed_outcomes_same_sha, n_pass, n_fail, n_attempts
    """
    rtp_dir = Path(rtp_dir)
    try:
        sim = CIHistorySimulator(rtp_dir, proj, label_policy=label_policy)
    except Exception as e:
        logger.warning("Failed to initialize simulator for %s: %s", proj, e)
        return None

    rows = []
    for sha in sim.iter_decision_points():
        c_idx = sim.get_commit_order(sha)
        gt = sim.reveal_ground_truth(sha)
        for tname, tr in gt.test_results.items():
            rows.append({
                "commit": sha,
                "commit_idx": c_idx,
                "testName": tname,
                "label": tr.label,
                "actual_duration": tr.actual_duration,
                "duration_mean": tr.actual_duration,  # legacy alias
                "current_status": tr.current_status,
                "parent_status": tr.parent_status,
                "label_status": tr.label_status,
                "mixed_outcomes_same_sha": tr.mixed_outcomes_same_sha,
                "intra_commit_flaky": tr.mixed_outcomes_same_sha,  # legacy alias
                "commit_failed_strict": tr.current_status == "fail",
                "any_fail": tr.n_fail > 0,
                "n_pass": tr.n_pass,
                "n_fail": tr.n_fail,
                "n_attempts": tr.n_attempts,
                "project": proj,
            })

    if not rows:
        return None
    return pd.DataFrame(rows)


def load_patches(proj: str, rtp_dir: str | Path) -> Dict[str, List[str]]:
    """Returns {commit_sha: [changed_file_path, ...]} for a project."""
    rtp_dir = Path(rtp_dir)
    patches_csv = rtp_dir / proj / f"{proj}-patches.csv"
    if not patches_csv.exists():
        return {}
    df = pd.read_csv(patches_csv, usecols=["sha", "name"])
    df = df.dropna(subset=["sha", "name"])
    return df.groupby("sha")["name"].apply(list).to_dict()


def load_all_projects(
    projects: List[str],
    rtp_dir: str | Path,
    job_to_commit: Optional[dict] = None,
    label_policy: str = "confirmed_new_fail",
) -> Dict[str, pd.DataFrame]:
    """Load all listed projects, returning {proj_name: DataFrame}."""
    datasets: Dict[str, pd.DataFrame] = {}
    for proj in projects:
        df = load_project(proj, rtp_dir, job_to_commit, label_policy=label_policy)
        if df is not None:
            datasets[proj] = df
            defective = df["label"].sum()
            total_commits = df["commit"].nunique()
            print(
                f"  [{proj}] commits={total_commits}  "
                f"(commit,test) pairs={len(df)}  "
                f"label=1 pairs={defective}"
            )
    return datasets
