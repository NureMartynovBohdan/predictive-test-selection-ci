"""
src/data_builder.py
===================
Loads and preprocesses RTPTorrent data for the conformal test-selection pipeline.

Outputs per-project a DataFrame of (commit, testName) pairs with:
  - commit chronological index
  - binary label y (1 = genuine pass->fail transition)
  - commit_failed_strict (bool): test stably failed on this commit (all jobs)
  - intra_commit_flaky   (bool): mixed results across jobs on same commit
  - duration             (float): mean duration across all jobs
  - changed_files        (list): file paths from patches.csv
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Project catalogue
# ---------------------------------------------------------------------------
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

# Top-5 by strict-defective count (from RESULT_filter_defective_commits.txt)
BEST_PROJECTS = [
    "apache@sling",           # 342 strict-defective commits
    "square@okhttp",          # 159
    "SonarSource@sonarqube",  # 131
    "l0rdn1kk0n@wicket-bootstrap",  # 120
    "CloudifySource@cloudify",       # 115
]


def load_project(
    proj: str,
    rtp_dir: str | Path,
    job_to_commit: dict[int, str],
) -> Optional[pd.DataFrame]:
    """
    Load and label (commit, testName) pairs for one project.

    Returns a DataFrame with columns:
        commit, testName, commit_idx, commit_failed_strict,
        intra_commit_flaky, any_fail, prev_strict_fail,
        label (y), duration_mean

    Returns None if data files are missing.
    """
    rtp_dir = Path(rtp_dir)
    proj_dir = rtp_dir / proj
    main_csv = proj_dir / f"{proj}.csv"
    patches_csv = proj_dir / f"{proj}-patches.csv"

    if not main_csv.exists() or not patches_csv.exists():
        print(f"[WARN] {proj}: missing CSV files, skipping.")
        return None

    # --- Load test-run log ---
    df = pd.read_csv(
        main_csv,
        usecols=["travisJobId", "testName", "duration", "failures", "errors"],
        dtype={"travisJobId": "int64", "duration": "float64",
               "failures": "int64", "errors": "int64"},
    )
    df["failed"] = (df["failures"] + df["errors"]) > 0
    df["commit"] = df["travisJobId"].map(job_to_commit)

    # --- Load changed-file manifest ---
    df_patches = pd.read_csv(patches_csv, usecols=["sha", "name"])
    valid_shas = set(df_patches["sha"].dropna().unique())

    # Keep only commits that have both test results AND a patch diff
    df = df[df["commit"].isin(valid_shas)].copy()
    if df.empty:
        print(f"[WARN] {proj}: no overlap between test log and patches.")
        return None

    # --- Chronological commit order ---
    commit_order = (
        df.groupby("commit")["travisJobId"].min()
        .sort_values()
        .index.tolist()
    )
    commit_rank = {c: i for i, c in enumerate(commit_order)}
    df["commit_idx"] = df["commit"].map(commit_rank)

    # --- Aggregate to (commit, testName) level ---
    # any_fail = True  if test failed in ANY job for this commit
    # all_fail = True  if test failed in ALL jobs for this commit
    agg = (
        df.groupby(["commit_idx", "commit", "testName"])
        .agg(
            any_fail=("failed", "max"),
            all_fail=("failed", "min"),
            duration_mean=("duration", "mean"),
        )
        .reset_index()
    )

    # Intra-commit flaky: different outcomes across matrix jobs
    agg["intra_commit_flaky"] = agg["any_fail"] & (~agg["all_fail"])
    # Strict (non-flaky) failure on this commit
    agg["commit_failed_strict"] = agg["all_fail"]

    # --- Pass→Fail transition label ---
    agg = agg.sort_values(["testName", "commit_idx"])
    agg["prev_strict_fail"] = (
        agg.groupby("testName")["commit_failed_strict"]
        .shift(1, fill_value=False)
    )

    # y = 1  <=>  stably failed NOW  AND  did NOT fail on previous appearance
    agg["label"] = (agg["commit_failed_strict"] & ~agg["prev_strict_fail"]).astype(int)
    agg["project"] = proj

    return agg.reset_index(drop=True)


def load_all_projects(
    projects: list[str],
    rtp_dir: str | Path,
    job_to_commit: Optional[dict] = None,
) -> dict[str, pd.DataFrame]:
    """
    Load all listed projects, returning {proj_name: DataFrame}.
    """
    rtp_dir = Path(rtp_dir)
    commits_csv = rtp_dir / "tr_all_built_commits.csv"

    if job_to_commit is None:
        df_commits = pd.read_csv(commits_csv, dtype={"tr_job_id": "int64"})
        job_to_commit = dict(
            zip(df_commits["tr_job_id"], df_commits["git_commit_id"])
        )

    datasets: dict[str, pd.DataFrame] = {}
    for proj in projects:
        df = load_project(proj, rtp_dir, job_to_commit)
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


def load_patches(proj: str, rtp_dir: str | Path) -> dict[str, list[str]]:
    """
    Returns {commit_sha: [changed_file_path, ...]} for a project.
    """
    rtp_dir = Path(rtp_dir)
    patches_csv = rtp_dir / proj / f"{proj}-patches.csv"
    df = pd.read_csv(patches_csv, usecols=["sha", "name"])
    df = df.dropna(subset=["sha", "name"])
    return df.groupby("sha")["name"].apply(list).to_dict()


if __name__ == "__main__":
    BASE = Path(__file__).parent.parent
    RTP_DIR = BASE / "rtp-torrent"

    print("Loading top-5 projects …")
    datasets = load_all_projects(BEST_PROJECTS, RTP_DIR)

    for proj, df in datasets.items():
        defective_commits = df[df["label"] == 1]["commit"].nunique()
        print(f"  {proj}: {defective_commits} defective commits (strict pass→fail)")
