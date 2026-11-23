"""
src/simulator.py — CIHistorySimulator (Temporal Data Layer v2)
=============================================================
The ONLY interface between raw RTPTorrent data and all downstream
components (features, models, calibration, evaluation).

Key Architecture:
1. No module outside this one may directly read raw aggregated tables
   with labels or future durations.
2. Each record is a single *attempt* (job_id, commit_sha, parent_sha,
   test_name, passed/failed, duration, monotonic job_order).
3. Ground-truth parents extracted from bare Git repos (rtp-torrent/repo/{project})
   via `git rev-list --parents --all` (with C^1 first parent for merge commits).
4. `get_decision_context` never exposes labels or actual_duration.
5. `get_test_history` returns only runs strictly before `as_of_order`.
6. `reveal_ground_truth` is called only AFTER the selector has made
   its decision (or for completed past steps during training).
7. Diagnostic fields: commit_sha, parent_sha, parent_status, current_status,
   label_status (confirmed_new_fail, no_new_fail, already_failing, unknown_parent,
   parent_mixed), parent_source (git_parent | chronological_fallback | none).
8. Sensitivity analysis support: label_policy in ('confirmed_new_fail', 'strict_all_fail', 'any_fail').
9. parent_status can now be 'pass' | 'fail' | 'mixed' | 'unknown'.
   parent=mixed -> current=fail produces label_status='parent_mixed' (label=0 in
   confirmed_new_fail), not 'already_failing', so audits can distinguish old breaks
   from ambiguous mixed-parent situations.
"""

from __future__ import annotations

import logging
import subprocess
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Global cache for job->sha to avoid re-reading 295MB file multiple times
_GLOBAL_JOB_TO_SHA_CACHE: Optional[Dict[int, str]] = None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TestRun:
    """A single test-execution attempt."""
    job_id: int
    commit_sha: str
    test_name: str
    passed: bool
    failed: bool
    duration: float
    job_order: int          # monotonic ordering proxy for time
    parent_sha: Optional[str] = None


class ParentStatus(Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


@dataclass
class DecisionContext:
    """Information known BEFORE running tests on a commit.

    Critically, this object contains NO labels and NO
    `actual_duration` for the current commit's tests.
    """
    commit_sha: str
    parent_sha: Optional[str]
    changed_files: List[str]
    n_changed_files: int
    candidate_tests: List[str]
    commit_order: int


@dataclass
class TestResult:
    """Ground-truth outcome for one (commit, test) pair."""
    label: int                    # 1 = target failure by policy
    actual_duration: float
    current_status: str           # 'pass' | 'fail' | 'mixed'
    parent_status: str            # 'pass' | 'fail' | 'mixed' | 'unknown'
    label_status: str             # 'confirmed_new_fail' | 'no_new_fail'
                                  # | 'already_failing' | 'unknown_parent'
                                  # | 'parent_mixed'
    n_pass: int
    n_fail: int
    n_attempts: int
    mixed_outcomes_same_sha: bool
    parent_sha: Optional[str] = None
    commit_sha: Optional[str] = None
    parent_source: str = "none"   # 'git_parent' | 'chronological_fallback' | 'none'

    @property
    def label_attribution(self) -> str:
        """Alias for backward compatibility."""
        return self.label_status


@dataclass
class GroundTruth:
    """Full ground-truth for a commit — available only after decision."""
    commit_sha: str
    test_results: Dict[str, TestResult]


# ---------------------------------------------------------------------------
# CIHistorySimulator
# ---------------------------------------------------------------------------

class CIHistorySimulator:
    """Temporal data layer for one RTPTorrent project.

    Parameters
    ----------
    rtp_dir : str | Path
        Root of the extracted RTPTorrent dataset.
    project : str
        Project identifier, e.g. "apache@sling".
    label_policy : str
        Labeling policy:
          - "confirmed_new_fail": strict pass -> fail transition by parent
          - "strict_all_fail": all runs on current commit failed
          - "any_fail": at least one run on current commit failed
    """

    def __init__(
        self,
        rtp_dir: str | Path,
        project: str,
        label_policy: str = "confirmed_new_fail",
    ) -> None:
        self.rtp_dir = Path(rtp_dir)
        self.project = project
        self.label_policy = label_policy

        # --- 1. Load patches (changed files per SHA) ---
        self._patches: Dict[str, List[str]] = self._load_patches()

        # --- 2. Extract TRUE parents from bare Git repo (if available) ---
        self._git_parents: Dict[str, Optional[str]] = self._load_git_parents()

        # --- 3. Job -> SHA mapping (many-to-many safe) ---
        self._job_to_sha = self._load_job_to_sha()

        # --- 4. Load raw per-attempt test records ---
        self._runs: List[TestRun] = []
        self._load_test_runs()

        # --- 5. Build indexes ---
        self._runs_by_commit: Dict[str, List[TestRun]] = {}
        self._runs_by_test: Dict[str, List[TestRun]] = {}
        for r in self._runs:
            self._runs_by_commit.setdefault(r.commit_sha, []).append(r)
            self._runs_by_test.setdefault(r.test_name, []).append(r)

        # --- 6. Chronological commit order (by min job_id) ---
        commit_min_job: Dict[str, int] = {}
        for r in self._runs:
            if r.commit_sha not in commit_min_job or r.job_id < commit_min_job[r.commit_sha]:
                commit_min_job[r.commit_sha] = r.job_id

        # Keep only commits that also have patch data
        valid_shas = set(self._patches.keys()) & set(commit_min_job.keys())
        sorted_commits = sorted(valid_shas, key=lambda s: commit_min_job[s])

        self._commit_list: List[str] = sorted_commits
        self._commit_order: Dict[str, int] = {
            sha: idx for idx, sha in enumerate(sorted_commits)
        }
        self._commit_min_job: Dict[str, int] = {
            sha: commit_min_job[sha] for sha in sorted_commits
        }

        # --- 7. Resolve Parent SHA ---
        # Prioritize true Git parent (first parent C^1 for merge commits);
        # fallback to chronological predecessor if absent from git repo.
        self._parent_sha: Dict[str, Optional[str]] = {}
        self._parent_source: Dict[str, str] = {}  # 'git_parent' | 'chronological_fallback' | 'none'
        for idx, sha in enumerate(self._commit_list):
            if sha in self._git_parents:
                self._parent_sha[sha] = self._git_parents[sha]
                self._parent_source[sha] = "git_parent"
            elif idx > 0:
                self._parent_sha[sha] = self._commit_list[idx - 1]
                self._parent_source[sha] = "chronological_fallback"
            else:
                self._parent_sha[sha] = None
                self._parent_source[sha] = "none"

        # Update parent_sha in test runs
        for r in self._runs:
            r.parent_sha = self._parent_sha.get(r.commit_sha)

        logger.info(
            "%s: %d commits, %d test runs, %d unique tests (Git parents: %d/%d)",
            project, len(self._commit_list), len(self._runs),
            len(self._runs_by_test),
            sum(1 for s in self._commit_list if s in self._git_parents),
            len(self._commit_list),
        )

    # ------------------------------------------------------ internal loaders

    def _load_git_parents(self) -> Dict[str, Optional[str]]:
        """Extract real Git parents using `git rev-list --parents --all`.

        For merge commits, the first parent in `--parents` is C^1.
        """
        repo_dir = self.rtp_dir / "repo" / self.project
        parents: Dict[str, Optional[str]] = {}
        if repo_dir.exists():
            try:
                res = subprocess.run(
                    ["git", f"--git-dir={repo_dir}", "rev-list", "--parents", "--all"],
                    capture_output=True, text=True, check=True
                )
                for line in res.stdout.strip().split("\n"):
                    if not line:
                        continue
                    parts = line.split()
                    commit = parts[0]
                    # First parent C^1
                    p1 = parts[1] if len(parts) > 1 else None
                    parents[commit] = p1
                logger.info("%s: loaded %d git parent mappings from bare repo", self.project, len(parents))
            except Exception as e:
                logger.warning("%s: could not extract git parents from bare repo: %s", self.project, e)
        return parents

    def _load_job_to_sha(self) -> Dict[int, str]:
        """Load job->SHA mapping, handling duplicates safely and caching."""
        global _GLOBAL_JOB_TO_SHA_CACHE
        if _GLOBAL_JOB_TO_SHA_CACHE is not None:
            return _GLOBAL_JOB_TO_SHA_CACHE

        csv_path = self.rtp_dir / "tr_all_built_commits.csv"
        # Read column names first
        preview = pd.read_csv(csv_path, nrows=2)
        cols = list(preview.columns)
        sha_col = "tr_original_commit" if "tr_original_commit" in cols else "git_commit_id"

        df = pd.read_csv(
            csv_path,
            usecols=["tr_job_id", sha_col],
            dtype={"tr_job_id": "int64", sha_col: "str"},
        )
        df = df.dropna(subset=[sha_col])

        # Drop duplicates keeping first
        df = df.drop_duplicates(subset=["tr_job_id"], keep="first")
        mapping = dict(zip(df["tr_job_id"], df[sha_col]))
        _GLOBAL_JOB_TO_SHA_CACHE = mapping
        return mapping

    def _load_test_runs(self) -> None:
        """Load raw per-attempt test records."""
        proj_dir = self.rtp_dir / self.project
        main_csv = proj_dir / f"{self.project}.csv"
        if not main_csv.exists():
            raise FileNotFoundError(f"Test-run CSV not found: {main_csv}")

        df = pd.read_csv(
            main_csv,
            usecols=["travisJobId", "testName", "duration", "failures", "errors"],
            dtype={
                "travisJobId": "int64",
                "duration": "float64",
                "failures": "int64",
                "errors": "int64",
            },
        )
        df["commit_sha"] = df["travisJobId"].map(self._job_to_sha)
        df = df.dropna(subset=["commit_sha"])
        df["failed_flag"] = (df["failures"] + df["errors"]) > 0

        # Fast tuple iteration instead of slow df.iterrows()
        for row in df.itertuples(index=False):
            jid = int(row.travisJobId)
            failed = bool(row.failed_flag)
            dur = float(row.duration) if not pd.isna(row.duration) else 0.0
            self._runs.append(TestRun(
                job_id=jid,
                commit_sha=str(row.commit_sha),
                test_name=str(row.testName),
                passed=not failed,
                failed=failed,
                duration=dur,
                job_order=jid,
            ))

        self._runs.sort(key=lambda r: r.job_order)

    def _load_patches(self) -> Dict[str, List[str]]:
        """Load changed-file manifest per SHA."""
        proj_dir = self.rtp_dir / self.project
        patches_csv = proj_dir / f"{self.project}-patches.csv"
        if not patches_csv.exists():
            logger.warning("%s: patches CSV not found", self.project)
            return {}
        df = pd.read_csv(patches_csv, usecols=["sha", "name"])
        df = df.dropna(subset=["sha", "name"])
        return df.groupby("sha")["name"].apply(list).to_dict()

    # --------------------------------------------------- internal validation

    def _validate_no_future_leak(
        self,
        test_name: str,
        as_of_order: int,
        returned_runs: List[TestRun],
    ) -> None:
        """Assert that no returned run has job_order >= as_of_order."""
        for r in returned_runs:
            assert r.job_order < as_of_order, (
                f"Future leak: test={test_name}, run job_order={r.job_order} "
                f">= as_of_order={as_of_order}"
            )

    # ---------------------------------------------------------- public API

    def iter_decision_points(self) -> Iterator[str]:
        """Yield commit SHAs in chronological order."""
        yield from self._commit_list

    def get_decision_context(self, commit_sha: str) -> DecisionContext:
        """Return pre-test information for a commit.

        Contains NO labels and NO actual_duration.
        """
        if commit_sha not in self._commit_order:
            raise KeyError(f"Unknown commit: {commit_sha}")

        runs = self._runs_by_commit.get(commit_sha, [])
        candidate_tests = sorted(set(r.test_name for r in runs))
        changed_files = self._patches.get(commit_sha, [])

        return DecisionContext(
            commit_sha=commit_sha,
            parent_sha=self._parent_sha.get(commit_sha),
            changed_files=changed_files,
            n_changed_files=len(changed_files),
            candidate_tests=candidate_tests,
            commit_order=self._commit_order[commit_sha],
        )

    def get_test_history(
        self,
        test_name: str,
        as_of_order: int,
    ) -> List[TestRun]:
        """Return runs of test_name strictly before as_of_order."""
        all_runs = self._runs_by_test.get(test_name, [])
        result = [r for r in all_runs if r.job_order < as_of_order]
        self._validate_no_future_leak(test_name, as_of_order, result)
        return result

    def get_parent_test_status(
        self,
        parent_sha: Optional[str],
        test_name: str,
        as_of_order: int,
    ) -> ParentStatus:
        """Status of test_name on parent_sha, if known before as_of_order."""
        if parent_sha is None:
            return ParentStatus.UNKNOWN

        parent_runs = self._runs_by_commit.get(parent_sha, [])
        relevant = [
            r for r in parent_runs
            if r.test_name == test_name and r.job_order < as_of_order
        ]

        if not relevant:
            return ParentStatus.UNKNOWN

        all_passed = all(r.passed for r in relevant)
        if all_passed:
            return ParentStatus.PASS
        return ParentStatus.FAIL

    def reveal_ground_truth(
        self,
        commit_sha: str,
        policy: Optional[str] = None,
    ) -> GroundTruth:
        """Return labels and actual durations for commit_sha.

        Called ONLY after the selector has made its choice.

        Rules:
        - If policy == "confirmed_new_fail":
            label = 1 iff current == 'fail' and parent == 'pass'
        - If policy == "strict_all_fail":
            label = 1 iff current == 'fail'
        - If policy == "any_fail":
            label = 1 iff n_fail > 0
        """
        active_policy = policy or self.label_policy

        if commit_sha not in self._commit_order:
            raise KeyError(f"Unknown commit: {commit_sha}")

        runs = self._runs_by_commit.get(commit_sha, [])
        parent_sha = self._parent_sha.get(commit_sha)
        as_of_order = self._commit_min_job[commit_sha]

        by_test: Dict[str, List[TestRun]] = {}
        for r in runs:
            by_test.setdefault(r.test_name, []).append(r)

        results: Dict[str, TestResult] = {}
        for test_name, test_runs in by_test.items():
            n_pass = sum(1 for r in test_runs if r.passed)
            n_fail = sum(1 for r in test_runs if r.failed)
            n_attempts = len(test_runs)
            mean_dur = float(np.mean([r.duration for r in test_runs]))

            # Current status
            if n_fail == n_attempts:
                current_status = "fail"
            elif n_pass == n_attempts:
                current_status = "pass"
            else:
                current_status = "mixed"

            mixed = current_status == "mixed"

            # Parent status (now detects mixed parent outcomes)
            if parent_sha is None:
                p_status_str = "unknown"
            else:
                parent_runs_all = [
                    r for r in self._runs_by_commit.get(parent_sha, [])
                    if r.test_name == test_name and r.job_order < as_of_order
                ]
                if not parent_runs_all:
                    p_status_str = "unknown"
                else:
                    p_pass = sum(1 for r in parent_runs_all if r.passed)
                    p_fail = sum(1 for r in parent_runs_all if r.failed)
                    if p_fail == 0:
                        p_status_str = "pass"
                    elif p_pass == 0:
                        p_status_str = "fail"
                    else:
                        p_status_str = "mixed"

            # Label status attribution
            if current_status == "fail" and p_status_str == "pass":
                label_status = "confirmed_new_fail"
            elif current_status == "fail" and p_status_str == "fail":
                label_status = "already_failing"
            elif current_status == "fail" and p_status_str == "mixed":
                label_status = "parent_mixed"
            elif current_status == "fail" and p_status_str == "unknown":
                label_status = "unknown_parent"
            else:
                label_status = "no_new_fail"

            # Policy-based label assignment
            if active_policy == "confirmed_new_fail":
                label = 1 if label_status == "confirmed_new_fail" else 0
            elif active_policy == "strict_all_fail":
                label = 1 if current_status == "fail" else 0
            elif active_policy == "any_fail":
                label = 1 if n_fail > 0 else 0
            else:
                raise ValueError(f"Unknown policy: {active_policy}")

            p_source = self._parent_source.get(commit_sha, "none")

            results[test_name] = TestResult(
                label=label,
                actual_duration=mean_dur,
                current_status=current_status,
                parent_status=p_status_str,
                label_status=label_status,
                n_pass=n_pass,
                n_fail=n_fail,
                n_attempts=n_attempts,
                mixed_outcomes_same_sha=mixed,
                parent_sha=parent_sha,
                commit_sha=commit_sha,
                parent_source=p_source,
            )

        return GroundTruth(commit_sha=commit_sha, test_results=results)

    # -------------------------------------------------------- convenience

    def get_commit_order(self, commit_sha: str) -> int:
        return self._commit_order[commit_sha]

    def get_commit_min_job(self, commit_sha: str) -> int:
        """Return the minimum job_id for commit_sha (used as temporal cut-off).

        Public alternative to accessing ``_commit_min_job`` directly.
        """
        return self._commit_min_job[commit_sha]

    def get_changed_files(self, commit_sha: str) -> List[str]:
        """Return list of changed files for commit_sha.

        Public alternative to accessing ``_patches`` directly.
        """
        return self._patches.get(commit_sha, [])

    def get_parent_source(self, commit_sha: str) -> str:
        """Return the source of the parent SHA for commit_sha.

        Returns 'git_parent', 'chronological_fallback', or 'none'.
        """
        return self._parent_source.get(commit_sha, "none")

    def get_all_commits(self) -> List[str]:
        return list(self._commit_list)

    def get_project_stats(self) -> dict:
        n_commits = len(self._commit_list)
        n_runs = len(self._runs)
        n_tests = len(self._runs_by_test)
        n_new_fail_commits = 0
        n_unknown_parent = 0
        for sha in self._commit_list:
            gt = self.reveal_ground_truth(sha)
            if any(tr.label_status == "confirmed_new_fail" for tr in gt.test_results.values()):
                n_new_fail_commits += 1
            if any(tr.label_status == "unknown_parent" for tr in gt.test_results.values()):
                n_unknown_parent += 1

        return {
            "project": self.project,
            "n_commits": n_commits,
            "n_runs": n_runs,
            "n_unique_tests": n_tests,
            "n_confirmed_new_fail_commits": n_new_fail_commits,
            "n_commits_with_unknown_parent": n_unknown_parent,
        }
