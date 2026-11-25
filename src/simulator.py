"""
src/simulator.py - CIHistorySimulator (Temporal Data Layer v3)
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
4. `get_candidate_tests(commit_sha)` builds the pre-execution test inventory
   STRICTLY from information available before running CI on `commit_sha`
   (external manifest if supplied, prior CI runs with job_order < as_of_order,
   and test classes discovered in `commit_sha`'s changed files). It NEVER reads
   `self._runs_by_commit[commit_sha]`.
5. `get_decision_context` never exposes labels or actual_duration.
6. `get_test_history` returns only runs strictly before `as_of_order`.
7. `reveal_ground_truth` is called only AFTER the selector has made
   its decision (or for completed past steps during training).
8. Three-state labeling under `confirmed_new_fail`:
     - label = 1    : confirmed new regression (parent='pass' -> current='fail')
     - label = 0    : confirmed non-regression (current='pass' with non-mixed
                      parent, or parent='fail' -> current='fail')
     - label = None : ambiguous / unknown outcome ('ambiguous_current',
                      'parent_mixed', 'unknown_parent'); excluded from binary
                      training and CMR evaluation rather than coerced to 0.
9. Diagnostic fields: commit_sha, parent_sha, parent_status, current_status,
   label_status, ambiguous_outcome_same_sha (alias: mixed_outcomes_same_sha),
   parent_source ('git_parent' | 'chronological_fallback' | 'none').
10. Sensitivity analysis support:
    label_policy in ('confirmed_new_fail', 'strict_all_fail', 'any_fail').
"""

from __future__ import annotations

from bisect import bisect_left
import logging
import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Global cache for job->sha to avoid re-reading 295MB file multiple times
_GLOBAL_JOB_TO_SHA_CACHE: Optional[Dict[int, str]] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_test_class_from_path(path: str) -> Optional[str]:
    """Extract a Java test class name from a changed file path, if it looks
    like a test file (e.g. *Test.java, *Tests.java, *TestCase.java, *IT.java,
    *Spec.java, Test*.java).
    """
    norm = path.replace("\\", "/")
    if not norm.lower().endswith(".java"):
        return None
    stem = Path(norm).stem
    is_test_name = (
        stem.endswith("Test")
        or stem.endswith("Tests")
        or stem.endswith("TestCase")
        or stem.endswith("IT")
        or stem.endswith("Spec")
        or stem == "Test"
        or (stem.startswith("Test") and len(stem) > 4 and (stem[4].isupper() or stem[4] == "_"))
    )
    if not is_test_name:
        return None
    idx = norm.find("/java/")
    if idx >= 0:
        rel = norm[idx + 6:]
    else:
        rel = norm.lstrip("/")
    rel = re.sub(r"\.java$", "", rel, flags=re.I)
    return rel.replace("/", ".")


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
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class DecisionContext:
    """Information known BEFORE running tests on a commit.

    Critically, this object contains NO labels, NO ``actual_duration``,
    and NO dependency on the current commit's CI execution results.
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
    label: Optional[int]          # 1 = confirmed regression, 0 = non-regression,
                                  # None = ambiguous / unknown (excluded from binary eval)
    actual_duration: float
    current_status: str           # 'pass' | 'fail' | 'mixed'
    parent_status: str            # 'pass' | 'fail' | 'mixed' | 'unknown'
    label_status: str             # 'confirmed_new_fail' | 'no_new_fail'
                                  # | 'already_failing' | 'unknown_parent'
                                  # | 'parent_mixed' | 'ambiguous_current'
    n_pass: int
    n_fail: int
    n_attempts: int
    mixed_outcomes_same_sha: bool
    parent_sha: Optional[str] = None
    commit_sha: Optional[str] = None
    parent_source: str = "none"   # 'git_parent' | 'chronological_fallback' | 'none'

    @property
    def ambiguous_outcome_same_sha(self) -> bool:
        """Primary descriptive alias for mixed outcomes within the same SHA
        (without claiming proven flakiness, as OS/JDK environment metadata are absent).
        """
        return self.mixed_outcomes_same_sha

    @property
    def label_attribution(self) -> str:
        """Alias for backward compatibility."""
        return self.label_status


@dataclass
class GroundTruth:
    """Full ground-truth for a commit - available only after decision."""
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
          - "confirmed_new_fail": three-state strict pass -> fail transition
            (ambiguous / unknown outcomes receive label=None)
          - "strict_all_fail": binary sensitivity policy (1 iff current == 'fail')
          - "any_fail": binary sensitivity policy (1 iff n_fail > 0)
    inventory_window : int
        Number of preceding commits strictly before ``as_of_order`` used to
        reconstruct the pre-execution active test suite (in addition to
        ``parent_sha`` and changed test files in the diff).
    test_inventory : dict[str, list[str]] | None
        Optional explicit pre-execution test manifest per commit_sha.
    """

    def __init__(
        self,
        rtp_dir: str | Path,
        project: str,
        label_policy: str = "confirmed_new_fail",
        inventory_window: int = 10,
        test_inventory: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.rtp_dir = Path(rtp_dir)
        self.project = project
        self.label_policy = label_policy
        self.inventory_window = max(1, int(inventory_window))
        self._explicit_inventory: Dict[str, List[str]] = (
            {k: sorted(set(v)) for k, v in test_inventory.items()}
            if test_inventory is not None else {}
        )

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
        self._test_job_orders: Dict[str, List[int]] = {}
        self._commit_test_min_job: Dict[str, Dict[str, int]] = {}
        for r in self._runs:
            self._runs_by_commit.setdefault(r.commit_sha, []).append(r)
            self._runs_by_test.setdefault(r.test_name, []).append(r)
            self._test_job_orders.setdefault(r.test_name, []).append(r.job_order)
            tmap = self._commit_test_min_job.setdefault(r.commit_sha, {})
            if r.test_name not in tmap or r.job_order < tmap[r.test_name]:
                tmap[r.test_name] = r.job_order

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
        # root commits in Git (parent is None) receive parent_source='none';
        # commits absent from bare Git repo use chronological_fallback.
        self._parent_sha: Dict[str, Optional[str]] = {}
        self._parent_source: Dict[str, str] = {}  # 'git_parent' | 'chronological_fallback' | 'none'
        for idx, sha in enumerate(self._commit_list):
            if sha in self._git_parents:
                git_p = self._git_parents[sha]
                self._parent_sha[sha] = git_p
                self._parent_source[sha] = "git_parent" if git_p is not None else "none"
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
            sum(1 for s in self._commit_list if self._parent_source.get(s) == "git_parent"),
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
        preview = pd.read_csv(csv_path, nrows=2)
        cols = list(preview.columns)
        sha_col = "tr_original_commit" if "tr_original_commit" in cols else "git_commit_id"

        df = pd.read_csv(
            csv_path,
            usecols=["tr_job_id", sha_col],
            dtype={"tr_job_id": "int64", sha_col: "str"},
        )
        df = df.dropna(subset=[sha_col])
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

    def get_candidate_tests(self, commit_sha: str) -> List[str]:
        """Return candidate tests known BEFORE running CI on ``commit_sha``.

        NEVER inspects ``self._runs_by_commit[commit_sha]`` or any run with
        ``job_order >= as_of_order``.

        Inventory sources (in order):
        1. Explicit ``test_inventory[commit_sha]`` if supplied at init.
        2. Active pre-execution inventory reconstructed strictly before
           ``as_of_order = get_commit_min_job(commit_sha)``:
           - tests observed on ``parent_sha`` with ``job_order < as_of_order``;
           - tests observed in the last ``inventory_window`` prior commits
             with ``job_order < as_of_order``;
           - test classes extracted from changed files in ``self._patches[commit_sha]``.
        """
        if commit_sha not in self._commit_order:
            raise KeyError(f"Unknown commit: {commit_sha}")

        if commit_sha in self._explicit_inventory:
            return list(self._explicit_inventory[commit_sha])

        as_of_order = self._commit_min_job[commit_sha]
        idx = self._commit_order[commit_sha]
        candidates: Set[str] = set()

        # 1. Preceding commits within inventory_window (strictly before commit_sha)
        prior_shas: List[str] = list(
            self._commit_list[max(0, idx - self.inventory_window):idx]
        )
        parent_sha = self._parent_sha.get(commit_sha)
        if parent_sha is not None and parent_sha != commit_sha and parent_sha not in prior_shas:
            prior_shas.append(parent_sha)

        for p_sha in prior_shas:
            tmap = self._commit_test_min_job.get(p_sha)
            if not tmap:
                continue
            for t_name, min_ord in tmap.items():
                if min_ord < as_of_order:
                    candidates.add(t_name)

        # 2. Newly added / modified test classes discovered from current patch diff
        for fpath in self._patches.get(commit_sha, []):
            cls_name = _extract_test_class_from_path(fpath)
            if cls_name:
                candidates.add(cls_name)

        return sorted(candidates)

    def get_decision_context(self, commit_sha: str) -> DecisionContext:
        """Return pre-test information for a commit.

        Contains NO labels, NO actual_duration, and builds ``candidate_tests``
        strictly from pre-execution information via ``get_candidate_tests``.
        """
        if commit_sha not in self._commit_order:
            raise KeyError(f"Unknown commit: {commit_sha}")

        candidate_tests = self.get_candidate_tests(commit_sha)
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
        all_runs = self._runs_by_test.get(test_name)
        if not all_runs:
            return []
        orders = self._test_job_orders[test_name]
        cut = bisect_left(orders, as_of_order)
        result = all_runs[:cut]
        self._validate_no_future_leak(test_name, as_of_order, result)
        return result

    def get_parent_test_status(
        self,
        parent_sha: Optional[str],
        test_name: str,
        as_of_order: int,
    ) -> ParentStatus:
        """Status of test_name on parent_sha, if known strictly before as_of_order."""
        if parent_sha is None:
            return ParentStatus.UNKNOWN

        parent_runs = self._runs_by_commit.get(parent_sha, [])
        relevant = [
            r for r in parent_runs
            if r.test_name == test_name and r.job_order < as_of_order
        ]

        if not relevant:
            return ParentStatus.UNKNOWN

        p_pass = sum(1 for r in relevant if r.passed)
        p_fail = sum(1 for r in relevant if r.failed)
        if p_fail == 0:
            return ParentStatus.PASS
        if p_pass == 0:
            return ParentStatus.FAIL
        return ParentStatus.MIXED

    def reveal_ground_truth(
        self,
        commit_sha: str,
        policy: Optional[str] = None,
    ) -> GroundTruth:
        """Return labels and actual durations for commit_sha.

        Called ONLY after the selector has made its choice.

        Rules:
        - If policy == "confirmed_new_fail" (three-state):
            * current == 'mixed'                    -> label=None ('ambiguous_current')
            * parent  == 'mixed'                    -> label=None ('parent_mixed')
            * current == 'fail' and parent == 'pass'-> label=1    ('confirmed_new_fail')
            * current == 'pass'                     -> label=0    ('no_new_fail')
            * current == 'fail' and parent == 'fail'-> label=0    ('already_failing')
            * current == 'fail' and parent == 'unknown' -> label=None ('unknown_parent')
        - If policy == "strict_all_fail" (binary sensitivity):
            * label = 1 iff current == 'fail' else 0
        - If policy == "any_fail" (binary sensitivity):
            * label = 1 iff n_fail > 0 else 0
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

        parent_by_test: Dict[str, List[TestRun]] = {}
        if parent_sha is not None:
            for r in self._runs_by_commit.get(parent_sha, []):
                if r.job_order < as_of_order:
                    parent_by_test.setdefault(r.test_name, []).append(r)

        results: Dict[str, TestResult] = {}
        p_source = self._parent_source.get(commit_sha, "none")

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

            # Parent status strictly before as_of_order (detects mixed outcomes)
            if parent_sha is None:
                p_status_str = "unknown"
            else:
                parent_runs_all = parent_by_test.get(test_name)
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

            # Label status attribution (three-state aware)
            if current_status == "mixed":
                label_status = "ambiguous_current"
            elif p_status_str == "mixed":
                label_status = "parent_mixed"
            elif current_status == "fail" and p_status_str == "pass":
                label_status = "confirmed_new_fail"
            elif current_status == "pass":
                label_status = "no_new_fail"
            elif current_status == "fail" and p_status_str == "fail":
                label_status = "already_failing"
            else:
                label_status = "unknown_parent"

            # Policy-based label assignment
            label: Optional[int]
            if active_policy == "confirmed_new_fail":
                if label_status == "confirmed_new_fail":
                    label = 1
                elif label_status in ("no_new_fail", "already_failing"):
                    label = 0
                else:
                    # ambiguous_current, parent_mixed, unknown_parent -> None
                    label = None
            elif active_policy == "strict_all_fail":
                label = 1 if current_status == "fail" else 0
            elif active_policy == "any_fail":
                label = 1 if n_fail > 0 else 0
            else:
                raise ValueError(f"Unknown policy: {active_policy}")

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
        """Return the minimum job_id for commit_sha (used as temporal cut-off)."""
        return self._commit_min_job[commit_sha]

    def get_changed_files(self, commit_sha: str) -> List[str]:
        """Return list of changed files for commit_sha."""
        return self._patches.get(commit_sha, [])

    def get_parent_sha(self, commit_sha: str) -> Optional[str]:
        """Return resolved parent SHA for commit_sha (or None)."""
        return self._parent_sha.get(commit_sha)

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
        n_ambiguous = 0
        n_parent_mixed = 0
        for sha in self._commit_list:
            gt = self.reveal_ground_truth(sha)
            vals = gt.test_results.values()
            if any(tr.label_status == "confirmed_new_fail" for tr in vals):
                n_new_fail_commits += 1
            if any(tr.label_status == "unknown_parent" for tr in vals):
                n_unknown_parent += 1
            if any(tr.label_status == "ambiguous_current" for tr in vals):
                n_ambiguous += 1
            if any(tr.label_status == "parent_mixed" for tr in vals):
                n_parent_mixed += 1

        return {
            "project": self.project,
            "n_commits": n_commits,
            "n_runs": n_runs,
            "n_unique_tests": n_tests,
            "n_confirmed_new_fail_commits": n_new_fail_commits,
            "n_commits_with_unknown_parent": n_unknown_parent,
            "n_commits_with_ambiguous_current": n_ambiguous,
            "n_commits_with_parent_mixed": n_parent_mixed,
        }