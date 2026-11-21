"""
src/features.py — Feature engineering (v2)
==========================================
Builds feature vectors for (commit, test) pairs using **only** the
``CIHistorySimulator`` API — no direct data access.

Feature groups (all computed from PAST data only — no data leakage):

Group 1 — Test history (multi-window):
  fail_rate_5 / 20 / 50, new_failure_rate_50, new_failure_support,
  last_fail_dist, last_observed_status, run_count_total,
  duration_median_20, duration_p90_20, duration_history_known

Group 2 — Parent status:
  parent_status_code, parent_result_known

Group 3 — Commit characteristics:
  n_changed_files, lines_added, lines_deleted,
  n_prod_files_changed, n_test_files_changed,
  has_java, has_xml, has_build_file

Group 4 — Test ↔ Commit coupling:
  test_file_changed, same_package_changed, same_module_changed,
  path_min_dist, path_max_overlap, co_fail_rate, co_fail_support
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Set, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from simulator import CIHistorySimulator, TestRun

# ---------------------------------------------------------------------------
# Helpers — path / name tokenisation
# ---------------------------------------------------------------------------

_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _split_camel(name: str) -> List[str]:
    """Split CamelCase: 'FooBarTest' → ['Foo', 'Bar', 'Test']."""
    return _CAMEL_RE.sub(" ", name).split()


def _tokenise_java_class(class_name: str) -> Set[str]:
    """Tokenise a Java class name into a set of lower-case tokens.

    ``'com.example.foo.FooBarTest'``
      → ``{'com', 'example', 'foo', 'bar', 'test'}``
    """
    class_name = re.sub(r"\.java$", "", class_name, flags=re.I)
    parts = re.split(r"[./\\]", class_name)
    tokens: list[str] = []
    for p in parts:
        tokens.extend(_split_camel(p))
    return set(t.lower() for t in tokens if t)


def _tokenise_path(path: str) -> Set[str]:
    """Tokenise a file path into lower-case tokens (with CamelCase split)."""
    # Remove extension
    stem = re.sub(r"\.[^./\\]+$", "", path)
    parts = re.split(r"[./\\]", stem)
    tokens: list[str] = []
    for p in parts:
        tokens.extend(_split_camel(p))
    return set(t.lower() for t in tokens if t)


def _jaccard_distance(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return 1.0 - inter / union


def _overlap_fraction(test_tokens: Set[str], file_tokens: Set[str]) -> float:
    if not test_tokens:
        return 0.0
    return len(test_tokens & file_tokens) / len(test_tokens)


def _java_package(class_name: str) -> str:
    """Extract Java package: 'com.example.Foo' → 'com.example'."""
    class_name = re.sub(r"\.java$", "", class_name, flags=re.I)
    parts = class_name.rsplit(".", 1)
    return parts[0] if len(parts) > 1 else ""


def _file_package(path: str) -> str:
    """Approximate Java package from file path."""
    # e.g. src/main/java/com/example/Foo.java → com/example
    path = path.replace("\\", "/")
    # Try to find after /java/
    idx = path.find("/java/")
    if idx >= 0:
        rest = path[idx + 6:]
    else:
        rest = path
    rest = re.sub(r"\.[^/]+$", "", rest)
    parts = rest.rsplit("/", 1)
    return parts[0] if len(parts) > 1 else ""


def _file_module(path: str) -> str:
    """Approximate module from path: first directory component."""
    path = path.replace("\\", "/").strip("/")
    return path.split("/")[0] if "/" in path else ""


def _is_test_file(path: str) -> bool:
    """Heuristic: file path likely contains a test."""
    lower = path.lower()
    return "test" in lower or "spec" in lower


# ---------------------------------------------------------------------------
# Main feature-engineering functions
# ---------------------------------------------------------------------------

def build_features(
    simulator: "CIHistorySimulator",
    commit_sha: str,
) -> pd.DataFrame:
    """Build feature vectors for all candidate tests on one commit.

    Uses ``simulator`` API exclusively — no direct data access.

    Returns a DataFrame with one row per (commit, test) containing
    feature columns.  Does **not** include ``label`` or
    ``actual_duration`` (those come from ``reveal_ground_truth``).
    """
    ctx = simulator.get_decision_context(commit_sha)
    as_of = simulator.get_commit_order(commit_sha)
    # Use min job_id of this commit as the temporal cut-off for history
    as_of_job = simulator._commit_min_job.get(commit_sha, as_of)

    changed_files = ctx.changed_files
    n_changed = ctx.n_changed_files

    # Pre-compute commit-level features (Group 3)
    has_java = int(any(f.endswith(".java") for f in changed_files))
    has_xml  = int(any(f.endswith(".xml") for f in changed_files))
    has_build = int(any(
        Path(f).name.lower() in {"pom.xml", "build.gradle", "build.gradle.kts", "build"}
        for f in changed_files
    ))
    n_test_files = sum(1 for f in changed_files if _is_test_file(f))
    n_prod_files = n_changed - n_test_files

    # Tokenise changed files (for coupling features)
    file_token_sets = [_tokenise_path(f) for f in changed_files]
    file_packages = [_file_package(f) for f in changed_files]
    file_modules  = set(_file_module(f) for f in changed_files)
    file_names_lower = set(
        Path(f).stem.lower() for f in changed_files
    )

    rows: list[dict] = []

    for test_name in ctx.candidate_tests:
        # --- History (Group 1) ---
        history = simulator.get_test_history(test_name, as_of_job)
        n_past = len(history)

        # Multi-window fail rates
        def _fail_rate(window: List) -> float:
            if not window:
                return 0.0
            return sum(1 for r in window if r.failed) / len(window)

        fail_rate_5 = _fail_rate(history[-5:]) if history else 0.0
        fail_rate_20 = _fail_rate(history[-20:]) if history else 0.0
        fail_rate_50 = _fail_rate(history[-50:]) if history else 0.0

        # New failure rate (transitions in past 50)
        window_50 = history[-50:]
        new_fail_count = 0
        for i in range(1, len(window_50)):
            if window_50[i].failed and window_50[i - 1].passed:
                new_fail_count += 1
        new_failure_rate_50 = new_fail_count / max(len(window_50) - 1, 1) if len(window_50) > 1 else 0.0
        new_failure_support = new_fail_count

        # Last fail distance
        last_fail_indices = [i for i, r in enumerate(history) if r.failed]
        if last_fail_indices:
            last_fail_order = history[last_fail_indices[-1]].job_order
            last_fail_dist = as_of_job - last_fail_order
        else:
            last_fail_dist = -1

        # Last observed status
        last_observed = history[-1].failed if history else -1
        last_observed_status = int(last_observed) if isinstance(last_observed, bool) else -1

        # Duration history
        if history:
            durs = [r.duration for r in history[-20:] if r.duration > 0]
            duration_median_20 = float(np.median(durs)) if durs else np.nan
            duration_p90_20 = float(np.percentile(durs, 90)) if durs else np.nan
            duration_history_known = 1
        else:
            duration_median_20 = np.nan
            duration_p90_20 = np.nan
            duration_history_known = 0

        # --- Parent status (Group 2) ---
        parent_status = simulator.get_parent_test_status(
            ctx.parent_sha, test_name, as_of_job
        )
        parent_status_code = {
            "pass": 0, "fail": 1, "unknown": -1
        }.get(parent_status.value, -1)
        parent_result_known = 1 if parent_status.value != "unknown" else 0

        # --- Coupling (Group 4) ---
        test_tokens = _tokenise_java_class(test_name)
        test_package = _java_package(test_name)
        test_class_lower = test_name.rsplit(".", 1)[-1].lower() if "." in test_name else test_name.lower()
        test_module = ""  # Will check against file modules

        # test_file_changed: exact test class name in changed files
        test_file_changed = int(test_class_lower in file_names_lower)

        # same_package_changed
        same_package = int(any(
            _file_package(f).replace("/", ".") == test_package
            or _file_package(f) == test_package.replace(".", "/")
            for f in changed_files
        )) if test_package and changed_files else 0

        # same_module_changed
        same_module = 0
        # Try to infer test module from class name or history
        if file_modules and changed_files:
            # Heuristic: check if any token from first package segment matches a file module
            pkg_parts = test_package.split(".") if test_package else []
            if pkg_parts:
                same_module = int(any(p.lower() in file_modules for p in pkg_parts[:2]))

        # Jaccard / overlap
        if file_token_sets:
            dists = [_jaccard_distance(test_tokens, ft) for ft in file_token_sets]
            path_min_dist = min(dists)
            overlaps = [_overlap_fraction(test_tokens, ft) for ft in file_token_sets]
            path_max_overlap = max(overlaps)
        else:
            path_min_dist = 1.0
            path_max_overlap = 0.0

        # Co-fail rate (from history)
        co_seen, co_fail = 0, 0
        if history and changed_files:
            curr_set = set(changed_files)
            # Look at past commits where this test ran
            seen_commits: dict[str, list] = {}
            for r in history[-50:]:
                seen_commits.setdefault(r.commit_sha, []).append(r)
            for past_sha, past_runs in seen_commits.items():
                past_files = simulator._patches.get(past_sha, [])
                if set(past_files) & curr_set:
                    co_seen += 1
                    if any(r.failed for r in past_runs):
                        co_fail += 1
        co_fail_rate = co_fail / co_seen if co_seen > 0 else 0.0
        co_fail_support = co_seen

        rows.append({
            # Identifiers (not features)
            "commit": commit_sha,
            "commit_idx": ctx.commit_order,
            "testName": test_name,
            # Group 1 — test history
            "fail_rate_5": fail_rate_5,
            "fail_rate_20": fail_rate_20,
            "fail_rate_50": fail_rate_50,
            "new_failure_rate_50": new_failure_rate_50,
            "new_failure_support": new_failure_support,
            "last_fail_dist": last_fail_dist,
            "last_observed_status": last_observed_status,
            "run_count_total": n_past,
            "duration_median_20": duration_median_20,
            "duration_p90_20": duration_p90_20,
            "duration_history_known": duration_history_known,
            # Group 2 — parent status
            "parent_status_code": parent_status_code,
            "parent_result_known": parent_result_known,
            # Group 3 — commit characteristics
            "n_changed_files": n_changed,
            "lines_added": 0,       # Not available in RTPTorrent
            "lines_deleted": 0,     # Not available in RTPTorrent
            "n_prod_files_changed": n_prod_files,
            "n_test_files_changed": n_test_files,
            "has_java": has_java,
            "has_xml": has_xml,
            "has_build_file": has_build,
            # Group 4 — coupling
            "test_file_changed": test_file_changed,
            "same_package_changed": same_package,
            "same_module_changed": same_module,
            "path_min_dist": path_min_dist,
            "path_max_overlap": path_max_overlap,
            "co_fail_rate": co_fail_rate,
            "co_fail_support": co_fail_support,
        })

    return pd.DataFrame(rows)


def build_features_batch(
    simulator: "CIHistorySimulator",
    commit_shas: List[str],
    verbose: bool = False,
) -> pd.DataFrame:
    """Build features for multiple commits."""
    parts: list[pd.DataFrame] = []
    for i, sha in enumerate(commit_shas):
        if verbose and (i + 1) % 100 == 0:
            print(f"  features: {i+1}/{len(commit_shas)}", flush=True)
        parts.append(build_features(simulator, sha))
    if parts:
        return pd.concat(parts, ignore_index=True)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Feature column list (for model training)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    # Group 1 — test history
    "fail_rate_5", "fail_rate_20", "fail_rate_50",
    "new_failure_rate_50", "new_failure_support",
    "last_fail_dist", "last_observed_status", "run_count_total",
    "duration_median_20", "duration_p90_20", "duration_history_known",
    # Group 2 — parent status
    "parent_status_code", "parent_result_known",
    # Group 3 — commit characteristics
    "n_changed_files", "lines_added", "lines_deleted",
    "n_prod_files_changed", "n_test_files_changed",
    "has_java", "has_xml", "has_build_file",
    # Group 4 — coupling
    "test_file_changed", "same_package_changed", "same_module_changed",
    "path_min_dist", "path_max_overlap",
    "co_fail_rate", "co_fail_support",
]
