"""
src/features.py
===============
Feature engineering for (commit, testName) pairs.

Feature groups (all computed from PAST data only — no data leakage):

Group 1 — Test history (rolling window over past N commits where test ran):
  - fail_rate_N      : fraction of past N appearances where test failed (strict)
  - last_fail_dist   : commits since test last failed (strict), inf → −1
  - duration_mean    : historical mean duration (seconds)
  - run_count_total  : how many times this test has been seen before this commit

Group 2 — Commit characteristics:
  - n_changed_files  : number of files touched in this commit's patch
  - has_java         : at least one .java file changed (bool → int)
  - has_test_file    : at least one file with "test" in name changed (bool → int)
  - has_xml          : at least one .xml file changed (bool → int)
  - has_build_file   : pom.xml / build.gradle / BUILD changed

Group 3 — Test ↔ Commit coupling (key features):
  - path_min_dist    : min Jaccard distance between test-class path tokens
                       and each changed file path tokens
  - path_max_overlap : max fraction of test-path tokens covered by any changed file
  - co_fail_rate     : historical fraction of commits where the same changed files
                       co-occurred with this test failing (approximated as ratio
                       of co-failure events to total commits with this test seen)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers — path tokenisation
# ---------------------------------------------------------------------------

def _tokenise_java_class(class_name: str) -> list[str]:
    """
    'com.example.foo.BarTest' → ['com', 'example', 'foo', 'BarTest']
    'com/example/foo/BarTest.java' → same
    """
    class_name = re.sub(r"\.java$", "", class_name, flags=re.I)
    # Replace dots that look like package separators (between lower-case tokens)
    tokens = re.split(r"[./\\]", class_name)
    # Further split CamelCase for broader coverage (optional)
    return [t.lower() for t in tokens if t]


def _tokenise_path(path: str) -> set[str]:
    tokens = re.split(r"[./\\]", path.lower())
    return set(t for t in tokens if t)


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 1.0  # maximum distance when one is empty
    inter = len(a & b)
    union = len(a | b)
    return 1.0 - inter / union  # distance (0 = identical, 1 = disjoint)


def _overlap_fraction(test_tokens: set, file_tokens: set) -> float:
    """Fraction of test_tokens covered by file_tokens."""
    if not test_tokens:
        return 0.0
    return len(test_tokens & file_tokens) / len(test_tokens)


# ---------------------------------------------------------------------------
# Main feature-engineering function
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    patches: dict[str, list[str]],
    history_window: int = 20,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : DataFrame from data_builder.load_project()
         Required columns: commit, testName, commit_idx, commit_failed_strict,
                           any_fail, intra_commit_flaky, duration_mean, label
    patches : {sha: [file_path, ...]}
    history_window : number of past appearances to use for rolling stats

    Returns
    -------
    DataFrame with original columns PLUS feature columns and 'label'.
    Rows are sorted by commit_idx.
    """
    df = df.sort_values(["commit_idx", "testName"]).copy()

    # Map commit → list of changed files
    df["changed_files"] = df["commit"].map(patches).apply(
        lambda x: x if isinstance(x, list) else []
    )

    # Pre-tokenise test names
    test_tokens_cache: dict[str, set] = {}
    def get_test_tokens(name: str) -> set:
        if name not in test_tokens_cache:
            test_tokens_cache[name] = set(_tokenise_java_class(name))
        return test_tokens_cache[name]

    # ---------------------------------------------------------------------------
    # Group 1 & 3 rolling-history features (computed per test, chronologically)
    # ---------------------------------------------------------------------------
    rows_out = []

    # Collect test-level history incrementally
    # history[testName] = list of (commit_idx, commit_failed_strict) in seen order
    test_history: dict[str, list[tuple[int, bool, float]]] = {}  # (idx, failed, dur)

    # Also track (testName, file_set) co-occurrence for Group 3
    # co_fail_map[testName][frozenset(changed_files)] = (total_commits_seen, fail_count)
    # Too memory-heavy to track per-file-set; use simpler "any shared token" heuristic
    # co_fail_map[testName] = {commit_idx: (changed_files, failed)}
    test_commit_history: dict[str, list[dict]] = {}

    for _, row in df.iterrows():
        test = row["testName"]
        c_idx = int(row["commit_idx"])
        commit = row["commit"]
        strict_fail = bool(row["commit_failed_strict"])
        dur = float(row["duration_mean"]) if not pd.isna(row["duration_mean"]) else 0.0
        changed = row["changed_files"]

        # --- Retrieve past history (before this commit) ---
        past = test_history.get(test, [])  # [(idx, failed, dur), ...]
        past_commit_hist = test_commit_history.get(test, [])

        # Group 1 features
        n_past = len(past)
        window = past[-history_window:] if past else []

        fail_rate_N = sum(1 for _, f, _ in window if f) / max(len(window), 1)
        last_failed_indices = [i for i, (idx, f, _) in enumerate(past) if f]
        if last_failed_indices:
            last_seen_idx = past[last_failed_indices[-1]][0]
            last_fail_dist = c_idx - last_seen_idx
        else:
            last_fail_dist = -1  # Never failed before

        dur_hist = np.mean([d for _, _, d in past]) if past else dur
        run_count_total = n_past

        # Group 2 — Commit characteristics
        n_changed = len(changed)
        has_java = int(any(f.endswith(".java") for f in changed))
        has_test_file = int(any("test" in f.lower() for f in changed))
        has_xml = int(any(f.endswith(".xml") for f in changed))
        has_build_file = int(any(
            Path(f).name.lower() in {"pom.xml", "build.gradle", "build"}
            for f in changed
        ))

        # Group 3 — Test ↔ file coupling
        test_toks = get_test_tokens(test)

        if changed:
            file_tok_sets = [_tokenise_path(f) for f in changed]
            dists = [_jaccard(test_toks, ft) for ft in file_tok_sets]
            path_min_dist = min(dists)
            overlaps = [_overlap_fraction(test_toks, ft) for ft in file_tok_sets]
            path_max_overlap = max(overlaps)
        else:
            path_min_dist = 1.0
            path_max_overlap = 0.0

        # Co-fail rate: among past commits where THIS test ran AND the current
        # changed-file set overlaps with what was changed then, how often did it fail?
        if past_commit_hist and changed:
            curr_set = set(changed)
            co_seen, co_fail = 0, 0
            for rec in past_commit_hist[-50:]:  # limit lookback
                overlap = curr_set & set(rec["changed"])
                if overlap:
                    co_seen += 1
                    if rec["failed"]:
                        co_fail += 1
            co_fail_rate = co_fail / co_seen if co_seen > 0 else 0.0
        else:
            co_fail_rate = 0.0

        rows_out.append({
            # identifiers
            "commit": commit,
            "commit_idx": c_idx,
            "testName": test,
            "label": int(row["label"]),
            # group 1
            "fail_rate_N": fail_rate_N,
            "last_fail_dist": last_fail_dist,
            "duration_mean": dur_hist,
            "run_count_total": run_count_total,
            # group 2
            "n_changed_files": n_changed,
            "has_java": has_java,
            "has_test_file": has_test_file,
            "has_xml": has_xml,
            "has_build_file": has_build_file,
            # group 3
            "path_min_dist": path_min_dist,
            "path_max_overlap": path_max_overlap,
            "co_fail_rate": co_fail_rate,
            # carry-through for conformal / evaluation
            "commit_failed_strict": strict_fail,
            "intra_commit_flaky": bool(row["intra_commit_flaky"]),
            "changed_files": changed,
        })

        # Update incremental history AFTER computing features (no leakage)
        if test not in test_history:
            test_history[test] = []
        test_history[test].append((c_idx, strict_fail, dur))

        if test not in test_commit_history:
            test_commit_history[test] = []
        test_commit_history[test].append({
            "commit_idx": c_idx,
            "changed": changed,
            "failed": strict_fail,
        })

    return pd.DataFrame(rows_out)


FEATURE_COLS = [
    "fail_rate_N", "last_fail_dist", "duration_mean", "run_count_total",
    "n_changed_files", "has_java", "has_test_file", "has_xml", "has_build_file",
    "path_min_dist", "path_max_overlap", "co_fail_rate",
]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from data_builder import load_project, load_patches, BEST_PROJECTS
    from pathlib import Path as P

    BASE = P(__file__).parent.parent
    RTP = BASE / "rtp-torrent"

    proj = "apache@sling"
    commits_csv = RTP / "tr_all_built_commits.csv"
    df_c = __import__("pandas").read_csv(commits_csv, dtype={"tr_job_id": "int64"})
    j2c = dict(zip(df_c["tr_job_id"], df_c["git_commit_id"]))

    df = load_project(proj, RTP, j2c)
    patches = load_patches(proj, RTP)
    feat_df = build_features(df, patches)
    print(feat_df[FEATURE_COLS + ["label"]].describe())
    print("label distribution:", feat_df["label"].value_counts().to_dict())
