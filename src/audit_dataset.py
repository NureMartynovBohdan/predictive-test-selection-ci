"""
src/audit_dataset.py — Dataset audit for RTPTorrent projects
=============================================================
Counts confirmed pass→fail transitions using CIHistorySimulator.

Usage:
    python src/audit_dataset.py

Output:
    Prints table to stdout, saves to outputs/dataset_audit.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

_this_dir = Path(__file__).resolve().parent
_root_dir = _this_dir.parent
sys.path.insert(0, str(_this_dir))

from simulator import CIHistorySimulator

RTP_ROOT = _root_dir / "rtp-torrent"
OUT_DIR = _root_dir / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# All projects in RTPTorrent
ALL_PROJECTS = [
    d.name for d in sorted(RTP_ROOT.iterdir())
    if d.is_dir() and (d / f"{d.name}.csv").exists()
]

MIN_DEFECTIVE_THRESHOLD = 150  # for calib + test combined


def audit_project(project: str) -> dict:
    """Audit one project and return summary statistics."""
    t0 = time.time()
    try:
        sim = CIHistorySimulator(RTP_ROOT, project)
    except Exception as e:
        return {"project": project, "error": str(e)}

    n_commits = len(sim.get_all_commits())
    n_confirmed_new_fail = 0
    n_already_failing = 0
    n_unknown_parent = 0
    n_no_new_fail = 0
    n_mixed_outcomes = 0
    n_total_pairs = 0

    commits_with_new_fail = set()

    for sha in sim.iter_decision_points():
        gt = sim.reveal_ground_truth(sha)
        for test_name, tr in gt.test_results.items():
            n_total_pairs += 1
            if tr.label_attribution == "confirmed_new_fail":
                n_confirmed_new_fail += 1
                commits_with_new_fail.add(sha)
            elif tr.label_attribution == "already_failing":
                n_already_failing += 1
            elif tr.label_attribution == "unknown_parent":
                n_unknown_parent += 1
            else:
                n_no_new_fail += 1
            if tr.mixed_outcomes_same_sha:
                n_mixed_outcomes += 1

    n_defective_commits = len(commits_with_new_fail)
    meets_threshold = n_defective_commits >= MIN_DEFECTIVE_THRESHOLD
    elapsed = time.time() - t0

    return {
        "project": project,
        "n_commits": n_commits,
        "n_total_pairs": n_total_pairs,
        "n_defective_commits": n_defective_commits,
        "n_confirmed_new_fail_pairs": n_confirmed_new_fail,
        "n_already_failing_pairs": n_already_failing,
        "n_unknown_parent_pairs": n_unknown_parent,
        "n_mixed_outcomes_pairs": n_mixed_outcomes,
        f"meets_{MIN_DEFECTIVE_THRESHOLD}_threshold": meets_threshold,
        "audit_time_s": round(elapsed, 1),
    }


if __name__ == "__main__":
    print(f"Auditing {len(ALL_PROJECTS)} projects in {RTP_ROOT}\n")

    results = []
    for proj in ALL_PROJECTS:
        print(f"  {proj} ...", end="", flush=True)
        row = audit_project(proj)
        results.append(row)
        if "error" in row:
            print(f" ERROR: {row['error']}")
        else:
            print(
                f" commits={row['n_commits']}, "
                f"defective={row['n_defective_commits']}, "
                f"unknown_parent={row['n_unknown_parent_pairs']}, "
                f"mixed={row['n_mixed_outcomes_pairs']} "
                f"({row['audit_time_s']}s)"
            )

    df = pd.DataFrame(results)
    if "error" in df.columns:
        df_ok = df[df["error"].isna()].drop(columns=["error"])
    else:
        df_ok = df
    df_ok = df_ok.sort_values("n_defective_commits", ascending=False)

    print("\n" + "=" * 80)
    print(df_ok.to_string(index=False))
    print("=" * 80)

    out_path = OUT_DIR / "dataset_audit.csv"
    df_ok.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
