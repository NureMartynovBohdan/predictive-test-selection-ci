"""
src/audit_dataset.py — Full 20-project RTPTorrent audit aligned with run_experiment.py
=======================================================================================
Applies the exact same filtering rules as ``run_experiment.py`` at commit level
(memory-safe for multi-million-pair repositories like SonarSource@sonarqube):
  - ``parent_source == 'git_parent'`` only (excludes ``chronological_fallback`` and ``none``)
  - Three-valued ``confirmed_new_fail`` labeling (excludes ``ambiguous_current``,
    ``parent_mixed``, ``unknown_parent``, ``unobserved_candidate``)
  - Unions ``get_candidate_tests(sha)`` and ``reveal_ground_truth(sha)`` to measure
    candidate inventory completeness (``in_candidate_inventory``)
  - Uses ``chrono_split_4`` and ``rolling_time_splits`` from ``models.py`` to report
    unique defective commits in ``Train``, ``ProbCalib``, ``ConfCalib``, ``Test``,
    ``ConfCalib + Test``, and across 3 expanding rolling windows, plus raw
    ``failures`` vs ``errors`` diagnostics.

Usage:
    python src/audit_dataset.py

Output:
    Prints summary table to stdout and saves to outputs/dataset_audit.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict

import pandas as pd

_this_dir = Path(__file__).resolve().parent
_root_dir = _this_dir.parent
sys.path.insert(0, str(_this_dir))

from models import chrono_split_4, rolling_time_splits
from run_experiment import (
    CONF_CALIB_FRAC,
    MIN_DEFECTIVE_CONF_CALIB,
    MIN_DEFECTIVE_TEST,
    PROB_CALIB_FRAC,
    TRAIN_FRAC,
)
from simulator import CIHistorySimulator

RTP_ROOT = _root_dir / "rtp-torrent"
OUT_DIR = _root_dir / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_PROJECTS = [
    d.name for d in sorted(RTP_ROOT.iterdir())
    if d.is_dir() and (d / f"{d.name}.csv").exists()
]

MIN_DEFECTIVE_THRESHOLD = 150  # target combined ConfCalib + Test defective commits


def _count_failures_vs_errors(project: str) -> Dict[str, float]:
    """Summarize raw CSV failing records: failures>0 vs errors>0 (without guessing infra cause)."""
    csv_path = RTP_ROOT / project / f"{project}.csv"
    df = pd.read_csv(csv_path, usecols=["failures", "errors"])
    fail_rows = df[(df["failures"] + df["errors"]) > 0]
    n_tot = len(fail_rows)
    if n_tot == 0:
        return {
            "raw_failing_attempts": 0,
            "raw_failures_only_pct": 0.0,
            "raw_errors_present_pct": 0.0,
        }
    n_only_fail = int(((fail_rows["failures"] > 0) & (fail_rows["errors"] == 0)).sum())
    n_err_present = int((fail_rows["errors"] > 0).sum())
    return {
        "raw_failing_attempts": n_tot,
        "raw_failures_only_pct": round(100.0 * n_only_fail / n_tot, 1),
        "raw_errors_present_pct": round(100.0 * n_err_present / n_tot, 1),
    }


def _def_commits(cdf: pd.DataFrame) -> int:
    if len(cdf) == 0:
        return 0
    return int(cdf["is_defective"].sum())


def audit_project(project: str, policy: str = "confirmed_new_fail") -> dict:
    """Audit one project using the exact ground-truth & filtering logic of run_experiment.py."""
    t0 = time.time()
    try:
        sim = CIHistorySimulator(RTP_ROOT, project, label_policy=policy)
    except Exception as e:
        return {"project": project, "error": str(e)}

    all_commits = sim.get_all_commits()
    commit_rows = []
    n_git_commits_raw = 0
    n_fallback_commits_raw = 0
    n_none_commits_raw = 0
    n_unknown_parent_pairs = 0
    n_ambiguous_current_pairs = 0
    n_parent_mixed_pairs = 0

    for sha in all_commits:
        p_src = sim.get_parent_source(sha)
        if p_src == "git_parent":
            n_git_commits_raw += 1
        elif p_src == "chronological_fallback":
            n_fallback_commits_raw += 1
            continue
        else:
            n_none_commits_raw += 1
            continue

        c_idx = sim.get_commit_order(sha)
        cand_set = set(sim.get_candidate_tests(sha))
        gt = sim.reveal_ground_truth(sha, policy=policy)

        c_valid = 0
        c_def = 0
        c_cand_pos = False
        c_cand_neg = False
        c_out_pairs = 0
        c_out_fail = 0

        for test_name, tr in gt.test_results.items():
            if tr.label is None:
                if tr.label_status == "unknown_parent":
                    n_unknown_parent_pairs += 1
                elif tr.label_status == "ambiguous_current":
                    n_ambiguous_current_pairs += 1
                elif tr.label_status == "parent_mixed":
                    n_parent_mixed_pairs += 1
                continue

            c_valid += 1
            in_cand = test_name in cand_set
            if not in_cand:
                c_out_pairs += 1
                if tr.label == 1:
                    c_out_fail += 1
            else:
                if tr.label == 1:
                    c_cand_pos = True
                else:
                    c_cand_neg = True
            if tr.label == 1:
                c_def = 1

        if c_valid > 0:
            commit_rows.append({
                "commit_idx": c_idx,
                "commit": sha,
                "is_defective": c_def,
                "has_cand_pos": c_cand_pos,
                "has_cand_neg": c_cand_neg,
                "n_valid_pairs": c_valid,
                "n_out_inv_pairs": c_out_pairs,
                "n_out_inv_fail": c_out_fail,
            })

    if not commit_rows:
        return {"project": project, "error": "no valid git_parent commits"}

    cdf = pd.DataFrame(commit_rows)
    n_commits_eval = int(len(cdf))
    n_valid_pairs = int(cdf["n_valid_pairs"].sum())
    n_out_inv_pairs = int(cdf["n_out_inv_pairs"].sum())
    n_out_inv_fail = int(cdf["n_out_inv_fail"].sum())
    n_def_total = _def_commits(cdf)

    if n_commits_eval >= 4:
        tr_df, pr_df, co_df, te_df = chrono_split_4(
            cdf, TRAIN_FRAC, PROB_CALIB_FRAC, CONF_CALIB_FRAC
        )
        n_def_train = _def_commits(tr_df)
        n_def_prob = _def_commits(pr_df)
        n_def_conf = _def_commits(co_df)
        n_def_test = _def_commits(te_df)
        holdout_two_class_ok = bool(
            tr_df["has_cand_pos"].any()
            and tr_df["has_cand_neg"].any()
            and pr_df["has_cand_pos"].any()
            and pr_df["has_cand_neg"].any()
        )
    else:
        n_def_train = n_def_prob = n_def_conf = n_def_test = 0
        holdout_two_class_ok = False

    n_def_conf_plus_test = n_def_conf + n_def_test
    holdout_viable = bool(
        n_def_total >= 15
        and n_def_conf >= MIN_DEFECTIVE_CONF_CALIB
        and n_def_test >= MIN_DEFECTIVE_TEST
        and holdout_two_class_ok
    )

    # Rolling windows (3 windows matching run_experiment.run_rolling)
    rolling_ok = 0
    r_conf_list = []
    r_test_list = []
    if n_commits_eval >= 10:
        splits = rolling_time_splits(
            cdf,
            n_splits=3,
            min_train_frac=0.25,
            prob_calib_frac=PROB_CALIB_FRAC,
            conf_calib_frac=CONF_CALIB_FRAC,
            test_frac=0.15,
        )
        for r_tr, r_pr, r_co, r_te in splits:
            wc = _def_commits(r_co)
            wt = _def_commits(r_te)
            r_conf_list.append(str(wc))
            r_test_list.append(str(wt))
            two_cls = bool(
                len(r_tr) > 0
                and r_tr["has_cand_pos"].any()
                and r_tr["has_cand_neg"].any()
                and len(r_pr) > 0
                and r_pr["has_cand_pos"].any()
                and r_pr["has_cand_neg"].any()
            )
            if wc >= MIN_DEFECTIVE_CONF_CALIB and wt >= MIN_DEFECTIVE_TEST and two_cls:
                rolling_ok += 1

    fail_diag = _count_failures_vs_errors(project)
    elapsed = time.time() - t0

    return {
        "project": project,
        "n_commits_raw": len(all_commits),
        "n_commits_git": n_git_commits_raw,
        "n_commits_fallback": n_fallback_commits_raw,
        "n_commits_none": n_none_commits_raw,
        "n_commits_eval": n_commits_eval,
        "n_valid_pairs": n_valid_pairs,
        "n_out_inv_pairs": n_out_inv_pairs,
        "n_out_inv_fail": n_out_inv_fail,
        "n_unknown_parent_pairs": n_unknown_parent_pairs,
        "n_ambiguous_current_pairs": n_ambiguous_current_pairs,
        "n_parent_mixed_pairs": n_parent_mixed_pairs,
        "n_def_total": n_def_total,
        "n_def_train": n_def_train,
        "n_def_prob_calib": n_def_prob,
        "n_def_conf_calib": n_def_conf,
        "n_def_test": n_def_test,
        "n_def_conf_plus_test": n_def_conf_plus_test,
        f"meets_{MIN_DEFECTIVE_THRESHOLD}_conf_test": n_def_conf_plus_test >= MIN_DEFECTIVE_THRESHOLD,
        "holdout_viable": holdout_viable,
        "rolling_ok_windows": rolling_ok,
        "rolling_def_conf": "/".join(r_conf_list) if r_conf_list else "-",
        "rolling_def_test": "/".join(r_test_list) if r_test_list else "-",
        **fail_diag,
        "audit_time_s": round(elapsed, 1),
    }


if __name__ == "__main__":
    print(
        f"Auditing {len(ALL_PROJECTS)} projects in {RTP_ROOT} "
        f"(git_parent ONLY, confirmed_new_fail)\n",
        flush=True,
    )

    results = []
    for proj in ALL_PROJECTS:
        print(f"  {proj} ...", end="", flush=True)
        row = audit_project(proj)
        results.append(row)
        if "error" in row:
            print(f" ERROR: {row['error']}", flush=True)
        else:
            print(
                f" git_commits={row['n_commits_git']}/{row['n_commits_raw']}, "
                f"def_total={row['n_def_total']} "
                f"(train={row['n_def_train']}, prob={row['n_def_prob_calib']}, "
                f"conf={row['n_def_conf_calib']}, test={row['n_def_test']}, "
                f"conf+test={row['n_def_conf_plus_test']}), "
                f"rolling_ok={row['rolling_ok_windows']}/3 "
                f"[conf={row['rolling_def_conf']} | test={row['rolling_def_test']}] "
                f"({row['audit_time_s']}s)",
                flush=True,
            )

    df = pd.DataFrame(results)
    if "error" in df.columns:
        df_ok = df[df["error"].isna()].drop(columns=["error"])
    else:
        df_ok = df

    df_ok = df_ok.sort_values(
        ["holdout_viable", "rolling_ok_windows", "n_def_conf_plus_test", "n_def_total"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    summary_cols = [
        "project",
        "n_commits_eval",
        "n_def_total",
        "n_def_train",
        "n_def_prob_calib",
        "n_def_conf_calib",
        "n_def_test",
        "n_def_conf_plus_test",
        "holdout_viable",
        "rolling_ok_windows",
        "rolling_def_conf",
        "rolling_def_test",
        "n_out_inv_fail",
    ]
    print("\n" + "=" * 100)
    print(df_ok[summary_cols].to_string(index=False))
    print("=" * 100)

    out_path = OUT_DIR / "dataset_audit.csv"
    df_ok.to_csv(out_path, index=False)
    print(f"\nSaved full audit to {out_path}")
