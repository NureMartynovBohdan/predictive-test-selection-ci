"""
tests/test_all.py — Unit & Integration tests for the refactored pipeline (v2.1)
================================================================================
Covers:
  Stage 1: Simulator leakage prevention, pre-execution candidate test inventory,
           real bare Git repo C^1 parent extraction, three-valued ground truth
  Stage 3: Feature tokenization and schema audit
  Stage 4: Conformal calibration, input validation, chronological & rolling splits
  Stage 5: Metrics (CMR, Wilson CI, block bootstrap CI, TTFF, Conditional APFDc)
  Stage 6: Experiment filtering (git_parent vs fallback, indeterminate exclusion)

All tests use synthetic data — no RTPTorrent files required.
Run with:  python -m pytest tests/test_all.py -v
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ===================================================================
# Helpers
# ===================================================================

def _setup_rtp_project(
    tmp_path: Path,
    project: str,
    job_sha_rows: list,
    patch_rows: list,
    run_rows: list,
) -> None:
    """Create a minimal synthetic RTPTorrent project directory."""
    import simulator
    simulator._GLOBAL_JOB_TO_SHA_CACHE = None

    df_map = pd.DataFrame(job_sha_rows, columns=["tr_job_id", "git_commit_id"])
    df_map.to_csv(tmp_path / "tr_all_built_commits.csv", index=False)

    proj_dir = tmp_path / project
    proj_dir.mkdir(parents=True, exist_ok=True)

    df_patches = pd.DataFrame(patch_rows, columns=["sha", "name"])
    df_patches.to_csv(proj_dir / f"{project}-patches.csv", index=False)

    df_runs = pd.DataFrame(
        run_rows,
        columns=["travisJobId", "testName", "duration", "failures", "errors"],
    )
    df_runs.to_csv(proj_dir / f"{project}.csv", index=False)


# ===================================================================
# Stage 1: Simulator leakage-prevention & candidate inventory tests
# ===================================================================

class TestSimulatorLeakagePrevention:

    def test_late_rerun_does_not_change_earlier_commit(self, tmp_path: Path):
        """
        Scenario: A/job1: fail -> B/job2: fail -> A/job100: pass

        Prove with real CIHistorySimulator and build_features_batch that adding
        late rerun A/job100 does NOT change:
        1. History returned for test 'T' as-of commit B (get_test_history)
        2. Feature vector computed for commit B (build_features_batch)
        3. Ground truth revealed for commit B (reveal_ground_truth)
        """
        from features import FEATURE_COLS, build_features_batch
        from simulator import CIHistorySimulator

        project = "test_proj"

        # 1. Without late rerun
        dir_before = tmp_path / "run_before"
        dir_before.mkdir()
        _setup_rtp_project(
            dir_before,
            project,
            job_sha_rows=[(1, "A"), (2, "B")],
            patch_rows=[("A", "File1.java"), ("B", "File2.java")],
            run_rows=[
                (1, "T", 5.0, 1, 0),
                (2, "T", 3.0, 1, 0),
            ],
        )
        sim_before = CIHistorySimulator(dir_before, project)

        # 2. With late rerun (A/job100: pass)
        dir_after = tmp_path / "run_after"
        dir_after.mkdir()
        _setup_rtp_project(
            dir_after,
            project,
            job_sha_rows=[(1, "A"), (2, "B"), (100, "A")],
            patch_rows=[("A", "File1.java"), ("B", "File2.java")],
            run_rows=[
                (1, "T", 5.0, 1, 0),
                (2, "T", 3.0, 1, 0),
                (100, "T", 4.0, 0, 0),
            ],
        )
        sim_after = CIHistorySimulator(dir_after, project)

        # --- A. Check get_test_history ---
        as_of_B_before = sim_before.get_commit_min_job("B")
        hist_before = sim_before.get_test_history("T", as_of_B_before)

        as_of_B_after = sim_after.get_commit_min_job("B")
        hist_after = sim_after.get_test_history("T", as_of_B_after)

        assert len(hist_before) == len(hist_after) == 1
        assert hist_before[0].job_id == hist_after[0].job_id == 1
        assert hist_before[0].failed == hist_after[0].failed is True
        assert hist_before[0].duration == hist_after[0].duration == 5.0

        # --- B. Check feature vector for commit B ---
        feat_before = build_features_batch(sim_before, ["B"])
        feat_after = build_features_batch(sim_after, ["B"])

        assert len(feat_before) == 1
        assert len(feat_after) == 1
        pd.testing.assert_frame_equal(feat_before[FEATURE_COLS], feat_after[FEATURE_COLS])

        # --- C. Check ground truth for commit B ---
        gt_before = sim_before.reveal_ground_truth("B")
        gt_after = sim_after.reveal_ground_truth("B")

        res_before = gt_before.test_results["T"]
        res_after = gt_after.test_results["T"]

        assert res_before.label == res_after.label
        assert res_before.current_status == res_after.current_status
        assert res_before.parent_status == res_after.parent_status
        assert res_before.label_status == res_after.label_status
        assert res_before.actual_duration == res_after.actual_duration

    def test_actual_duration_not_in_features(self, tmp_path: Path):
        """
        Prove with real CIHistorySimulator and build_features_batch that changing
        actual_duration of a newly introduced test on commit C (1.0 vs 1000.0)
        does NOT change the feature vector submitted to the model.

        Uses 'XTest.java' in patches so pre-execution candidate discovery finds
        'XTest' without inspecting current commit CI results.
        """
        from features import FEATURE_COLS, build_features_batch
        from simulator import CIHistorySimulator

        project = "test_dur_proj"

        # Case 1: duration = 1.0
        dir_1 = tmp_path / "dur_1"
        dir_1.mkdir()
        _setup_rtp_project(
            dir_1,
            project,
            job_sha_rows=[(10, "C")],
            patch_rows=[("C", "XTest.java")],
            run_rows=[(10, "XTest", 1.0, 1, 0)],
        )
        sim_1 = CIHistorySimulator(dir_1, project)

        # Case 2: duration = 1000.0
        dir_2 = tmp_path / "dur_2"
        dir_2.mkdir()
        _setup_rtp_project(
            dir_2,
            project,
            job_sha_rows=[(10, "C")],
            patch_rows=[("C", "XTest.java")],
            run_rows=[(10, "XTest", 1000.0, 1, 0)],
        )
        sim_2 = CIHistorySimulator(dir_2, project)

        feat_1 = build_features_batch(sim_1, ["C"])
        feat_2 = build_features_batch(sim_2, ["C"])

        assert len(feat_1) == 1
        assert len(feat_2) == 1

        # All FEATURE_COLS must be identical
        pd.testing.assert_frame_equal(feat_1[FEATURE_COLS], feat_2[FEATURE_COLS])

        # Verify duration features for newly introduced test with empty history
        assert np.isnan(feat_1["duration_median_20"].iloc[0])
        assert np.isnan(feat_2["duration_median_20"].iloc[0])
        assert np.isnan(feat_1["duration_p90_20"].iloc[0])
        assert np.isnan(feat_2["duration_p90_20"].iloc[0])
        assert feat_1["duration_history_known"].iloc[0] == 0
        assert feat_2["duration_history_known"].iloc[0] == 0

    def test_candidate_tests_independent_of_current_runs(self, tmp_path: Path):
        """
        Prove that get_candidate_tests(commit_sha) depends ONLY on pre-execution
        information (past test history strictly before commit_sha + test files in
        commit_sha's diff), and is completely invariant to which tests actually
        ran or appeared in CI results for commit_sha.
        """
        from simulator import CIHistorySimulator

        project = "cand_proj"

        # Project 1: commit B runs TestAlpha, TestGamma, and an unseen TestSneakyFuture
        dir_1 = tmp_path / "cand_1"
        dir_1.mkdir()
        _setup_rtp_project(
            dir_1,
            project,
            job_sha_rows=[(1, "A"), (2, "B")],
            patch_rows=[("A", "App.java"), ("B", "TestGamma.java")],
            run_rows=[
                (1, "TestAlpha", 1.0, 0, 0),
                (1, "TestBeta", 2.0, 0, 0),
                (2, "TestAlpha", 1.0, 0, 0),
                (2, "TestGamma", 1.0, 1, 0),
                (2, "TestSneakyFuture", 5.0, 1, 0),
            ],
        )
        sim_1 = CIHistorySimulator(dir_1, project)

        # Project 2: identical history on A and identical patch on B,
        # but commit B only recorded TestAlpha in CI results
        dir_2 = tmp_path / "cand_2"
        dir_2.mkdir()
        _setup_rtp_project(
            dir_2,
            project,
            job_sha_rows=[(1, "A"), (2, "B")],
            patch_rows=[("A", "App.java"), ("B", "TestGamma.java")],
            run_rows=[
                (1, "TestAlpha", 1.0, 0, 0),
                (1, "TestBeta", 2.0, 0, 0),
                (2, "TestAlpha", 1.0, 0, 0),
            ],
        )
        sim_2 = CIHistorySimulator(dir_2, project)

        cand_1 = sim_1.get_candidate_tests("B")
        cand_2 = sim_2.get_candidate_tests("B")

        assert cand_1 == cand_2
        assert "TestAlpha" in cand_1
        assert "TestBeta" in cand_1
        assert "TestGamma" in cand_1
        assert "TestSneakyFuture" not in cand_1
        assert sim_1.get_decision_context("B").candidate_tests == cand_1

    def test_unobserved_candidate_handled_without_keyerror(self, tmp_path: Path):
        """
        If get_candidate_tests(commit) includes a test class (e.g. from patch diff
        or prior history) that has no run record in reveal_ground_truth(commit),
        build_project_data must mark it as label=None / label_status='unobserved_candidate'
        instead of raising KeyError, and filter_valid_pairs must exclude it.
        """
        from run_experiment import build_project_data, filter_valid_pairs
        from simulator import CIHistorySimulator

        project = "unobs_proj"
        _setup_rtp_project(
            tmp_path,
            project,
            job_sha_rows=[(1, "A"), (2, "B")],
            patch_rows=[("A", "App.java"), ("B", "GhostTest.java")],
            run_rows=[
                (1, "TestAlpha", 1.0, 0, 0),
                (2, "TestAlpha", 1.0, 0, 0),
                # GhostTest never ran in commit B CI logs
            ],
        )
        sim = CIHistorySimulator(tmp_path, project)

        assert "GhostTest" in sim.get_candidate_tests("B")
        gt_b = sim.reveal_ground_truth("B")
        assert "GhostTest" not in gt_b.test_results

        df_raw = build_project_data(sim, policy="confirmed_new_fail")
        ghost_rows = df_raw[(df_raw["commit"] == "B") & (df_raw["testName"] == "GhostTest")]
        assert len(ghost_rows) == 1
        assert pd.isna(ghost_rows["label"].iloc[0])
        assert ghost_rows["label_status"].iloc[0] == "unobserved_candidate"

        df_clean, stats = filter_valid_pairs(
            df_raw, include_fallback_parents=True, policy="confirmed_new_fail", verbose=False
        )
        assert "unobserved_candidate" in stats
        assert not ((df_clean["commit"] == "B") & (df_clean["testName"] == "GhostTest")).any()


# ===================================================================
# Stage 4: Conformal calibration tests
# ===================================================================

class TestConformalCalibration:

    def test_conformal_threshold_audit_example(self):
        from conformal import compute_global_threshold

        scores = np.array([0.05, 0.10, 0.15, 0.20, 0.25,
                           0.30, 0.35, 0.40, 0.45, 0.50])
        threshold = compute_global_threshold(scores, 0.10)
        assert threshold == pytest.approx(0.05)

    def test_compute_tau_residual_audit_example(self):
        from conformal import compute_tau_residual

        df = pd.DataFrame({
            "commit_idx": list(range(10)),
            "testName": [f"test_{i}" for i in range(10)],
            "label": [1] * 10,
        })
        probs = np.array([0.05, 0.10, 0.15, 0.20, 0.25,
                          0.30, 0.35, 0.40, 0.45, 0.50])

        tau = compute_tau_residual(df, probs, alpha=0.10)
        assert tau == pytest.approx(0.05)
        assert tau != pytest.approx(0.10)

    def test_empty_prefix(self):
        from conformal import select_tests_residual

        df = pd.DataFrame({
            "commit_idx": [0, 0],
            "testName": ["A", "B"],
            "label": [0, 0],
        })
        selected = select_tests_residual(df, np.array([0.01, 0.01]), tau=0.1)
        assert len(selected) == 0

    def test_tie_breaking_consistency(self):
        from conformal import _deterministic_sort

        df = pd.DataFrame({
            "commit_idx": [0, 0, 0],
            "testName": ["C_test", "A_test", "B_test"],
            "label": [0, 1, 0],
        })
        probs = np.array([0.5, 0.5, 0.5])

        sorted1 = _deterministic_sort(df, probs)
        sorted2 = _deterministic_sort(df.sample(frac=1, random_state=42), probs)

        assert sorted1["testName"].tolist() == ["A_test", "B_test", "C_test"]
        assert sorted2["testName"].tolist() == ["A_test", "B_test", "C_test"]

    def test_global_threshold_computation(self):
        from conformal import compute_global_threshold

        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        assert compute_global_threshold(scores, 0.10) == 0.0
        assert compute_global_threshold(scores, 0.20) == pytest.approx(0.1)
        assert compute_global_threshold(scores, 0.50) == pytest.approx(0.3)

    def test_residual_risk_function(self):
        from conformal import _residual_risk

        probs = np.array([0.9, 0.5, 0.1])
        assert _residual_risk(probs, 0) == pytest.approx(0.955)
        assert _residual_risk(probs, 1) == pytest.approx(0.55)
        assert _residual_risk(probs, 2) == pytest.approx(0.1)
        assert _residual_risk(probs, 3) == pytest.approx(0.0)


# ===================================================================
# Stage 5: Metrics tests
# ===================================================================

class TestMetrics:

    def test_ttff_and_apfdc_controlled(self):
        from evaluate import compute_ttff, compute_apfdc, compute_conditional_apfdc

        df = pd.DataFrame({
            "commit_idx": [0, 0],
            "testName": ["test_A", "test_B"],
            "label": [0, 1],
            "actual_duration": [1000.0, 1.0],
            "_rank": [1, 2],
        })

        assert compute_ttff(df, duration_col="actual_duration") == pytest.approx(1001.0)

        expected = 1.0 - (1001.0 - 0.5) / (1 * 1001.0)
        apfdc = compute_apfdc(df, duration_col="actual_duration")
        cond_apfdc = compute_conditional_apfdc(df, duration_col="actual_duration")
        assert apfdc == pytest.approx(expected, abs=1e-6)
        assert cond_apfdc == pytest.approx(apfdc)
        assert apfdc < 0.001

    def test_ttff_first_test_fails(self):
        from evaluate import compute_ttff, compute_apfdc

        df = pd.DataFrame({
            "commit_idx": [0, 0],
            "testName": ["test_A", "test_B"],
            "label": [1, 0],
            "actual_duration": [5.0, 100.0],
            "_rank": [1, 2],
        })

        assert compute_ttff(df, duration_col="actual_duration") == pytest.approx(5.0)
        expected = 1.0 - (5.0 - 2.5) / (1 * 105.0)
        apfdc = compute_apfdc(df, duration_col="actual_duration")
        assert apfdc == pytest.approx(expected, abs=1e-6)
        assert apfdc > 0.95

    def test_multiple_faults_apfdc(self):
        from evaluate import compute_apfdc

        df = pd.DataFrame({
            "commit_idx": [0, 0, 0],
            "testName": ["A", "B", "C"],
            "label": [0, 1, 1],
            "actual_duration": [10.0, 2.0, 3.0],
            "_rank": [1, 2, 3],
        })

        expected = 1.0 - ((12.0 - 1.0) + (15.0 - 1.5)) / (2 * 15.0)
        assert compute_apfdc(df, duration_col="actual_duration") == pytest.approx(expected, abs=1e-6)

    def test_cmr_all_detected(self):
        from evaluate import compute_cmr

        df = pd.DataFrame({"commit_idx": [0, 0, 1, 1], "label": [0, 1, 1, 0]})
        assert compute_cmr(df, np.array([True, True, True, True])) == pytest.approx(0.0)

    def test_cmr_all_missed(self):
        from evaluate import compute_cmr

        df = pd.DataFrame({"commit_idx": [0, 0, 1, 1], "label": [0, 1, 1, 0]})
        assert compute_cmr(df, np.array([True, False, False, True])) == pytest.approx(1.0)

    def test_cmr_partial(self):
        from evaluate import compute_cmr

        df = pd.DataFrame({"commit_idx": [0, 0, 1, 1], "label": [0, 1, 1, 0]})
        assert compute_cmr(df, np.array([True, True, False, True])) == pytest.approx(0.5)

    def test_metrics_safely_ignore_none_labels(self):
        """Rows with label=None must not break CMR, TTFF, or Conditional APFDc."""
        from evaluate import compute_cmr, compute_ttff, compute_conditional_apfdc

        df = pd.DataFrame({
            "commit_idx": [0, 0, 1, 1],
            "testName": ["a", "b", "c", "d"],
            "label": [None, 1, None, 0],
            "actual_duration": [10.0, 2.0, 5.0, 3.0],
            "_rank": [1, 2, 1, 2],
        })
        selected = np.array([True, True, True, True])
        assert compute_cmr(df, selected) == pytest.approx(0.0)
        assert compute_ttff(df) == pytest.approx(2.0)
        assert 0.0 <= compute_conditional_apfdc(df) <= 1.0


# ===================================================================
# Stage 4 (additional): chrono_split_4 & rolling_time_splits tests
# ===================================================================

class TestChronoSplit:

    def test_4_way_split_no_overlap(self):
        from models import chrono_split_4

        df = pd.DataFrame({
            "commit_idx": np.repeat(np.arange(100), 2),
            "testName": ["A", "B"] * 100,
            "label": np.random.randint(0, 2, 200),
        })

        train, prob_cal, conf_cal, test = chrono_split_4(df)

        train_commits = set(train["commit_idx"])
        prob_commits = set(prob_cal["commit_idx"])
        conf_commits = set(conf_cal["commit_idx"])
        test_commits = set(test["commit_idx"])

        assert len(train_commits & prob_commits) == 0
        assert len(train_commits & conf_commits) == 0
        assert len(train_commits & test_commits) == 0
        assert len(prob_commits & conf_commits) == 0
        assert len(prob_commits & test_commits) == 0
        assert len(conf_commits & test_commits) == 0

        if train_commits and prob_commits:
            assert max(train_commits) < min(prob_commits)
        if prob_commits and conf_commits:
            assert max(prob_commits) < min(conf_commits)
        if conf_commits and test_commits:
            assert max(conf_commits) < min(test_commits)

        assert (train_commits | prob_commits | conf_commits | test_commits) == set(range(100))

    def test_rolling_time_splits_non_overlapping(self):
        """Rolling splits must produce distinct, strictly chronological,
        non-overlapping test windows."""
        from models import rolling_time_splits

        df = pd.DataFrame({
            "commit_idx": np.repeat(np.arange(200), 2),
            "testName": ["A", "B"] * 200,
            "label": np.random.randint(0, 2, 400),
        })
        splits = rolling_time_splits(
            df, n_splits=3, min_train_frac=0.25,
            prob_calib_frac=0.15, conf_calib_frac=0.15, test_frac=0.15,
        )
        assert len(splits) == 3

        test_sets = []
        train_sizes = []
        for train_df, prob_df, conf_df, test_df in splits:
            t_c = set(train_df["commit_idx"])
            p_c = set(prob_df["commit_idx"])
            c_c = set(conf_df["commit_idx"])
            te_c = set(test_df["commit_idx"])
            assert max(t_c) < min(p_c)
            assert max(p_c) < min(c_c)
            assert max(c_c) < min(te_c)
            test_sets.append(te_c)
            train_sizes.append(len(t_c))

        assert train_sizes[0] < train_sizes[1] < train_sizes[2]
        assert len(test_sets[0] & test_sets[1]) == 0
        assert len(test_sets[1] & test_sets[2]) == 0
        assert len(test_sets[0] & test_sets[2]) == 0


# ===================================================================
# Stage 3: Feature tokenization tests
# ===================================================================

class TestTokenization:

    def test_camel_case_split(self):
        from features import _split_camel
        assert _split_camel("FooBarTest") == ["Foo", "Bar", "Test"]
        assert _split_camel("XMLParser") == ["XML", "Parser"]
        assert _split_camel("simple") == ["simple"]

    def test_java_class_tokenize(self):
        from features import _tokenise_java_class
        tokens = _tokenise_java_class("com.example.foo.FooBarTest")
        for tok in ("com", "example", "foo", "bar", "test"):
            assert tok in tokens


# ===================================================================
# Stage 4 & 5: Cost-aware prioritization & Three-valued Label Policy tests
# ===================================================================

class TestCostAwareAndPolicy:

    def test_cost_aware_improves_ttff_and_apfdc(self):
        from evaluate import evaluate_selection

        df = pd.DataFrame({
            "commit_idx": [0, 0],
            "testName": ["slow_pass", "fast_fail"],
            "label": [0, 1],
            "actual_duration": [100.0, 1.0],
            "duration_median_20": [100.0, 1.0],
            "_prob": [0.90, 0.89],
        })
        selected_mask = np.array([True, True])

        res_prob = evaluate_selection(df, selected_mask, "prob_order", alpha=0.10, order_by="prob")
        res_cost = evaluate_selection(df, selected_mask, "cost_order", alpha=0.10, order_by="cost_aware")

        assert res_cost["ttff"] == pytest.approx(1.0)
        assert res_prob["ttff"] == pytest.approx(101.0)
        assert res_cost["apfdc"] > res_prob["apfdc"]
        assert res_cost["conditional_apfdc"] == pytest.approx(res_cost["apfdc"])

    def test_label_policies(self, tmp_path: Path):
        """
        Verify three-valued label policies with real CIHistorySimulator:
        - pass -> fail transition gives label == 1 (confirmed_new_fail)
        - fail -> fail transition gives label == 0 (already_failing)
        - root / no-parent commit gives label is None (unknown_parent)
        - mixed status on current commit gives label is None (ambiguous_current)
        - policy differences for strict_all_fail and any_fail
        """
        from simulator import CIHistorySimulator

        project = "label_test_proj"
        proj_dir = tmp_path / "label_proj"
        proj_dir.mkdir()

        _setup_rtp_project(
            proj_dir,
            project,
            job_sha_rows=[
                (1, "C0"),
                (2, "C1"),
                (3, "C2"),
                (4, "C3"),
                (5, "C3"),
            ],
            patch_rows=[
                ("C0", "File0.java"),
                ("C1", "File1.java"),
                ("C2", "File2.java"),
                ("C3", "File3.java"),
            ],
            run_rows=[
                (1, "test_A", 1.0, 0, 0),          # C0: test_A pass
                (1, "test_root_fail", 1.0, 1, 0),  # C0: test_root_fail fail (no parent)
                (2, "test_A", 1.0, 1, 0),          # C1: test_A fail (pass -> fail)
                (3, "test_A", 1.0, 1, 0),          # C2: test_A fail (fail -> fail)
                (4, "test_flaky", 1.0, 0, 0),      # C3: test_flaky pass
                (5, "test_flaky", 1.0, 1, 0),      # C3: test_flaky fail (mixed)
            ],
        )
        sim = CIHistorySimulator(proj_dir, project)

        # 1. C0: root commit, no parent -> unknown_parent, label is None under confirmed_new_fail
        gt_c0_conf = sim.reveal_ground_truth("C0", policy="confirmed_new_fail")
        res_root = gt_c0_conf.test_results["test_root_fail"]
        assert res_root.parent_status == "unknown"
        assert res_root.label_status == "unknown_parent"
        assert res_root.label is None

        gt_c0_strict = sim.reveal_ground_truth("C0", policy="strict_all_fail")
        assert gt_c0_strict.test_results["test_root_fail"].label == 1

        gt_c0_any = sim.reveal_ground_truth("C0", policy="any_fail")
        assert gt_c0_any.test_results["test_root_fail"].label == 1

        # 2. C1: pass -> fail transition -> confirmed_new_fail, label == 1
        gt_c1_conf = sim.reveal_ground_truth("C1", policy="confirmed_new_fail")
        res_c1 = gt_c1_conf.test_results["test_A"]
        assert res_c1.current_status == "fail"
        assert res_c1.parent_status == "pass"
        assert res_c1.label_status == "confirmed_new_fail"
        assert res_c1.label == 1

        assert sim.reveal_ground_truth("C1", policy="strict_all_fail").test_results["test_A"].label == 1
        assert sim.reveal_ground_truth("C1", policy="any_fail").test_results["test_A"].label == 1

        # 3. C2: fail -> fail transition -> already_failing, label == 0
        gt_c2_conf = sim.reveal_ground_truth("C2", policy="confirmed_new_fail")
        res_c2 = gt_c2_conf.test_results["test_A"]
        assert res_c2.current_status == "fail"
        assert res_c2.parent_status == "fail"
        assert res_c2.label_status == "already_failing"
        assert res_c2.label == 0

        assert sim.reveal_ground_truth("C2", policy="strict_all_fail").test_results["test_A"].label == 1
        assert sim.reveal_ground_truth("C2", policy="any_fail").test_results["test_A"].label == 1

        # 4. C3: mixed status (1 pass, 1 fail on same SHA) -> label is None under confirmed_new_fail
        gt_c3_conf = sim.reveal_ground_truth("C3", policy="confirmed_new_fail")
        res_c3_conf = gt_c3_conf.test_results["test_flaky"]
        assert res_c3_conf.current_status == "mixed"
        ambig = getattr(
            res_c3_conf,
            "ambiguous_outcome_same_sha",
            getattr(res_c3_conf, "mixed_outcomes_same_sha", False),
        )
        assert ambig is True
        assert res_c3_conf.n_pass == 1
        assert res_c3_conf.n_fail == 1
        assert res_c3_conf.label is None
        assert res_c3_conf.label_status == "ambiguous_current"

        assert sim.reveal_ground_truth("C3", policy="strict_all_fail").test_results["test_flaky"].label == 0
        assert sim.reveal_ground_truth("C3", policy="any_fail").test_results["test_flaky"].label == 1

    def test_order_by_modes(self):
        from evaluate import evaluate_selection, baseline_recently_failed

        df = pd.DataFrame({
            "commit_idx": [0, 0, 0],
            "testName": ["z_test", "a_test", "m_test"],
            "label": [1, 0, 0],
            "actual_duration": [10.0, 10.0, 10.0],
            "_prob": [0.1, 0.9, 0.5],
            "fail_rate_20": [0.5, 0.0, 0.2],
            "last_fail_dist": [2, -1, 10],
        })
        selected = np.array([True, True, True])

        res_default = evaluate_selection(df, selected, "default", alpha=0.10, order_by="default")
        assert res_default["ttff"] == pytest.approx(30.0)

        res_prob = evaluate_selection(df, selected, "prob", alpha=0.10, order_by="prob")
        assert res_prob["ttff"] == pytest.approx(30.0)

        res_recent = evaluate_selection(df, selected, "recent", alpha=0.10, order_by="recently_failed")
        assert res_recent["ttff"] == pytest.approx(10.0)
        assert res_recent["apfdc"] > res_default["apfdc"]

        mask_rf = baseline_recently_failed(df)
        assert mask_rf[0] == True
        assert mask_rf[1] == False
        assert mask_rf[2] == True

        res_random_1 = evaluate_selection(
            df, selected, "random", alpha=0.10, order_by="random", rng=np.random.default_rng(42)
        )
        res_random_2 = evaluate_selection(
            df, selected, "random", alpha=0.10, order_by="random", rng=np.random.default_rng(42)
        )
        assert res_random_1["ttff"] == pytest.approx(res_random_2["ttff"])
        assert res_random_1["apfdc"] == pytest.approx(res_random_2["apfdc"])
        assert 10.0 <= res_random_1["ttff"] <= 30.0
        assert 0.0 <= res_random_1["apfdc"] <= 1.0


# ===================================================================
# Audit tests: Git parents, three-valued parent_mixed, filtering
# ===================================================================

class TestParentSource:
    """parent_source tracking, real bare Git repo integration, and parent_mixed."""

    def test_parent_source_git_vs_chronological(self, tmp_path: Path):
        from simulator import CIHistorySimulator

        project = "ps_proj"
        _setup_rtp_project(
            tmp_path,
            project,
            job_sha_rows=[(1, "A"), (2, "B"), (3, "C")],
            patch_rows=[("A", "F.java"), ("B", "F.java"), ("C", "F.java")],
            run_rows=[
                (1, "t1", 1.0, 0, 0),
                (2, "t1", 1.0, 1, 0),
                (3, "t1", 1.0, 1, 0),
            ],
        )
        sim = CIHistorySimulator(tmp_path, project)

        assert sim.get_parent_source("A") == "none"
        assert sim.get_parent_source("B") == "chronological_fallback"
        assert sim.get_parent_source("C") == "chronological_fallback"

        assert sim.reveal_ground_truth("B").test_results["t1"].parent_source == "chronological_fallback"
        assert sim.reveal_ground_truth("A").test_results["t1"].parent_source == "none"

    @pytest.mark.skipif(shutil.which("git") is None, reason="git executable not available")
    def test_real_bare_git_repo_parents(self, tmp_path: Path):
        """
        Integration test with a real bare Git repository including a merge commit:
            C0 -> C1 (main)
              \\-> C2 (feature)
            M = merge(C2 into C1)  => first parent C^1 of M is C1, NOT C2!

        Even when job_id(C2) > job_id(C1) so C2 is the chronological predecessor
        of M, CIHistorySimulator must resolve M's parent to C1 via `git rev-list`
        with parent_source == 'git_parent'.
        """
        from simulator import CIHistorySimulator

        project = "git_merge_proj"
        work_repo = tmp_path / "work_repo"
        work_repo.mkdir()

        def _git(args, cwd=work_repo):
            return subprocess.run(
                ["git"] + args,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()

        _git(["init", "-b", "main"])
        _git(["config", "user.email", "test@example.com"])
        _git(["config", "user.name", "Test User"])

        # C0 on main
        (work_repo / "base.java").write_text("class Base {}\n", encoding="utf-8")
        _git(["add", "base.java"])
        _git(["commit", "-m", "C0"])
        sha_c0 = _git(["rev-parse", "HEAD"])

        # C1 on main
        (work_repo / "main.java").write_text("class Main {}\n", encoding="utf-8")
        _git(["add", "main.java"])
        _git(["commit", "-m", "C1"])
        sha_c1 = _git(["rev-parse", "HEAD"])

        # Branch feature from C0 -> C2
        _git(["checkout", "-b", "feature", sha_c0])
        (work_repo / "feat.java").write_text("class Feat {}\n", encoding="utf-8")
        _git(["add", "feat.java"])
        _git(["commit", "-m", "C2"])
        sha_c2 = _git(["rev-parse", "HEAD"])

        # Merge feature into main -> M (first parent C^1 is C1!)
        _git(["checkout", "main"])
        _git(["merge", "--no-ff", "feature", "-m", "Merge feature"])
        sha_m = _git(["rev-parse", "HEAD"])

        # Create bare clone in rtp_dir/repo/{project}
        rtp_dir = tmp_path / "rtp_root"
        bare_dir = rtp_dir / "repo" / project
        bare_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--bare", str(work_repo), str(bare_dir)],
            capture_output=True,
            text=True,
            check=True,
        )

        # Assign job_ids so C2 (job 3) is chronologically right before M (job 4),
        # while C1 (job 2) is the true first parent C^1 of M.
        # C1: T passes; C2: T fails; M: T fails -> relative to C1 (pass), M is confirmed_new_fail!
        _setup_rtp_project(
            rtp_dir,
            project,
            job_sha_rows=[(1, sha_c0), (2, sha_c1), (3, sha_c2), (4, sha_m)],
            patch_rows=[
                (sha_c0, "base.java"),
                (sha_c1, "main.java"),
                (sha_c2, "feat.java"),
                (sha_m, "feat.java"),
            ],
            run_rows=[
                (1, "T", 1.0, 0, 0),  # C0: pass
                (2, "T", 1.0, 0, 0),  # C1: pass (true parent C^1 of M)
                (3, "T", 1.0, 1, 0),  # C2: fail (chronological predecessor of M)
                (4, "T", 1.0, 1, 0),  # M:  fail
            ],
        )

        sim = CIHistorySimulator(rtp_dir, project)

        assert sim.get_parent_source(sha_c1) == "git_parent"
        assert sim.get_parent_source(sha_c2) == "git_parent"
        assert sim.get_parent_source(sha_m) == "git_parent"

        gt_m = sim.reveal_ground_truth(sha_m, policy="confirmed_new_fail")
        res_m = gt_m.test_results["T"]
        assert res_m.parent_sha == sha_c1
        assert res_m.parent_source == "git_parent"
        assert res_m.parent_status == "pass"
        assert res_m.label_status == "confirmed_new_fail"
        assert res_m.label == 1

    def test_parent_mixed_label_status(self, tmp_path: Path):
        """parent=[pass, pass, fail] -> current=[fail] -> label_status='parent_mixed'
        and label is None under confirmed_new_fail (NOT 0 / already_failing)."""
        from simulator import CIHistorySimulator

        project = "pm_proj"
        _setup_rtp_project(
            tmp_path,
            project,
            job_sha_rows=[(1, "A"), (2, "A"), (3, "A"), (4, "B")],
            patch_rows=[("A", "File.java"), ("B", "File.java")],
            run_rows=[
                (1, "test_X", 1.0, 0, 0),
                (2, "test_X", 1.0, 0, 0),
                (3, "test_X", 1.0, 1, 0),
                (4, "test_X", 1.0, 1, 0),
            ],
        )
        sim = CIHistorySimulator(tmp_path, project)

        gt = sim.reveal_ground_truth("B", policy="confirmed_new_fail")
        res = gt.test_results["test_X"]

        assert res.parent_status == "mixed"
        assert res.label_status == "parent_mixed"
        assert res.label is None

        # Sensitivity policies still treat current fail as 1
        assert sim.reveal_ground_truth("B", policy="strict_all_fail").test_results["test_X"].label == 1
        assert sim.reveal_ground_truth("B", policy="any_fail").test_results["test_X"].label == 1


class TestConformalValidation:
    """Input validation in conformal.py."""

    def test_global_threshold_alpha_zero_raises(self):
        from conformal import compute_global_threshold
        with pytest.raises(ValueError, match="alpha"):
            compute_global_threshold(np.array([0.1, 0.2]), alpha=0.0)

    def test_global_threshold_alpha_one_raises(self):
        from conformal import compute_global_threshold
        with pytest.raises(ValueError, match="alpha"):
            compute_global_threshold(np.array([0.1, 0.2]), alpha=1.0)

    def test_global_threshold_alpha_negative_raises(self):
        from conformal import compute_global_threshold
        with pytest.raises(ValueError, match="alpha"):
            compute_global_threshold(np.array([0.1, 0.2]), alpha=-0.1)

    def test_global_threshold_empty_scores_raises(self):
        from conformal import compute_global_threshold
        with pytest.raises(ValueError, match="empty"):
            compute_global_threshold(np.array([]), alpha=0.10)

    def test_global_threshold_nan_scores_raises(self):
        from conformal import compute_global_threshold
        with pytest.raises(ValueError, match="NaN"):
            compute_global_threshold(np.array([0.1, float("nan")]), alpha=0.10)

    def test_global_threshold_inf_scores_raises(self):
        from conformal import compute_global_threshold
        with pytest.raises(ValueError, match="NaN"):
            compute_global_threshold(np.array([0.1, float("inf")]), alpha=0.10)

    def test_tau_residual_alpha_zero_raises(self):
        from conformal import compute_tau_residual
        df = pd.DataFrame({"commit_idx": [0], "testName": ["t"], "label": [1]})
        with pytest.raises(ValueError, match="alpha"):
            compute_tau_residual(df, np.array([0.5]), alpha=0.0)

    def test_tau_residual_alpha_one_raises(self):
        from conformal import compute_tau_residual
        df = pd.DataFrame({"commit_idx": [0], "testName": ["t"], "label": [1]})
        with pytest.raises(ValueError, match="alpha"):
            compute_tau_residual(df, np.array([0.5]), alpha=1.0)

    def test_tau_residual_empty_raises(self):
        from conformal import compute_tau_residual
        df = pd.DataFrame({"commit_idx": [0], "testName": ["t"], "label": [1]})
        with pytest.raises(ValueError, match="empty"):
            compute_tau_residual(df, np.array([]), alpha=0.10)

    def test_tau_residual_nan_raises(self):
        from conformal import compute_tau_residual
        df = pd.DataFrame({"commit_idx": [0], "testName": ["t"], "label": [1]})
        with pytest.raises(ValueError, match="NaN"):
            compute_tau_residual(df, np.array([float("nan")]), alpha=0.10)


class TestBrierScoreAndWilsonCI:
    """Brier score, Wilson CI, and block bootstrap CI for CMR."""

    def test_brier_perfect_prediction(self):
        from models import compute_brier_score
        assert compute_brier_score(np.array([1, 0, 1, 0]), np.array([1.0, 0.0, 1.0, 0.0])) == pytest.approx(0.0)

    def test_brier_worst_prediction(self):
        from models import compute_brier_score
        assert compute_brier_score(np.array([1, 0]), np.array([0.0, 1.0])) == pytest.approx(1.0)

    def test_brier_uninformative(self):
        from models import compute_brier_score
        assert compute_brier_score(np.array([1, 0, 1, 0]), np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.25)

    def test_brier_empty(self):
        from models import compute_brier_score
        assert np.isnan(compute_brier_score(np.array([]), np.array([])))

    def test_wilson_ci_zero_misses_11_defective(self):
        from evaluate import compute_cmr_wilson_ci

        n = 11
        df = pd.DataFrame({"commit_idx": list(range(n)), "label": [1] * n})
        cmr_val, ci_low, ci_high = compute_cmr_wilson_ci(df, np.ones(n, dtype=bool))
        assert cmr_val == pytest.approx(0.0)
        assert ci_low == pytest.approx(0.0)
        assert 0.20 < ci_high < 0.32

    def test_wilson_ci_all_misses(self):
        from evaluate import compute_cmr_wilson_ci

        n = 5
        df = pd.DataFrame({"commit_idx": list(range(n)), "label": [1] * n})
        cmr_val, ci_low, ci_high = compute_cmr_wilson_ci(df, np.zeros(n, dtype=bool))
        assert cmr_val == pytest.approx(1.0)
        assert ci_low > 0.5
        assert ci_high == pytest.approx(1.0)

    def test_block_bootstrap_ci(self):
        from evaluate import compute_cmr_block_bootstrap_ci

        n = 20
        df = pd.DataFrame({"commit_idx": list(range(n)), "label": [1] * n})
        selected = np.array([i % 5 != 0 for i in range(n)], dtype=bool)
        cmr_val, lo, hi = compute_cmr_block_bootstrap_ci(
            df, selected, block_size=4, n_bootstrap=500, rng=np.random.default_rng(7)
        )
        assert cmr_val == pytest.approx(0.20)
        assert 0.0 <= lo <= cmr_val <= hi <= 1.0

    def test_evaluate_selection_includes_ci(self):
        from evaluate import evaluate_selection

        df = pd.DataFrame({
            "commit_idx": [0, 1],
            "testName": ["t1", "t2"],
            "label": [1, 1],
            "actual_duration": [10.0, 10.0],
            "_prob": [0.9, 0.9],
        })
        res = evaluate_selection(df, np.array([True, True]), "test_method", alpha=0.10)
        assert "cmr_ci_low" in res and "cmr_ci_high" in res
        assert "cmr_boot_ci_low" in res and "cmr_boot_ci_high" in res
        assert "conditional_apfdc" in res
        assert 0.0 <= res["cmr_ci_low"] <= res["cmr_ci_high"] <= 1.0


class TestFeatureColsAudit:
    """Feature schema audit and simulator public API."""

    def test_lines_added_not_in_feature_cols(self):
        from features import FEATURE_COLS
        assert "lines_added" not in FEATURE_COLS

    def test_lines_deleted_not_in_feature_cols(self):
        from features import FEATURE_COLS
        assert "lines_deleted" not in FEATURE_COLS

    def test_public_api_no_private_access_needed(self, tmp_path: Path):
        from simulator import CIHistorySimulator

        project = "pub_api_proj"
        _setup_rtp_project(
            tmp_path,
            project,
            job_sha_rows=[(5, "X"), (10, "Y")],
            patch_rows=[("X", "foo.java"), ("X", "bar.java"), ("Y", "baz.java")],
            run_rows=[(5, "t1", 1.0, 0, 0), (10, "t1", 1.0, 1, 0)],
        )
        sim = CIHistorySimulator(tmp_path, project)

        assert sim.get_commit_min_job("X") == sim._commit_min_job["X"]
        assert sim.get_commit_min_job("Y") == sim._commit_min_job["Y"]
        assert set(sim.get_changed_files("X")) == set(sim._patches.get("X", []))
        assert sim.get_changed_files("Y") == sim._patches.get("Y", [])


class TestExperimentFiltering:
    """Verify run_experiment.filter_valid_pairs and threshold minimum helpers."""

    def test_filter_valid_pairs_excludes_none_and_fallback(self):
        from run_experiment import filter_valid_pairs

        df = pd.DataFrame({
            "commit": ["c1", "c2", "c3", "c4", "c5", "c6"],
            "commit_idx": [0, 1, 2, 3, 4, 5],
            "testName": ["t"] * 6,
            "label": [1, 0, None, None, None, 1],
            "label_status": [
                "confirmed_new_fail",
                "no_new_fail",
                "ambiguous_current",
                "parent_mixed",
                "unobserved_candidate",
                "confirmed_new_fail",
            ],
            "parent_source": [
                "git_parent",
                "git_parent",
                "git_parent",
                "git_parent",
                "git_parent",
                "chronological_fallback",
            ],
        })

        # Main experiment: git_parent only, drop all None/indeterminate labels
        df_main, stats_main = filter_valid_pairs(
            df, include_fallback_parents=False, policy="confirmed_new_fail", verbose=False
        )
        assert df_main["commit"].tolist() == ["c1", "c2"]
        assert df_main["label"].tolist() == [1, 0]
        assert "parent_source:chronological_fallback" in stats_main
        assert "ambiguous_current" in stats_main
        assert "parent_mixed" in stats_main
        assert "unobserved_candidate" in stats_main

        # Sensitivity mode: allow chronological_fallback, still drop None/indeterminate
        df_sens, _ = filter_valid_pairs(
            df, include_fallback_parents=True, policy="confirmed_new_fail", verbose=False
        )
        assert df_sens["commit"].tolist() == ["c1", "c2", "c6"]

    def test_min_defective_for_nonzero_threshold(self):
        from run_experiment import min_defective_for_nonzero_threshold

        assert min_defective_for_nonzero_threshold(0.05) == 19
        assert min_defective_for_nonzero_threshold(0.10) == 9
        assert min_defective_for_nonzero_threshold(0.15) == 6
        assert min_defective_for_nonzero_threshold(0.20) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
