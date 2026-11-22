"""
tests/test_all.py — Unit tests for the refactored pipeline
===========================================================
Covers:
  Stage 1: Simulator leakage prevention
  Stage 4: Conformal calibration correctness
  Stage 5: Metric correctness (TTFF, APFDc)

All tests use synthetic data — no RTPTorrent files required.
Run with:  pytest tests/test_all.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ===================================================================
# Stage 1: Simulator leakage-prevention tests
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

    # tr_all_built_commits.csv
    df_map = pd.DataFrame(job_sha_rows, columns=["tr_job_id", "git_commit_id"])
    df_map.to_csv(tmp_path / "tr_all_built_commits.csv", index=False)

    proj_dir = tmp_path / project
    proj_dir.mkdir(parents=True, exist_ok=True)

    # patches
    df_patches = pd.DataFrame(patch_rows, columns=["sha", "name"])
    df_patches.to_csv(proj_dir / f"{project}-patches.csv", index=False)

    # runs
    df_runs = pd.DataFrame(
        run_rows,
        columns=["travisJobId", "testName", "duration", "failures", "errors"],
    )
    df_runs.to_csv(proj_dir / f"{project}.csv", index=False)


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
        as_of_B_before = sim_before._commit_min_job["B"]
        hist_before = sim_before.get_test_history("T", as_of_B_before)

        as_of_B_after = sim_after._commit_min_job["B"]
        hist_after = sim_after.get_test_history("T", as_of_B_after)

        assert len(hist_before) == len(hist_after) == 1
        assert hist_before[0].job_id == hist_after[0].job_id == 1
        assert hist_before[0].failed == hist_after[0].failed == True
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
        actual_duration of a new test on commit C (1.0 vs 1000.0) does NOT change
        the feature vector submitted to the model.

        Also verify:
        - duration_median_20 and duration_p90_20 are NaN
        - duration_history_known == 0
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
            patch_rows=[("C", "File.java")],
            run_rows=[(10, "X", 1.0, 1, 0)],
        )
        sim_1 = CIHistorySimulator(dir_1, project)

        # Case 2: duration = 1000.0
        dir_2 = tmp_path / "dur_2"
        dir_2.mkdir()
        _setup_rtp_project(
            dir_2,
            project,
            job_sha_rows=[(10, "C")],
            patch_rows=[("C", "File.java")],
            run_rows=[(10, "X", 1000.0, 1, 0)],
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


# ===================================================================
# Stage 4: Conformal calibration tests
# ===================================================================

class TestConformalCalibration:

    def test_conformal_threshold_audit_example(self):
        """
        10 defective calibration commits with residual risks 0.05..0.50
        at alpha=0.10 should give conformal threshold 0.05.

        k = floor(0.10 * 11) = 1
        tau = R_(1) = 0.05 (1st order statistic)
        """
        from conformal import compute_global_threshold

        scores = np.array([0.05, 0.10, 0.15, 0.20, 0.25,
                           0.30, 0.35, 0.40, 0.45, 0.50])
        alpha = 0.10

        # k = floor(0.10 * 11) = 1
        threshold = compute_global_threshold(scores, alpha)
        assert threshold == pytest.approx(0.05)

    def test_compute_tau_residual_audit_example(self):
        """
        Directly invoke compute_tau_residual(df, probs, alpha=0.10) on a synthetic
        DataFrame of 10 defective calibration commits with residual risks 0.05..0.50.
        Verify it returns 0.05 (lower quantile with n+1 correction: k = floor(0.10 * 11) = 1),
        not 0.10.
        """
        from conformal import compute_tau_residual

        # 10 defective calibration commits, 1 test per commit
        df = pd.DataFrame({
            "commit_idx": list(range(10)),
            "testName": [f"test_{i}" for i in range(10)],
            "label": [1] * 10,
        })
        probs = np.array([0.05, 0.10, 0.15, 0.20, 0.25,
                          0.30, 0.35, 0.40, 0.45, 0.50])
        alpha = 0.10

        tau = compute_tau_residual(df, probs, alpha=alpha)

        # With 10 commits and alpha=0.10, k = floor(0.10 * 11) = 1 -> returns 1st order statistic (0.05)
        assert tau == pytest.approx(0.05)
        assert tau != pytest.approx(0.10)

    def test_empty_prefix(self):
        """
        p = [0.01, 0.01], tau = 0.1
        R(0) = 1 - (1-0.01)*(1-0.01) = 0.0199 < 0.1
        → select 0 tests
        """
        from conformal import select_tests_residual

        df = pd.DataFrame({
            "commit_idx": [0, 0],
            "testName": ["A", "B"],
            "label": [0, 0],
        })
        probs = np.array([0.01, 0.01])
        tau = 0.1

        selected = select_tests_residual(df, probs, tau)
        assert len(selected) == 0

    def test_tie_breaking_consistency(self):
        """Calibration and application must produce identical test ordering
        when probabilities are tied."""
        from conformal import _deterministic_sort

        df = pd.DataFrame({
            "commit_idx": [0, 0, 0],
            "testName": ["C_test", "A_test", "B_test"],
            "label": [0, 1, 0],
        })
        probs = np.array([0.5, 0.5, 0.5])  # all tied

        sorted1 = _deterministic_sort(df, probs)
        sorted2 = _deterministic_sort(df.sample(frac=1, random_state=42), probs)

        # Both should sort alphabetically by testName when probs are tied
        assert sorted1["testName"].tolist() == ["A_test", "B_test", "C_test"]
        assert sorted2["testName"].tolist() == ["A_test", "B_test", "C_test"]

    def test_global_threshold_computation(self):
        """Test global threshold with known examples."""
        from conformal import compute_global_threshold

        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        # alpha=0.10, k = floor(0.10 * 6) = 0 → threshold = 0.0
        assert compute_global_threshold(scores, 0.10) == 0.0

        # alpha=0.20, k = floor(0.20 * 6) = 1 → threshold = 0.1
        assert compute_global_threshold(scores, 0.20) == pytest.approx(0.1)

        # alpha=0.50, k = floor(0.50 * 6) = 3 → threshold = 0.3
        assert compute_global_threshold(scores, 0.50) == pytest.approx(0.3)

    def test_residual_risk_function(self):
        """Verify R(j) calculation."""
        from conformal import _residual_risk

        probs = np.array([0.9, 0.5, 0.1])  # sorted descending

        # R(0) = 1 - (1-0.9)*(1-0.5)*(1-0.1) = 1 - 0.1*0.5*0.9 = 1 - 0.045 = 0.955
        assert _residual_risk(probs, 0) == pytest.approx(0.955)

        # R(1) = 1 - (1-0.5)*(1-0.1) = 1 - 0.45 = 0.55
        assert _residual_risk(probs, 1) == pytest.approx(0.55)

        # R(2) = 1 - (1-0.1) = 0.1
        assert _residual_risk(probs, 2) == pytest.approx(0.1)

        # R(3) = 0 (no remaining)
        assert _residual_risk(probs, 3) == pytest.approx(0.0)


# ===================================================================
# Stage 5: Metrics tests
# ===================================================================

class TestMetrics:

    def test_ttff_and_apfdc_controlled(self):
        """
        Test case: 2 tests ordered [test_A (rank 1), test_B (rank 2)]
        test_A: duration=1000s, label=0 (pass)
        test_B: duration=1s,    label=1 (fail)

        Expected:
        - TTFF = 1000 + 1 = 1001.0 seconds
        - APFDc: m=1 fault, T=1001 total duration
          C_1 = 1001 (cumulative to rank 2 where fault found)
          t_1 = 1 (duration of fault-detecting test)
          APFDc = 1 - (1001 - 0.5) / (1 * 1001)
                = 1 - 1000.5/1001
                ≈ 0.000499...
        """
        from evaluate import compute_ttff, compute_apfdc

        df = pd.DataFrame({
            "commit_idx": [0, 0],
            "testName": ["test_A", "test_B"],
            "label": [0, 1],
            "actual_duration": [1000.0, 1.0],
            "_rank": [1, 2],
        })

        ttff = compute_ttff(df, duration_col="actual_duration")
        assert ttff == pytest.approx(1001.0)

        apfdc = compute_apfdc(df, duration_col="actual_duration")
        expected = 1.0 - (1001.0 - 0.5) / (1 * 1001.0)
        assert apfdc == pytest.approx(expected, abs=1e-6)
        # Should be very small, approximately 0.0005
        assert apfdc < 0.001

    def test_ttff_first_test_fails(self):
        """
        test_A: duration=5s, label=1 (fail), rank=1
        test_B: duration=100s, label=0 (pass), rank=2

        TTFF = 5.0
        APFDc = 1 - (5 - 5/2) / (1 * 105) = 1 - 2.5/105 ≈ 0.976
        """
        from evaluate import compute_ttff, compute_apfdc

        df = pd.DataFrame({
            "commit_idx": [0, 0],
            "testName": ["test_A", "test_B"],
            "label": [1, 0],
            "actual_duration": [5.0, 100.0],
            "_rank": [1, 2],
        })

        ttff = compute_ttff(df, duration_col="actual_duration")
        assert ttff == pytest.approx(5.0)

        apfdc = compute_apfdc(df, duration_col="actual_duration")
        expected = 1.0 - (5.0 - 2.5) / (1 * 105.0)
        assert apfdc == pytest.approx(expected, abs=1e-6)
        assert apfdc > 0.95

    def test_multiple_faults_apfdc(self):
        """
        3 tests: rank 1 (pass, 10s), rank 2 (fail, 2s), rank 3 (fail, 3s)
        m=2, T=15
        C_1=12 (10+2), t_1=2;  C_2=15 (10+2+3), t_2=3
        APFDc = 1 - ((12-1) + (15-1.5)) / (2*15) = 1 - (11+13.5)/30 = 1 - 24.5/30
        """
        from evaluate import compute_apfdc

        df = pd.DataFrame({
            "commit_idx": [0, 0, 0],
            "testName": ["A", "B", "C"],
            "label": [0, 1, 1],
            "actual_duration": [10.0, 2.0, 3.0],
            "_rank": [1, 2, 3],
        })

        apfdc = compute_apfdc(df, duration_col="actual_duration")
        expected = 1.0 - ((12.0 - 1.0) + (15.0 - 1.5)) / (2 * 15.0)
        assert apfdc == pytest.approx(expected, abs=1e-6)

    def test_cmr_all_detected(self):
        """All defective commits have at least one selected failing test → CMR = 0."""
        from evaluate import compute_cmr

        df = pd.DataFrame({
            "commit_idx": [0, 0, 1, 1],
            "label": [0, 1, 1, 0],
        })
        selected = np.array([True, True, True, True])

        cmr = compute_cmr(df, selected)
        assert cmr == pytest.approx(0.0)

    def test_cmr_all_missed(self):
        """No selected test catches any failure → CMR = 1.0."""
        from evaluate import compute_cmr

        df = pd.DataFrame({
            "commit_idx": [0, 0, 1, 1],
            "label": [0, 1, 1, 0],
        })
        # Select only the passing tests
        selected = np.array([True, False, False, True])

        cmr = compute_cmr(df, selected)
        assert cmr == pytest.approx(1.0)

    def test_cmr_partial(self):
        """One of two defective commits missed → CMR = 0.5."""
        from evaluate import compute_cmr

        df = pd.DataFrame({
            "commit_idx": [0, 0, 1, 1],
            "label": [0, 1, 1, 0],
        })
        # Catch first defective commit, miss second
        selected = np.array([True, True, False, True])

        cmr = compute_cmr(df, selected)
        assert cmr == pytest.approx(0.5)


# ===================================================================
# Stage 4 (additional): chrono_split_4 test
# ===================================================================

class TestChronoSplit:

    def test_4_way_split_no_overlap(self):
        """4-way split produces non-overlapping, chronological partitions."""
        from models import chrono_split_4

        # 100 commits, 2 tests each
        df = pd.DataFrame({
            "commit_idx": np.repeat(np.arange(100), 2),
            "testName": ["A", "B"] * 100,
            "label": np.random.randint(0, 2, 200),
        })

        train, prob_cal, conf_cal, test = chrono_split_4(df)

        # No overlap
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

        # Chronological ordering
        if train_commits and prob_commits:
            assert max(train_commits) < min(prob_commits)
        if prob_commits and conf_commits:
            assert max(prob_commits) < min(conf_commits)
        if conf_commits and test_commits:
            assert max(conf_commits) < min(test_commits)

        # All commits covered
        all_covered = train_commits | prob_commits | conf_commits | test_commits
        assert all_covered == set(range(100))


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
        assert "com" in tokens
        assert "example" in tokens
        assert "foo" in tokens
        assert "bar" in tokens
        assert "test" in tokens


# ===================================================================
# Stage 4 & 5: Cost-aware prioritization & Label policy tests
# ===================================================================

class TestCostAwareAndPolicy:

    def test_cost_aware_improves_ttff_and_apfdc(self):
        """When probabilities are equal or nearly equal, cost-aware ordering
        (p / duration) prioritizes fast tests over slow ones, reducing TTFF
        and improving APFDc.
        """
        from evaluate import evaluate_selection

        # Two tests both with high prob, one slow (100s, pass), one fast (1s, fail)
        df = pd.DataFrame({
            "commit_idx": [0, 0],
            "testName": ["slow_pass", "fast_fail"],
            "label": [0, 1],
            "actual_duration": [100.0, 1.0],
            "duration_median_20": [100.0, 1.0],
            "_prob": [0.90, 0.89],  # slow test has slightly higher raw probability
        })
        selected_mask = np.array([True, True])

        # Probability ordering: puts slow_pass first
        res_prob = evaluate_selection(
            df, selected_mask, "prob_order", alpha=0.10, order_by="prob"
        )

        # Cost-aware ordering: fast_fail priority = 0.89/1 = 0.89, slow priority = 0.90/100 = 0.009
        # Puts fast_fail first!
        res_cost = evaluate_selection(
            df, selected_mask, "cost_order", alpha=0.10, order_by="cost_aware"
        )

        # TTFF with cost-aware should be 1.0s, while with prob it is 101.0s
        assert res_cost["ttff"] == pytest.approx(1.0)
        assert res_prob["ttff"] == pytest.approx(101.0)
        assert res_cost["apfdc"] > res_prob["apfdc"]

    def test_label_policies(self, tmp_path: Path):
        """
        Verify label policies with real CIHistorySimulator:
        - pass -> fail transition gives label == 1 (confirmed_new_fail)
        - fail -> fail transition gives label == 0 (already_failing)
        - root / no-parent commit gives unknown_parent
        - policy differences for strict_all_fail and any_fail, including mixed status.
        """
        from simulator import CIHistorySimulator

        project = "label_test_proj"
        proj_dir = tmp_path / "label_proj"
        proj_dir.mkdir()

        # Commits:
        # C0 (root): job 1: test_A pass, test_root_fail fail
        # C1 (parent C0): job 2: test_A fail (pass -> fail)
        # C2 (parent C1): job 3: test_A fail (fail -> fail)
        # C3 (parent C2): job 4: test_flaky pass, job 5: test_flaky fail (mixed on same sha)
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
                (1, "test_A", 1.0, 0, 0),         # C0: test_A pass
                (1, "test_root_fail", 1.0, 1, 0), # C0: test_root_fail fail
                (2, "test_A", 1.0, 1, 0),         # C1: test_A fail (pass -> fail)
                (3, "test_A", 1.0, 1, 0),         # C2: test_A fail (fail -> fail)
                (4, "test_flaky", 1.0, 0, 0),     # C3: test_flaky pass
                (5, "test_flaky", 1.0, 1, 0),     # C3: test_flaky fail
            ],
        )
        sim = CIHistorySimulator(proj_dir, project)

        # 1. C0: root commit, no parent -> unknown_parent
        gt_c0_conf = sim.reveal_ground_truth("C0", policy="confirmed_new_fail")
        res_root = gt_c0_conf.test_results["test_root_fail"]
        assert res_root.parent_status == "unknown"
        assert res_root.label_status == "unknown_parent"
        assert res_root.label == 0

        gt_c0_strict = sim.reveal_ground_truth("C0", policy="strict_all_fail")
        assert gt_c0_strict.test_results["test_root_fail"].label == 1

        gt_c0_any = sim.reveal_ground_truth("C0", policy="any_fail")
        assert gt_c0_any.test_results["test_root_fail"].label == 1

        # 2. C1: pass -> fail transition -> confirmed_new_fail
        gt_c1_conf = sim.reveal_ground_truth("C1", policy="confirmed_new_fail")
        res_c1 = gt_c1_conf.test_results["test_A"]
        assert res_c1.current_status == "fail"
        assert res_c1.parent_status == "pass"
        assert res_c1.label_status == "confirmed_new_fail"
        assert res_c1.label == 1

        gt_c1_strict = sim.reveal_ground_truth("C1", policy="strict_all_fail")
        assert gt_c1_strict.test_results["test_A"].label == 1

        gt_c1_any = sim.reveal_ground_truth("C1", policy="any_fail")
        assert gt_c1_any.test_results["test_A"].label == 1

        # 3. C2: fail -> fail transition -> already_failing
        gt_c2_conf = sim.reveal_ground_truth("C2", policy="confirmed_new_fail")
        res_c2 = gt_c2_conf.test_results["test_A"]
        assert res_c2.current_status == "fail"
        assert res_c2.parent_status == "fail"
        assert res_c2.label_status == "already_failing"
        assert res_c2.label == 0  # Confirmed new fail ignores already failing tests

        gt_c2_strict = sim.reveal_ground_truth("C2", policy="strict_all_fail")
        assert gt_c2_strict.test_results["test_A"].label == 1

        gt_c2_any = sim.reveal_ground_truth("C2", policy="any_fail")
        assert gt_c2_any.test_results["test_A"].label == 1

        # 4. C3: mixed status (1 pass, 1 fail on same SHA)
        gt_c3_conf = sim.reveal_ground_truth("C3", policy="confirmed_new_fail")
        res_c3_conf = gt_c3_conf.test_results["test_flaky"]
        assert res_c3_conf.current_status == "mixed"
        assert res_c3_conf.mixed_outcomes_same_sha == True
        assert res_c3_conf.n_pass == 1
        assert res_c3_conf.n_fail == 1
        assert res_c3_conf.label == 0

        gt_c3_strict = sim.reveal_ground_truth("C3", policy="strict_all_fail")
        res_c3_strict = gt_c3_strict.test_results["test_flaky"]
        assert res_c3_strict.label == 0  # strict_all_fail requires current_status == 'fail'

        gt_c3_any = sim.reveal_ground_truth("C3", policy="any_fail")
        res_c3_any = gt_c3_any.test_results["test_flaky"]
        assert res_c3_any.label == 1  # any_fail requires n_fail > 0

    def test_order_by_modes(self):
        """Verify order_by default, random, recently_failed, and prob in evaluate_selection."""
        from evaluate import evaluate_selection, baseline_recently_failed

        df = pd.DataFrame({
            "commit_idx": [0, 0, 0],
            "testName": ["z_test", "a_test", "m_test"],
            "label": [1, 0, 0],
            "actual_duration": [10.0, 10.0, 10.0],
            "_prob": [0.1, 0.9, 0.5],  # a_test has highest prob, z_test has lowest
            "fail_rate_20": [0.5, 0.0, 0.2],
            "last_fail_dist": [2, -1, 10],  # z_test failed 2 jobs ago, m_test 10, a_test never
        })
        selected = np.array([True, True, True])

        # 1. default: alphabetical by testName -> a_test (rank 1), m_test (rank 2), z_test (rank 3)
        # z_test (label=1) is at rank 3 -> TTFF = 30s
        res_default = evaluate_selection(df, selected, "default", alpha=0.10, order_by="default")
        assert res_default["ttff"] == pytest.approx(30.0)

        # 2. prob: sorted by _prob descending -> a_test (0.9), m_test (0.5), z_test (0.1)
        res_prob = evaluate_selection(df, selected, "prob", alpha=0.10, order_by="prob")
        assert res_prob["ttff"] == pytest.approx(30.0)

        # 3. recently_failed: sorted by last_fail_dist ascending -> z_test (dist 2, rank 1!)
        # z_test is at rank 1 -> TTFF = 10s
        res_recent = evaluate_selection(df, selected, "recent", alpha=0.10, order_by="recently_failed")
        assert res_recent["ttff"] == pytest.approx(10.0)
        assert res_recent["apfdc"] > res_default["apfdc"]

        # 4. baseline_recently_failed selector selects tests with fail_rate_20 > 0
        mask_rf = baseline_recently_failed(df)
        assert mask_rf[0] == True   # z_test (0.5)
        assert mask_rf[1] == False  # a_test (0.0)
        assert mask_rf[2] == True   # m_test (0.2)

        # 5. random: order_by="random" with explicitly passed rng
        rng_seed = 42
        res_random_1 = evaluate_selection(
            df, selected, "random", alpha=0.10, order_by="random", rng=np.random.default_rng(rng_seed)
        )
        res_random_2 = evaluate_selection(
            df, selected, "random", alpha=0.10, order_by="random", rng=np.random.default_rng(rng_seed)
        )
        # Deterministic with identical rng
        assert res_random_1["ttff"] == pytest.approx(res_random_2["ttff"])
        assert res_random_1["apfdc"] == pytest.approx(res_random_2["apfdc"])
        assert 10.0 <= res_random_1["ttff"] <= 30.0
        assert 0.0 <= res_random_1["apfdc"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

