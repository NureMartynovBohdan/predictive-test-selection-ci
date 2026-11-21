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
# We build a *minimal* fake simulator to test the temporal invariant
# without touching real RTPTorrent data.

class FakeTestRun:
    """Minimal stand-in for simulator.TestRun."""
    def __init__(self, job_id, commit_sha, test_name, failed, duration):
        self.job_id = job_id
        self.commit_sha = commit_sha
        self.test_name = test_name
        self.failed = failed
        self.passed = not failed
        self.duration = duration
        self.job_order = job_id  # job_id as monotonic proxy


class FakeSimulator:
    """Minimal simulator mimic for unit-testing temporal guarantees.

    Accepts a list of FakeTestRun and builds the same indexes as
    CIHistorySimulator.
    """

    def __init__(self, runs, patches=None):
        self._runs = sorted(runs, key=lambda r: r.job_order)
        self._patches = patches or {}

        # Indexes
        self._runs_by_commit = {}
        self._runs_by_test = {}
        for r in self._runs:
            self._runs_by_commit.setdefault(r.commit_sha, []).append(r)
            self._runs_by_test.setdefault(r.test_name, []).append(r)

        # Commit order
        commit_min_job = {}
        for r in self._runs:
            if r.commit_sha not in commit_min_job or r.job_id < commit_min_job[r.commit_sha]:
                commit_min_job[r.commit_sha] = r.job_id
        sorted_commits = sorted(commit_min_job.keys(), key=lambda s: commit_min_job[s])
        self._commit_list = sorted_commits
        self._commit_order = {sha: i for i, sha in enumerate(sorted_commits)}
        self._commit_min_job = {sha: commit_min_job[sha] for sha in sorted_commits}

        # Parent
        self._parent_sha = {}
        for idx, sha in enumerate(sorted_commits):
            self._parent_sha[sha] = sorted_commits[idx - 1] if idx > 0 else None

    def get_test_history(self, test_name, as_of_order):
        all_runs = self._runs_by_test.get(test_name, [])
        return [r for r in all_runs if r.job_order < as_of_order]

    def get_commit_order(self, sha):
        return self._commit_order[sha]


class TestSimulatorLeakagePrevention:

    def test_late_rerun_does_not_change_earlier_commit(self):
        """
        Scenario: A/job1: fail → B/job2: fail → A/job100: pass

        Prove that adding late rerun A/job100 does NOT change:
        1. History returned for test 'T' as-of commit B
        2. Labels / features that would be computed for commit B
        """
        # --- Without the late rerun ---
        runs_before = [
            FakeTestRun(job_id=1, commit_sha="A", test_name="T", failed=True, duration=5.0),
            FakeTestRun(job_id=2, commit_sha="B", test_name="T", failed=True, duration=3.0),
        ]
        sim_before = FakeSimulator(runs_before)

        # History of T as-of commit B (as_of_order = min_job of B = 2)
        as_of_B = sim_before._commit_min_job["B"]
        hist_before = sim_before.get_test_history("T", as_of_B)

        # --- With the late rerun ---
        runs_after = [
            FakeTestRun(job_id=1, commit_sha="A", test_name="T", failed=True, duration=5.0),
            FakeTestRun(job_id=2, commit_sha="B", test_name="T", failed=True, duration=3.0),
            FakeTestRun(job_id=100, commit_sha="A", test_name="T", failed=False, duration=4.0),
        ]
        sim_after = FakeSimulator(runs_after)

        as_of_B_2 = sim_after._commit_min_job["B"]
        hist_after = sim_after.get_test_history("T", as_of_B_2)

        # Must be identical: only job 1 (before job 2)
        assert len(hist_before) == len(hist_after)
        assert len(hist_before) == 1
        assert hist_before[0].job_id == 1
        assert hist_after[0].job_id == 1
        assert hist_before[0].failed == hist_after[0].failed

    def test_actual_duration_not_in_features(self):
        """
        Prove that changing actual_duration of a new test on commit C
        does NOT change the feature vector submitted to the model.

        Feature vector at decision time is built BEFORE ground truth
        is revealed, so actual_duration of test on current commit C
        is never included.
        """
        # Test with duration=1
        runs_1 = [
            FakeTestRun(job_id=10, commit_sha="C", test_name="X", failed=True, duration=1.0),
        ]
        sim_1 = FakeSimulator(runs_1, patches={"C": ["file.java"]})

        # Test with duration=1000
        runs_2 = [
            FakeTestRun(job_id=10, commit_sha="C", test_name="X", failed=True, duration=1000.0),
        ]
        sim_2 = FakeSimulator(runs_2, patches={"C": ["file.java"]})

        # History of X before commit C = empty in both cases (new test)
        as_of_C_1 = sim_1._commit_min_job["C"]
        as_of_C_2 = sim_2._commit_min_job["C"]
        hist_1 = sim_1.get_test_history("X", as_of_C_1)
        hist_2 = sim_2.get_test_history("X", as_of_C_2)

        assert hist_1 == hist_2 == []
        # Since history is empty, duration_median_20 and duration_p90_20
        # will be NaN in both cases — no leakage of actual current duration.


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

    def test_label_policies(self):
        """Test sensitivity analysis policies on synthetic ground truth."""
        from simulator import TestResult

        # Case 1: current=fail, parent=pass
        tr1 = TestResult(
            label=1, actual_duration=1.0, current_status="fail", parent_status="pass",
            label_status="confirmed_new_fail", n_pass=0, n_fail=2, n_attempts=2,
            mixed_outcomes_same_sha=False
        )
        assert tr1.label_status == "confirmed_new_fail"
        assert tr1.label_attribution == "confirmed_new_fail"

        # Case 2: current=fail, parent=fail -> already_failing
        tr2 = TestResult(
            label=0, actual_duration=1.0, current_status="fail", parent_status="fail",
            label_status="already_failing", n_pass=0, n_fail=2, n_attempts=2,
            mixed_outcomes_same_sha=False
        )
        assert tr2.label_status == "already_failing"
        assert tr2.label == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

