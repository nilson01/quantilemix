"""Small independent and fault-path checks for the optional study tools."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import warnings

import numpy as np
import pytest

from quantilemix import fit_poisson, fit_poisson_bootstrap


ROOT = Path(__file__).resolve().parents[1] / "validation"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load("poisson_bootstrap_offset_calibration")
checker = _load("check_poisson_bootstrap_offset_calibration")
aggregator = _load("audit_poisson_bootstrap_offset_summary")


def test_independent_score_solution_matches_both_public_methods():
    x, z, exposure_time, y = runner.generate("pilot", 3, 1)
    seed = runner.bootstrap_seed("pilot", 3, 1)
    independent = checker.independent_case(x, z, exposure_time, y, seed, 12)
    fitted = fit_poisson_bootstrap(y, x, z, q=None, B=12, seed=seed, exposure_time=exposure_time)
    baseline = fit_poisson(y, x, z, q=None, exposure_time=exposure_time)
    assert np.allclose(independent["bootstrap"]["bootstrap_effects"], fitted.bootstrap_effects, atol=1e-7)
    assert np.isclose(independent["bootstrap"]["se"], fitted.se, atol=1e-7)
    assert np.isclose(independent["model_based"]["se"], baseline.se, atol=1e-7)
    assert independent["max_score_residual"] < 1e-7


def test_checker_flags_fabricated_failure_and_changed_effect():
    x, z, exposure_time, y = runner.generate("pilot", 1, 0)
    seed = runner.bootstrap_seed("pilot", 1, 0)
    failure = {"status": "failed", "estimate": None, "se": None, "ci": None,
               "p_value": None, "failure_class": "ValueError", "failure_detail": "invented",
               "warnings": [], "bootstrap_effects": None, "intervention_means": None}
    issues = checker.audit_saved_method(failure, x, z, exposure_time, y, seed, 12, 1, 0, "bootstrap")
    assert {issue["code"] for issue in issues} == {"fabricated_failure", "package_fit_failed"}

    reference = checker.independent_case(x, z, exposure_time, y, seed, 12)["bootstrap"]
    changed = dict(reference)
    changed["bootstrap_effects"] = reference["bootstrap_effects"].copy()
    changed["bootstrap_effects"][5] += 0.1
    _, differences = checker.compare_method(changed, reference, 1, 0, "bootstrap")
    assert differences == [{"field": "bootstrap_effects", "indices": [5]}]


def test_plan_seeds_and_input_hash_are_distinct():
    x1, z1, t1, y1 = runner.generate("pilot", 0, 0)
    x2, z2, t2, y2 = runner.generate("study", 0, 0)
    assert runner.input_hash(x1, z1, t1, y1) != runner.input_hash(x2, z2, t2, y2)
    assert runner.bootstrap_seed("pilot", 0, 0) != runner.bootstrap_seed("study", 0, 0)
    assert runner.design("study")["total_bootstrap_fits"] == 800_000


def test_fixed_input_oracle_includes_unequal_times():
    x, z, t, y = runner.generate("pilot", 0, 0)
    assert x[:3].tolist() == [[0, 2, 2, 2], [3, 0, 1, 3], [3, 1, 3, 1]]
    assert z[:3, 0].tolist() == [0, 0, 1]
    assert t[:8].tolist() == [2.0, 2.0, 2.0, 0.5, 2.0, 0.25, 2.0, 2.0]
    assert y[:8].tolist() == [0, 0, 2, 0, 0, 0, 1, 1]
    assert runner.input_hash(x, z, t, y) == (
        "10909799b45813c0d960ab25897986f81abcc698d33cd4ce9bcbeddee7f7ca57")
    assert runner.bootstrap_seed("pilot", 0, 0) == 12971183127358791558


@pytest.fixture(scope="module")
def pilot(tmp_path_factory):
    path = tmp_path_factory.mktemp("poisson-bootstrap-offset") / "pilot.json"
    runner.run("pilot", path)
    assert checker.check(path)["status"] == "passed"
    return path


@pytest.mark.parametrize("damage", ["input_hash", "time_total", "method_record", "missing_case"])
def test_late_damage_preserves_all_available_audits(pilot, tmp_path, damage):
    payload = json.loads(pilot.read_text())
    if damage == "input_hash":
        payload["records"][-1]["input_sha256"] = "0" * 64
    elif damage == "time_total":
        payload["records"][-1]["time_total"] += 0.25
    elif damage == "method_record":
        payload["records"][-1]["bootstrap"] = None
    else:
        payload["records"].pop()
    damaged = tmp_path / "damaged.json"
    damaged.write_text(json.dumps(payload))
    report = checker.check(damaged)
    assert report["status"] == "needs_review"
    assert len(report["case_audit"]) == len(payload["records"])
    assert report["method_pairs"] >= 38
    if damage == "method_record":
        assert any(issue["code"] == "method_integrity_error" for issue in report["issues"])


def test_checker_rejects_different_numpy_version(pilot, tmp_path):
    payload = json.loads(pilot.read_text())
    payload["source_identity"]["numpy"] = "0.0"
    changed = tmp_path / "version.json"
    changed.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="NumPy version differs"):
        checker.check(changed)


def test_checker_records_independent_solver_warning(pilot, monkeypatch):
    original = checker.independent_case

    def warn_once(*args, **kwargs):
        warnings.warn("trial overflow", RuntimeWarning)
        monkeypatch.setattr(checker, "independent_case", original)
        return original(*args, **kwargs)

    monkeypatch.setattr(checker, "independent_case", warn_once)
    report = checker.check(pilot)
    assert report["status"] == "needs_review"
    assert any(issue["code"] == "independent_warning" for issue in report["issues"])
    assert len(report["case_audit"]) == 20


def test_separate_aggregator_rejects_changed_coverage(pilot, tmp_path):
    expected = aggregator.aggregate(pilot)
    assert expected["0"]["bootstrap"]["successful"] == 5
    payload = json.loads(pilot.read_text())
    payload["summaries"]["0"]["bootstrap"]["covered"] += 1
    damaged = tmp_path / "changed-summary.json"
    damaged.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="summary differs"):
        aggregator.aggregate(damaged)
