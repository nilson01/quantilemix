"""Fault-injection checks for optional repeated-data validation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "validation" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("poisson_calibration")
checker = load("check_poisson_calibration")


@pytest.fixture
def pilot(tmp_path):
    path = tmp_path / "pilot.json"
    runner.run("pilot", path)
    assert checker.check(path)["paired"] == 20
    return path


@pytest.mark.parametrize("field,value", [
    ("input_sha256", "0" * 64),
    ("estimate", 100.0),
    ("variance", 100.0),
    ("count_total", -1),
    ("repetition", 1000),
])
def test_checker_rejects_changed_case(pilot, field, value):
    payload = json.loads(pilot.read_text())
    payload["records"][0][field] = value
    if field in ("estimate", "variance"):
        payload["summaries"] = runner.summarize(payload["records"], payload["plan"])
    pilot.write_text(json.dumps(payload))
    if field in ("estimate", "variance"):
        report = checker.check(pilot)
        assert report["status"] == "needs_review"
        assert report["issues"][0]["cell"] == 0
    else:
        with pytest.raises(ValueError):
            checker.check(pilot)


def test_checker_rejects_changed_summary(pilot):
    payload = json.loads(pilot.read_text())
    payload["summaries"]["0"]["coverage"] = -1
    pilot.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="summary"):
        checker.check(pilot)


def test_pilot_and_study_ids_generate_different_inputs():
    pilot_x, pilot_y = runner.generate("pilot", 0, 0)
    study_x, study_y = runner.generate("study", 0, 0)
    assert runner.input_hash(pilot_x, pilot_y) != runner.input_hash(study_x, study_y)
    assert runner.design("study")["total_fits"] == 4000


def test_new_output_required(pilot):
    with pytest.raises(ValueError, match="new file"):
        runner.run("pilot", pilot)


def test_checker_audits_package_failure(pilot):
    payload = json.loads(pilot.read_text())
    row = payload["records"][0]
    row.update({"failure": "ValueError", "failure_detail": "synthetic failure",
                "estimate": None, "variance": None,
                "ci": None, "p_value": None})
    payload["summaries"] = runner.summarize(payload["records"], payload["plan"])
    pilot.write_text(json.dumps(payload))
    report = checker.check(pilot)
    assert report["status"] == "needs_review"
    assert report["fit_failures"] == 1
    assert report["per_cell"]["0"]["package_failures"] == 1
    assert report["issues"][0]["code"] == "fabricated_package_failure"
    assert report["issues"][1]["code"] == "package_fit_failed"
    assert len(report["case_audit"]) == 20


def test_checker_retains_multiple_mismatches(pilot):
    payload = json.loads(pilot.read_text())
    payload["records"][0]["estimate"] = 100.0
    payload["records"][1]["estimate"] = 100.0
    payload["summaries"] = runner.summarize(payload["records"], payload["plan"])
    pilot.write_text(json.dumps(payload))
    report = checker.check(pilot)
    assert report["status"] == "needs_review"
    assert len(report["issues"]) == 2
    assert report["per_cell"]["0"]["mismatches"] == 2


def test_checker_retains_independent_failure(pilot, monkeypatch):
    actual = checker.independent_fit
    calls = 0

    def fail_first(x, y):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("injected independent failure")
        return actual(x, y)

    monkeypatch.setattr(checker, "independent_fit", fail_first)
    report = checker.check(pilot)
    assert report["status"] == "needs_review"
    assert report["independent_failures"] == 1
    assert report["paired"] == 19
    assert len(report["case_audit"]) == 20
    assert report["issues"][0]["code"] == "independent_fit_failed"


def test_checker_flags_warning(pilot):
    payload = json.loads(pilot.read_text())
    payload["records"][0]["warnings"] = [{"category": "QGCompWarning", "message": "injected warning"}]
    payload["summaries"] = runner.summarize(payload["records"], payload["plan"])
    pilot.write_text(json.dumps(payload))
    report = checker.check(pilot)
    assert report["status"] == "needs_review"
    assert report["issues"][0]["code"] == "package_warning"


def test_runner_saves_unexpected_case_error(tmp_path, monkeypatch):
    real = runner.fit_poisson
    calls = 0

    def fail_first(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected unexpected fit error")
        return real(*args, **kwargs)

    monkeypatch.setattr(runner, "fit_poisson", fail_first)
    output = tmp_path / "unexpected.json"
    result = runner.run("pilot", output)
    assert len(result["records"]) == 20
    assert result["records"][0]["failure"] == "RuntimeError"
    assert result["summaries"]["0"]["failed"] == 1


def test_interrupted_run_keeps_case_journal(tmp_path, monkeypatch):
    real = runner.generate

    def stop_after_first(mode, cell, repetition):
        if cell == 0 and repetition == 1:
            raise RuntimeError("injected interruption")
        return real(mode, cell, repetition)

    monkeypatch.setattr(runner, "generate", stop_after_first)
    output = tmp_path / "interrupted.json"
    with pytest.raises(RuntimeError, match="injected interruption"):
        runner.run("pilot", output)
    assert not output.exists()
    lines = (tmp_path / "interrupted.json.partial.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["repetition"] == 0


def test_solver_success_flag_cannot_hide_bad_residual(monkeypatch):
    x, y = runner.generate("pilot", 0, 0)
    real = checker.optimize.root

    def false_success(*args, **kwargs):
        result = real(*args, **kwargs)
        result.x[:] = 0
        result.success = True
        return result

    monkeypatch.setattr(checker.optimize, "root", false_success)
    with pytest.raises(ValueError, match="residual"):
        checker.independent_fit(x, y)
