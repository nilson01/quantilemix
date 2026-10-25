"""Fault-injection checks for optional repeated-data validation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "validation" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("poisson_offset_calibration")
checker = load("check_poisson_offset_calibration")


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
    ("exposure_time_total", -1),
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
    pilot_x, pilot_time, pilot_y = runner.generate("pilot", 0, 0)
    study_x, study_time, study_y = runner.generate("study", 0, 0)
    assert runner.input_hash(pilot_x, pilot_time, pilot_y) != runner.input_hash(study_x, study_time, study_y)
    assert runner.design("study")["total_fits"] == 4000
    assert set(pilot_time).issubset({0.25, 0.5, 1.0, 2.0})
    assert np.all(pilot_time > 0)


@pytest.mark.parametrize("mode,expected", [
    ("pilot", "1208ffe0ea486b950b1746e7e03efc5bfd483aeeef8354e3eeebea5923a52a73"),
    ("study", "bf412c830383b78525c9a65fe38ac8e9faa5787cd99d0d54584959296fd6ca19"),
])
def test_protocol_seed_identity(mode, expected):
    generated = runner.generate(mode, 0, 0)
    independently_generated = checker.expected_input(mode, 0, 0)
    assert runner.input_hash(*generated) == expected
    assert checker.expected_hash(*independently_generated) == expected
    assert all(np.array_equal(a, b) for a, b in zip(generated, independently_generated))


def test_checker_rejects_shared_runner_generator_error(pilot, monkeypatch):
    payload = json.loads(pilot.read_text())
    payload["records"][0]["input_sha256"] = "0" * 64
    pilot.write_text(json.dumps(payload))
    monkeypatch.setattr(runner, "generate", lambda *_: (_ for _ in ()).throw(
        AssertionError("checker must not call runner generator")))
    with pytest.raises(ValueError, match="case input differs"):
        checker.check(pilot)


def test_checker_recomputes_summary_denominators(pilot):
    payload = json.loads(pilot.read_text())
    payload["summaries"]["0"]["successful"] -= 1
    pilot.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="summary differs"):
        checker.check(pilot)


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

    def fail_first(x, exposure_time, y):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("injected independent failure")
        return actual(x, exposure_time, y)

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
    audit = checker.check_partial(tmp_path / "interrupted.json.partial.jsonl")
    assert audit["status"] == "partial"
    assert audit["complete_study"] is False
    assert audit["completed"] == [[0, 0]]
    assert len(audit["missing"]) == 19


def test_partial_audit_rejects_wrong_identity(tmp_path, monkeypatch):
    real = runner.generate

    def stop_after_first(mode, cell, repetition):
        if cell == 0 and repetition == 1:
            raise RuntimeError("stop")
        return real(mode, cell, repetition)

    monkeypatch.setattr(runner, "generate", stop_after_first)
    with pytest.raises(RuntimeError, match="stop"):
        runner.run("pilot", tmp_path / "interrupted.json")
    partial = tmp_path / "interrupted.json.partial.jsonl"
    lines = partial.read_text().splitlines()
    row = json.loads(lines[1])
    row["repetition"] = 1
    partial.write_text(lines[0] + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="out of order"):
        checker.check_partial(partial)


def test_solver_success_flag_cannot_hide_bad_residual(monkeypatch):
    x, exposure_time, y = runner.generate("pilot", 0, 0)
    real = checker.optimize.root

    def false_success(*args, **kwargs):
        result = real(*args, **kwargs)
        result.x[:] = 0
        result.success = True
        return result

    monkeypatch.setattr(checker.optimize, "root", false_success)
    with pytest.raises(ValueError, match="residual"):
        checker.independent_fit(x, exposure_time, y)
