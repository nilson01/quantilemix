"""Pinned marginal Cox stage comparison and optional-checker failures."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_cox_bootstrap_r", ROOT / "validation/check_cox_bootstrap_r.py")
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def records():
    return (json.loads(checker.INPUT.read_text()), json.loads(checker.EXPECTED.read_text()),
            json.loads(checker.SOURCE.read_text()))


def test_saved_installed_r_marginal_stages():
    inputs, expected, source = records()
    result = checker.check(inputs, expected, copy.deepcopy(expected), source)
    assert result["status"] == "passed"
    assert [item["degree"] for item in result["cases"]] == [1, 2]
    assert all(max(item["max_absolute_differences"].values()) < 2e-8
               for item in result["cases"])


@pytest.mark.parametrize("change,match", [
    ("version", "2.19.6"), ("count", "two named"),
    ("degree", "degree"), ("effect", "saved_R"),
    ("outcome", "Python_input"), ("python", "Python_R"),
])
def test_checker_rejects_changed_evidence(change, match):
    inputs, expected, source = records()
    live = copy.deepcopy(expected)
    if change == "version":
        live["qgcomp_version"] = "0.0"
    elif change == "count":
        live["cases"].pop()
    elif change == "degree":
        live["cases"][0]["degree"] = 2
    elif change == "effect":
        live["cases"][0]["expected"]["psi"] += .1
    elif change == "outcome":
        inputs["cases"][0]["simulated_time"][0] += 1
    else:
        expected["cases"][0]["expected"]["psi"] += .1
        live = copy.deepcopy(expected)
    with pytest.raises(ValueError, match=match):
        checker.check(inputs, expected, live, source)


def test_checker_requires_rscript(tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(sys, "argv", ["check_cox_bootstrap_r.py",
                        "--rscript", str(tmp_path / "missing-Rscript"),
                        "--library", str(tmp_path),
                        "--output", str(tmp_path / "report.json")])
    with pytest.raises(FileNotFoundError):
        checker.main()
    assert not (tmp_path / "report.json").exists()
