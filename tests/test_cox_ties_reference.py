"""Saved installed-R Efron comparisons and optional-checker failures."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_cox_ties_r", ROOT / "validation/check_cox_ties_r.py")
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def fixture():
    return json.loads(checker.FIXTURE.read_text())


def test_saved_installed_r_tied_cases():
    report = checker.check(fixture(), fixture())
    assert report["status"] == "passed"
    assert len(report["cases"]) == 2
    assert all(c["max_absolute_differences"]["log_likelihood"] < 1e-9
               for c in report["cases"])


@pytest.mark.parametrize("change,match", [
    ("version", "2.19.6"), ("count", "two fixed"),
    ("input", "fixed inputs"), ("method", "tie method"),
    ("effect", "saved_R"), ("python", "Python"),
])
def test_checker_rejects_changed_reference(change, match):
    saved = fixture()
    live = copy.deepcopy(saved)
    if change == "version":
        live["qgcomp_version"] = "0.0"
    elif change == "count":
        live["cases"].pop()
    elif change == "input":
        live["cases"][0]["inputs"]["time"][0] += 1
    elif change == "method":
        live["cases"][0]["expected"]["tie_method"] = "breslow"
    elif change == "effect":
        live["cases"][0]["expected"]["psi"] += 0.1
    else:
        both = copy.deepcopy(saved)
        both["cases"][0]["expected"]["psi"] += 0.1
        with pytest.raises(ValueError, match=match):
            checker.check(both, copy.deepcopy(both))
        return
    with pytest.raises(ValueError, match=match):
        checker.check(saved, live)


def test_checker_requires_rscript(tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(sys, "argv", ["check_cox_ties_r.py",
                      "--rscript", str(tmp_path / "missing-Rscript"),
                      "--library", str(tmp_path),
                      "--output", str(tmp_path / "result.json")])
    with pytest.raises(FileNotFoundError):
        checker.main()
    assert not (tmp_path / "result.json").exists()
