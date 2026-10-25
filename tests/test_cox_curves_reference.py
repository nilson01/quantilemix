"""Installed-R conditional Cox curve fixture and checker failures."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_cox_curves_r", ROOT / "validation/check_cox_curves_r.py")
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def records():
    return json.loads(checker.FIXTURE.read_text()), json.loads(checker.INPUT.read_text())


def test_saved_installed_r_curves():
    curves, inputs = records()
    report = checker.check(curves, copy.deepcopy(curves), inputs)
    assert report["status"] == "passed"
    assert len(report["cases"]) == 2
    assert all(case["max_absolute_differences"]["survival"] < 1e-12
               for case in report["cases"])


@pytest.mark.parametrize("change,match", [
    ("version", "2.19.6"), ("count", "two named"),
    ("rows", "profile rows"), ("reference", "saved_R"),
    ("python", "Python"),
])
def test_checker_rejects_changed_evidence(change, match):
    curves, inputs = records()
    live = copy.deepcopy(curves)
    if change == "version":
        live["qgcomp_version"] = "0.0"
    elif change == "count":
        live["cases"].pop()
    elif change == "rows":
        live["cases"][0]["profile_rows"][0] += 1
    elif change == "reference":
        live["cases"][0]["expected"]["survival"][0][0] += 0.1
    else:
        curves["cases"][0]["expected"]["survival"][0][0] += 0.1
        live = copy.deepcopy(curves)
    with pytest.raises(ValueError, match=match):
        checker.check(curves, live, inputs)


def test_checker_requires_rscript(tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(sys, "argv", ["check_cox_curves_r.py",
                        "--rscript", str(tmp_path / "missing-Rscript"),
                        "--library", str(tmp_path),
                        "--output", str(tmp_path / "report.json")])
    with pytest.raises(FileNotFoundError):
        checker.main()
    assert not (tmp_path / "report.json").exists()
