"""Pinned optional-checker success and changed-evidence failures."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_cox_survcurve_r", ROOT / "validation/check_cox_survcurve_r.py")
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def records():
    return (json.loads(checker.EXPECTED.read_text()), json.loads(checker.SOURCE.read_text()),
            json.loads(checker.STAGE.read_text()))


def test_saved_eight_installed_r_tables():
    expected, source, stage = records()
    report = checker.check(expected, copy.deepcopy(expected), source, stage)
    assert report["status"] == "passed"
    assert [case["degree"] for case in report["cases"]] == [1, 2]
    assert all(max(case["maximum_absolute_differences"].values()) < 2e-8
               for case in report["cases"])


@pytest.mark.parametrize("change,match", [
    ("version", "2.19.6"), ("count", "degree-one"),
    ("r_curve", "saved_R"), ("source", "fixed survival"),
    ("stage", "saved_stage"), ("python", "Python_R"),
])
def test_changed_evidence_rejected(change, match):
    expected, source, stage = records()
    live = copy.deepcopy(expected)
    if change == "version":
        live["qgcomp_version"] = "0.0"
    elif change == "count":
        live["cases"].pop()
    elif change == "r_curve":
        live["cases"][0]["curves"]["mdfpop"]["surv"][0] += .1
    elif change == "source":
        source["cases"][1]["name"] = "wrong"
    elif change == "stage":
        stage["cases"][1]["simulated_time"][0] += 1
    else:
        expected["cases"][0]["psi"] += .1
        live = copy.deepcopy(expected)
    with pytest.raises(ValueError, match=match):
        checker.check(expected, live, source, stage)


def test_missing_rscript_does_not_write_report(tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(sys, "argv", ["check_cox_survcurve_r.py",
                        "--rscript", str(tmp_path / "missing-Rscript"),
                        "--library", str(tmp_path),
                        "--output", str(tmp_path / "report.json")])
    with pytest.raises(FileNotFoundError):
        checker.main()
    assert not (tmp_path / "report.json").exists()
