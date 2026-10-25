"""Saved installed-R replay and failure checks without an R test dependency."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from quantilemix import fit_quadratic_gaussian_ee


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_quadratic_gaussian_ee_r",
    ROOT / "validation/check_quadratic_gaussian_ee_r.py")
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def fixture():
    return json.loads(checker.FIXTURE.read_text())


def test_saved_installed_r_and_fresh_cluster_arithmetic():
    saved = fixture()
    report = checker.check(saved, copy.deepcopy(saved))
    assert report["status"] == "passed"
    assert len(report["cases"]) == 3
    assert all(case["r_stale_term_meat_difference"] > 1 for case in report["cases"])
    assert callable(fit_quadratic_gaussian_ee)


@pytest.mark.parametrize("change,match", [
    ("version", "2.19.6"), ("case", "three fixed"),
    ("input", "inputs changed"), ("effect", "saved_R"),
    ("fresh", "fresh_meat.Python"),
])
def test_reference_checker_rejects_changed_evidence(change, match):
    saved = fixture()
    live = copy.deepcopy(saved)
    if change == "version":
        live["qgcomp_version"] = "0.0"
    elif change == "case":
        live["cases"].pop()
    elif change == "input":
        live["cases"][0]["inputs"]["y"][0] += 1
    elif change == "effect":
        live["cases"][0]["expected"]["psi"][0] += 0.1
    else:
        both = copy.deepcopy(saved)
        both["cases"][0]["expected"]["fresh_meat"][0][0] += 1
        with pytest.raises(ValueError, match=match):
            checker.check(both, copy.deepcopy(both))
        return
    with pytest.raises(ValueError, match=match):
        checker.check(saved, live)


def test_reference_checker_requires_exporter(tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(sys, "argv", ["check_quadratic_gaussian_ee_r.py",
                      "--rscript", str(tmp_path / "missing-Rscript"),
                      "--library", str(tmp_path),
                      "--output", str(tmp_path / "result.json")])
    with pytest.raises(FileNotFoundError):
        checker.main()
    assert not (tmp_path / "result.json").exists()
