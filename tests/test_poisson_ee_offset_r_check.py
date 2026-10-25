"""Reproduce the installed-R Poisson EE offset limitation without requiring R."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))
import check_poisson_ee_offset_r as checker  # noqa: E402


def test_saved_probe_identifies_r_ignored_offset_and_python_rate_fit():
    saved = json.loads(checker.FIXTURE.read_text())
    report = checker.check(saved, saved)
    assert report["status"] == "passed"
    assert report["absolute_effect_difference"] > 0.1


def test_changed_reference_effect_is_rejected():
    saved = json.loads(checker.FIXTURE.read_text())
    changed = json.loads(json.dumps(saved))
    changed["supplied_offset"]["psi"] += .01
    with pytest.raises(ValueError, match="saved_R"):
        checker.check(saved, changed)


def test_optional_checker_requires_pinned_fixture_and_exporter(monkeypatch, tmp_path):
    monkeypatch.setattr(checker, "DRIVER_SHA", "wrong")
    monkeypatch.setattr(sys, "argv", ["check_poisson_ee_offset_r.py", "--rscript",
                                   "/missing/Rscript", "--library", "/missing/library",
                                   "--output", str(tmp_path / "report.json")])
    with pytest.raises(ValueError, match="hash"):
        checker.main()
