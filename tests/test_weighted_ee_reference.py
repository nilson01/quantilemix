"""Saved installed-R weighted EE cases and checker failure paths."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))
import check_weighted_ee_r  # noqa: E402


def test_saved_reference_compares_with_itself():
    saved = json.loads(check_weighted_ee_r.FIXTURE.read_text())
    report = check_weighted_ee_r.check(saved, saved)
    assert report["status"] == "passed"
    assert len(report["cases"]) == 4


def test_changed_weighted_effect_detected():
    saved = json.loads(check_weighted_ee_r.FIXTURE.read_text())
    changed = json.loads(json.dumps(saved))
    changed["cases"][3]["expected"]["psi"] += .01
    with pytest.raises(ValueError, match="saved_R"):
        check_weighted_ee_r.check(saved, changed)


def test_changed_case_weight_detected():
    saved = json.loads(check_weighted_ee_r.FIXTURE.read_text())
    changed = json.loads(json.dumps(saved))
    changed["cases"][0]["inputs"]["w"][0] += 1
    with pytest.raises(ValueError, match="inputs changed"):
        check_weighted_ee_r.check(saved, changed)


def test_cli_fails_before_R_if_hash_changes(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["check_weighted_ee_r.py", "--rscript",
                                   "/missing/Rscript", "--library", "/missing/library",
                                   "--output", str(tmp_path / "report.json")])
    monkeypatch.setattr(check_weighted_ee_r, "FIXTURE_SHA", "wrong")
    with pytest.raises(ValueError, match="hash"):
        check_weighted_ee_r.main()
