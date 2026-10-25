"""Saved installed-R binary EE cases remain reproducible."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

from quantilemix import fit_binomial_ee

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))
import check_binomial_ee_r  # noqa: E402


def test_saved_reference_compares_with_itself():
    saved = json.loads(check_binomial_ee_r.FIXTURE.read_text())
    report = check_binomial_ee_r.check(saved, saved)
    assert report["status"] == "passed"
    assert len(report["cases"]) == 4


def test_changed_effect_detected():
    saved = json.loads(check_binomial_ee_r.FIXTURE.read_text())
    changed = json.loads(json.dumps(saved))
    changed["cases"][3]["expected"]["psi"] += .01
    with pytest.raises(ValueError, match="saved_R"):
        check_binomial_ee_r.check(saved, changed)


def test_four_level_slope_is_not_one_adjacent_risk_ratio():
    saved = json.loads(check_binomial_ee_r.FIXTURE.read_text())
    inputs = saved["cases"][3]["inputs"]
    x = np.column_stack((inputs["x1"], inputs["x2"]))
    result = fit_binomial_ee(inputs["y"], x, inputs["id"], inputs["z"], q=4, rr=True)
    assert abs(result.effect_ratio - result.intervention_risks[1]/result.intervention_risks[0]) > .1
    assert result.effect_ratio == pytest.approx(1.349464081924916, abs=2e-10)


def test_cli_fails_before_R_if_hash_changes(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["check_binomial_ee_r.py", "--rscript",
                                   "/missing/Rscript", "--library", "/missing/library",
                                   "--output", str(tmp_path / "report.json")])
    monkeypatch.setattr(check_binomial_ee_r, "FIXTURE_SHA", "wrong")
    with pytest.raises(ValueError, match="hash"):
        check_binomial_ee_r.main()
