"""Pinned odds-ratio reference checker rejects altered or missing evidence."""

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "validation/check_odds_ratio_r.py"
SPEC = importlib.util.spec_from_file_location("check_odds_ratio_r", SCRIPT)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def test_saved_reference_passes_checker():
    saved = json.loads(checker.FIXTURE.read_text())
    assert checker.sha(checker.FIXTURE) == checker.FIXTURE_SHA
    assert checker.sha(checker.DRIVER) == checker.DRIVER_SHA
    report = checker.check(saved, copy.deepcopy(saved))
    assert report["status"] == "passed"
    assert len(report["cases"]) == 2
    assert all(case["score_difference"] == 0 for case in report["cases"])


@pytest.mark.parametrize("change,pattern", [
    ("inputs", "reference inputs changed"),
    ("effect", "saved_R"),
    ("version", "installed R qgcomp"),
    ("shape", "shape or finiteness differs"),
])
def test_changed_reference_is_rejected(change, pattern):
    saved = json.loads(checker.FIXTURE.read_text())
    live = copy.deepcopy(saved)
    if change == "inputs":
        live["cases"][0]["inputs"]["y"][0] = 1 - live["cases"][0]["inputs"]["y"][0]
    elif change == "effect":
        live["cases"][0]["expected"]["psi"] += .01
    elif change == "version":
        live["qgcomp_version"] = "other"
    else:
        live["cases"][0]["expected"]["ci"] = [1]
    with pytest.raises(ValueError, match=pattern):
        checker.check(saved, live)


def test_missing_rscript_leaves_no_report(tmp_path, monkeypatch):
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--rscript",
                                     str(tmp_path / "missing-Rscript"),
                                     "--library", str(tmp_path),
                                     "--output", str(output)])
    with pytest.raises(FileNotFoundError):
        checker.main()
    assert not output.exists()
