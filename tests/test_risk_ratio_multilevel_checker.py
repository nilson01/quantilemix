"""Pinned installed-R checker rejects changed reference inputs and outputs."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "validation/check_risk_ratio_multilevel_r.py"
SPEC = importlib.util.spec_from_file_location("check_risk_ratio_multilevel_r", SCRIPT)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def test_saved_reference_passes_checker():
    saved = json.loads(checker.FIXTURE.read_text())
    assert checker.sha(checker.FIXTURE) == checker.FIXTURE_SHA
    assert checker.sha(checker.DRIVER) == checker.DRIVER_SHA
    result = checker.check(saved, copy.deepcopy(saved))
    assert result["status"] == "passed"
    assert len(result["cases"]) == 2


@pytest.mark.parametrize("change,pattern", [
    ("inputs", "reference inputs changed"),
    ("effect", "saved_R"),
    ("version", "installed R qgcomp"),
])
def test_changed_reference_is_rejected(change, pattern):
    saved = json.loads(checker.FIXTURE.read_text())
    live = copy.deepcopy(saved)
    if change == "inputs":
        live["cases"][0]["inputs"]["y"][0] = 1
    elif change == "effect":
        live["cases"][0]["expected"]["psi"] += 0.01
    else:
        live["qgcomp_version"] = "other"
    with pytest.raises(ValueError, match=pattern):
        checker.check(saved, live)
