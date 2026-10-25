"""The optional checker rejects changed R cases without requiring R in CI."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "validation/check_interactions_r.py"
SPEC = importlib.util.spec_from_file_location("check_interactions_r", SCRIPT)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


@pytest.mark.parametrize("change", ["input", "schedule", "r_output", "version"])
def test_reference_checker_rejects_changed_live_case(change):
    saved = json.loads(CHECK.FIXTURE.read_text())
    live = copy.deepcopy(saved)
    if change == "input":
        live["cases"][0]["inputs"]["x1"][0] += 1
    elif change == "schedule":
        live["cases"][0]["resample_indices"][0][0] += 1
    elif change == "r_output":
        live["cases"][0]["expected"]["psi"][0] += .1
    else:
        live["qgcomp_version"] = "2.19.5"
    with pytest.raises(ValueError):
        CHECK.check(saved, live)


def test_reference_checker_accepts_unchanged_saved_case():
    saved = json.loads(CHECK.FIXTURE.read_text())
    result = CHECK.check(saved, copy.deepcopy(saved))
    assert result["status"] == "passed"
    assert len(result["cases"]) == 2
