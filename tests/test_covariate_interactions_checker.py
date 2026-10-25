"""Optional checker rejects changed installed-R interaction evidence."""

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "validation/check_covariate_interactions_r.py"
SPEC = importlib.util.spec_from_file_location("check_covariate_interactions_r", SCRIPT)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def payload():
    return json.loads(CHECK.FIXTURE.read_text())


def test_unchanged_saved_cases_pass():
    report = CHECK.check(payload(), payload())
    assert report["status"] == "passed"
    assert len(report["cases"]) == 2


@pytest.mark.parametrize("change,pattern", [
    ("version", "2.19.6"), ("input", "inputs or bootstrap schedule changed"),
    ("schedule", "inputs or bootstrap schedule changed"), ("output", "saved_R"),
])
def test_changed_reference_rejected(change, pattern):
    live = payload()
    if change == "version":
        live["qgcomp_version"] = "2.19.5"
    elif change == "input":
        live["cases"][0]["inputs"]["z"][0] += 1
    elif change == "schedule":
        live["cases"][0]["resample_indices"][0][0] += 1
    else:
        live["cases"][0]["expected"]["psi"][0] += .01
    with pytest.raises(ValueError, match=pattern):
        CHECK.check(payload(), live)
