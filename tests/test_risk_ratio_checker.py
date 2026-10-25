"""The optional R checker rejects altered or incomplete evidence."""

import json
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "validation/check_risk_ratio_r.py"
SPEC = importlib.util.spec_from_file_location("check_risk_ratio_r", SCRIPT)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)
check = CHECK.check


FIXTURE = Path(__file__).parent / "fixtures/r_risk_ratio_2_19_6.json"


def payload():
    return json.loads(FIXTURE.read_text())


def test_saved_payload_passes():
    report = check(payload(), payload())
    assert report["status"] == "passed"
    assert len(report["cases"]) == 2


def test_wrong_version_rejected():
    live = payload()
    live["qgcomp_version"] = "2.19.5"
    with pytest.raises(ValueError, match="2.19.6"):
        check(payload(), live)


def test_changed_r_value_rejected():
    live = payload()
    live["cases"][0]["expected"]["psi"] += .01
    with pytest.raises(ValueError, match="saved_R"):
        check(payload(), live)


def test_changed_inputs_rejected():
    live = payload()
    live["cases"][0]["inputs"]["y"][0] = 0
    with pytest.raises(ValueError, match="inputs changed"):
        check(payload(), live)
