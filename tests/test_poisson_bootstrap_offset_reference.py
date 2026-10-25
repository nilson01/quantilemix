"""Pinned installed-R unequal-time cases are checked without R in CI."""

import importlib.util
import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest

from quantilemix._poisson_bootstrap import _from_indices


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/r_poisson_bootstrap_offset_2_19_6.json"
SCRIPT = ROOT / "validation/check_poisson_bootstrap_offset_r.py"
SPEC = importlib.util.spec_from_file_location("check_poisson_bootstrap_offset_r", SCRIPT)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def test_fixed_offset_cases_and_independent_arithmetic():
    cases = json.loads(FIXTURE.read_text())["cases"]
    assert len(cases) == 2
    for case in cases:
        data, expected = case["inputs"], case["expected"]
        x = np.column_stack((data["x1"], data["x2"]))
        z = None if data["z"] is None else np.asarray(data["z"])[:, None]
        result = _from_indices(data["y"], x, data["indices"], z,
                               q=data["q"], exposure_time=data["time"])
        for actual, reference in ((result.psi, expected["psi"]),
                                  (result.se, expected["se"]),
                                  (result.ci, expected["ci"]),
                                  (result.p_value, expected["p"]),
                                  (result.intervention_means, expected["means"]),
                                  (result.bootstrap_effects, expected["effects"]),
                                  (result.point_fit.scores, expected["scores"])):
            assert_allclose(actual, reference, rtol=0, atol=2e-9)
        independent_psi, effects, means = CHECK.independent_effects(data)
        assert_allclose(independent_psi, expected["psi"], rtol=0, atol=2e-9)
        assert_allclose(effects, expected["effects"], rtol=0, atol=2e-9)
        assert_allclose(means, expected["means"], rtol=0, atol=2e-9)


def test_optional_checker_accepts_unchanged_export():
    saved = json.loads(FIXTURE.read_text())
    live = {"qgcomp_version": "2.19.6", "cases": saved["cases"]}
    assert CHECK.check(saved, live)["status"] == "passed"


@pytest.mark.parametrize("field,pattern", [("version", "2.19.6"),
                                            ("input", "inputs changed"),
                                            ("output", "saved_R")])
def test_optional_checker_rejects_changed_reference(field, pattern):
    saved = json.loads(FIXTURE.read_text())
    live = {"qgcomp_version": "2.19.6", "cases": json.loads(json.dumps(saved["cases"]))}
    if field == "version":
        live["qgcomp_version"] = "2.19.5"
    elif field == "input":
        live["cases"][0]["inputs"]["time"][0] += 1
    else:
        live["cases"][0]["expected"]["psi"] += .01
    with pytest.raises(ValueError, match=pattern):
        CHECK.check(saved, live)
