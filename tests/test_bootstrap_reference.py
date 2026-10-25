"""Pinned installed-R Gaussian bootstrap cases with shared row schedules."""

import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from quantilemix._bootstrap import _fit_bootstrap_with_indices


FIXTURE = Path(__file__).parent / "fixtures/r_bootstrap_2_19_6.json"
SHA256 = "3cb75d425742963c99d41a359cb94c4180df840c9c1887eb27a8cea3126cf062"


def test_bootstrap_reference_identity():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == SHA256
    payload = json.loads(FIXTURE.read_text())
    assert payload["reference"]["version"] == "2.19.6"
    assert {case["name"] for case in payload["cases"]} == {
        "bootstrap_prequantized", "bootstrap_quantized_adjusted"
    }


@pytest.mark.parametrize("case", json.loads(FIXTURE.read_text())["cases"],
                         ids=lambda case: case["name"])
def test_bootstrap_cases_agree_with_installed_r(case):
    inputs, expected = case["inputs"], case["expected"]
    x = np.column_stack((inputs["x1"], inputs["x2"]))
    result = _fit_bootstrap_with_indices(
        inputs["y"], x, case["resample_indices"], inputs["z"], q=inputs["q"]
    )
    assert result.B == inputs["B"]
    assert result.p_value_df == expected["df"]
    assert_array_equal(result.point_fit.scores, expected["scores"])
    assert_array_equal(result.intervention_levels, expected["intervention_levels"])
    for key, actual in (
        ("psi", result.psi), ("variance", result.var_psi),
        ("se", result.se), ("ci", result.ci),
        ("statistic", result.test_statistic), ("p", result.p_value),
        ("bootstrap_effects", result.bootstrap_effects),
        ("intervention_means", result.intervention_means),
    ):
        assert_allclose(actual, expected[key], rtol=5e-10, atol=5e-10,
                        err_msg=f"{case['name']}.{key}")
    if inputs["q"] is not None:
        assert len(result.point_fit.breaks) == len(expected["breaks"])
        for a, b in zip(result.point_fit.breaks, expected["breaks"]):
            assert_allclose(a, b, rtol=5e-12, atol=5e-12)
