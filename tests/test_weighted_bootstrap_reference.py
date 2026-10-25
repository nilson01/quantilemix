"""Fixed installed-R weighted bootstrap values without an R runtime dependency."""

import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from quantilemix._bootstrap import _fit_bootstrap_with_indices


PAYLOAD = json.loads(
    (Path(__file__).parent / "fixtures/r_weighted_bootstrap_2_19_6.json").read_text()
)
CASES = {case["name"]: case for case in PAYLOAD["cases"]}


@pytest.mark.parametrize("name", ("weighted_prequantized", "weighted_quantized"))
def test_fixed_installed_r_weighted_bootstrap_case(name):
    assert PAYLOAD["reference"]["package"] == "qgcomp"
    assert PAYLOAD["reference"]["version"] == "2.19.6"
    assert set(CASES) == {"weighted_prequantized", "weighted_quantized"}
    case = CASES[name]
    inputs = case["inputs"]
    expected = case["expected"]
    result = _fit_bootstrap_with_indices(
        inputs["y"], np.column_stack((inputs["x1"], inputs["x2"])),
        case["resample_indices"], inputs["z"], q=inputs["q"],
        weights=inputs["weights"],
    )
    assert result.B == inputs["B"]
    assert_allclose(result.psi, expected["psi"], rtol=5e-10, atol=5e-10)
    assert_allclose(result.se, expected["se"], rtol=5e-10, atol=5e-10)
    assert_allclose(result.ci, expected["ci"], rtol=5e-10, atol=5e-10)
    assert_allclose(result.bootstrap_effects, expected["bootstrap_effects"],
                    rtol=5e-10, atol=5e-10)
    assert_allclose(result.intervention_means, expected["intervention_means"],
                    rtol=5e-10, atol=5e-10)
    assert_array_equal(result.point_fit.scores, expected["scores"])
    if inputs["q"] is not None:
        for actual, saved in zip(result.point_fit.breaks, expected["breaks"]):
            assert_allclose(actual, saved, rtol=5e-10, atol=5e-10)
