"""Pinned installed-R Poisson offset cases and their numerical limit."""

import hashlib
import json
from pathlib import Path

from numpy.testing import assert_allclose, assert_array_equal
import pytest

from quantilemix import fit_poisson_formula


FIXTURE = Path(__file__).parent / "fixtures/r_poisson_offset_2_19_6.json"
SHA256 = "fbeb2ba9e7b52c6b8e83f182972d2d479e0a34f4bf483a29894118c56c4c1739"


def test_poisson_reference_identity():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == SHA256
    payload = json.loads(FIXTURE.read_text())
    assert payload["reference"]["version"] == "2.19.6"
    assert {case["name"] for case in payload["cases"]} == {
        "offset_exact_scores", "offset_quantized_adjusted"
    }


@pytest.mark.parametrize("case", json.loads(FIXTURE.read_text())["cases"],
                         ids=lambda case: case["name"])
def test_poisson_cases_agree_with_installed_r(case):
    inputs, expected = case["inputs"], case["expected"]
    data = {"y": inputs["y"], "x1": inputs["x1"], "x2": inputs["x2"], "time": inputs["time"]}
    new = {"x1": inputs["new_x1"], "x2": inputs["new_x2"], "time": inputs["new_time"]}
    formula = "y ~ x1 + x2"
    if inputs["z"] is not None:
        data["z"], new["z"] = inputs["z"], inputs["new_z"]
        formula += " + z"
    result = fit_poisson_formula(formula, data, ("x1", "x2"), q=inputs["q"], exposure_time="time")
    assert result.coefficient_names == tuple(expected["coefficient_names"])
    assert result.p_value_df is None
    assert result.residual_df == expected["residual_df"]
    assert_array_equal(result.scores, expected["scores"])
    for key, actual in (
        ("coefficients", result.coefficients), ("psi", result.psi),
        ("fitted", result.fitted_values),
        ("predictions", result.predict_data(new)),
    ):
        assert_allclose(actual, expected[key], rtol=5e-10, atol=5e-10,
                        err_msg=f"{case['name']}.{key}")
    # R glm.fit stops on deviance change before the final Fisher information
    # stabilizes. Python iterates further, so model-based covariance differs
    # by at most about 2e-7 in these cases; the exact-cell test checks the
    # Python information calculation independently at much tighter precision.
    for key, actual in (
        ("covariance", result.covariance), ("variance", result.var_psi),
        ("se", result.se), ("ci", result.ci),
        ("zstat", result.test_statistic), ("p", result.p_value),
    ):
        assert_allclose(actual, expected[key], rtol=2e-6, atol=2e-6,
                        err_msg=f"{case['name']}.{key}")
    assert (result.breaks is None) == (expected["breaks"] is None)
    if result.breaks is not None:
        for a, b in zip(result.breaks, expected["breaks"]):
            assert_allclose(a, b, rtol=5e-12, atol=5e-12)
