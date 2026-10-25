"""Pinned installed-R comparisons for named categorical Gaussian inputs."""

import hashlib
import json
from pathlib import Path

from numpy.testing import assert_allclose, assert_array_equal
import pytest

from quantilemix import fit_formula


FIXTURE = Path(__file__).parent / "fixtures/r_formula_gaussian_2_19_6.json"
SHA256 = "6910f0bf2b30c9b3cf5fa6bee52ddc19aa78e5927fc8264f10bedf2c88b08619"


def test_formula_reference_identity():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == SHA256
    payload = json.loads(FIXTURE.read_text())
    assert payload["reference"]["version"] == "2.19.6"
    assert {case["name"] for case in payload["cases"]} == {
        "formula_categorical_quantized", "formula_categorical_prequantized_weighted"
    }


@pytest.mark.parametrize(
    "case", json.loads(FIXTURE.read_text())["cases"], ids=lambda case: case["name"]
)
def test_formula_cases_agree_with_installed_r(case):
    inputs, expected = case["inputs"], case["expected"]
    result = fit_formula(inputs["formula"], inputs["data"], inputs["exposures"],
                         q=inputs["q"], weights=inputs["weights"])
    assert result.coefficient_names[:3] == tuple(expected["coefficient_names"][:3])
    assert result.coefficient_names[3:] == ("g[T.B]", "g[T.C]")
    for key, actual in (
        ("coefficients", result.coefficients),
        ("covariance", result.covariance),
        ("psi", result.psi), ("variance", result.var_psi),
        ("se", result.se), ("ci", result.ci),
        ("t", result.test_statistic), ("p", result.p_value),
        ("dispersion", result.residual_variance),
        ("fitted", result.fitted_values),
        ("predictions", result.predict_data(inputs["new_data"])),
    ):
        assert_allclose(actual, expected[key], rtol=5e-12, atol=5e-12,
                        err_msg=f"{case['name']}.{key}")
    assert_array_equal(result.scores, expected["scores"])
    assert result.p_value_df == expected["p_df"]
    assert result.residual_df == expected["residual_df"]
    assert (result.breaks is None) == (expected["breaks"] is None)
    if expected["breaks"] is not None:
        for a, b in zip(result.breaks, expected["breaks"]):
            assert_allclose(a, b, rtol=5e-12, atol=5e-12)
