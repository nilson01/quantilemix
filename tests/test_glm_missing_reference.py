"""Fixed installed-R references for complete-case binomial and Poisson formulas."""

import hashlib
import json
from pathlib import Path
import warnings

from numpy.testing import assert_allclose, assert_array_equal
import pytest

from quantilemix import fit_binomial_formula, fit_poisson_formula


FIXTURE = Path(__file__).parent / "fixtures/r_glm_missing_2_19_6.json"
SHA256 = "b2ffc8da33baaa7cb86837859f26da251adfcf190fea7f8b1a3311ad0f5de9d7"
CASES = json.loads(FIXTURE.read_text())["cases"]


def test_reference_identity():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == SHA256
    payload = json.loads(FIXTURE.read_text())
    assert payload["reference"]["version"] == "2.19.6"
    assert {case["name"] for case in payload["cases"]} == {
        "binomial_quantized", "binomial_prequantized",
        "poisson_quantized", "poisson_prequantized",
    }


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_complete_case_reference(case):
    inputs, expected = case["inputs"], case["expected"]
    fit = fit_binomial_formula if inputs["family"] == "binomial" else fit_poisson_formula
    result = fit(inputs["formula"], inputs["data"], inputs["exposures"],
                 q=inputs["q"], missing="drop")
    assert result.coefficient_names == ("(Intercept)", "x1", "x2", "z", "g[T.B]", "g[T.C]")
    assert result.training_row_indices == tuple(expected["training_row_indices"])
    assert result.original_row_count == expected["original_row_count"]
    assert result.residual_df == expected["residual_df"]
    assert result.family == inputs["family"]
    assert_array_equal(result.scores, expected["scores"])
    assert (result.breaks is None) == (expected["breaks"] is None)
    if result.breaks is not None:
        assert result.quantization_mode == "quantile"
        for actual, reference in zip(result.breaks, expected["breaks"]):
            assert_allclose(actual, reference, atol=1e-12, rtol=1e-12)
    for field, actual in (
        ("coefficients", result.coefficients), ("covariance", result.covariance),
        ("psi", result.psi), ("variance", result.var_psi), ("se", result.se),
        ("ci", result.ci), ("zstat", result.test_statistic), ("p", result.p_value),
        ("fitted", result.fitted_values),
    ):
        # R's GLM IRLS convergence affects its covariance at about 1e-8 here.
        # Point estimates agree much more closely; retain separate tolerances.
        tolerance = 1e-7 if field in {"covariance", "variance", "se", "ci", "zstat", "p"} else 1e-10
        assert_allclose(actual, expected[field], atol=tolerance, rtol=tolerance,
                        err_msg=f"{case['name']}.{field}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prediction = result.predict_data(inputs["new_data"])
    assert_allclose(prediction, expected["predictions"], atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("family", ("binomial", "poisson"))
def test_pre_omission_cutpoint_and_explicit_failure_boundaries(family):
    case = next(item for item in CASES if item["name"] == f"{family}_quantized")
    inputs = case["inputs"]
    fit = fit_binomial_formula if family == "binomial" else fit_poisson_formula
    result = fit(inputs["formula"], inputs["data"], inputs["exposures"],
                 q=4, missing="drop")
    rows = result.training_row_indices
    reduced = {name: [values[i] for i in rows] for name, values in inputs["data"].items()}
    # This deliberately different operation quantizes only retained rows.
    after_drop = fit(inputs["formula"], reduced, inputs["exposures"], q=4)
    assert result.breaks[0][1] == 0.26171875
    assert after_drop.breaks[0][1] == 0.390625
    assert abs(after_drop.psi - result.psi) > 0.01
    with pytest.raises(ValueError):
        fit(inputs["formula"], inputs["data"], inputs["exposures"], q=4)
    with pytest.raises(ValueError, match="missing must"):
        fit(inputs["formula"], inputs["data"], inputs["exposures"], missing="omit")
    assert result.training_row_indices == tuple(range(23, 128))


@pytest.mark.parametrize("family", ("binomial", "poisson"))
def test_no_complete_rows_rejected(family):
    case = next(item for item in CASES if item["name"] == f"{family}_prequantized")
    inputs = case["inputs"]
    data = dict(inputs["data"])
    data["g"] = [None] * len(data["y"])
    fit = fit_binomial_formula if family == "binomial" else fit_poisson_formula
    with pytest.raises(ValueError, match="no complete"):
        fit(inputs["formula"], data, inputs["exposures"], q=None, missing="drop")
