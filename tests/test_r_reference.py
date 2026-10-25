"""Fixed comparisons with unmodified, pinned R qgcomp source outputs."""
import json
from pathlib import Path

import numpy as np
import pytest

from quantilemix import fit

PAYLOAD = json.loads((Path(__file__).parent / "fixtures/r_gaussian_2_19_6.json").read_text())


def test_reference_identity_and_case_inventory():
    reference = PAYLOAD["reference"]
    assert reference["package"] == "qgcomp"
    assert reference["version"] == "2.19.6"
    assert reference["mode"] == "unmodified source functions; not installed package"
    assert reference["source_md5"] == {
        "R/base.R": "853fcb3a29f9477bcc68e1b8e588ca44",
        "R/base_utility.R": "8b3aafbeb2f51bb26757634feaf5f36b",
    }
    assert {case["name"] for case in PAYLOAD["cases"]} == {
        "single_exposure", "prediction_boundaries", "adjusted_positive",
        "adjusted_mixed", "adjusted_cancellation", "tied_endpoints",
        "explicit_breaks", "prequantized", "quantile_rounding_boundaries",
    }
    assert len(PAYLOAD["cases"]) == 9


@pytest.mark.parametrize("case", PAYLOAD["cases"], ids=lambda case: case["name"])
def test_pinned_r_gaussian_reference(case):
    inputs, expected = case["inputs"], case["expected"]
    result = fit(inputs["y"], inputs["exposures"], inputs["covariates"],
                 q=inputs["q"], breaks=inputs["breaks"], alpha=inputs["alpha"])
    pairs = {
        "coefficients": result.coefficients,
        "covariance": result.covariance,
        "psi": result.psi,
        "variance": result.var_psi,
        "se": result.se,
        "ci": result.ci,
        "statistic": result.test_statistic,
        "positive_psi": result.pos_psi,
        "negative_psi": result.neg_psi,
        "fitted": result.fitted_values,
        "predictions": result.predict(inputs["new_exposures"], inputs["new_covariates"]),
    }
    for field, actual in pairs.items():
        np.testing.assert_allclose(actual, expected[field], atol=1e-10, rtol=1e-8,
                                   err_msg=f"{case['name']}: {field}")
    assert result.p_value == pytest.approx(expected["p_value"], abs=1e-10)
    assert result.residual_df == expected["df_residual"]
    assert result.p_value_df == expected["df_pvalue"]
    np.testing.assert_array_equal(result.transform(inputs["exposures"]).scores,
                                  expected["scores"])
    if expected["breaks"] is None:
        assert result.breaks is None
    else:
        for actual, ref in zip(result.breaks, expected["breaks"], strict=True):
            np.testing.assert_allclose(actual, ref, atol=1e-12, rtol=1e-12)
    for actual, ref in [(result.pos_weights, expected["positive_weights"]),
                        (result.neg_weights, expected["negative_weights"])]:
        assert set(actual) == set(ref)
        for name in actual:
            assert actual[name] == pytest.approx(ref[name], abs=1e-10, rel=1e-8)
