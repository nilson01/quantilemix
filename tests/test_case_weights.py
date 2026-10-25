"""Case-weighted Gaussian checks with exact arithmetic and installed-R values."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from quantilemix import fit


FIXTURE = Path(__file__).parent / "fixtures/r_weighted_gaussian_2_19_6.json"
FIXTURE_SHA256 = "9594e90d09ace4d5c7705498ffb1e6cfb503ad9e59b121ed0bd517aee7e013f2"


@pytest.mark.parametrize("case", json.loads(FIXTURE.read_text())["cases"],
                         ids=lambda case: case["name"])
def test_positive_case_weights_agree_with_installed_r(case):
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    inputs, expected = case["inputs"], case["expected"]
    result = fit(inputs["y"], inputs["exposures"], inputs["covariates"],
                 q=inputs["q"], weights=inputs["weights"])
    for key, actual in {
        "coefficients": result.coefficients,
        "covariance": result.covariance,
        "scores": result.scores,
        "psi": result.psi,
        "variance": result.var_psi,
        "se": result.se,
        "ci": result.ci,
        "t": result.test_statistic,
        "p": result.p_value,
        "p_df": result.p_value_df,
        "residual_df": result.residual_df,
        "dispersion": result.residual_variance,
        "fitted": result.fitted_values,
        "predictions": result.predict(inputs["new_exposures"], inputs["new_covariates"]),
        "positive_psi": result.pos_psi,
        "negative_psi": result.neg_psi,
    }.items():
        assert_allclose(actual, expected[key], rtol=5e-12, atol=5e-12,
                        err_msg=f"{case['name']}: {key}")
    assert_allclose(result.case_weights, inputs["weights"], rtol=0, atol=0)
    assert result.case_weights.flags.writeable is False
    assert result.diagnostics["case_weighted"] is True
    assert "Case weights: supplied" in result.summary()
    if expected["breaks"] is None:
        assert result.breaks is None
    else:
        for actual, target in zip(result.breaks, expected["breaks"]):
            assert_allclose(actual, target, rtol=5e-12, atol=5e-12)
    for key, actual in (("positive_weights", result.pos_weights),
                        ("negative_weights", result.neg_weights)):
        assert set(actual) == set(expected[key])
        for name, value in expected[key].items():
            assert_allclose(actual[name], value, rtol=5e-12, atol=5e-12)


def test_correlated_exposures_have_exact_weighted_joint_variance():
    # Direct rational normal-equation solution for the fixed 8-row table.
    # X'WX = [[36,26,22],[26,26,15],[22,15,22]], X'Wy = [135,105,108].
    x = np.column_stack([[0, 0, 0, 0, 1, 1, 1, 1],
                         [0, 0, 1, 1, 0, 0, 1, 1]])
    y = [1, 2, 3, 4, 2, 2, 5, 6]
    w = np.arange(1, 9)
    result = fit(y, x, q=None, weights=w)
    assert_allclose(result.coefficients, [593 / 732, 521 / 366, 1145 / 366],
                    rtol=0, atol=1e-12)
    assert_allclose(result.psi, 833 / 183, rtol=0, atol=1e-12)
    assert_allclose(result.residual_variance, 2513 / 1220, rtol=0, atol=1e-12)
    assert_allclose(result.var_psi, 198527 / 334890, rtol=0, atol=1e-12)
    # Omitting the off-diagonal covariance would give a different answer.
    diagonal_only = result.covariance[1, 1] + result.covariance[2, 2]
    assert abs(diagonal_only - result.var_psi) > 0.05


def test_uniform_scale_of_case_weights_preserves_model_based_inference():
    case = json.loads(FIXTURE.read_text())["cases"][0]["inputs"]
    x, y, z, w = case["exposures"], case["y"], case["covariates"], np.asarray(case["weights"])
    baseline = fit(y, x, z, weights=w)
    doubled = fit(y, x, z, weights=2 * w)
    for attr in ("coefficients", "covariance", "psi", "var_psi", "ci", "p_value"):
        assert_allclose(getattr(doubled, attr), getattr(baseline, attr),
                        rtol=5e-12, atol=5e-12)
    assert_allclose(doubled.residual_variance, 2 * baseline.residual_variance,
                    rtol=5e-12)
    assert_array_equal(doubled.scores, baseline.scores)
    assert "Sign weights" in doubled.summary()


def test_unit_weights_preserve_unweighted_fit_and_default():
    case = json.loads(FIXTURE.read_text())["cases"][0]["inputs"]
    x, y, z = case["exposures"], case["y"], case["covariates"]
    default = fit(y, x, z)
    units = fit(y, x, z, weights=np.ones(len(y)))
    assert default.case_weights is None
    assert default.diagnostics["case_weighted"] is False
    for attr in ("coefficients", "covariance", "psi", "var_psi", "ci", "p_value"):
        assert_allclose(getattr(units, attr), getattr(default, attr),
                        rtol=5e-12, atol=5e-12)
    assert "Case weights: none" in default.summary()
    assert_array_equal(units.scores, default.scores)


@pytest.mark.parametrize("bad, message", [
    ([1, 2], "one value per row"),
    ([[1]] * 8, "one-dimensional"),
    ([1, 1, 1, 0, 1, 1, 1, 1], "strictly positive"),
    ([1, 1, 1, -1, 1, 1, 1, 1], "strictly positive"),
    ([1, 1, 1, float("nan"), 1, 1, 1, 1], "nonfinite"),
    ([1, 1, 1, float("inf"), 1, 1, 1, 1], "nonfinite"),
    ([1, 1, 1, "bad", 1, 1, 1, 1], "numeric"),
])
def test_bad_case_weights_fail_without_silent_row_removal(bad, message):
    x = np.column_stack([[0, 0, 0, 0, 1, 1, 1, 1],
                         [0, 0, 1, 1, 0, 0, 1, 1]])
    with pytest.raises(ValueError, match=message):
        fit([1, 2, 3, 4, 2, 2, 5, 6], x, q=None, weights=bad)


def test_masked_case_weight_is_rejected_before_numpy_drops_mask():
    x = np.arange(8)
    w = np.ma.array(np.ones(8), mask=[0, 0, 1, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="masked"):
        fit(1 + x + np.sin(x), x, q=None, weights=w)


def test_caller_weights_are_copied_and_result_is_immutable():
    x = np.arange(8)
    w = np.arange(1, 9, dtype=float)
    result = fit(1 + x + np.sin(x), x, q=None, weights=w)
    w[:] = 100
    assert_array_equal(result.case_weights, np.arange(1, 9))
    with pytest.raises(ValueError):
        result.case_weights[0] = 2
