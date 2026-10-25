"""Checks for whole-mixture exposure placement in conditional count fits."""

import json
from pathlib import Path

import numpy as np
import pytest

from quantilemix import fit_hurdle_poisson, fit_zero_inflated_poisson


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("kind,filename", [
    ("zip", "r_zero_inflated_2_19_6.json"),
    ("hurdle", "r_hurdle_2_19_6.json"),
])
@pytest.mark.parametrize("component", ["count", "zero"])
def test_only_one_component_has_joint_exposure_effect(kind, filename, component):
    rows = json.loads((FIXTURES / filename).read_text())["cases"][0]["data"]
    y = [r["y"] for r in rows]
    x = [[r["x1"], r["x2"]] for r in rows]
    fitter = fit_zero_inflated_poisson if kind == "zip" else fit_hurdle_poisson
    result = fitter(y, x, q=None, exposure_component=component)
    assert result.exposure_component == component
    assert len(result.coefficients) == 4
    assert result.coefficients.shape == result.covariance.shape[:1]
    mu, probability = result.predict_components([[0, 0], [3, 3]])
    if component == "count":
        assert len(result.coefficient_names) == 3
        assert len(result.zero_coefficient_names) == 1
        assert result.psi_count is not None
        assert result.se_count > 0
        assert result.ci_count is not None
        assert result.p_value_count is not None
        assert (result.psi_zero if kind == "zip" else result.psi_nonzero) is None
        assert (result.structural_zero_odds_ratio if kind == "zip" else result.nonzero_odds_ratio) is None
        np.testing.assert_allclose(probability[0], probability[1], atol=1e-12)
        assert mu[0] != pytest.approx(mu[1], abs=1e-9)
        np.testing.assert_allclose(result.psi_count, sum(result.coefficients[1:3]))
        np.testing.assert_allclose(result.se_count,
                                   np.sqrt(np.sum(result.covariance[1:3, 1:3])), atol=1e-12)
        assert "no joint exposure effect fitted" in result.summary()
    else:
        assert len(result.coefficient_names) == 1
        assert len(result.zero_coefficient_names) == 3
        assert result.psi_count is None
        assert result.se_count is None
        assert result.ci_count is None
        assert result.p_value_count is None
        assert (result.count_ratio if kind == "zip" else result.count_parameter_ratio) is None
        np.testing.assert_allclose(mu[0], mu[1], atol=1e-12)
        assert probability[0] != pytest.approx(probability[1], abs=1e-9)
        slope = result.psi_zero if kind == "zip" else result.psi_nonzero
        np.testing.assert_allclose(slope, sum(result.coefficients[2:4]))
        assert "no joint exposure effect fitted" in result.summary()


@pytest.mark.parametrize("fitter", [fit_zero_inflated_poisson, fit_hurdle_poisson])
def test_reject_unknown_component(fitter):
    with pytest.raises(ValueError, match="exposure_component"):
        fitter([0, 1], [[0], [1]], q=None, exposure_component="partial")
