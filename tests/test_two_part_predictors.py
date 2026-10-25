"""Independent component-design checks for conditional two-part count fits."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import special

from quantilemix import fit_hurdle_poisson, fit_zero_inflated_poisson
from quantilemix._hurdle import _positive_likelihood_score_information
from quantilemix._zero_inflated import _likelihood_score_information


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("kind", ["zip", "hurdle"])
def test_separate_zero_adjustment_prediction_and_score(kind):
    filename = "r_zero_inflated_2_19_6.json" if kind == "zip" else "r_hurdle_2_19_6.json"
    rows = json.loads((FIXTURES / filename).read_text())["cases"][0]["data"]
    y = np.array([r["y"] for r in rows])
    x = np.array([[r["x1"], r["x2"]] for r in rows])
    zc = np.array([r["z"] for r in rows])
    zz = np.cos(np.arange(len(y))*0.17)
    fitter = fit_zero_inflated_poisson if kind == "zip" else fit_hurdle_poisson
    result = fitter(y, x, zc, zero_covariates=zz, q=None)
    assert result.covariate_names == ("z1",)
    assert result.zero_covariate_names == ("zz1",)
    assert len(result.coefficient_names) == len(result.zero_coefficient_names) == 4
    count = np.column_stack((np.ones(len(y)), x, zc))
    zero = np.column_stack((np.ones(len(y)), x, zz))
    p = count.shape[1]
    mu, probability = result.predict_components(x, zc, zero_covariates=zz)
    np.testing.assert_allclose(mu, np.exp(count @ result.coefficients[:p]), atol=1e-12)
    np.testing.assert_allclose(probability, special.expit(zero @ result.coefficients[p:]), atol=1e-12)
    if kind == "zip":
        np.testing.assert_allclose(result.predict(x, zc, zero_covariates=zz),
                                   (1-probability)*mu, atol=1e-12)
        _, score, information = _likelihood_score_information(count, zero, y,
                                                               result.coefficients)
        assert np.max(np.abs(score)) < 1e-5
        assert np.linalg.eigvalsh(information)[0] > 0
        steps = np.eye(len(result.coefficients))*1e-5
        numerical_score = np.array([
            (_likelihood_score_information(count, zero, y, result.coefficients+step)[0]
             - _likelihood_score_information(count, zero, y, result.coefficients-step)[0])/2e-5
            for step in steps])
        np.testing.assert_allclose(score, numerical_score, atol=1e-6, rtol=0)
    else:
        np.testing.assert_allclose(result.predict(x, zc, zero_covariates=zz),
                                   probability*mu/(-np.expm1(-mu)), atol=1e-12)
        _, score, information = _positive_likelihood_score_information(
            count[y > 0], y[y > 0], result.coefficients[:p])
        assert np.max(np.abs(score)) < 1e-5
        assert np.linalg.eigvalsh(information)[0] > 0
    contrast = np.zeros((2, len(result.coefficients)))
    contrast[0, 1:3] = 1
    contrast[1, p+1:p+3] = 1
    np.testing.assert_allclose(contrast @ result.coefficients,
                               [result.psi_count, result.psi_zero if kind == "zip" else result.psi_nonzero])
    np.testing.assert_allclose(np.sqrt(np.diag(contrast @ result.covariance @ contrast.T)),
                               [result.se_count, result.se_zero if kind == "zip" else result.se_nonzero])
    with pytest.raises(ValueError, match="zero_covariates must be supplied"):
        result.predict_components(x, zc)
    with pytest.raises(ValueError, match="fitted rows and columns"):
        result.predict_components(x, zc, zero_covariates=None)


@pytest.mark.parametrize("kind", ["zip", "hurdle"])
def test_explicit_no_zero_adjustment_and_bad_inputs(kind):
    filename = "r_zero_inflated_2_19_6.json" if kind == "zip" else "r_hurdle_2_19_6.json"
    rows = json.loads((FIXTURES / filename).read_text())["cases"][0]["data"]
    y = [r["y"] for r in rows]
    x = [[r["x1"], r["x2"]] for r in rows]
    zc = [r["z"] for r in rows]
    fitter = fit_zero_inflated_poisson if kind == "zip" else fit_hurdle_poisson
    result = fitter(y, x, zc, zero_covariates=None, q=None)
    assert result.zero_covariate_names == ()
    assert len(result.zero_coefficient_names) == 3
    result.predict_components(x, zc, zero_covariates=None)
    with pytest.raises(ValueError, match="zero_covariates must be supplied"):
        result.predict_components(x, zc)
    with pytest.raises(ValueError, match="zero_covariates"):
        fitter(y, x, zc, zero_covariates="bad", q=None)
    with pytest.raises(ValueError, match="same rows"):
        fitter(y, x, zc, zero_covariates=[1, 2], q=None)
