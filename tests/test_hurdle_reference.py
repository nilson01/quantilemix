"""Fixed installed-R cases and independent hurdle likelihood checks."""

import json
from pathlib import Path

import numpy as np
import pytest

from quantilemix import fit_hurdle_poisson
from quantilemix._hurdle import _positive_likelihood_score_information


CASES = json.loads((Path(__file__).parent / "fixtures/r_hurdle_2_19_6.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES)
def test_installed_r_fixed_case(case):
    rows = case["data"]
    x = [[r["x1"], r["x2"]] for r in rows]
    z = [r["z"] for r in rows]
    fit = fit_hurdle_poisson([r["y"] for r in rows], x, z, q=None)
    np.testing.assert_allclose(
        [fit.psi_count, fit.psi_nonzero, fit.se_count, fit.se_nonzero],
        [case["psi"]["count"], case["psi"]["zero"],
         case["se"]["count"], case["se"]["zero"]], atol=3e-7, rtol=0)
    np.testing.assert_allclose(
        fit.coefficients, case["coefficients"]["count"]+case["coefficients"]["zero"],
        atol=1e-7, rtol=0)
    np.testing.assert_allclose(fit.covariance, case["covariance"], atol=1e-7, rtol=0)
    mu, p_nonzero = fit.predict_components(x, z)
    np.testing.assert_allclose(fit.predict(x, z),
                               p_nonzero*mu/(-np.expm1(-mu)), atol=1e-12)
    assert "separate component effects" in fit.summary()
    assert not fit.coefficients.flags.writeable


def test_positive_truncated_poisson_derivatives():
    rows = [r for r in CASES[0]["data"] if r["y"] > 0]
    design = np.asarray([[1, r["x1"], r["x2"], r["z"]] for r in rows])
    y = np.asarray([r["y"] for r in rows])
    beta = np.array([.4, .2, -.1, .1])
    likelihood, score, information = _positive_likelihood_score_information(design, y, beta)
    assert np.isfinite(likelihood)
    unit = np.eye(len(beta))*1e-5
    score_numerical = np.array([
        (_positive_likelihood_score_information(design, y, beta+v)[0]
         - _positive_likelihood_score_information(design, y, beta-v)[0]) / 2e-5
        for v in unit])
    hessian_numerical = np.column_stack([
        (_positive_likelihood_score_information(design, y, beta+v)[1]
         - _positive_likelihood_score_information(design, y, beta-v)[1]) / 2e-5
        for v in unit])
    np.testing.assert_allclose(score, score_numerical, atol=1e-6, rtol=0)
    np.testing.assert_allclose(information, -hessian_numerical, atol=1e-6, rtol=0)


@pytest.mark.parametrize("outcome", [[0]*32, [1]*32, [0, 1.5]+[0]*30])
def test_reject_bad_count_outcomes(outcome):
    with pytest.raises(ValueError):
        fit_hurdle_poisson(outcome, np.arange(32) % 4, q=None)


def test_component_ratios_are_not_observed_mean_ratio():
    rows = CASES[0]["data"]
    fit = fit_hurdle_poisson([r["y"] for r in rows],
                             [[r["x1"], r["x2"]] for r in rows],
                             [r["z"] for r in rows], q=None)
    mu0, p0 = fit.predict_components([[0, 0]], [0])
    mu1, p1 = fit.predict_components([[1, 1]], [0])
    np.testing.assert_allclose(mu1/mu0, fit.count_parameter_ratio, atol=1e-12)
    np.testing.assert_allclose((p1/(1-p1))/(p0/(1-p0)), fit.nonzero_odds_ratio, atol=1e-12)
    assert not np.isclose((fit.predict([[1, 1]], [0])/fit.predict([[0, 0]], [0]))[0],
                          fit.count_parameter_ratio)
