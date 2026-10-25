"""Fixed installed-R cases and independent ZIP likelihood checks."""

import json
from pathlib import Path

import numpy as np
import pytest

from quantilemix import fit_zero_inflated_poisson
from quantilemix._zero_inflated import _likelihood_score_information


CASES = json.loads((Path(__file__).parent / "fixtures/r_zero_inflated_2_19_6.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES)
def test_installed_r_fixed_case(case):
    rows = case["data"]
    x = [[r["x1"], r["x2"]] for r in rows]
    z = [r["z"] for r in rows]
    fit = fit_zero_inflated_poisson([r["y"] for r in rows], x, z, q=None)
    np.testing.assert_allclose(
        [fit.psi_count, fit.psi_zero, fit.se_count, fit.se_zero],
        [case["psi"]["count"], case["psi"]["zero"],
         case["se"]["count"], case["se"]["zero"]], atol=3e-6, rtol=0)
    np.testing.assert_allclose(
        fit.coefficients, case["coefficients"]["count"]+case["coefficients"]["zero"],
        atol=5e-6, rtol=0)
    np.testing.assert_allclose(fit.covariance, case["covariance"], atol=3e-6, rtol=0)
    mu, pi = fit.predict_components(x, z)
    np.testing.assert_allclose(fit.predict(x, z), (1-pi)*mu, atol=1e-12)
    assert "separate component effects" in fit.summary()
    assert not fit.coefficients.flags.writeable


def test_likelihood_derivatives_by_independent_central_difference():
    rows = CASES[0]["data"]
    design = np.asarray([[1, r["x1"], r["x2"], r["z"]] for r in rows])
    y = np.asarray([r["y"] for r in rows])
    beta = np.array([.4, .2, -.1, .1, -1, .15, .2, -.1])
    likelihood, score, information = _likelihood_score_information(design, design, y, beta)
    assert np.isfinite(likelihood)
    unit = np.eye(len(beta))*1e-5
    score_numerical = np.array([
        (_likelihood_score_information(design, design, y, beta+v)[0]
         - _likelihood_score_information(design, design, y, beta-v)[0]) / 2e-5
        for v in unit])
    hessian_numerical = np.column_stack([
        (_likelihood_score_information(design, design, y, beta+v)[1]
         - _likelihood_score_information(design, design, y, beta-v)[1]) / 2e-5
        for v in unit])
    np.testing.assert_allclose(score, score_numerical, atol=1e-6, rtol=0)
    np.testing.assert_allclose(information, -hessian_numerical, atol=1e-6, rtol=0)


@pytest.mark.parametrize("outcome", [[0]*32, [1]*32, [0, 1.5]+[0]*30])
def test_reject_bad_count_outcomes(outcome):
    x = np.arange(32) % 4
    with pytest.raises(ValueError):
        fit_zero_inflated_poisson(outcome, x, q=None)


def test_two_component_effects_are_not_observed_mean_ratio():
    rows = CASES[1]["data"]
    x = np.asarray([[r["x1"], r["x2"]] for r in rows])
    z = np.asarray([r["z"] for r in rows])
    fit = fit_zero_inflated_poisson([r["y"] for r in rows], x, z, q=None)
    mu0, pi0 = fit.predict_components([[0, 0]], [0])
    mu1, pi1 = fit.predict_components([[1, 1]], [0])
    np.testing.assert_allclose(mu1/mu0, fit.count_ratio, atol=1e-12)
    np.testing.assert_allclose((pi1/(1-pi1))/(pi0/(1-pi0)),
                               fit.structural_zero_odds_ratio, atol=1e-12)
    assert not np.isclose(((1-pi1)*mu1/((1-pi0)*mu0))[0], fit.count_ratio)
