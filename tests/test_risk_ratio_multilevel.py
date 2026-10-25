"""Four-level marginal binary effect, independent equations and R cases."""

import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy import optimize, special

from quantilemix import BinomialRiskRatioResult, fit_binomial_risk_ratio, qgcomp
from quantilemix._risk_ratio import _from_indices


FIXTURE = Path(__file__).parent / "fixtures/r_risk_ratio_multilevel_2_19_6.json"


def _case(case):
    data = case["inputs"]
    x = np.column_stack((data["x1"], data["x2"]))
    z = None if data["z"] is None else np.asarray(data["z"])[:, None]
    result = _from_indices(data["y"], x, data["indices"], z, q=4)
    return data, x, z, result


@pytest.mark.parametrize("index", [0, 1])
def test_four_level_installed_r_fixture(index):
    case = json.loads(FIXTURE.read_text())["cases"][index]
    data, x, z, result = _case(case)
    expected = case["expected"]
    assert_array_equal(result.point_fit.scores, data["scores"])
    assert_array_equal(result.intervention_levels, np.arange(4))
    for name, actual in {
        "psi": result.psi, "se": result.se, "ci": result.ci,
        "p": result.p_value, "risks": result.intervention_risks,
        "msm_coefficients": result.msm_coefficients,
        "msm_fitted_risks": result.msm_fitted_risks,
        "effects": result.bootstrap_effects,
    }.items():
        assert_allclose(actual, expected[name], rtol=0,
                        atol=1e-7 if name == "effects" else 2e-8)
    assert not result.msm_coefficients.flags.writeable
    assert not result.msm_fitted_risks.flags.writeable


def test_independent_fractional_binomial_score_equations():
    case = json.loads(FIXTURE.read_text())["cases"][1]
    data, x, z, result = _case(case)
    beta = result.point_fit.coefficients
    levels = np.arange(4, dtype=float)
    base = beta[0] + z[:, 0]*beta[-1]
    standardized = np.array([special.expit(base + a*(beta[1]+beta[2])).mean()
                             for a in levels])
    assert_allclose(result.intervention_risks, standardized, atol=1e-13)

    def equations(theta):
        fitted = np.exp(theta[0] + theta[1]*levels)
        return np.array([np.sum((standardized-fitted)/(1-fitted)),
                         np.dot(levels, (standardized-fitted)/(1-fitted))])

    root = optimize.root(equations, [np.log(standardized.mean()), 0.0],
                         method="hybr", tol=1e-12)
    assert root.success or np.linalg.norm(equations(root.x)) < 1e-10
    assert_allclose(equations(root.x), 0, atol=1e-10)
    assert_allclose(result.msm_coefficients, root.x, atol=1e-9)
    assert_allclose(result.msm_fitted_risks,
                    np.exp(root.x[0] + root.x[1]*levels), atol=1e-9)
    assert abs(result.psi - np.log(standardized[1]/standardized[0])) > 0.005
    assert_allclose(result.se, np.std(result.bootstrap_effects, ddof=1), atol=1e-13)
    assert_allclose(result.ci, result.psi + np.array([-1, 1])*
                    special.ndtri(.975)*result.se, atol=1e-13)
    assert "score 3" in result.summary()


def test_public_default_q4_dispatch_and_unquantized_scores():
    case = json.loads(FIXTURE.read_text())["cases"][0]
    data, x, _, _ = _case(case)
    public = qgcomp("y ~ x1 + x2", {key: data[key] for key in ("y", "x1", "x2")},
                    ("x1", "x2"), family="binomial", B=12, seed=17)
    direct = fit_binomial_risk_ratio(data["y"], x, q=4, B=12, seed=17)
    assert isinstance(public, BinomialRiskRatioResult)
    assert_allclose(public.psi, direct.psi, atol=1e-13)
    assert_array_equal(public.bootstrap_effects, direct.bootstrap_effects)
    unquantized = fit_binomial_risk_ratio(data["y"], x, q=None, B=12, seed=17)
    assert_allclose(unquantized.psi, public.psi, atol=1e-12)
    assert_array_equal(public.intervention_levels, np.arange(4))


def test_noncontiguous_unquantized_levels_fail():
    case = json.loads(FIXTURE.read_text())["cases"][0]
    data, x, _, _ = _case(case)
    x = 2*x
    with pytest.raises(ValueError, match="contiguous score levels"):
        fit_binomial_risk_ratio(data["y"], x, q=None, B=2, seed=1)


def test_explicit_four_level_boundaries_and_unequal_counts():
    case = json.loads(FIXTURE.read_text())["cases"][0]
    data, x, _, _ = _case(case)
    cuts = [-1, .5, 1.5, 2.5, 4]
    explicit = fit_binomial_risk_ratio(data["y"], x,
                                       breaks=[cuts, cuts], B=12, seed=17)
    automatic = fit_binomial_risk_ratio(data["y"], x, q=4, B=12, seed=17)
    assert_array_equal(explicit.intervention_levels, np.arange(4))
    assert_allclose(explicit.psi, automatic.psi, atol=1e-12)
    with pytest.raises(ValueError, match="same number of supplied levels"):
        fit_binomial_risk_ratio(data["y"], x,
                                breaks=[cuts, [-1, 1.5, 4]], B=2, seed=17)
