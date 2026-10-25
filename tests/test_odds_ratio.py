"""Independent marginal odds arithmetic and public API checks."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy import special

from quantilemix import fit_binomial_odds_ratio
from quantilemix._odds_ratio import _logistic_msm


def cells():
    i = np.arange(160)
    x = np.column_stack((i % 2, (i // 2) % 2)).astype(float)
    z = ((i*7) % 13 / 12)[:, None]
    y = (((i*19+3) % 29) / 29 < .22 + .1*x[:, 0] + .09*x[:, 1] + .11*z[:, 0]).astype(float)
    return y, x, z


def test_two_level_standardization_and_bootstrap_interval():
    y, x, z = cells()
    result = fit_binomial_odds_ratio(y, x, z, q=None, B=12, seed=7)
    beta = result.point_fit.coefficients
    eta = beta[0] + z[:, 0]*beta[3]
    risks = np.array([np.mean(special.expit(eta + level*(beta[1]+beta[2])))
                      for level in (0, 1)])
    odds = risks/(1-risks)
    assert_allclose(result.intervention_risks, risks, atol=1e-13)
    assert_allclose(result.odds_ratio, odds[1]/odds[0], atol=1e-13)
    assert_allclose(result.psi, np.log(odds[1]/odds[0]), atol=1e-13)
    assert_allclose(result.se, np.std(result.bootstrap_effects, ddof=1), atol=1e-13)
    assert_allclose(result.ci, result.psi+np.array([-1, 1])*special.ndtri(.975)*result.se)
    assert_allclose(result.odds_ratio_ci, np.exp(result.ci))
    assert "marginal binary odds ratio" in result.summary()
    assert "With two levels, this compares" in result.summary()
    assert not result.bootstrap_effects.flags.writeable
    assert_array_equal(result.intervention_levels, [0, 1])


def test_four_level_marginal_logistic_score_equations():
    risks = np.array([.12, .17, .28, .42])
    levels = np.arange(4, dtype=float)
    beta = _logistic_msm(risks, levels)
    fitted = special.expit(beta[0]+beta[1]*levels)
    assert_allclose(np.sum(risks-fitted), 0, atol=1e-12)
    assert_allclose(np.sum(levels*(risks-fitted)), 0, atol=1e-12)
    assert beta[1] > 0
    assert not np.allclose(np.diff(special.logit(risks)),beta[1])


def test_conditional_and_marginal_odds_differ_with_adjustment():
    y, x, z = cells()
    result = fit_binomial_odds_ratio(y, x, z, q=None, B=10, seed=2)
    conditional = np.exp(np.sum(result.point_fit.coefficients[1:3]))
    assert abs(result.odds_ratio-conditional) > 1e-5
    again = fit_binomial_odds_ratio(y, x, z, q=None, B=10, seed=2)
    assert_array_equal(result.bootstrap_effects, again.bootstrap_effects)


@pytest.mark.parametrize("change,pattern", [
    ("B", "B must"), ("seed", "seed must"),
    ("levels", "contiguous score levels"), ("summary", "digits"),
])
def test_invalid_inputs_are_named(change, pattern):
    y, x, z = cells()
    with pytest.raises(ValueError, match=pattern):
        if change == "B":
            fit_binomial_odds_ratio(y, x, z, q=None, B=1)
        elif change == "seed":
            fit_binomial_odds_ratio(y, x, z, q=None, seed=-1)
        elif change == "levels":
            fit_binomial_odds_ratio(y, x*2, z, q=None, B=5)
        else:
            fit_binomial_odds_ratio(y, x, z, q=None, B=5, seed=3).summary(0)
