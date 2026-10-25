"""Independent polynomial arithmetic for pairwise exposure interactions."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from quantilemix import fit_quadratic_bootstrap
from quantilemix._quadratic import _fit_quadratic_with_indices


def cells():
    x = np.array([(a, b, c) for a in range(4) for b in range(4)
                  for c in range(4)] * 4, dtype=float)
    z = np.tile([0, 1, 2, 1], 64).astype(float)
    n = len(x)
    y = (1 + .2*x[:, 0] - .1*x[:, 1] + .35*x[:, 2]
         + .08*x[:, 0]**2 - .06*x[:, 1]**2 + .04*x[:, 2]**2
         + .12*x[:, 0]*x[:, 1] - .09*x[:, 0]*x[:, 2]
         + .07*x[:, 1]*x[:, 2] + .25*z
         + .03*((7*np.arange(n)) % 11 - 5))
    return y, x, z


def test_three_exposure_interaction_design_and_shared_schedule():
    y, x, z = cells()
    n = len(y)
    schedule = np.tile(np.arange(n), (6, 1))
    schedule[0, :12] = np.arange(20, 32)
    schedule[1, 40:55] = np.arange(100, 115)
    schedule[2, 80:96] = np.arange(150, 166)
    result = _fit_quadratic_with_indices(
        y, x, schedule, z, q=None, pairwise_interactions=True)
    design = np.column_stack((np.ones(n), x, x*x,
                              x[:, 0]*x[:, 1], x[:, 0]*x[:, 2],
                              x[:, 1]*x[:, 2], z))
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    expected_psi = [sum(beta[1:4]), sum(beta[4:10])]
    pairs = []
    for sampled in schedule:
        replica = np.linalg.lstsq(design[sampled], y[sampled], rcond=None)[0]
        pairs.append([sum(replica[1:4]), sum(replica[4:10])])
    assert_allclose(result.conditional_coefficients, beta, atol=1e-12)
    assert_allclose(result.psi, expected_psi, atol=1e-12)
    assert_allclose(result.bootstrap_coefficients, pairs, atol=1e-12)
    assert_allclose(result.covariance, np.cov(pairs, rowvar=False, ddof=1), atol=1e-12)
    levels = np.arange(4)
    assert_allclose(result.intervention_means,
                    beta[0] + beta[10]*np.mean(z)
                    + expected_psi[0]*levels + expected_psi[1]*levels**2,
                    atol=1e-12)
    for level in range(3):
        assert_allclose(result.effect_at(level),
                        result.intervention_means[level+1]
                        - result.intervention_means[level], atol=1e-12)
    assert result.p_value_df == n - 10 - 3
    assert result.pairwise_interactions is True
    assert "pairwise exposure products" in result.summary()
    assert_array_equal(result.intervention_levels, levels)


def test_interaction_public_seed_and_opt_in():
    y, x, z = cells()
    a = fit_quadratic_bootstrap(y, x, z, q=None, B=5, seed=77,
                                pairwise_interactions=True)
    b = fit_quadratic_bootstrap(y, x, z, q=None, B=5, seed=77,
                                pairwise_interactions=True)
    base = fit_quadratic_bootstrap(y, x, z, q=None, B=5, seed=77)
    assert_array_equal(a.bootstrap_coefficients, b.bootstrap_coefficients)
    assert not np.array_equal(a.psi, base.psi)
    assert base.pairwise_interactions is False
    assert "exposure squares" in base.summary()


@pytest.mark.parametrize("bad", [0, 1, "true", None])
def test_interaction_flag_must_be_boolean(bad):
    y, x, z = cells()
    with pytest.raises(ValueError, match="Boolean"):
        fit_quadratic_bootstrap(y, x, z, q=None, B=3,
                                pairwise_interactions=bad)


def test_interaction_requires_two_exposures():
    y, x, z = cells()
    with pytest.raises(ValueError, match="two exposures"):
        fit_quadratic_bootstrap(y, x[:, :1], z, q=None, B=3,
                                pairwise_interactions=True)
