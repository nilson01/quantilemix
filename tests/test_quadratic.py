"""Independent polynomial arithmetic and bounded failure checks."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy import stats

from quantilemix import fit_quadratic_bootstrap
from quantilemix._quadratic import _fit_quadratic_with_indices


def cells():
    x = np.array([(a, b) for a in range(4) for b in range(4)] * 5, dtype=float)
    z = np.tile([0, 1, 2, 1, 0], 16).astype(float)
    n = len(x)
    y = (1 + .4*x[:, 0] - .2*x[:, 1] + .3*x[:, 0]**2 - .1*x[:, 1]**2
         + .5*z + .05*((7*np.arange(n)) % 9 - 4))
    return y, x, z


def test_quadratic_known_design_and_shared_row_schedule():
    y, x, z = cells()
    n = len(y)
    schedules = np.tile(np.arange(n), (8, 1))
    schedules[0, :8] = np.arange(8, 16)
    schedules[1, 20:28] = np.arange(29, 37)
    schedules[2, 40:48] = np.arange(48, 56)
    result = _fit_quadratic_with_indices(y, x, schedules, z, q=None)
    design = np.column_stack((np.ones(n), x, x*x, z))
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    manual = np.empty((len(schedules), 2))
    for b, sampled in enumerate(schedules):
        rep = np.linalg.lstsq(design[sampled], y[sampled], rcond=None)[0]
        manual[b] = [sum(rep[1:3]), sum(rep[3:5])]
    psi = np.array([sum(beta[1:3]), sum(beta[3:5])])
    covariance = np.cov(manual, rowvar=False, ddof=1)
    assert_allclose(result.conditional_coefficients, beta, atol=1e-12)
    assert_allclose(result.psi, psi, atol=1e-12)
    assert_allclose(result.bootstrap_coefficients, manual, atol=1e-12)
    assert_allclose(result.covariance, covariance, atol=1e-12)
    assert_allclose(result.se, np.sqrt(np.diag(covariance)), atol=1e-12)
    assert_allclose(result.var_psi, np.diag(covariance), atol=1e-12)
    assert_allclose(result.intervention_means,
                    beta[0] + beta[5]*np.mean(z) + psi[0]*np.arange(4)
                    + psi[1]*np.arange(4)**2, atol=1e-12)
    for level in range(3):
        assert_allclose(result.effect_at(level),
                        result.intervention_means[level+1] - result.intervention_means[level],
                        atol=1e-12)
    assert result.p_value_df == n - 5 - 1 - 2
    assert_allclose(result.p_value, 2*stats.t.sf(abs(result.psi/result.se),
                                                result.p_value_df), atol=1e-12)
    assert_array_equal(result.intervention_levels, np.arange(4))
    assert not result.psi.flags.writeable
    assert "starting joint score" in result.summary()
    assert f"Two-sided Student t p-values (df {result.p_value_df})" in result.summary()


def test_public_seed_repeats_and_global_rng_unchanged():
    y, x, z = cells()
    before = np.random.get_state()
    a = fit_quadratic_bootstrap(y, x, z, q=None, B=8, seed=11)
    b = fit_quadratic_bootstrap(y, x, z, q=None, B=8, seed=11)
    assert_array_equal(a.bootstrap_coefficients, b.bootstrap_coefficients)
    after = np.random.get_state()
    assert before[0] == after[0]
    assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize("change", ["two_levels", "rank", "invalid_B", "invalid_seed",
                                     "bad_schedule", "failed_replicate", "bad_level"])
def test_quadratic_rejects_invalid_inputs(change):
    y, x, z = cells()
    if change == "two_levels":
        with pytest.raises(ValueError, match="three"):
            fit_quadratic_bootstrap(y, x, z, q=2, B=5)
    elif change == "rank":
        with pytest.raises(ValueError, match="rank"):
            fit_quadratic_bootstrap(y, np.column_stack((x[:, 0], x[:, 0])), z,
                                    q=None, B=5)
    elif change == "invalid_B":
        with pytest.raises(ValueError, match="B must"):
            fit_quadratic_bootstrap(y, x, z, q=None, B=1)
    elif change == "invalid_seed":
        with pytest.raises(ValueError, match="seed"):
            fit_quadratic_bootstrap(y, x, z, q=None, B=5, seed=True)
    elif change == "bad_schedule":
        with pytest.raises(ValueError, match="resample indices"):
            _fit_quadratic_with_indices(y, x, [[0, 1]], z, q=None)
    elif change == "failed_replicate":
        schedule = np.tile(np.arange(len(y)), (3, 1))
        schedule[1] = 0
        with pytest.raises(ValueError, match="replicate 2 failed"):
            _fit_quadratic_with_indices(y, x, schedule, z, q=None)
    else:
        result = fit_quadratic_bootstrap(y, x, z, q=None, B=5, seed=1)
        with pytest.raises(ValueError, match="level"):
            result.effect_at(3)
