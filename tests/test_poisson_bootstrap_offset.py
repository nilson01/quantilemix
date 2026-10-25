"""Unequal observation times in the marginal Poisson bootstrap."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from quantilemix import fit_poisson_bootstrap
from quantilemix._poisson_bootstrap import _from_indices


def offset_cells():
    cells = ((0, 0, 1), (1, 0, 2), (0, 1, 3), (1, 1, 6))
    x = np.vstack([np.tile([a, b], (5, 1)) for a, b, _ in cells]).astype(float)
    time = np.tile(np.arange(1, 6), 4).astype(float)
    y = np.concatenate([mean*np.arange(1, 6) + [0, 1, -1, 2, -2]
                        for _, _, mean in cells])
    return y, x, time


def test_known_means_keep_each_period_under_both_interventions():
    y, x, time = offset_cells()
    result = fit_poisson_bootstrap(y, x, q=None, exposure_time=time, B=24, seed=991)
    assert_allclose(result.intervention_means, [3, 18], atol=1e-8)
    assert_allclose(result.rate_ratio, 6, atol=1e-8)
    assert_allclose(result.psi, np.log(6), atol=1e-8)
    assert_allclose(result.psi, result.point_fit.psi, atol=1e-8)
    assert_array_equal(result.point_fit.exposure_time, time)
    assert "Observation times are held" in result.summary()
    assert not result.intervention_means.flags.writeable
    assert not result.point_fit.exposure_time.flags.writeable


def test_row_schedule_carries_time_with_counts_and_covariates():
    y, x, time = offset_cells()
    schedule = np.random.default_rng(72).integers(0, len(y), size=(12, len(y)))
    scheduled = _from_indices(y, x, schedule, q=None, exposure_time=time)
    direct = fit_poisson_bootstrap(y, x, q=None, exposure_time=time, B=12, seed=72)
    assert_array_equal(scheduled.bootstrap_effects, direct.bootstrap_effects)
    assert_allclose(scheduled.intervention_means, direct.intervention_means)


@pytest.mark.parametrize("time", [0, -1, np.nan, np.inf, True, "x"])
def test_invalid_observation_time_rejected(time):
    y, x, _ = offset_cells()
    with pytest.raises((ValueError, TypeError), match="exposure_time"):
        fit_poisson_bootstrap(y, x, q=None,
                              exposure_time=np.repeat(time, len(y)), B=3, seed=2)


def test_time_length_and_shape_rejected():
    y, x, time = offset_cells()
    for invalid in (time[:-1], time[:, None]):
        with pytest.raises(ValueError, match="exposure_time"):
            fit_poisson_bootstrap(y, x, q=None, exposure_time=invalid, B=3, seed=2)


def test_equal_time_option_preserves_no_offset_behavior():
    y, x, _ = offset_cells()
    ordinary = fit_poisson_bootstrap(y, x, q=None, B=12, seed=77)
    explicit = fit_poisson_bootstrap(y, x, q=None,
                                     exposure_time=np.ones(len(y)), B=12, seed=77)
    assert_allclose(ordinary.psi, explicit.psi, atol=1e-11)
    assert_allclose(ordinary.bootstrap_effects, explicit.bootstrap_effects, atol=1e-11)
    assert_allclose(ordinary.intervention_means, explicit.intervention_means, atol=1e-11)
