"""Known Poisson means, bootstrap arithmetic and input boundaries."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy import special

from quantilemix import fit_poisson_bootstrap
from quantilemix._poisson_bootstrap import _from_indices


def cells():
    groups = ((0, 0, 1), (1, 0, 2), (0, 1, 3), (1, 1, 6))
    x = np.vstack([np.tile([a, b], (20, 1)) for a, b, _ in groups]).astype(float)
    y = np.concatenate([np.repeat(mean, 20) for _, _, mean in groups])
    return y, x


def test_known_marginal_means_and_bootstrap_inference():
    y, x = cells()
    result = fit_poisson_bootstrap(y, x, q=None, B=20, seed=43)
    assert_allclose(result.intervention_means, [1, 6], atol=1e-11)
    assert_allclose(result.psi, np.log(6), atol=1e-11)
    assert_allclose(result.rate_ratio, 6, atol=1e-11)
    assert_allclose(result.se, np.std(result.bootstrap_effects, ddof=1), atol=1e-13)
    assert_allclose(result.ci, result.psi + np.array([-1, 1])*special.ndtri(.975)*result.se,
                    atol=1e-13)
    assert_allclose(result.rate_ratio_ci, np.exp(result.ci), atol=1e-13)
    assert_allclose(result.p_value, 2*special.ndtr(-abs(result.psi/result.se)), atol=1e-13)
    assert "marginal Poisson" in result.summary()
    assert not result.bootstrap_effects.flags.writeable
    assert not result.intervention_means.flags.writeable
    assert_array_equal(result.intervention_levels, [0, 1])


def test_adjusted_standardization_and_reproducibility():
    i = np.arange(120)
    x = np.column_stack((i % 3, (i // 3) % 3)).astype(float)
    z = ((i*7) % 11 / 10)[:, None]
    y = ((i*5 + 3*x[:, 0] + 2*x[:, 1]) % 7 + 1).astype(float)
    result = fit_poisson_bootstrap(y, x, z, q=None, B=12, seed=9)
    beta = result.point_fit.coefficients
    means = [np.mean(np.exp(beta[0]+z[:, 0]*beta[3]+a*(beta[1]+beta[2])))
             for a in (0, 1, 2)]
    assert_allclose(result.intervention_means, means, atol=1e-12)
    assert_allclose(np.diff(np.log(result.intervention_means)), result.psi, atol=1e-12)
    other = fit_poisson_bootstrap(y, x, z, q=None, B=12, seed=9)
    assert_array_equal(result.bootstrap_effects, other.bootstrap_effects)


@pytest.mark.parametrize("change,pattern", [
    ("B", "B must"), ("seed", "seed must"), ("levels", "consecutive levels"),
    ("schedule", "B-by-n"), ("replicate", "bootstrap replicate 1 failed"),
    ("summary", "digits"),
])
def test_invalid_inputs_are_explicit(change, pattern):
    y, x = cells()
    with pytest.raises(ValueError, match=pattern):
        if change == "B":
            fit_poisson_bootstrap(y, x, q=None, B=1)
        elif change == "seed":
            fit_poisson_bootstrap(y, x, q=None, seed=-1)
        elif change == "levels":
            fit_poisson_bootstrap(y, x+1, q=None, B=5)
        elif change == "schedule":
            _from_indices(y, x, np.ones((2, 3)), q=None)
        elif change == "replicate":
            schedule = np.tile(0, (2, len(y)))
            _from_indices(y, x, schedule, q=None)
        else:
            fit_poisson_bootstrap(y, x, q=None, B=5, seed=3).summary(0)
