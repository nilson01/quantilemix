"""Independent arithmetic for Gaussian exposure-covariate products."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from quantilemix import fit_quadratic_bootstrap
from quantilemix._quadratic import _fit_quadratic_with_indices


def cells():
    x = np.array([(a, b, c) for a in range(4) for b in range(4)
                  for c in range(4)] * 6, dtype=float)
    n = len(x)
    z = np.column_stack((np.repeat([0, 1, 2, 1, 0, 2], 64),
                         np.repeat([0, 0, 1, 1, 2, 2], 64))).astype(float)
    y = (1 + .2*x[:, 0] - .1*x[:, 1] + .35*x[:, 2]
         + .08*x[:, 0]**2 - .06*x[:, 1]**2 + .04*x[:, 2]**2
         + .12*x[:, 0]*x[:, 1] - .09*x[:, 0]*x[:, 2]
         + .07*x[:, 1]*x[:, 2] + .25*z[:, 0] - .16*z[:, 1]
         + .1*x[:, 0]*z[:, 0] - .08*x[:, 0]*z[:, 1]
         + .07*x[:, 1]*z[:, 0] + .04*x[:, 1]*z[:, 1]
         - .06*x[:, 2]*z[:, 0] + .09*x[:, 2]*z[:, 1]
         + .03*((7*np.arange(n)) % 11 - 5))
    return y, x, z


def design(x, z):
    return np.column_stack((np.ones(len(x)), x, x*x,
                            x[:, 0]*x[:, 1], x[:, 0]*x[:, 2],
                            x[:, 1]*x[:, 2],
                            *(x[:, i]*z[:, j] for i in range(3) for j in range(2)), z))


def test_two_covariate_marginalization_and_shared_schedule():
    y, x, z = cells()
    n = len(y)
    schedule = np.tile(np.arange(n), (6, 1))
    schedule[0, :12] = np.arange(20, 32)
    schedule[1, 40:55] = np.arange(100, 115)
    schedule[2, 80:96] = np.arange(150, 166)
    result = _fit_quadratic_with_indices(
        y, x, schedule, z, q=None, pairwise_interactions=True,
        covariate_interactions=True)
    full = design(x, z)
    beta = np.linalg.lstsq(full, y, rcond=None)[0]
    means_z = np.mean(z, axis=0)
    linear = sum(beta[1:4]) + np.sum(beta[10:16].reshape(3, 2)*means_z)
    curved = sum(beta[4:10])
    expected = [linear, curved]
    replicates = []
    for sampled in schedule:
        replica = np.linalg.lstsq(full[sampled], y[sampled], rcond=None)[0]
        replicas_z = np.mean(z[sampled], axis=0)
        replicates.append([sum(replica[1:4])
                           + np.sum(replica[10:16].reshape(3, 2)*replicas_z),
                           sum(replica[4:10])])
    assert_allclose(result.conditional_coefficients, beta, atol=1e-12)
    assert_allclose(result.psi, expected, atol=1e-12)
    assert_allclose(result.bootstrap_coefficients, replicates, atol=1e-12)
    assert_allclose(result.covariance, np.cov(replicates, rowvar=False, ddof=1),
                    atol=1e-12)
    levels = np.arange(4)
    assert_allclose(result.intervention_means,
                    beta[0]+means_z@beta[16:18]+linear*levels+curved*levels**2,
                    atol=1e-12)
    assert result.p_value_df == n - 2*3 - 3 - 3*2 - 2 - 3
    assert result.covariate_interactions is True
    assert "exposure-covariate products" in result.summary()
    assert_array_equal(result.intervention_levels, levels)


def test_public_opt_in_and_reproducibility():
    y, x, z = cells()
    a = fit_quadratic_bootstrap(y, x, z, q=None, B=5, seed=19,
                                covariate_interactions=True)
    b = fit_quadratic_bootstrap(y, x, z, q=None, B=5, seed=19,
                                covariate_interactions=True)
    base = fit_quadratic_bootstrap(y, x, z, q=None, B=5, seed=19)
    assert_array_equal(a.bootstrap_coefficients, b.bootstrap_coefficients)
    assert a.covariate_interactions is True
    assert base.covariate_interactions is False
    assert not np.array_equal(a.psi, base.psi)


@pytest.mark.parametrize("bad", [0, 1, "true", None])
def test_option_requires_boolean(bad):
    y, x, z = cells()
    with pytest.raises(ValueError, match="Boolean"):
        fit_quadratic_bootstrap(y, x, z, q=None, B=3, covariate_interactions=bad)


def test_option_requires_covariates():
    y, x, _ = cells()
    with pytest.raises(ValueError, match="require covariates"):
        fit_quadratic_bootstrap(y, x, q=None, B=3, covariate_interactions=True)
