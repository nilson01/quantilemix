"""Independent polynomial contrasts and cluster-sandwich checks."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy import stats

from quantilemix import fit_quadratic_gaussian_ee


def cells():
    x = np.array([(a, b) for a in range(4) for b in range(4)] * 8, dtype=float)
    n = len(x)
    ids = np.arange(n) // 4
    z = ((np.arange(n) * 7 + ids) % 5).astype(float)
    y = (1 + .35*x[:, 0] - .2*x[:, 1] + .11*x[:, 0]**2
         + .04*x[:, 1]**2 + .08*x[:, 0]*x[:, 1] + .3*z
         + .13*((ids*3) % 7 - 3) + .02*((np.arange(n)*11) % 9 - 4))
    return y, x, z, ids


def test_polynomial_contrasts_and_independent_cluster_covariance():
    y, x, z, ids = cells()
    result = fit_quadratic_gaussian_ee(
        y, x, ids, z, q=None, pairwise_interactions=True)
    design = np.column_stack((np.ones(len(y)), x, x**2,
                              x[:, 0]*x[:, 1], z))
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y-design @ beta
    group_scores = np.array([design[ids == group].T @ residual[ids == group]
                             for group in np.unique(ids)])
    bread_inverse = np.linalg.inv(design.T @ design)
    beta_covariance = bread_inverse @ (group_scores.T @ group_scores) @ bread_inverse
    contrast = np.zeros((2, len(beta)))
    contrast[0, 1:3] = 1
    contrast[1, 3:6] = 1
    expected_psi = contrast @ beta
    expected_covariance = contrast @ beta_covariance @ contrast.T
    assert_allclose(result.conditional_coefficients, beta, atol=1e-11)
    assert_allclose(result.psi, expected_psi, atol=1e-11)
    assert_allclose(result.covariance, expected_covariance, atol=1e-10)
    assert_allclose(result.se, np.sqrt(np.diag(expected_covariance)), atol=1e-11)
    assert_allclose(result.msm_coefficients[0], beta[0]+beta[-1]*np.mean(z), atol=1e-11)
    assert_allclose(result.intervention_means,
                    result.msm_coefficients[0]+expected_psi[0]*np.arange(4)
                    +expected_psi[1]*np.arange(4)**2, atol=1e-11)
    assert_allclose(result.effect_at(2), expected_psi[0]+5*expected_psi[1], atol=1e-11)
    assert_allclose(result.p_value,
                    2*stats.t.sf(abs(result.test_statistic), len(y)-len(beta)-2), atol=1e-12)
    assert result.cluster_count == len(np.unique(ids))
    assert "mean change depends" in result.summary()
    assert not result.psi.flags.writeable
    assert not result.full_covariance.flags.writeable
    assert_array_equal(result.intervention_levels, np.arange(4))


def test_integer_weights_replicate_rows_within_clusters_and_interactions():
    y, x, z, ids = cells()
    weights = 1+(np.arange(len(y))*3+ids) % 3
    weighted = fit_quadratic_gaussian_ee(
        y, x, ids, z, q=None, weights=weights,
        pairwise_interactions=True, covariate_interactions=True)
    rows = np.repeat(np.arange(len(y)), weights)
    repeated = fit_quadratic_gaussian_ee(
        y[rows], x[rows], ids[rows], z[rows], q=None,
        pairwise_interactions=True, covariate_interactions=True)
    assert_allclose(weighted.conditional_coefficients,
                    repeated.conditional_coefficients, atol=1e-11)
    assert_allclose(weighted.msm_coefficients, repeated.msm_coefficients, atol=1e-11)
    assert_allclose(weighted.full_covariance, repeated.full_covariance, atol=1e-10)
    assert_allclose(weighted.intervention_means, repeated.intervention_means, atol=1e-11)
    assert_allclose(weighted.psi, repeated.psi, atol=1e-11)
    assert weighted.case_weights is not None and not weighted.case_weights.flags.writeable


@pytest.mark.parametrize("change,match", [
    ("two_levels", "at least three"), ("pairwise_one", "at least two exposures"),
    ("missing_covariates", "require covariates"), ("one_cluster", "two clusters"),
    ("fraction_id", "exact integer"), ("bad_weights", "strictly positive"),
    ("bad_flag", "Boolean"), ("bad_summary", "digits"),
])
def test_rejects_unsupported_cases(change, match):
    y, x, z, ids = cells()
    if change == "bad_summary":
        result = fit_quadratic_gaussian_ee(y, x, ids, z, q=None)
        with pytest.raises(ValueError, match=match):
            result.summary(0)
        return
    kwargs = {"q": None}
    if change == "two_levels":
        x = x % 2
    elif change == "pairwise_one":
        x = x[:, :1]
        kwargs["pairwise_interactions"] = True
    elif change == "missing_covariates":
        z = None
        kwargs["covariate_interactions"] = True
    elif change == "one_cluster":
        ids = np.zeros_like(ids)
    elif change == "fraction_id":
        ids = ids.astype(float)
        ids[0] = 0.5
    elif change == "bad_weights":
        kwargs["weights"] = np.zeros(len(y))
    else:
        kwargs["pairwise_interactions"] = "yes"
    with pytest.raises(ValueError, match=match):
        fit_quadratic_gaussian_ee(y, x, ids, z, **kwargs)


def test_effect_at_checks_level():
    y, x, z, ids = cells()
    result = fit_quadratic_gaussian_ee(y, x, ids, z, q=None)
    for level in (-1, 3, 0.5, True):
        with pytest.raises(ValueError, match="consecutive intervention"):
            result.effect_at(level)
