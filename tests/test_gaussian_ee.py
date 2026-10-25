"""Independent clustered-contrast arithmetic and bounded input checks."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy import stats

from quantilemix import fit_gaussian_ee


def cells():
    x = np.array([(a, b) for a in range(4) for b in range(4)] * 5, dtype=float)
    n = len(x)
    ids = np.arange(n) // 2
    z = np.arange(n) % 3
    y = (1 + .4*x[:, 0] - .2*x[:, 1] + .3*z
         + .07*((ids*3) % 5 - 2) + .015*((np.arange(n)*7) % 11 - 5))
    return y, x, z.astype(float), ids


def test_cluster_robust_joint_contrast_from_conditional_scores():
    y, x, z, ids = cells()
    result = fit_gaussian_ee(y, x, ids, z, q=None)
    design = np.column_stack((np.ones(len(y)), x, z))
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ beta
    scores = np.array([np.sum(design[ids == group] * residual[ids == group, None], axis=0)
                       for group in np.unique(ids)])
    cond_bread = np.linalg.inv(design.T @ design)
    cond_cov = cond_bread @ (scores.T @ scores) @ cond_bread
    contrast = np.array([0., 1., 1., 0.])
    expected_psi = float(contrast @ beta)
    expected_var = float(contrast @ cond_cov @ contrast)
    assert_allclose(result.conditional_coefficients, beta, atol=1e-12)
    assert_allclose(result.conditional_covariance, cond_cov, atol=1e-12)
    assert_allclose(result.psi, expected_psi, atol=1e-12)
    assert_allclose(result.var_psi, expected_var, atol=1e-12)
    assert_allclose(result.se, np.sqrt(expected_var), atol=1e-12)
    assert_allclose(result.msm_coefficients,
                    [beta[0] + beta[3]*np.mean(z), expected_psi], atol=1e-12)
    assert_allclose(np.diff(result.intervention_means), expected_psi, atol=1e-12)
    assert_allclose(result.p_value,
                    2*stats.t.sf(abs(expected_psi/np.sqrt(expected_var)),
                                 len(y)-2-1-2), atol=1e-12)
    assert result.cluster_count == len(np.unique(ids))
    assert result.p_value_df == len(y)-2-1-2
    assert "cluster-robust SE" in result.summary()
    assert not result.cluster_ids.flags.writeable
    assert_array_equal(result.intervention_levels, np.arange(4))


def test_gaussian_ee_without_adjustment_columns():
    y, x, _, ids = cells()
    result = fit_gaussian_ee(y, x, ids, q=None)
    design = np.column_stack((np.ones(len(y)), x))
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    assert_allclose(result.psi, sum(beta[1:3]), atol=1e-12)
    assert_allclose(result.intervention_means,
                    beta[0] + np.arange(4)*sum(beta[1:3]), atol=1e-12)


@pytest.mark.parametrize("change", ["id_length", "id_fraction", "id_large", "one_cluster",
                                     "missing_id", "two_levels", "rank", "bad_summary"])
def test_gaussian_ee_rejects_invalid_cases(change):
    y, x, z, ids = cells()
    if change == "id_length":
        with pytest.raises(ValueError, match="cluster_ids"):
            fit_gaussian_ee(y, x, ids[:-1], z, q=None)
    elif change == "id_fraction":
        ids = ids.astype(float)
        ids[2] = .5
        with pytest.raises(ValueError, match="exact integer"):
            fit_gaussian_ee(y, x, ids, z, q=None)
    elif change == "id_large":
        ids = ids.astype(float)
        ids[2] = 2**53
        with pytest.raises(ValueError, match="exact integer"):
            fit_gaussian_ee(y, x, ids, z, q=None)
    elif change == "one_cluster":
        with pytest.raises(ValueError, match="two clusters"):
            fit_gaussian_ee(y, x, np.zeros(len(ids)), z, q=None)
    elif change == "missing_id":
        ids = ids.astype(float)
        ids[2] = np.nan
        with pytest.raises(ValueError, match="finite"):
            fit_gaussian_ee(y, x, ids, z, q=None)
    elif change == "two_levels":
        with pytest.raises(ValueError, match="2 through 9"):
            fit_gaussian_ee(y, np.zeros_like(x), ids, z, q=None)
    elif change == "rank":
        with pytest.raises(ValueError, match="rank"):
            fit_gaussian_ee(y, np.column_stack((x[:, 0], x[:, 0])), ids, z, q=None)
    else:
        result = fit_gaussian_ee(y, x, ids, z, q=None)
        with pytest.raises(ValueError, match="digits"):
            result.summary(0)
