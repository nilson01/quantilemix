"""Known-answer checks for one-row conditional fit summaries."""

import numpy as np
import pytest
from scipy import special, stats

from quantilemix import QGCompWarning, fit, fit_binomial, fit_cox, fit_poisson, glance


def test_gaussian_fit_statistics_from_known_group_means():
    x = np.tile([0., 1.], 8)
    residual = np.tile([0.2, -0.2, -0.2, 0.2], 4)
    result = fit(1+2*x+residual, x[:, None], q=None)
    row = glance(result)
    sse, tss, n, k = 0.64, 16.64, 16, 2
    expected_log_likelihood = -n/2*(np.log(2*np.pi)+1+np.log(sse/n))
    assert row.family == "gaussian"
    assert row["nobs"] == n and row["df_residual"] == 14
    np.testing.assert_allclose(row["deviance"], sse, atol=1e-12)
    np.testing.assert_allclose(row["null_deviance"], tss, atol=1e-12)
    assert row["df_null"] == n-1
    np.testing.assert_allclose(row["r_squared"], 1-sse/tss, atol=1e-12)
    np.testing.assert_allclose(row["adjusted_r_squared"],
                               1-(sse/tss)*(n-1)/(n-k), atol=1e-12)
    np.testing.assert_allclose(row["sigma"], np.sqrt(sse/(n-k)), atol=1e-12)
    np.testing.assert_allclose(row["f_statistic"], 16/(sse/(n-k)), atol=1e-10)
    np.testing.assert_allclose(row["f_p_value"],
                               stats.f.sf(16/(sse/(n-k)), 1, n-k), atol=1e-15)
    np.testing.assert_allclose(row["log_likelihood"], expected_log_likelihood,
                               atol=1e-12)
    np.testing.assert_allclose(row["aic"], -2*expected_log_likelihood+2*(k+1))
    np.testing.assert_allclose(row["bic"], -2*expected_log_likelihood+np.log(n)*(k+1))
    assert row.to_dict() == dict(row.table)
    with pytest.raises(TypeError):
        row.table["nobs"] = 1


def test_weighted_gaussian_likelihood_uses_log_weights():
    x = np.tile([0., 1.], 8)
    y = 1+2*x+np.tile([0.2, -0.2, -0.2, 0.2], 4)
    weights = np.tile([1., 1., 2., 2.], 4)
    result = fit(y, x[:, None], q=None, weights=weights)
    row = glance(result)
    weighted_sse = np.sum(weights*result.residuals**2)
    expected = 0.5*(np.sum(np.log(weights))
                    -len(y)*(np.log(2*np.pi)+1-np.log(len(y))+np.log(weighted_sse)))
    np.testing.assert_allclose(row["log_likelihood"], expected, atol=1e-12)
    assert row["nobs"] == len(y)


def test_gaussian_numerical_zero_marks_likelihood_undefined():
    x = np.tile([0., 1.], 8)
    with pytest.warns(QGCompWarning, match="numerical zero"):
        result = fit(1+2*x, x[:, None], q=None)
    row = glance(result)
    assert row["sigma"] == 0
    for field in ("log_likelihood", "aic", "bic", "f_statistic", "f_p_value"):
        assert row[field] is None


def test_binary_group_likelihood_and_null_deviance():
    x = np.repeat([0., 1.], 8)
    y = np.r_[[0]*6+[1]*2, [0]*2+[1]*6]
    row = glance(fit_binomial(y, x[:, None], q=None))
    log_likelihood = 4*np.log(0.25)+12*np.log(0.75)
    null_log_likelihood = 16*np.log(0.5)
    assert row.family == "binomial"
    assert row["df_null"] == 15
    np.testing.assert_allclose(row["log_likelihood"], log_likelihood, atol=1e-12)
    np.testing.assert_allclose(row["deviance"], -2*log_likelihood, atol=1e-12)
    np.testing.assert_allclose(row["null_deviance"], -2*null_log_likelihood,
                               atol=1e-12)
    np.testing.assert_allclose(row["aic"], -2*log_likelihood+4)
    np.testing.assert_allclose(row["bic"], -2*log_likelihood+2*np.log(16))


def test_poisson_offset_null_model_retains_observation_times():
    x = np.repeat([0., 1.], 8)
    time = np.tile([1., 2.], 8)
    y = np.r_[[0, 1, 2, 0, 1, 2, 2, 4], [0, 2, 4, 0, 2, 4, 4, 8]]
    result = fit_poisson(y, x[:, None], q=None, exposure_time=time)
    row = glance(result)
    model_mean = time*np.where(x == 0, 1., 2.)
    null_mean = 1.5*time
    saturated = np.sum(special.xlogy(y, y)-y-special.gammaln(y+1))
    model_log_likelihood = np.sum(y*np.log(model_mean)-model_mean-special.gammaln(y+1))
    null_log_likelihood = np.sum(y*np.log(null_mean)-null_mean-special.gammaln(y+1))
    assert row.family == "poisson"
    np.testing.assert_allclose(result.fitted_values, model_mean, atol=1e-10)
    np.testing.assert_allclose(row["log_likelihood"], model_log_likelihood, atol=1e-10)
    np.testing.assert_allclose(row["deviance"],
                               2*(saturated-model_log_likelihood), atol=1e-10)
    np.testing.assert_allclose(row["null_deviance"],
                               2*(saturated-null_log_likelihood), atol=1e-10)


@pytest.mark.parametrize("family", ["binomial", "poisson"])
def test_integer_weights_match_independent_row_replication(family):
    x = np.repeat([0., 1.], 8)
    weights = np.tile([1, 2, 3, 1], 4)
    if family == "binomial":
        y = np.r_[[0]*6+[1]*2, [0]*2+[1]*6]
        weighted = fit_binomial(y, x[:, None], q=None, weights=weights)
        repeated = fit_binomial(np.repeat(y, weights),
                                np.repeat(x, weights)[:, None], q=None)
    else:
        y = np.r_[[0, 1, 2, 0, 1, 2, 2, 4], [0, 2, 4, 0, 2, 4, 4, 8]]
        time = np.tile([1., 2.], 8)
        weighted = fit_poisson(y, x[:, None], q=None, weights=weights,
                               exposure_time=time)
        repeated = fit_poisson(np.repeat(y, weights),
                               np.repeat(x, weights)[:, None], q=None,
                               exposure_time=np.repeat(time, weights))
    left, right = glance(weighted), glance(repeated)
    for field in ("log_likelihood", "deviance", "null_deviance"):
        np.testing.assert_allclose(left[field], right[field], atol=1e-9)
    assert left["nobs"] == len(y)
    assert right["nobs"] == int(np.sum(weights))


def test_unsupported_result_does_not_inherit_conditional_likelihood():
    x = np.tile([0., 1.], 20)
    time = np.arange(1, len(x)+1, dtype=float)
    event = np.tile([1, 0, 0, 1], 10)
    result = fit_cox(time, event, x[:, None], q=None)
    with pytest.raises(TypeError, match="additive conditional"):
        glance(result)
