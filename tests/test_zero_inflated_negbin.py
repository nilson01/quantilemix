"""Installed-pscl known answer and zero-inflated NB likelihood checks."""

import numpy as np
import pytest
from scipy import special, stats

from quantilemix import fit_zero_inflated_negative_binomial
from quantilemix._zero_inflated_negbin import _negbin_mixture_likelihood_score


def _case():
    i = np.arange(720)
    x = np.column_stack((i % 4, (i // 4) % 4))
    z = np.sin(i*.17)
    mu = np.exp(.3+.18*x[:, 0]-.11*x[:, 1]+.12*z)
    pi = special.expit(-.8+.15*x[:, 0]-.12*x[:, 1]+.1*z)
    u0 = ((i*37) % 719+.5)/719
    u1 = ((i*113) % 719+.5)/719
    y = np.where(u0 < pi, 0, stats.nbinom.ppf(u1, 1.2, 1.2/(1.2+mu)))
    return y, x, z


def test_installed_pscl_zero_inflated_negbin_case():
    # pscl 1.5.9, zeroinfl(..., dist="negbin", EM=FALSE, reltol=1e-12).
    y, x, z = _case()
    fit = fit_zero_inflated_negative_binomial(y, x, z, q=None)
    expected = [.248453142291354, .223493344183126, -.0718189425664094,
                -.00136056810629047, -.960137077409183, .223235386548354,
                -.0543722290050239, -.198870581595725]
    np.testing.assert_allclose(fit.coefficients, expected, atol=2e-6, rtol=0)
    np.testing.assert_allclose(fit.theta, 1.297567555397, atol=2e-6, rtol=0)
    np.testing.assert_allclose(np.diag(fit.covariance),
                               [.0327427302219605, .00383950347077368,
                                .0037262503559417, .0109476883134774,
                                .255312980557782, .0241071715364941,
                                .0178370871863353, .0509565763844511],
                               atol=2e-6, rtol=0)
    np.testing.assert_allclose(fit.se_logtheta, .314860887439843, atol=2e-6, rtol=0)
    mu, pi = fit.predict_components(x[:6], z[:6])
    np.testing.assert_allclose(fit.predict(x[:6], z[:6]), (1-pi)*mu, atol=1e-12)
    assert "theta" in fit.summary() and not fit.coefficients.flags.writeable


def test_zero_inflated_negbin_score_matches_numerical_derivative():
    y, x, z = _case()
    count = np.column_stack((np.ones(len(y)), x, z))
    zero = count.copy()
    parameters = np.array([.3, .18, -.11, .12, -.8, .15, -.12, .1, np.log(1.2)])
    _, score = _negbin_mixture_likelihood_score(count, zero, y, parameters)
    steps = np.eye(len(parameters))*1e-5
    numerical = [(_negbin_mixture_likelihood_score(count, zero, y, parameters+v)[0]
                  - _negbin_mixture_likelihood_score(count, zero, y, parameters-v)[0])
                 / 2e-5 for v in steps]
    np.testing.assert_allclose(score, numerical, atol=2e-6, rtol=0)


@pytest.mark.parametrize("component", ["count", "zero"])
def test_exposure_placement_with_distinct_zero_adjustment(component):
    y, x, z = _case()
    fit = fit_zero_inflated_negative_binomial(
        y, x, z, zero_covariates=z*z, exposure_component=component, q=None)
    if component == "count":
        assert fit.psi_count is not None and fit.psi_zero is None
    else:
        assert fit.psi_count is None and fit.psi_zero is not None
    predicted = fit.predict(x[:4], z[:4], zero_covariates=z[:4]**2)
    assert np.all(np.isfinite(predicted)) and np.all(predicted > 0)
    with pytest.raises(ValueError, match="zero_covariates"):
        fit.predict(x[:4], z[:4])


@pytest.mark.parametrize("outcome", [[0]*32, [1]*32, [0, 1.5]+[0]*30])
def test_reject_bad_count_outcomes(outcome):
    with pytest.raises(ValueError):
        fit_zero_inflated_negative_binomial(outcome, np.arange(32) % 4, q=None)
