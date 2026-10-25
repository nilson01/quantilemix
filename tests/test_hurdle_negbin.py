"""Known-answer pscl comparison and independent NB hurdle checks."""

import numpy as np
import pytest
from scipy import special, stats

from quantilemix import fit_hurdle_negative_binomial
from quantilemix._hurdle_negbin import _positive_negbin_likelihood_score
import quantilemix._hurdle_negbin as hurdle_negbin


def _case():
    i = np.arange(480)
    x = np.column_stack((i % 4, (i // 4) % 4))
    z = np.sin(i * .17)
    mu = np.exp(.3 + .15*x[:, 0] - .1*x[:, 1] + .2*z)
    theta = 1.3
    chance = special.expit(.2 + .2*x[:, 0] - .15*x[:, 1] - .1*z)
    u0 = ((i*37) % 479 + .5)/479
    u1 = ((i*113) % 479 + .5)/479
    p0 = stats.nbinom.pmf(0, theta, theta/(theta+mu))
    y = (u0 < chance)*np.maximum(
        1, stats.nbinom.ppf(np.minimum(p0+u1*(1-p0), np.nextafter(1., 0.)),
                             theta, theta/(theta+mu)))
    return y, x, z


def test_installed_pscl_hurdle_negbin_fixed_case():
    # pscl 1.5.9, hurdle(y ~ x1+x2+z, dist="negbin"); see validation report.
    y, x, z = _case()
    fit = fit_hurdle_negative_binomial(y, x, z, q=None)
    expected = [.345410811494391, .0971004346637867, -.140509069321271,
                .220006359367639, .203586088294525, .196018614072195,
                -.155111580924585, -.0717122050133931]
    np.testing.assert_allclose(fit.coefficients, expected, atol=8e-7, rtol=0)
    np.testing.assert_allclose(fit.theta, 1.38974573101345, atol=8e-7, rtol=0)
    np.testing.assert_allclose(np.diag(fit.covariance),
                               [.0411544278502264, .00472846659041075,
                                .00463965478807451, .0117394548091194,
                                .0392570752087437, .00697020349419767,
                                .00694917456595351, .0172716003851807],
                               atol=2e-6, rtol=0)
    np.testing.assert_allclose(fit.psi_count, expected[1]+expected[2], atol=8e-7)
    np.testing.assert_allclose(fit.psi_nonzero, expected[5]+expected[6], atol=8e-7)
    mu_hat, p_hat = fit.predict_components(x[:5], z[:5])
    p0_hat = np.exp(-fit.theta*np.log1p(mu_hat/fit.theta))
    np.testing.assert_allclose(fit.predict(x[:5], z[:5]),
                               p_hat*mu_hat/(1-p0_hat), atol=1e-12)
    assert "theta" in fit.summary()
    assert not fit.coefficients.flags.writeable


def test_positive_negbin_score_matches_likelihood_difference():
    y, x, z = _case()
    selected = y > 0
    design = np.column_stack((np.ones(sum(selected)), x[selected], z[selected]))
    parameters = np.array([.35, .1, -.14, .2, np.log(1.3)])
    _, score = _positive_negbin_likelihood_score(design, y[selected], parameters)
    steps = np.eye(len(parameters))*1e-5
    numerical = [(_positive_negbin_likelihood_score(design, y[selected], parameters+v)[0]
                  - _positive_negbin_likelihood_score(design, y[selected], parameters-v)[0])
                 / 2e-5 for v in steps]
    np.testing.assert_allclose(score, numerical, atol=2e-6, rtol=0)


def test_count_score_refinement_repairs_early_optimizer_stop(monkeypatch):
    y, x, z = _case()
    expected = fit_hurdle_negative_binomial(y, x, z, q=None)
    original = hurdle_negbin.optimize.minimize
    def early_stop(*args, **kwargs):
        result = original(*args, **kwargs)
        result.x = result.x.copy()
        result.x[0] += 1e-3
        return result
    monkeypatch.setattr(hurdle_negbin.optimize, "minimize", early_stop)
    repaired = fit_hurdle_negative_binomial(y, x, z, q=None)
    np.testing.assert_allclose(repaired.coefficients, expected.coefficients,
                               atol=1e-7, rtol=0)
    np.testing.assert_allclose(repaired.theta, expected.theta, atol=1e-7, rtol=0)


@pytest.mark.parametrize("component", ["count", "zero"])
def test_exposure_placement_and_separate_adjustment(component):
    y, x, z = _case()
    fit = fit_hurdle_negative_binomial(y, x, z, zero_covariates=z*z,
                                        exposure_component=component, q=None)
    if component == "count":
        assert fit.psi_count is not None and fit.psi_nonzero is None
    else:
        assert fit.psi_count is None and fit.psi_nonzero is not None
    predicted = fit.predict(x[:4], z[:4], zero_covariates=z[:4]**2)
    assert np.all(np.isfinite(predicted)) and np.all(predicted > 0)
    with pytest.raises(ValueError, match="zero_covariates"):
        fit.predict(x[:4], z[:4])


@pytest.mark.parametrize("outcome", [[0]*32, [1]*32, [0, 1.5]+[0]*30])
def test_reject_bad_count_outcomes(outcome):
    with pytest.raises(ValueError):
        fit_hurdle_negative_binomial(outcome, np.arange(32) % 4, q=None)
