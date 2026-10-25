"""Public contract and failure checks for marginal bootstrap curve bounds."""

from dataclasses import replace

import numpy as np
import pytest
from scipy import stats

from quantilemix import fit_bootstrap, fit_poisson_bootstrap, modelbound_boot


def _rows():
    x = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), (12, 1))
    i = np.arange(len(x))
    return x, i


def test_gaussian_pointwise_and_simultaneous_scale():
    x, i = _rows()
    y = 2+1.2*x[:, 0]-.3*x[:, 1]+.11*(i % 7)
    fit = fit_bootstrap(y, x, q=2, B=20, seed=7)
    result = modelbound_boot(fit)
    se = np.std(fit.bootstrap_means, axis=0, ddof=1)
    critical = stats.norm.isf(.025)
    np.testing.assert_allclose(result["m"], fit.intervention_means)
    np.testing.assert_allclose(result["se.pw"], se)
    np.testing.assert_allclose(result["ll.pw"], fit.intervention_means-critical*se)
    np.testing.assert_allclose(result["ul.pw"], fit.intervention_means+critical*se)
    assert np.all(result["ll.simul"] <= result["ul.simul"])
    assert result.link == "identity"
    assert not result["ll.simul"].flags.writeable
    with pytest.raises(ValueError):
        result["ll.simul"][0] = -100


def test_poisson_offset_pointwise_link_and_pwonly():
    x, i = _rows()
    time = 1+i % 3
    y = (1+x[:, 0]+2*x[:, 1])*time+i % 2
    fit = fit_poisson_bootstrap(y, x, q=2, exposure_time=time, B=20, seed=9)
    result = modelbound_boot(fit, pwonly=True)
    profiles = np.log(fit.bootstrap_means)
    se = np.std(profiles, axis=0, ddof=1)
    np.testing.assert_allclose(result["linpred"], np.log(fit.intervention_means))
    np.testing.assert_allclose(result["r"], fit.intervention_means)
    np.testing.assert_allclose(result["se.pw"], se)
    np.testing.assert_allclose(result["ll.pw"],
                               np.exp(np.log(fit.intervention_means)-stats.norm.isf(.025)*se))
    assert np.all(np.isnan(result["ll.simul"]))
    assert np.all(np.isnan(result["ul.simul"]))
    assert result.link == "log"


@pytest.mark.parametrize("alpha", [0, 1, float("nan"), True, "0.05"])
def test_bad_alpha(alpha):
    x, i = _rows()
    fit = fit_bootstrap(2+x[:, 0]+i % 3, x, q=2, B=12, seed=3)
    with pytest.raises(ValueError, match="alpha"):
        modelbound_boot(fit, alpha=alpha)


@pytest.mark.parametrize("pwonly", [0, 1, "yes", None])
def test_bad_pwonly(pwonly):
    x, i = _rows()
    fit = fit_bootstrap(2+x[:, 0]+i % 3, x, q=2, B=12, seed=3)
    with pytest.raises(ValueError, match="pwonly"):
        modelbound_boot(fit, pwonly=pwonly)


def test_nonbootstrap_and_null_q_rejected():
    with pytest.raises(TypeError, match="bootstrap"):
        modelbound_boot(object())
    x, i = _rows()
    fit = fit_bootstrap(2+x[:, 0]+i % 3, x, q=None, B=12, seed=3)
    with pytest.raises(ValueError, match="non-null q"):
        modelbound_boot(fit)


def test_singular_covariance_and_nonlinear_profiles_rejected():
    x, i = _rows()
    fit = fit_bootstrap(2+x[:, 0]+i % 3, x, q=2, B=12, seed=3)
    constant = np.tile(fit.bootstrap_means[0], (fit.B, 1))
    with pytest.raises(ValueError, match="singular"):
        modelbound_boot(replace(fit, bootstrap_means=constant))
    x3 = np.tile(np.array([(a, b) for a in range(3) for b in range(3)]), (5, 1))
    y3 = 1+x3[:, 0]+.2*x3[:, 1]+.1*(np.arange(len(x3)) % 5)
    fit3 = fit_bootstrap(y3, x3, q=3, B=25, seed=10)
    altered = fit3.bootstrap_means.copy()
    altered[:, 1] += 1
    with pytest.raises(ValueError, match="degree-one"):
        modelbound_boot(replace(fit3, bootstrap_means=altered))
