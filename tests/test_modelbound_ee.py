"""Public contract and failure checks for Gaussian EE curve bounds."""

from dataclasses import replace

import numpy as np
import pytest
from scipy import stats

from quantilemix import fit_gaussian_ee, modelbound_ee
from quantilemix._modelbound_ee import _from_coefficient_draws


def _fit():
    x = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), (16, 1))
    i = np.arange(len(x))
    y = 1 + .7*x[:, 0] - .2*x[:, 1] + .2*(i % 7) + .03*(i % 3)
    ids = i // 2
    return fit_gaussian_ee(y, x, ids, q=2)


def test_reproducible_pointwise_and_simultaneous_bounds():
    fitted = _fit()
    first = modelbound_ee(fitted, seed=42)
    second = modelbound_ee(fitted, seed=42)
    assert first.columns == ("quantile", "quantile.midpoint", "linpred", "m", "se.pw",
                             "ll.pw", "ul.pw", "ll.simul", "ul.simul")
    for name in first.columns:
        np.testing.assert_array_equal(first[name], second[name])
    np.testing.assert_allclose(first["linpred"], fitted.msm_coefficients @
                               np.array([[1, 1], [0, 1]]))
    np.testing.assert_allclose(first["ll.pw"],
                               first["linpred"] - stats.norm.isf(.025)*first["se.pw"])
    np.testing.assert_allclose(first["ul.pw"],
                               first["linpred"] + stats.norm.isf(.025)*first["se.pw"])
    assert np.all(first["ll.simul"] <= first["ul.simul"])
    assert not first["m"].flags.writeable
    with pytest.raises(ValueError):
        first["m"][0] = 0


def test_pwonly_and_mc_pointwise_se():
    fitted = _fit()
    table = modelbound_ee(fitted, seed=41, pwonly=True)
    design = np.array([[1, 0], [1, 1]], dtype=float)
    analytic = np.sqrt(np.diag(design @ fitted.msm_covariance @ design.T))
    np.testing.assert_allclose(table["se.pw"], analytic, rtol=.05)
    assert np.all(np.isnan(table["ll.simul"]))
    assert np.all(np.isnan(table["ul.simul"]))
    assert table.family == "gaussian"
    assert table.link == "identity"


@pytest.mark.parametrize("alpha", [0, 1, float("nan"), True, "0.05"])
def test_bad_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        modelbound_ee(_fit(), alpha=alpha)


@pytest.mark.parametrize("pwonly", [0, 1, "yes", None])
def test_bad_pwonly(pwonly):
    with pytest.raises(ValueError, match="pwonly"):
        modelbound_ee(_fit(), pwonly=pwonly)


@pytest.mark.parametrize("seed", [-1, 1.5, True, "42"])
def test_bad_seed(seed):
    with pytest.raises(ValueError, match="seed"):
        modelbound_ee(_fit(), seed=seed)


def test_wrong_result_and_singular_covariance():
    with pytest.raises(TypeError, match="Gaussian"):
        modelbound_ee(object())
    fitted = _fit()
    with pytest.raises(ValueError, match="positive definite"):
        modelbound_ee(replace(fitted, msm_covariance=np.zeros((2, 2))))


def test_draws_require_expected_finite_shape():
    fitted = _fit()
    with pytest.raises(ValueError, match="5000-by-2"):
        _from_coefficient_draws(fitted, np.ones((4, 2)), alpha=.05, pwonly=False)
