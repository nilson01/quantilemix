"""Independent pointwise contrast calculations and failure cases."""

from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import special

from quantilemix import fit, fit_binomial, fit_poisson, pointwisebound_noboot


def _gaussian_case():
    rows = np.array(list(product(range(4), range(4), range(3))), dtype=float)
    x, z = rows[:, :2], rows[:, 2]
    residual = 0.07 * ((3*x[:, 0] + x[:, 1] + 2*z) % 5)
    y = 1.5 + 0.8*x[:, 0] - 0.3*x[:, 1] + 0.2*z + residual
    return y, x, z


def test_gaussian_independent_full_covariance_and_reference_level():
    y, x, z = _gaussian_case()
    fitted = fit(y, x, z, q=4)
    observed = pointwisebound_noboot(fitted, pointwiseref=3, alpha=0.1)
    design = np.column_stack((np.ones(len(x)), x, z))
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    covariance = np.linalg.inv(design.T @ design) * np.sum((y-design@beta)**2)/(len(y)-design.shape[1])
    contrast = np.array([0, 1, 1, 0], dtype=float)
    joint = float(contrast @ beta)
    joint_se = float(np.sqrt(contrast @ covariance @ contrast))
    difference = (np.arange(4)-2)*joint
    errors = np.abs(np.arange(4)-2)*joint_se
    critical = -special.ndtri(.1/2)
    assert observed.columns == ("quantile", "quantile.midpoint", "linpred",
                                "mean.diff", "se.diff", "ll.diff", "ul.diff")
    assert_allclose(observed["linpred"], beta[0] + np.arange(4)*joint, atol=1e-13)
    assert_allclose(observed["mean.diff"], difference, atol=1e-13)
    assert_allclose(observed["se.diff"], errors, atol=1e-13)
    assert_allclose(observed["ll.diff"], difference-critical*errors, atol=1e-13)
    assert_allclose(observed["ul.diff"], difference+critical*errors, atol=1e-13)
    assert observed["se.diff"][2] == 0
    assert not observed["linpred"].flags.writeable
    with pytest.raises(TypeError):
        observed.table["extra"] = np.zeros(4)


def test_binary_exact_cell_odds_ratio():
    x, y = [], []
    for first, second in product((0, 1), repeat=2):
        odds = 3 ** (first + second)
        successes = round(20*odds/(1+odds))
        x.extend([(first, second)]*20)
        y.extend([1]*successes + [0]*(20-successes))
    fitted = fit_binomial(y, x, q=2)
    observed = pointwisebound_noboot(fitted)
    assert observed.link == "logit"
    assert observed.measure == "conditional odds ratio"
    assert observed.columns == ("quantile", "quantile.midpoint", "linpred",
                                "or", "se.lnor", "ll.or", "ul.or")
    assert_allclose(observed["or"], [1, 9], atol=1e-10)
    assert_allclose(observed["se.lnor"][1], fitted.se, atol=1e-12)


def test_poisson_exact_cell_mean_ratio():
    x = np.repeat(np.array(list(product((0, 1), repeat=2))), 6, axis=0)
    y = 2*2**(x[:, 0]+x[:, 1])
    fitted = fit_poisson(y, x, q=2)
    observed = pointwisebound_noboot(fitted, pointwiseref=2)
    assert observed.link == "log"
    assert observed.measure == "conditional mean ratio"
    assert observed.columns == ("quantile", "quantile.midpoint", "linpred",
                                "rr", "se.lnrr", "ll.rr", "ul.rr")
    assert_allclose(observed["rr"], [0.25, 1], atol=1e-11)
    assert observed["se.lnrr"][1] == 0


@pytest.mark.parametrize("reference,alpha", [(0, .05), (5, .05), (1.5, .05),
                                             (True, .05), (1, 0), (1, 1),
                                             (1, float("nan")), (1, False)])
def test_reject_invalid_reference_or_alpha(reference, alpha):
    y, x, z = _gaussian_case()
    fitted = fit(y, x, z, q=4)
    with pytest.raises(ValueError):
        pointwisebound_noboot(fitted, pointwiseref=reference, alpha=alpha)


def test_reject_prequantized_offset_and_unsupported_fit():
    y, x, z = _gaussian_case()
    with pytest.raises(ValueError, match="non-null"):
        pointwisebound_noboot(fit(y, x, z, q=None))
    count = np.arange(1, len(y)+1) % 7
    with pytest.raises(ValueError, match="offset"):
        pointwisebound_noboot(fit_poisson(count, x, z, q=4,
                                        exposure_time=np.ones(len(y))))
    with pytest.raises(TypeError, match="non-bootstrap"):
        pointwisebound_noboot(object())
