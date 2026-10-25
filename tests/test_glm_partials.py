"""Independent binary and count held-out partial-effect checks."""

import numpy as np
from numpy.testing import assert_allclose
import pytest

from quantilemix import GLMPartialResult, qgcomp_partials


def _binary_cells(multiplier, repeat):
    x, y = [], []
    for x1 in (0, 1):
        for x2 in (0, 1):
            odds = multiplier**(x1-x2)
            successes = round(repeat*odds/(1+odds))
            x.extend([(x1, x2)] * repeat)
            y.extend([1] * successes + [0] * (repeat-successes))
    return np.asarray(y), np.asarray(x)


def test_binomial_partials_have_exact_cell_targets():
    train_y, train_x = _binary_cells(2, 6)
    valid_y, valid_x = _binary_cells(3, 4)
    result = qgcomp_partials(train_y, train_x, valid_y, valid_x,
                             q=None, family="binomial")
    assert isinstance(result, GLMPartialResult)
    assert result.positive_exposures == ("x1",)
    assert result.negative_exposures == ("x2",)
    assert_allclose(result.positive_fit.psi, np.log(3), atol=1e-10)
    assert_allclose(result.negative_fit.psi, -np.log(3), atol=1e-10)
    # Fisher information from the four exact validation cell probabilities.
    design = np.column_stack((np.ones(len(valid_x)), valid_x))
    probability = 3.0**(valid_x[:, 0]-valid_x[:, 1])
    probability /= 1 + probability
    covariance = np.linalg.inv(design.T @ ((probability*(1-probability))[:, None] * design))
    assert_allclose(result.positive_fit.se, np.sqrt(covariance[1, 1]), atol=1e-10)
    assert_allclose(result.negative_fit.se, np.sqrt(covariance[2, 2]), atol=1e-10)
    assert "log odds ratio" in result.summary()


def _poisson_cells(multiplier, baseline, repeat):
    x = np.asarray([(x1, x2) for x1 in (0, 1) for x2 in (0, 1)
                    for _ in range(repeat)])
    y = baseline * float(multiplier)**(x[:, 0]-x[:, 1])
    return y.astype(int), x


def test_poisson_partials_have_exact_cell_targets():
    train_y, train_x = _poisson_cells(2, 4, 8)
    valid_y, valid_x = _poisson_cells(3, 3, 8)
    result = qgcomp_partials(train_y, train_x, valid_y, valid_x,
                             q=None, family="poisson")
    assert result.positive_exposures == ("x1",)
    assert result.negative_exposures == ("x2",)
    assert_allclose(result.positive_fit.psi, np.log(3), atol=1e-10)
    assert_allclose(result.negative_fit.psi, -np.log(3), atol=1e-10)
    design = np.column_stack((np.ones(len(valid_x)), valid_x))
    information = design.T @ (valid_y[:, None] * design)
    covariance = np.linalg.inv(information)
    assert_allclose(result.positive_fit.se, np.sqrt(covariance[1, 1]), atol=1e-10)
    assert_allclose(result.negative_fit.se, np.sqrt(covariance[2, 2]), atol=1e-10)
    assert "log rate ratio" in result.summary()


def test_validation_sign_reversal_keeps_training_groups():
    train_y, train_x = _binary_cells(2, 6)
    valid_y, valid_x = _binary_cells(1/3, 4)
    result = qgcomp_partials(train_y, train_x, valid_y, valid_x,
                             q=None, family="binomial")
    assert result.positive_exposures == ("x1",)
    assert result.negative_exposures == ("x2",)
    assert_allclose(result.positive_fit.psi, -np.log(3), atol=1e-10)
    assert_allclose(result.negative_fit.psi, np.log(3), atol=1e-10)


@pytest.mark.parametrize("family,bad_y", [("binomial", [0, 1, 2]),
                                            ("poisson", [0, -1, 2])])
def test_invalid_family_outcome_is_rejected(family, bad_y):
    x = np.array([[0, 0], [0, 1], [1, 0]])
    with pytest.raises(ValueError):
        qgcomp_partials(bad_y, x, bad_y, x, q=None, family=family)


def test_unsupported_family_is_rejected():
    with pytest.raises(ValueError, match="family"):
        qgcomp_partials([0, 1], [[0], [1]], [0, 1], [[0], [1]],
                        q=None, family="cox")
