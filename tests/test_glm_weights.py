"""Independent frequency-weight and input-contract checks for the two GLMs."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from quantilemix import (fit_binomial, fit_binomial_formula, fit_poisson,
                    fit_poisson_formula)


def _cases(family):
    x = np.tile([[0, 0], [0, 1], [1, 0], [1, 1]], (12, 1))
    if family == "binomial":
        y = np.tile([0, 1, 1, 0, 1, 0, 0, 1], 6)
        fitter, formula_fitter = fit_binomial, fit_binomial_formula
    else:
        y = np.tile([0, 1, 2, 3, 1, 3, 2, 0], 6)
        fitter, formula_fitter = fit_poisson, fit_poisson_formula
    weights = 1 + np.arange(len(y)) % 3
    return y, x, weights, fitter, formula_fitter


@pytest.mark.parametrize("family", ["binomial", "poisson"])
def test_integer_case_weights_equal_repeated_observations(family):
    y, x, w, fitter, _ = _cases(family)
    weighted = fitter(y, x, weights=w, q=None)
    repeated = fitter(np.repeat(y, w), np.repeat(x, w, axis=0), q=None)
    assert_allclose(weighted.coefficients, repeated.coefficients, atol=1e-11)
    assert_allclose(weighted.covariance, repeated.covariance, atol=1e-11)
    assert_allclose(weighted.psi, repeated.psi, atol=1e-11)
    assert weighted.diagnostics["case_weighted"] is True
    assert "Case weights: supplied; model-based SE." in weighted.summary()
    assert_array_equal(weighted.case_weights, w)
    assert not weighted.case_weights.flags.writeable
    unweighted = fitter(y, x, q=None)
    assert unweighted.case_weights is None
    assert "Case weights: not supplied; model-based SE." in unweighted.summary()


@pytest.mark.parametrize("family", ["binomial", "poisson"])
def test_weighted_formula_drops_missing_weight_but_keeps_original_cuts(family):
    y, x, w, fitter, formula_fitter = _cases(family)
    data = {"y": y, "x1": x[:, 0], "x2": x[:, 1]}
    incomplete = w.astype(float)
    incomplete[3] = np.nan
    result = formula_fitter("y ~ x1 + x2", data, ("x1", "x2"),
                            weights=incomplete, missing="drop", q=2)
    kept = np.arange(len(y)) != 3
    manual = fitter(y[kept], x[kept], weights=w[kept],
                    q=2, breaks=result.breaks)
    assert_allclose(result.coefficients, manual.coefficients, atol=1e-11)
    assert result.training_row_indices == tuple(np.flatnonzero(kept))
    assert result.original_row_count == len(y)
    assert_array_equal(result.case_weights, w[kept])
    with pytest.raises(ValueError):
        formula_fitter("y ~ x1 + x2", data, ("x1", "x2"),
                       weights=incomplete, q=2)


@pytest.mark.parametrize("family", ["binomial", "poisson"])
@pytest.mark.parametrize("bad", [0, -1, np.inf, np.nan, True])
def test_invalid_weights_rejected(family, bad):
    y, x, w, fitter, _ = _cases(family)
    invalid = w.astype(object)
    invalid[0] = bad
    with pytest.raises(ValueError, match="weights"):
        fitter(y, x, weights=invalid, q=None)
