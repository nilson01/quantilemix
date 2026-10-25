"""Independent Bernoulli-logit answers and strict family boundaries."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from quantilemix import fit_binomial, fit_binomial_formula


def exact_cells():
    # Odds are 1/3, 1, 1, and 3 in the four cells. Both exposure
    # coefficients are exactly log(3), so the joint odds ratio is 9.
    cells = ((0, 0, 5), (1, 0, 10), (0, 1, 10), (1, 1, 15))
    x = np.vstack([np.tile([a, b], (20, 1)) for a, b, _ in cells])
    y = np.concatenate([np.r_[np.ones(successes), np.zeros(20-successes)]
                        for _, _, successes in cells])
    return y, x


def test_exact_log_odds_and_independent_information():
    y, x = exact_cells()
    result = fit_binomial(y, x, q=None)
    assert_allclose(result.coefficients, [-np.log(3), np.log(3), np.log(3)], atol=1e-12)
    assert_allclose(result.psi, np.log(9), atol=1e-12)
    assert_allclose(result.odds_ratio, 9, atol=1e-12)
    assert_allclose(result.predict([[0, 0], [1, 0], [0, 1], [1, 1]]),
                    [.25, .5, .5, .75], atol=1e-12)
    design = np.column_stack((np.ones(len(x)), x))
    probability = result.fitted_values
    information = design.T @ ((probability*(1-probability))[:, None]*design)
    covariance = np.linalg.inv(information)
    contrast = np.array([0, 1, 1])
    assert_allclose(result.covariance, covariance, atol=1e-12)
    assert_allclose(result.var_psi, contrast @ covariance @ contrast, atol=1e-12)
    assert result.p_value_df is None
    assert "not a risk ratio" in result.summary()
    assert result.family == "binomial" and result.link == "logit"
    assert result.coefficients.flags.writeable is False


def test_quantile_and_formula_encoding_agree_with_numeric_design():
    y, x = exact_cells()
    group = np.tile(["A", "B"], 40)
    data = {"y": y, "x1": x[:, 0], "x2": x[:, 1], "group": group}
    formula = fit_binomial_formula("y ~ x1 + x2 + C(group)", data,
                                   ("x1", "x2"), q=2)
    manual = fit_binomial(y, x, (group == "B").astype(float), q=2)
    assert_allclose(formula.coefficients, manual.coefficients, atol=1e-12)
    assert_allclose(formula.covariance, manual.covariance, atol=1e-12)
    assert formula.coefficient_names == ("(Intercept)", "x1", "x2", "group[T.B]")
    assert dict(formula.categorical_levels) == {"group": ("A", "B")}
    assert_allclose(formula.predict_data({"x1": [0, 1], "x2": [1, 1],
                                          "group": ["A", "B"]}),
                    manual.predict([[0, 1], [1, 1]], [0, 1]), atol=1e-12)
    assert_array_equal(formula.transform_data({"x1": [0, 1], "x2": [1, 1]}).scores,
                       [[0, 1], [1, 1]])


@pytest.mark.parametrize("change", ("nonbinary", "missing", "single_class", "rank", "separated"))
def test_invalid_binary_fit_rejected(change):
    y, x = exact_cells()
    if change == "nonbinary":
        y[0] = 0.5
    elif change == "missing":
        y[0] = np.nan
    elif change == "single_class":
        y[:] = 1
    elif change == "rank":
        x[:, 1] = x[:, 0]
    else:
        y[:] = x[:, 0]
    with pytest.raises(ValueError):
        fit_binomial(y, x, q=None)


def test_prediction_inputs_and_display_digits_rejected():
    y, x = exact_cells()
    result = fit_binomial(y, x, q=None)
    with pytest.raises(ValueError, match="without covariates"):
        result.predict([[0, 1]], [1])
    with pytest.raises(ValueError, match="digits"):
        result.summary(digits=0)
    with pytest.raises(ValueError, match="requires"):
        result.predict_data({"x1": [0], "x2": [1]})
