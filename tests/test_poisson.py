"""Independent Poisson-log answers and strict count-family boundaries."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from quantilemix import fit_poisson, fit_poisson_formula


def exact_cells():
    # Four conditional means are 1, 2, 3 and 6. Therefore exposure
    # coefficients are log(2), log(3), and the joint mean ratio is 6.
    cells = ((0, 0, 1), (1, 0, 2), (0, 1, 3), (1, 1, 6))
    x = np.vstack([np.tile([a, b], (10, 1)) for a, b, _ in cells])
    y = np.concatenate([np.repeat(mean, 10) for _, _, mean in cells])
    return y, x


def test_exact_log_rate_and_independent_information():
    y, x = exact_cells()
    result = fit_poisson(y, x, q=None)
    assert_allclose(result.coefficients, [0, np.log(2), np.log(3)], atol=1e-12)
    assert_allclose(result.psi, np.log(6), atol=1e-12)
    assert_allclose(result.rate_ratio, 6, atol=1e-12)
    assert_allclose(result.predict([[0, 0], [1, 0], [0, 1], [1, 1]]),
                    [1, 2, 3, 6], atol=1e-12)
    design = np.column_stack((np.ones(len(x)), x))
    information = design.T @ (result.fitted_values[:, None] * design)
    covariance = np.linalg.inv(information)
    contrast = np.array([0, 1, 1])
    assert_allclose(result.covariance, covariance, atol=1e-12)
    assert_allclose(result.var_psi, contrast @ covariance @ contrast, atol=1e-12)
    assert result.p_value_df is None
    assert result.family == "poisson" and result.link == "log"
    assert result.coefficients.flags.writeable is False
    assert "equal observation periods" in result.summary()


def test_formula_category_encoding_and_prediction():
    y, x = exact_cells()
    group = np.tile(["A", "B"], 20)
    data = {"y": y, "x1": x[:, 0], "x2": x[:, 1], "group": group}
    formula = fit_poisson_formula("y ~ x1 + x2 + C(group)", data,
                                  ("x1", "x2"), q=2)
    manual = fit_poisson(y, x, (group == "B").astype(float), q=2)
    assert_allclose(formula.coefficients, manual.coefficients, atol=1e-12)
    assert_allclose(formula.covariance, manual.covariance, atol=1e-12)
    assert dict(formula.categorical_levels) == {"group": ("A", "B")}
    assert_allclose(formula.predict_data({"x1": [0, 1], "x2": [1, 1],
                                          "group": ["A", "B"]}),
                    manual.predict([[0, 1], [1, 1]], [0, 1]), atol=1e-12)
    assert_array_equal(formula.transform_data({"x1": [0, 1], "x2": [1, 1]}).scores,
                       [[0, 1], [1, 1]])


@pytest.mark.parametrize("change", ("noninteger", "negative", "missing", "all_zero", "rank"))
def test_invalid_count_fit_rejected(change):
    y, x = exact_cells()
    y = y.astype(float)
    if change == "noninteger":
        y[0] = 0.5
    elif change == "negative":
        y[0] = -1
    elif change == "missing":
        y[0] = np.nan
    elif change == "all_zero":
        y[:] = 0
    else:
        x[:, 1] = x[:, 0]
    with pytest.raises(ValueError):
        fit_poisson(y, x, q=None)


def test_prediction_inputs_and_display_digits_rejected():
    y, x = exact_cells()
    result = fit_poisson(y, x, q=None)
    with pytest.raises(ValueError, match="without covariates"):
        result.predict([[0, 1]], [1])
    with pytest.raises(ValueError, match="digits"):
        result.summary(digits=0)
    with pytest.raises(ValueError, match="requires"):
        result.predict_data({"x1": [0], "x2": [1]})


def test_exposure_time_exact_rates_information_and_predictions():
    cells = ((0, 0, 1), (1, 0, 2), (0, 1, 3), (1, 1, 6))
    x = np.vstack([np.tile([a, b], (5, 1)) for a, b, _ in cells])
    time = np.tile(np.arange(1, 6, dtype=float), 4)
    y = np.concatenate([rate * time[:5] for _, _, rate in cells])
    result = fit_poisson(y, x, q=None, exposure_time=time)
    assert_allclose(result.coefficients, [0, np.log(2), np.log(3)], atol=1e-12)
    assert_allclose(result.psi, np.log(6), atol=1e-12)
    assert_allclose(result.fitted_values, y, atol=1e-12)
    design = np.column_stack((np.ones(len(x)), x))
    information = design.T @ (y[:, None] * design)
    assert_allclose(result.covariance, np.linalg.inv(information), atol=1e-12)
    assert_allclose(result.predict([[1, 0], [1, 1]], exposure_time=[2, 3]),
                    [4, 18], atol=1e-12)
    assert result.exposure_time.flags.writeable is False
    assert result.diagnostics["exposure_time_offset"] is True
    assert "Rate ratio: 6" in result.summary()
    assert "time offset" in result.summary()


def test_exposure_time_formula_category_and_missing_rows():
    y, x = exact_cells()
    time = np.ones(len(y), dtype=object)
    time[4] = None
    group = np.tile(["A", "B"], 20)
    data = {"y": y, "x1": x[:, 0], "x2": x[:, 1],
            "group": group, "time": time}
    result = fit_poisson_formula("y ~ x1 + x2 + C(group)", data,
                                 ("x1", "x2"), q=2, missing="drop",
                                 exposure_time="time")
    assert 4 not in result.training_row_indices
    assert len(result.fitted_values) == len(y) - 1
    assert_allclose(result.predict_data({"x1": [1], "x2": [1],
                                         "group": ["A"], "time": [3]}),
                    [3 * np.exp(result.coefficients[:3].sum())])
    with pytest.raises(ValueError, match="exposure_time"):
        result.predict_data({"x1": [1], "x2": [1], "group": ["A"], "time": [0]})
    with pytest.raises(ValueError, match="time"):
        result.predict_data({"x1": [1], "x2": [1], "group": ["A"]})


def test_exposure_time_prior_weights_match_repeated_rows():
    y, x = exact_cells()
    time = 1 + np.arange(len(y)) % 5
    weight = 1 + np.arange(len(y)) % 3
    weighted = fit_poisson(y, x, q=None, weights=weight, exposure_time=time)
    repeated = fit_poisson(np.repeat(y, weight), np.repeat(x, weight, axis=0),
                           q=None, exposure_time=np.repeat(time, weight))
    assert_allclose(weighted.coefficients, repeated.coefficients, atol=1e-12)
    assert_allclose(weighted.covariance, repeated.covariance, atol=1e-12)


@pytest.mark.parametrize("time", ([0] * 40, [-1] * 40, [np.inf] * 40,
                                   [True] * 40, [1] * 39, [[1]] * 40))
def test_invalid_exposure_time_rejected(time):
    y, x = exact_cells()
    with pytest.raises(ValueError, match="exposure_time"):
        fit_poisson(y, x, q=None, exposure_time=time)
