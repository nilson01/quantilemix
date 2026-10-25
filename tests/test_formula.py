"""Independent checks for the restricted named-data formula interface."""

from itertools import product

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from quantilemix import fit, fit_bootstrap, fit_bootstrap_formula, fit_formula, quantize


def sample():
    rows = list(product(range(4), range(4), ("A", "B", "C")))
    a = np.asarray([row[0] for row in rows])
    b = np.asarray([row[1] for row in rows])
    group = [row[2] for row in rows]
    y = (2 + 1.5 * a - 0.4 * b
         + 0.7 * (np.asarray(group) == "B")
         - 0.3 * (np.asarray(group) == "C")
         + 0.2 * (a - 1.5) * (b - 1.5))
    return {"y": y, "x1": a, "x2": b, "group": group}


def test_formula_known_coefficients_and_manual_design():
    data = sample()
    result = fit_formula("y ~ x1 + x2 + C(group)", data, ("x1", "x2"), q=None)
    assert result.coefficient_names == (
        "(Intercept)", "x1", "x2", "group[T.B]", "group[T.C]"
    )
    assert result.formula == "y ~ x1 + x2 + C(group)"
    assert dict(result.categorical_levels) == {"group": ("A", "B", "C")}
    assert_allclose(result.coefficients, [2, 1.5, -0.4, 0.7, -0.3], atol=1e-12)
    assert_allclose(result.psi, 1.1, atol=1e-12)
    manual_z = np.column_stack((
        np.asarray(data["group"]) == "B",
        np.asarray(data["group"]) == "C",
    )).astype(float)
    manual = fit(data["y"], np.column_stack((data["x1"], data["x2"])),
                 manual_z, q=None)
    assert_allclose(result.coefficients, manual.coefficients, atol=1e-12)
    assert_allclose(result.covariance, manual.covariance, atol=1e-12)
    assert_allclose(result.var_psi, manual.var_psi, atol=1e-12)
    assert_allclose(
        result.predict_data({"x1": [1, 2], "x2": [2, 3], "group": ["B", "C"]}),
        [3.4, 3.5], atol=1e-12
    )
    assert_array_equal(result.transform_data({"x1": [1], "x2": [2]})
                       .scores, [[1, 2]])


def test_formula_quantile_cuts_match_numeric_fit():
    data = sample()
    a = fit_formula("y ~ x1 + x2 + C(group)", data, ("x1", "x2"))
    b = fit_formula("y ~ x1 + x2 + C(group)", data, ("x1", "x2"), q=None)
    assert_array_equal(a.scores, b.scores)
    assert_allclose(a.psi, b.psi, atol=1e-12)


@pytest.mark.parametrize("formula", (
    "y ~ x1 * x2", "y ~ x1:x2", "y ~ log(x1) + x2",
    "y ~ x1 + x2 - 1", "y ~ x1 + x2 +", "y ~ x1 + x1 + x2",
    "y ~ C(x1) + x2", "y ~ x1 + x2 + C(group) + C(group)",
))
def test_unsupported_formula_rejected(formula):
    with pytest.raises(ValueError):
        fit_formula(formula, sample(), ("x1", "x2"), q=None)


@pytest.mark.parametrize("exposures", (None, "x1", ("x1", [])))
def test_invalid_exposure_argument_rejected(exposures):
    with pytest.raises(ValueError, match="exposures|exposure names"):
        fit_formula("y ~ x1 + x2 + C(group)", sample(), exposures, q=None)


def test_missing_and_unknown_category_rejected():
    data = sample()
    data["group"][0] = None
    with pytest.raises(ValueError, match="complete"):
        fit_formula("y ~ x1 + x2 + C(group)", data, ("x1", "x2"), q=None)
    data = sample()
    result = fit_formula("y ~ x1 + x2 + C(group)", data, ("x1", "x2"), q=None)
    with pytest.raises(ValueError, match="unseen"):
        result.predict_data({"x1": [1], "x2": [2], "group": ["D"]})


def test_missing_numeric_and_unavailable_prediction_column_rejected():
    data = sample()
    data["y"][0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        fit_formula("y ~ x1 + x2 + C(group)", data, ("x1", "x2"), q=None)
    result = fit_formula("y ~ x1 + x2 + C(group)", sample(), ("x1", "x2"), q=None)
    with pytest.raises(ValueError, match="missing column"):
        result.predict_data({"x1": [1], "x2": [2]})


def test_plain_fit_has_no_named_prediction_schema():
    data = sample()
    result = fit(data["y"], np.column_stack((data["x1"], data["x2"])), q=None)
    with pytest.raises(ValueError, match="fit_formula"):
        result.predict_data({"x1": [1], "x2": [2]})


def test_dataframe_style_table_and_duplicate_columns():
    class Table:
        def __init__(self, values, columns=None):
            self.values = values
            self.columns = list(values) if columns is None else columns

        def __getitem__(self, name):
            return self.values[name]

    data = sample()
    result = fit_formula("y ~ x1 + x2 + C(group)", Table(data),
                         ("x1", "x2"), q=None)
    assert_allclose(result.psi, 1.1, atol=1e-12)
    with pytest.raises(ValueError, match="duplicate"):
        fit_formula("y ~ x1 + x2 + C(group)",
                    Table(data, ["y", "x1", "x1", "group"]),
                    ("x1", "x2"), q=None)


def test_gaussian_bootstrap_formula_matches_named_manual_design():
    data = sample()
    result = fit_bootstrap_formula(
        "y ~ x1 + x2 + C(group)", data, ("x1", "x2"),
        q=None, B=16, seed=19,
    )
    category = np.asarray(data["group"])
    z = np.column_stack((category == "B", category == "C")).astype(float)
    direct = fit_bootstrap(
        data["y"], np.column_stack((data["x1"], data["x2"])),
        z, q=None, B=16, seed=19,
    )
    assert_array_equal(result.bootstrap_effects, direct.bootstrap_effects)
    assert_allclose(result.intervention_means, direct.intervention_means, atol=1e-12)
    assert_allclose(result.ci, direct.ci, atol=1e-12)
    assert result.point_fit.formula == "y ~ x1 + x2 + C(group)"
    assert result.point_fit.coefficient_names == (
        "(Intercept)", "x1", "x2", "group[T.B]", "group[T.C]"
    )
    assert result.point_fit.training_row_indices == tuple(range(len(data["y"])))
    assert_allclose(
        result.point_fit.predict_data({"x1": [1], "x2": [2], "group": ["B"]}),
        direct.point_fit.predict(np.array([[1., 2.]]), np.array([[1., 0.]])),
        atol=1e-12,
    )


def test_gaussian_bootstrap_formula_drops_rows_after_training_cuts():
    n = 60
    x1 = np.arange(n, dtype=float)
    x2 = np.array([i % 4 for i in range(n)], dtype=float)
    group = ["A" if i % 3 else "B" for i in range(n)]
    y = 2 + 0.15*x1 - 0.3*x2 + 0.4*(np.asarray(group) == "B")
    y += 0.05*np.sin(x1)
    y[-1] = np.nan
    group[0] = None
    data = {"y": y, "x1": x1, "x2": x2, "group": group}
    result = fit_bootstrap_formula(
        "y ~ x1 + x2 + C(group)", data, ("x1", "x2"),
        q=4, missing="drop", B=12, seed=7,
    )
    kept = np.arange(1, n - 1)
    z = (np.asarray(group, dtype=object)[kept] == "B").astype(float)[:, None]
    breaks = (quantize(x1, q=4).breaks[0], quantize(x2, q=4).breaks[0])
    direct = fit_bootstrap(
        y[kept], np.column_stack((x1[kept], x2[kept])), z,
        q=4, breaks=breaks, B=12, seed=7,
    )
    assert result.point_fit.training_row_indices == tuple(kept)
    assert result.point_fit.original_row_count == n
    assert result.point_fit.quantization_mode == "quantile"
    assert_array_equal(result.point_fit.scores, direct.point_fit.scores)
    assert_array_equal(result.bootstrap_effects, direct.bootstrap_effects)
    assert_allclose(result.ci, direct.ci, atol=1e-12)


def test_gaussian_bootstrap_formula_keeps_weights_with_complete_rows():
    data = sample()
    n = len(data["y"])
    weights = 0.7 + (np.arange(n) % 5) / 4
    data["group"][0] = None
    data["y"][-1] = np.nan
    weights[2] = np.nan
    kept = np.asarray([i for i in range(n) if i not in (0, 2, n - 1)])
    result = fit_bootstrap_formula(
        "y ~ x1 + x2 + C(group)", data, ("x1", "x2"),
        q=None, missing="drop", weights=weights, B=12, seed=11,
    )
    category = np.asarray(data["group"], dtype=object)[kept]
    z = np.column_stack((category == "B", category == "C")).astype(float)
    direct = fit_bootstrap(
        data["y"][kept], np.column_stack((data["x1"][kept], data["x2"][kept])),
        z, q=None, weights=weights[kept], B=12, seed=11,
    )
    assert result.point_fit.training_row_indices == tuple(kept)
    assert_allclose(result.point_fit.case_weights, weights[kept])
    assert_allclose(result.bootstrap_effects, direct.bootstrap_effects, atol=1e-12)
    assert_allclose(result.intervention_means, direct.intervention_means, atol=1e-12)


def test_gaussian_bootstrap_formula_rejects_unsupported_inputs():
    data = sample()
    with pytest.raises(ValueError, match="only simple main effects"):
        fit_bootstrap_formula("y ~ x1 * x2", data, ("x1", "x2"), B=4)
    with pytest.raises(ValueError, match="missing must"):
        fit_bootstrap_formula("y ~ x1 + x2", data, ("x1", "x2"),
                              missing="impute", B=4)
    with pytest.raises(ValueError, match="B must"):
        fit_bootstrap_formula("y ~ x1 + x2", data, ("x1", "x2"), B=1)
