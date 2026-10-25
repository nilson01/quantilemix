"""Complete-case formulas and the pre-omission quantile rule."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from quantilemix import fit_formula


def sample():
    x1 = np.tile(np.arange(4), 12).astype(float)
    x2 = np.tile(np.repeat(np.arange(4), 4), 3).astype(float)
    group = np.repeat(["A", "B", "C"], 16).astype(object)
    y = (2 + 1.5*x1 - 0.4*x2 + 0.7*(group == "B")
         - 0.3*(group == "C") + 0.2*(x1 - 1.5)*(x2 - 1.5))
    z = (np.arange(1, 49) % 7).astype(float)
    return {"y": y, "x1": x1, "x2": x2, "z": z, "group": group}


def test_missing_rows_keep_original_positions_and_pre_omission_cuts():
    data = sample()
    data["y"][0] = np.nan
    data["z"][1] = np.nan
    data["x1"][2] = np.nan
    result = fit_formula("y ~ x1 + x2 + z + C(group)", data,
                         ("x1", "x2"), missing="drop")
    assert result.original_row_count == 48
    assert result.training_row_indices == tuple(range(3, 48))
    assert result.q == 4
    assert result.quantization_mode == "quantile"
    assert_allclose(result.breaks[0], [-1e64, 0.5, 1, 2.5, 1e64])
    assert_allclose(result.psi, 1.1346576138483668, rtol=1e-11)
    # Dropping first and then quantizing changes the first cut from 0.5 to 1.
    reduced = {name: values[3:] for name, values in data.items()}
    different = fit_formula("y ~ x1 + x2 + z + C(group)", reduced,
                            ("x1", "x2"))
    assert different.breaks[0][1] == 1
    assert abs(different.psi - result.psi) > 0.01


def test_default_rejects_missing_and_drop_does_not_change_complete_input():
    data = sample()
    ordinary = fit_formula("y ~ x1 + x2 + z + C(group)", data,
                           ("x1", "x2"))
    dropped = fit_formula("y ~ x1 + x2 + z + C(group)", data,
                          ("x1", "x2"), missing="drop")
    assert_array_equal(ordinary.scores, dropped.scores)
    assert_allclose(ordinary.psi, dropped.psi)
    assert ordinary.training_row_indices == tuple(range(48))
    data["y"][0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        fit_formula("y ~ x1 + x2 + z + C(group)", data, ("x1", "x2"))


def test_missing_category_and_weight_are_jointly_dropped():
    data = sample()
    data["group"][0] = None
    weights = np.ones(48)
    weights[1] = np.nan
    result = fit_formula("y ~ x1 + x2 + z + C(group)", data,
                         ("x1", "x2"), missing="drop", weights=weights, q=None)
    assert result.training_row_indices == tuple(range(2, 48))
    reduced = {name: values[2:] for name, values in data.items()}
    manual = fit_formula("y ~ x1 + x2 + z + C(group)", reduced,
                         ("x1", "x2"), weights=weights[2:], q=None)
    assert_allclose(result.coefficients, manual.coefficients)
    assert_allclose(result.covariance, manual.covariance)
    assert result.q is None


def test_missing_options_and_malformed_values_fail_clearly():
    data = sample()
    with pytest.raises(ValueError, match="missing must"):
        fit_formula("y ~ x1 + x2 + z + C(group)", data,
                    ("x1", "x2"), missing="omit")
    data["x1"][:] = np.nan
    with pytest.raises(ValueError):
        fit_formula("y ~ x1 + x2 + z + C(group)", data,
                    ("x1", "x2"), missing="drop")
    data = sample()
    data["x1"][0] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        fit_formula("y ~ x1 + x2 + z + C(group)", data,
                    ("x1", "x2"), missing="drop")
    data = sample()
    data["group"][:] = None
    with pytest.raises(ValueError, match="no complete"):
        fit_formula("y ~ x1 + x2 + z + C(group)", data,
                    ("x1", "x2"), missing="drop")


def test_explicit_cuts_keep_explicit_mode():
    data = sample()
    data["y"][0] = np.nan
    cuts = ([-1e64, 0.5, 1, 2.5, 1e64], [-1e64, 0.5, 1, 2.5, 1e64])
    result = fit_formula("y ~ x1 + x2 + z + C(group)", data,
                         ("x1", "x2"), missing="drop", breaks=cuts)
    assert result.quantization_mode == "explicit"
    assert result.q is None
