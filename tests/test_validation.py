"""Unsupported inputs fail explicitly; degenerate fits never claim significance."""

from collections import UserList
from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from quantilemix import fit, quantize


class LabeledArray:
    """Minimal labeled numeric input; test column identity without pandas."""

    def __init__(self, values, columns):
        self.values = np.asarray(values)
        self.columns = tuple(columns)

    def __array__(self, dtype=None, copy=None):
        array = np.asarray(self.values, dtype=dtype)
        return array.copy() if copy else array


@pytest.fixture
def complete_data():
    rows = np.asarray(list(product((0, 1), repeat=3)), dtype=float)
    q, z = rows[:, :2], rows[:, 2:]
    residual = (q[:, 0] - .5) * (q[:, 1] - .5)
    y = 3 + q[:, 0] - q[:, 1] + .5 * z[:, 0] + residual
    return y, q, z


@pytest.mark.parametrize("q", [True, False, 0, 1, -1, 2.5, "4", np.nan, np.inf])
def test_invalid_quantile_count_is_rejected(q):
    with pytest.raises(ValueError):
        quantize([0, 1, 2, 3], q=q)


def test_huge_quantile_count_fails_before_grid_allocation(monkeypatch):
    def forbidden_grid(*args, **kwargs):
        raise AssertionError("an oversized quantile grid must not be allocated")

    monkeypatch.setattr(np, "arange", forbidden_grid)
    with pytest.raises(ValueError):
        quantize([0, 1, 2, 3], q=10**12)


@pytest.mark.parametrize("alpha", [0, 1, -.1, 1.1, np.nan, np.inf])
def test_invalid_alpha_is_rejected(complete_data, alpha):
    y, q, z = complete_data
    with pytest.raises(ValueError):
        fit(y, q, z, q=None, alpha=alpha)


@pytest.mark.parametrize("field", ["outcome", "exposure", "covariate"])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_values_are_not_silently_dropped(complete_data, field, bad_value):
    y, q, z = complete_data
    target = {"outcome": y, "exposure": q, "covariate": z}[field]
    target.flat[0] = bad_value
    with pytest.raises(ValueError):
        fit(y, q, z, q=None)


@pytest.mark.parametrize("field", ["outcome", "exposure", "covariate"])
def test_masked_training_values_are_not_treated_as_observed(complete_data, field):
    inputs = dict(zip(("outcome", "exposure", "covariate"), complete_data))
    masked = np.ma.array(inputs[field], mask=False)
    masked.mask.flat[0] = True  # The backing value is finite but unobserved.
    inputs[field] = masked
    with pytest.raises(ValueError, match="masked"):
        fit(inputs["outcome"], inputs["exposure"], inputs["covariate"], q=None)


@pytest.mark.parametrize("field", ["exposure", "breaks"])
def test_quantization_rejects_masked_values_and_boundaries(field):
    exposure = np.arange(4, dtype=float)
    breaks = np.array([-1, 1.5, 4.])
    if field == "exposure":
        exposure = np.ma.array(exposure, mask=[False, True, False, False])
    else:
        breaks = np.ma.array(breaks, mask=[False, True, False])
    with pytest.raises(ValueError, match="masked"):
        quantize(exposure, breaks=breaks)


@pytest.mark.parametrize("field", ["exposure", "covariate"])
def test_prediction_rejects_masked_values(complete_data, field):
    y, q, z = complete_data
    result = fit(y, q, z, q=None)
    if field == "exposure":
        q = np.ma.array(q, mask=False)
        q.mask[0, 0] = True
        with pytest.raises(ValueError, match="masked"):
            result.transform(q)
    else:
        z = np.ma.array(z, mask=False)
        z.mask[0, 0] = True
    with pytest.raises(ValueError, match="masked"):
        result.predict(q, z)


def test_masked_array_containers_without_missing_values_remain_supported(complete_data):
    y, q, z = complete_data
    expected = fit(y, q, z, q=None)
    result = fit(*(np.ma.array(a, mask=False) for a in (y, q, z)), q=None)
    assert_allclose(result.coefficients, expected.coefficients, rtol=0, atol=0)
    assert_allclose(result.covariance, expected.covariance, rtol=0, atol=0)


@pytest.mark.parametrize("container", [list, tuple, UserList])
def test_nested_array_inputs_do_not_hide_masked_entries(container):
    rows = container([np.ma.array([0., 1.], mask=[False, True]), [2., 3.]])
    with pytest.raises(ValueError, match="masked"):
        quantize(rows, q=None)


def test_array_protocol_cannot_hide_masked_entries():
    class ArrayWrapper:
        def __array__(self, dtype=None, copy=None):
            values = np.ma.array([0., 1., 2.], mask=[False, True, False], dtype=dtype)
            return values.copy() if copy else values

    with pytest.raises(ValueError, match="masked"):
        quantize(ArrayWrapper(), q=None)


@pytest.mark.parametrize("field", ["outcome", "exposure", "covariate"])
def test_complex_values_are_not_silently_cast_to_real(complete_data, field):
    y, q, z = complete_data
    if field == "outcome":
        y = y.astype(complex) + 1j
    elif field == "exposure":
        q = q.astype(complex) + 1j
    else:
        z = z.astype(complex) + 1j
    with pytest.raises(ValueError):
        fit(y, q, z, q=None)


@pytest.mark.parametrize("case", ["outcome_matrix", "outcome_rows", "exposure_rows",
                                 "covariate_rows", "no_exposures", "scalar_exposure",
                                 "three_dimensions", "empty"])
def test_invalid_shapes_are_rejected(complete_data, case):
    y, q, z = complete_data
    if case == "outcome_matrix":
        y = y[:, None]
    elif case == "outcome_rows":
        y = y[:-1]
    elif case == "exposure_rows":
        q = q[:-1]
    elif case == "covariate_rows":
        z = z[:-1]
    elif case == "no_exposures":
        q = q[:, :0]
    elif case == "scalar_exposure":
        q = np.array(1)
    elif case == "three_dimensions":
        q = q[:, :, None]
    else:
        y, q, z = y[:0], q[:0], z[:0]
    with pytest.raises(ValueError):
        fit(y, q, z, q=None)


@pytest.mark.parametrize("bad_breaks", [[], [[0]], [[0, 1, 1, 4]], [[0, 2, 1, 4]],
                                      [[0, np.nan, 4]], [[-np.inf, 1, np.inf]],
                                      [[1, 2, 3]], [[0, 4], [0, 4]]])
def test_invalid_or_out_of_range_explicit_breaks_are_rejected(bad_breaks):
    with pytest.raises(ValueError):
        quantize([0, 1, 2, 3], breaks=bad_breaks)


@pytest.mark.parametrize("values", [[0, -1, 1, 2], [0, .5, 1, 2]])
def test_prequantized_mode_requires_nonnegative_integer_scores(values):
    with pytest.raises(ValueError):
        quantize(values, q=None)


@pytest.mark.parametrize("values", [[0, 1, 2, 1e64], [-1e64, 0, 1, 2], [1, 1, 1, 1]])
def test_automatic_quantization_rejects_sentinel_magnitudes_and_constant_exposures(values):
    with pytest.raises(ValueError):
        quantize(values)


def test_exact_rank_failure_and_nonpositive_residual_df_are_rejected(complete_data):
    y, q, z = complete_data
    with pytest.raises(ValueError):
        fit(y, np.column_stack([q[:, 0], q[:, 0]]), q=None)
    with pytest.raises(ValueError):
        fit(y, q, np.ones_like(z), q=None)  # Adjustment duplicates the intercept.
    with pytest.raises(ValueError):
        fit(y, np.ones((len(y), 1)), q=None)
    with pytest.raises(ValueError):
        fit([0, 1, 1], [[0, 0], [0, 1], [1, 0]], q=None)


def test_quantization_can_create_rank_failure_even_when_raw_columns_differ():
    raw = np.arange(12, dtype=float)
    with pytest.raises(ValueError):
        fit(raw + 1, np.column_stack([raw, raw**2]), q=4)


def test_prediction_rejects_missing_or_misaligned_adjustment(complete_data):
    y, q, z = complete_data
    result = fit(y, q, z, q=None)
    for new_q, new_z in [(q, None), (q, z[:-1]), (q[:, :1], z),
                         (q, np.column_stack([z, z]))]:
        with pytest.raises(ValueError):
            result.predict(new_q, new_z)
    with pytest.raises(ValueError):
        result.transform([[0, np.nan]])
    with pytest.raises(ValueError):
        result.transform([[-1, 0]])


def test_labeled_prediction_requires_original_exposure_and_covariate_identity():
    rows = np.asarray(list(product((0, 1), repeat=4)), dtype=float)
    q, z = rows[:, :2], rows[:, 2:]
    y = 3 + q[:, 0] - q[:, 1] + .5 * z[:, 0] - .25 * z[:, 1]
    y += (q[:, 0] - .5) * (q[:, 1] - .5)
    named_q = LabeledArray(q, ("exposure_a", "exposure_b"))
    named_z = LabeledArray(z, ("adjustment_a", "adjustment_b"))
    result = fit(y, named_q, named_z, q=None)
    assert tuple(result.exposure_names) == named_q.columns
    assert tuple(result.covariate_names) == named_z.columns
    assert_allclose(result.predict(named_q, named_z), result.fitted_values, atol=1e-11)
    assert dict(result.pos_weights) == pytest.approx({"exposure_a": 1.0})
    assert dict(result.neg_weights) == pytest.approx({"exposure_b": 1.0})

    invalid_inputs = [
        (LabeledArray(q[:, ::-1], named_q.columns[::-1]), named_z),
        (named_q, LabeledArray(z[:, ::-1], named_z.columns[::-1])),
        (LabeledArray(q, ("exposure_a", "different")), named_z),
        (named_q, LabeledArray(z, ("adjustment_a", "different"))),
        (q, named_z),  # Dropping labels cannot hide a changed column order.
        (named_q, z),
    ]
    for new_q, new_z in invalid_inputs:
        with pytest.raises(ValueError):
            result.predict(new_q, new_z)


@pytest.mark.parametrize("case", ["duplicate_exposures", "duplicate_covariates",
                                 "cross_group_duplicate", "intercept_collision"])
def test_ambiguous_column_identifiers_are_rejected(complete_data, case):
    y, q, z = complete_data
    exposure_names = ("a", "b")
    covariate_names = ("c",)
    if case == "duplicate_exposures":
        exposure_names = ("a", "a")
    elif case == "duplicate_covariates":
        z = np.column_stack([z, z])
        covariate_names = ("c", "c")
    elif case == "cross_group_duplicate":
        covariate_names = ("a",)
    else:
        exposure_names = ("(Intercept)", "b")
    with pytest.raises(ValueError):
        fit(y, LabeledArray(q, exposure_names), LabeledArray(z, covariate_names), q=None)


@pytest.mark.parametrize("zero_outcome", [False, True])
def test_zero_residual_variance_marks_inference_degenerate(complete_data, zero_outcome):
    _, q, z = complete_data
    y = np.zeros(len(q)) if zero_outcome else 3 + 2 * q[:, 0] - 2 * q[:, 1] + z[:, 0]
    result = fit(y, q, z, q=None)
    assert_allclose(result.predict(q, z), y, atol=1e-11)
    assert result.diagnostics["inference_degenerate"] is True
    assert result.residual_variance == 0
    assert result.se == 0
    assert result.var_psi == 0
    assert_array_equal(result.covariance, np.zeros((4, 4)))
    assert_allclose(result.ci, [result.psi, result.psi], atol=0, rtol=0)
    assert np.isnan(result.p_value)
    assert np.isnan(result.test_statistic)
    if zero_outcome:
        assert dict(result.pos_weights) == {}
        assert dict(result.neg_weights) == {}
        assert tuple(result.zero_exposures) == ("x1", "x2")
