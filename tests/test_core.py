"""Independent algebraic targets; no fitted package output defines the oracle."""

from fractions import Fraction as F
from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import norm, t

from quantilemix import fit, quantize


def rational_example():
    """A correlated design with an exactly orthogonal, nonzero residual.

    Q1=a, Q2=a+b for a,b in 0..3, crossed with two +/-1 covariates.
    e=(a-3/2)(b-3/2) is orthogonal to all five design columns.
    The 64 residual squares sum to 100; full-model residual df is 59.
    """
    rows = list(product(range(4), range(4), (-1, 1), (-1, 1)))
    design = [[F(1), F(a), F(a + b), F(z1), F(z2)]
              for a, b, z1, z2 in rows]
    residual = [(F(a) - F(3, 2)) * (F(b) - F(3, 2))
                for a, b, _, _ in rows]
    theta = [F(3), F(2), F(-1), F(1, 2), F(-1, 4)]
    signal = [sum(v * coefficient for v, coefficient in zip(row, theta))
              for row in design]
    y = np.array([float(m + e) for m, e in zip(signal, residual)])
    matrix = np.asarray(design, dtype=float)
    return y, matrix[:, 1:3], matrix[:, 3:], design, residual, theta, signal


def test_exact_rational_coefficients_covariance_and_reference_inference():
    y, q, z, design, residual, theta, signal = rational_example()
    # Verify the fixture's construction using exact arithmetic, not regression.
    assert all(sum(row[j] * e for row, e in zip(design, residual)) == 0
               for j in range(5))
    assert sum(e * e for e in residual) == 100
    gram = [[sum(row[i] * row[j] for row in design) for j in range(5)]
            for i in range(5)]
    inverse = [
        [F(23, 320), F(0), F(-3, 160), F(0), F(0)],
        [F(0), F(1, 40), F(-1, 80), F(0), F(0)],
        [F(-3, 160), F(-1, 80), F(1, 80), F(0), F(0)],
        [F(0), F(0), F(0), F(1, 64), F(0)],
        [F(0), F(0), F(0), F(0), F(1, 64)],
    ]
    for i, j in product(range(5), repeat=2):
        assert sum(gram[i][k] * inverse[k][j] for k in range(5)) == int(i == j)
    expected_covariance = np.array(
        [[float(F(100, 59) * entry) for entry in row] for row in inverse]
    )

    result = fit(y, q, z, q=None)
    assert_allclose(result.coefficients, np.asarray(theta, dtype=float), atol=1e-11)
    assert_allclose(result.fitted_values, np.asarray(signal, dtype=float), atol=1e-11)
    assert_allclose(result.residuals, np.asarray(residual, dtype=float), atol=1e-11)
    assert_allclose(result.covariance, expected_covariance, atol=1e-12, rtol=1e-11)
    assert result.residual_df == 59
    assert result.residual_variance == pytest.approx(100 / 59, abs=1e-12)
    assert result.psi == pytest.approx(1, abs=1e-11)
    assert result.var_psi == pytest.approx(5 / 236, abs=1e-12)
    assert result.se == pytest.approx(np.sqrt(5 / 236), abs=1e-12)

    # A diagonal-only calculation gives three times the correct variance.
    wrong_variance = expected_covariance[1, 1] + expected_covariance[2, 2]
    assert wrong_variance == pytest.approx(15 / 236)
    assert not np.isclose(result.var_psi, wrong_variance, rtol=0.1)
    expected_statistic = 1 / np.sqrt(5 / 236)
    assert result.p_value_df == 61  # R compatibility, not full-model df=59.
    assert result.test_statistic == pytest.approx(expected_statistic)
    assert result.p_value == pytest.approx(2 * t.sf(expected_statistic, 61), abs=1e-14)
    assert_allclose(result.ci, 1 + np.array([-1, 1]) * norm.ppf(.975) * np.sqrt(5 / 236),
                    atol=1e-11)
    assert result.diagnostics["inference_degenerate"] is False


def test_estimable_joint_variance_survives_nearly_collinear_covariance_cancellation():
    y, original_q, z, *_ = rational_example()
    scale = 100_000_000
    a = original_q[:, 0]
    b = original_q[:, 1] - a
    q = np.column_stack([scale * a, scale * a + b])
    # Reparameterize the *same* exact-rational experiment. The coefficients
    # become (1 + 1/M, -1), so psi=1/M. Its variance is 5/(236*M**2),
    # while the individual variances remain about 5/236. Summing rounded
    # covariance entries can erase the positive joint variance completely.
    with pytest.warns(RuntimeWarning):
        result = fit(y, q, z, q=None)
    assert result.psi == pytest.approx(1 / scale, abs=1e-12)
    assert result.var_psi == pytest.approx(5 / (236 * scale**2), rel=5e-6, abs=0)
    assert result.se > 0
    assert result.diagnostics["inference_degenerate"] is False
    assert np.isfinite(result.p_value)
    assert result.p_value > 0


def test_mixed_sign_cancellation_does_not_normalize_by_net_effect():
    _, q, z, _, residual, _, _ = rational_example()
    y = 3 + 2 * q[:, 0] - 2 * q[:, 1] + z @ np.array([.5, -.25])
    y += np.asarray(residual, dtype=float)
    result = fit(y, q, z, q=None)
    assert result.psi == pytest.approx(0, abs=1e-11)
    assert result.pos_psi == pytest.approx(2)
    assert result.neg_psi == pytest.approx(-2)
    assert dict(result.pos_weights) == pytest.approx({"x1": 1.0})
    assert dict(result.neg_weights) == pytest.approx({"x2": 1.0})
    assert result.se > 0
    assert result.p_value == pytest.approx(1, abs=1e-11)


def test_multiple_weights_are_normalized_within_each_sign_group():
    q = np.asarray(list(product((0, 1), repeat=4)), dtype=float)
    residual = (q[:, 0] - .5) * (q[:, 1] - .5)
    result = fit(2 + q @ np.array([2, 1, -3, -1]) + residual, q, q=None)
    assert result.psi == pytest.approx(-1)
    assert result.pos_psi == pytest.approx(3)
    assert result.neg_psi == pytest.approx(-4)
    assert dict(result.pos_weights) == pytest.approx({"x1": 2 / 3, "x2": 1 / 3})
    assert dict(result.neg_weights) == pytest.approx({"x3": 3 / 4, "x4": 1 / 4})


@pytest.mark.parametrize("direction", [-1, 1])
def test_absent_sign_group_is_empty(direction):
    _, q, z, _, residual, _, _ = rational_example()
    y = direction * (2 * q[:, 0] + q[:, 1]) + np.asarray(residual, dtype=float)
    result = fit(y, q, z, q=None)
    present = result.pos_weights if direction > 0 else result.neg_weights
    absent = result.neg_weights if direction > 0 else result.pos_weights
    assert dict(present) == pytest.approx({"x1": 2 / 3, "x2": 1 / 3})
    assert dict(absent) == {}


def test_alpha_changes_normal_interval_but_not_estimate_or_p_value():
    y, q, z, *_ = rational_example()
    result = fit(y, q, z, q=None, alpha=.1)
    assert_allclose(result.ci, 1 + np.array([-1, 1]) * norm.ppf(.95) * np.sqrt(5 / 236),
                    atol=1e-11)
    assert result.p_value == pytest.approx(2 * t.sf(1 / np.sqrt(5 / 236), 61), abs=1e-14)


def test_row_exposure_and_covariate_permutations_preserve_the_estimand():
    y, q, z, *_ = rational_example()
    original = fit(y, q, z, q=None)
    order = np.roll(np.arange(len(y))[::-1], 7)
    reordered = fit(y[order], q[order], z[order], q=None)
    assert_allclose(reordered.coefficients, original.coefficients, atol=1e-11)
    assert_allclose(reordered.covariance, original.covariance, atol=1e-11)
    assert_allclose(reordered.fitted_values, original.fitted_values[order], atol=1e-11)

    swapped = fit(y, q[:, ::-1], z[:, ::-1], q=None)
    coefficient_order = [0, 2, 1, 4, 3]
    assert_allclose(swapped.coefficients, original.coefficients[coefficient_order], atol=1e-11)
    assert_allclose(swapped.covariance,
                    original.covariance[np.ix_(coefficient_order, coefficient_order)], atol=1e-11)
    assert_allclose(swapped.predict(q[:, ::-1], z[:, ::-1]), original.predict(q, z), atol=1e-11)
    assert swapped.psi == pytest.approx(original.psi)
    assert swapped.var_psi == pytest.approx(original.var_psi)


@pytest.mark.parametrize("scale", [3.0, -2.0])
def test_outcome_rescaling_has_the_required_inference_and_weight_behavior(scale):
    y, q, z, *_ = rational_example()
    original = fit(y, q, z, q=None)
    changed = fit(scale * y + 7, q, z, q=None)
    expected = scale * original.coefficients
    expected[0] += 7
    assert_allclose(changed.coefficients, expected, atol=1e-11)
    assert_allclose(changed.covariance, scale**2 * original.covariance, atol=1e-11)
    assert changed.psi == pytest.approx(scale * original.psi)
    assert changed.se == pytest.approx(abs(scale) * original.se)
    assert changed.test_statistic == pytest.approx(np.sign(scale) * original.test_statistic)
    assert changed.p_value == pytest.approx(original.p_value)
    if scale < 0:
        assert changed.pos_psi == pytest.approx(scale * original.neg_psi)
        assert changed.neg_psi == pytest.approx(scale * original.pos_psi)
        assert dict(changed.pos_weights) == pytest.approx(dict(original.neg_weights))
        assert dict(changed.neg_weights) == pytest.approx(dict(original.pos_weights))


@pytest.mark.parametrize("offset", [-1e13, 1e13])
@pytest.mark.parametrize("slope", [0., 2.])
def test_large_outcome_offset_preserves_resolvable_residual_variation(offset, slope):
    x = np.repeat(np.arange(4), 2)
    residual = np.tile([-.125, .125], 4)
    # The residual is orthogonal to the intercept and x. Its sum of squares
    # is 1/8, residual df is 6, and centered x has sum of squares 10.
    y = offset + 1 + slope * x + residual
    result = fit(y, x, q=None)
    assert_allclose(result.coefficients, [offset + 1, slope], rtol=0, atol=1e-11)
    assert_allclose(result.residuals, residual, rtol=0, atol=1e-11)
    assert result.residual_variance == pytest.approx(1 / 48, rel=1e-12)
    assert result.var_psi == pytest.approx(1 / 480, rel=1e-12)
    assert result.psi == pytest.approx(slope, abs=1e-11)
    assert result.se == pytest.approx(np.sqrt(1 / 480), rel=1e-12)
    assert result.diagnostics["inference_degenerate"] is False
    assert np.isfinite(result.p_value)
    assert_allclose(result.predict(x), offset + 1 + slope * x, rtol=0, atol=.004)


def test_small_covariate_units_do_not_erase_residual_variation():
    x = np.repeat([0., 1.], 4)
    z = np.tile(np.repeat([0., 1.], 2), 2)
    residual = np.tile([-.125, .125], 4)
    y = 1 + 2 * x + 3 * z + residual
    # SSE=1/8 and residual df=5. The centered binary x has sum of
    # squares 2 and is orthogonal to centered z, so Var(beta_x)=1/80.
    with pytest.warns(RuntimeWarning, match="condition number"):
        result = fit(y, x, 1e-13 * z, q=None)
    assert result.psi == pytest.approx(2, abs=1e-11)
    assert result.residual_variance == pytest.approx(1 / 40, rel=1e-12)
    assert result.var_psi == pytest.approx(1 / 80, rel=1e-12)
    assert_allclose(result.residuals, residual, rtol=0, atol=1e-11)
    assert result.diagnostics["inference_degenerate"] is False
    assert np.isfinite(result.p_value)


def test_fitted_joint_score_contrast_equals_psi_with_adjustment_held_fixed():
    y, q, z, *_ = rational_example()
    result = fit(y, q, z, q=None)
    admissible = (q[:, 0] < 3) & (q[:, 1] < 6)
    change = result.predict(q[admissible] + 1, z[admissible]) - result.predict(q[admissible], z[admissible])
    assert_allclose(change, np.ones(admissible.sum()), atol=1e-11)


@pytest.mark.parametrize("raw,expected_scores,interior", [
    ([0, 0, 0, 1], [0, 0, 0, 1], .25),
    ([0, 1, 1, 1], [0, 1, 1, 1], .75),
])
def test_full_quantile_vector_deduplication_at_tied_endpoints(raw, expected_scores, interior):
    result = quantize(raw, q=4)
    assert_array_equal(result.scores, np.array(expected_scores)[:, None])
    assert_allclose(result.breaks[0], [-1e64, interior, 1e64], rtol=0, atol=0)
    assert tuple(result.category_counts) == (2,)


def test_type_seven_interpolation_and_explicit_right_closed_breaks():
    automatic = quantize([0, 2, 8, 10], q=4)
    assert_allclose(automatic.breaks[0], [-1e64, 1.5, 5, 8.5, 1e64], rtol=0, atol=0)
    assert_array_equal(automatic.scores[:, 0], [0, 1, 2, 3])
    values = [-10, .25, np.nextafter(.25, np.inf), 2, 10]
    explicit = quantize(values, q=2, breaks=[[-10, .25, 2, 10]])
    assert_array_equal(explicit.scores[:, 0], [0, 0, 1, 1, 2])
    assert tuple(explicit.category_counts) == (3,)  # Explicit breaks override q.


def test_prediction_reuses_training_cuts_instead_of_quantizing_each_new_batch():
    raw = np.arange(8, dtype=float)
    scores = np.repeat(np.arange(4), 2)
    y = 1 + 2 * scores + np.tile([-.5, .5], 4)
    result = fit(y, raw)
    saved_breaks = result.breaks[0].copy()
    tail = np.array([5.5, 6, 6.5, 7])
    assert_array_equal(result.transform(tail).scores[:, 0], [3, 3, 3, 3])
    assert_allclose(result.predict(tail), [7, 7, 7, 7], atol=1e-11)
    boundary = [1.75, np.nextafter(1.75, np.inf), 3.5, 5.25, 7]
    assert_array_equal(result.transform(boundary).scores[:, 0], [0, 1, 1, 2, 3])
    assert_array_equal(result.breaks[0], saved_breaks)
    assert result.predict([6])[0] == pytest.approx(result.predict(tail)[1])


def test_extrapolation_uses_extreme_bins_and_is_reported():
    raw = np.arange(8, dtype=float)
    y = 1 + 2 * np.repeat(np.arange(4), 2) + np.tile([-.5, .5], 4)
    result = fit(y, raw)
    transformed = result.transform([-100, 100])
    assert_array_equal(transformed.scores[:, 0], [0, 3])
    assert tuple(transformed.diagnostics["extrapolated_rows"]) == (0, 1)
    assert tuple(transformed.diagnostics["extrapolated_columns"]) == ("x1",)
    assert tuple(transformed.diagnostics["extrapolated_counts"]) == (2,)
    with pytest.warns(RuntimeWarning):
        prediction = result.predict([-100, 100])
    assert_allclose(prediction, [1, 7], atol=1e-11)


def test_training_array_mutation_cannot_change_an_existing_fit():
    raw = np.arange(8, dtype=float)
    original = raw.copy()
    y = 1 + 2 * np.repeat(np.arange(4), 2) + np.tile([-.5, .5], 4)
    result = fit(y, raw)
    expected_prediction = result.predict(original).copy()
    raw[:] = -100
    y[:] = 100
    assert_allclose(result.predict(original), expected_prediction, atol=0, rtol=0)
    with pytest.raises(ValueError):
        result.breaks[0][1] = 100
