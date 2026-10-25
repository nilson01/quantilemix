"""Independent fixed-cut Gaussian bootstrap arithmetic and failure checks."""

import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy import stats

from quantilemix import QGCompWarning, fit_bootstrap
from quantilemix._bootstrap import _fit_bootstrap_with_indices


def data():
    rows = np.array([(a, b, r) for r in range(6) for a in range(3) for b in range(3)])
    x = rows[:, :2].astype(float)
    z = (rows[:, 2] % 3).astype(float)
    y = 2 + 1.5*x[:, 0] - .5*x[:, 1] + .2*z + .17*((np.arange(len(x)) % 5) - 2)
    return y, x, z


def test_external_schedule_matches_independent_normal_equations():
    y, x, z = data()
    n = len(y)
    schedules = np.tile(np.arange(n), (4, 1))
    schedules[0, :5] = [9, 10, 11, 12, 13]
    schedules[1, 6:12] = [20, 21, 22, 23, 24, 25]
    schedules[2, 20:26] = [2, 3, 4, 5, 6, 7]
    schedules[3, 35:42] = [12, 13, 14, 15, 16, 17, 18]
    result = _fit_bootstrap_with_indices(y, x, schedules, z, q=None)
    design = np.column_stack((np.ones(n), x, z))
    point = np.linalg.solve(design.T @ design, design.T @ y)
    manual = []
    for sampled in schedules:
        d, target = design[sampled], y[sampled]
        coef = np.linalg.solve(d.T @ d, d.T @ target)
        manual.append(coef[1] + coef[2])
    manual = np.array(manual)
    se = manual.std(ddof=1)
    assert se > 0.001
    psi = point[1] + point[2]
    assert_allclose(result.psi, psi, atol=1e-12)
    assert_allclose(result.bootstrap_effects, manual, atol=1e-12)
    assert_allclose(result.se, se, atol=1e-12)
    assert_allclose(result.var_psi, se**2, atol=1e-12)
    assert_allclose(result.intervention_levels, [0, 1, 2])
    expected = [point[0] + level*psi + point[3]*z.mean() for level in [0, 1, 2]]
    assert_allclose(result.intervention_means, expected, atol=1e-12)
    critical = stats.norm.ppf(.975)
    assert_allclose(result.ci, (psi-critical*se, psi+critical*se), atol=1e-12)
    assert result.p_value_df == n - 2 - 1 - 2
    assert_allclose(result.p_value, 2*stats.t.sf(abs(psi/se), result.p_value_df), atol=1e-12)
    summary = result.summary()
    assert "fixed-cut" in summary
    assert f"Joint mean difference: {psi:.6g}; bootstrap SE: {se:.6g}" in summary
    assert f"Two-sided Student t p-value: {result.p_value:.6g}; df: {n-5}" in summary
    assert result.bootstrap_effects.flags.writeable is False


def test_weighted_schedule_matches_independent_normal_equations():
    y, x, z = data()
    n = len(y)
    weights = 0.6 + (np.arange(n) % 7) / 5
    schedules = np.tile(np.arange(n), (4, 1))
    schedules[0, :5] = [9, 10, 11, 12, 13]
    schedules[1, 6:12] = [20, 21, 22, 23, 24, 25]
    schedules[2, 20:26] = [2, 3, 4, 5, 6, 7]
    schedules[3, 35:42] = [12, 13, 14, 15, 16, 17, 18]
    result = _fit_bootstrap_with_indices(
        y, x, schedules, z, q=None, weights=weights
    )
    design = np.column_stack((np.ones(n), x, z))

    def coefficient(rows):
        selected_design = design[rows]
        selected_weights = weights[rows]
        return np.linalg.solve(
            selected_design.T @ (selected_weights[:, None] * selected_design),
            selected_design.T @ (selected_weights * y[rows]),
        )

    point = coefficient(np.arange(n))
    effects = np.asarray([sum(coefficient(rows)[1:3]) for rows in schedules])
    assert_allclose(result.psi, sum(point[1:3]), atol=1e-12)
    assert_allclose(result.bootstrap_effects, effects, atol=1e-12)
    assert_allclose(result.se, effects.std(ddof=1), atol=1e-12)
    levels = np.arange(3)
    expected_means = point[0] + levels * sum(point[1:3]) + point[3] * np.average(z, weights=weights)
    assert_allclose(result.intervention_means, expected_means, atol=1e-12)
    for b, rows in enumerate(schedules):
        beta = coefficient(rows)
        expected = beta[0] + levels * sum(beta[1:3]) + beta[3] * np.average(
            z[rows], weights=weights[rows]
        )
        assert_allclose(result.bootstrap_means[b], expected, atol=1e-12)
    assert_allclose(result.point_fit.case_weights, weights)


def test_weighted_bootstrap_public_seed_and_weight_scale():
    y, x, z = data()
    weights = 1 + (np.arange(len(y)) % 5)
    first = fit_bootstrap(y, x, z, q=None, B=12, seed=7, weights=weights)
    second = fit_bootstrap(y, x, z, q=None, B=12, seed=7, weights=3*weights)
    assert_allclose(first.psi, second.psi, atol=1e-12)
    assert_allclose(first.bootstrap_effects, second.bootstrap_effects, atol=1e-12)
    assert_allclose(first.intervention_means, second.intervention_means, atol=1e-12)
    plain = fit_bootstrap(y, x, z, q=None, B=12, seed=7)
    unit = fit_bootstrap(y, x, z, q=None, B=12, seed=7,
                         weights=np.ones(len(y)))
    assert_allclose(unit.bootstrap_effects, plain.bootstrap_effects, atol=1e-12)


@pytest.mark.parametrize("bad", (-1.0, 0.0, np.nan, np.inf))
def test_weighted_bootstrap_rejects_invalid_case_weights(bad):
    y, x, z = data()
    weights = np.ones(len(y))
    weights[3] = bad
    with pytest.raises(ValueError, match="weights"):
        fit_bootstrap(y, x, z, q=None, B=3, weights=weights)


def test_weighted_bootstrap_rejects_wrong_weight_length():
    y, x, z = data()
    with pytest.raises(ValueError, match="weights"):
        fit_bootstrap(y, x, z, q=None, B=3, weights=np.ones(len(y)-1))


def test_bootstrap_summary_rejects_invalid_display_precision():
    y, x, z = data()
    result = fit_bootstrap(y, x, z, q=None, B=5, seed=5)
    for digits in (True, 0, 18, 2.5):
        with pytest.raises(ValueError, match="digits"):
            result.summary(digits)


def test_public_seed_repeats_and_does_not_change_legacy_global_rng():
    y, x, z = data()
    before = np.random.get_state()
    first = fit_bootstrap(y, x, z, q=None, B=15, seed=42)
    second = fit_bootstrap(y, x, z, q=None, B=15, seed=42)
    assert_array_equal(first.bootstrap_effects, second.bootstrap_effects)
    after = np.random.get_state()
    assert before[0] == after[0]
    assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert first.B == 15 and first.seed == 42


@pytest.mark.parametrize("B,seed", [(1, None), (True, None), (2.5, None), (5, -1), (5, True)])
def test_invalid_resampling_controls(B, seed):
    y, x, z = data()
    with pytest.raises(ValueError):
        fit_bootstrap(y, x, z, q=None, B=B, seed=seed)


def test_failed_replicate_is_named_and_never_replaced():
    y, x, z = data()
    schedule = np.vstack((np.arange(len(y)), np.zeros(len(y), dtype=int)))
    with pytest.raises(ValueError, match="replicate 2 failed"):
        _fit_bootstrap_with_indices(y, x, schedule, z, q=None)


def test_bootstrap_uses_one_training_cut_map():
    y, x, z = data()
    with warnings.catch_warnings():
        warnings.simplefilter("error", QGCompWarning)
        result = fit_bootstrap(y, x, z, q=3, B=12, seed=11)
    assert result.point_fit.q == 3
    assert result.point_fit.breaks is not None
    assert_allclose(result.intervention_levels, [0, 1, 2])
