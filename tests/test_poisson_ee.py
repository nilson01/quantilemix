"""Known-answer and failure checks for clustered Poisson estimating equations."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from quantilemix import fit_poisson, fit_poisson_ee


def _exact_cells():
    # Each column has one count per cluster. Cell means are 2, 4, 6, 12.
    counts = np.array([[1, 2, 2, 3], [2, 3, 5, 6],
                       [4, 5, 7, 8], [9, 11, 13, 15]])
    rows = [(a, b, counts[cell, cluster], cluster)
            for cluster in range(4)
            for cell, (a, b) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1)))]
    array = np.asarray(rows, dtype=float)
    return array[:, 2], array[:, :2], array[:, 3]


def test_exact_multiplicative_cells_and_independent_cluster_sandwich():
    y, exposure, ids = _exact_cells()
    fitted = fit_poisson_ee(y, exposure, ids, q=None)
    np.testing.assert_allclose(fitted.conditional_coefficients,
                               [np.log(2), np.log(2), np.log(3)], atol=1e-12)
    np.testing.assert_allclose(fitted.msm_coefficients, [np.log(2), np.log(6)], atol=1e-12)
    np.testing.assert_allclose(fitted.intervention_means, [2, 12], atol=1e-12)
    assert fitted.psi == pytest.approx(np.log(6), abs=1e-12)
    assert fitted.mean_ratio == pytest.approx(6, abs=1e-12)
    x = np.column_stack((np.ones(len(y)), exposure))
    mu = 2 * (2 ** exposure[:, 0]) * (3 ** exposure[:, 1])
    cluster_scores = np.array([x[ids == cluster].T @ (y[ids == cluster] - mu[ids == cluster])
                               for cluster in np.unique(ids)])
    inverse_info = np.linalg.solve(x.T @ (mu[:, None] * x), np.eye(3))
    contrast = np.array([0., 1., 1.])
    expected_var = contrast @ inverse_info @ cluster_scores.T @ cluster_scores @ inverse_info @ contrast
    assert fitted.var_psi == pytest.approx(expected_var, abs=1e-12)
    critical = stats.norm.isf(.025)
    np.testing.assert_allclose(fitted.ci, [np.log(6)-critical*fitted.se,
                                           np.log(6)+critical*fitted.se], atol=1e-12)
    assert not fitted.msm_covariance.flags.writeable
    with pytest.raises(ValueError):
        fitted.msm_covariance[0, 0] = 0


def test_quantized_fit_and_summary():
    y, exposure, ids = _exact_cells()
    fitted = fit_poisson_ee(y, exposure, ids, q=2)
    assert fitted.breaks is not None
    assert fitted.cluster_count == 4
    assert "cluster-robust SE" in fitted.summary()
    assert "equal observation periods" in fitted.summary()
    np.testing.assert_allclose(fitted.transform(exposure).scores, fitted.scores)
    assert fitted.mean_ratio_ci[0] < fitted.mean_ratio < fitted.mean_ratio_ci[1]


def test_unequal_observation_time_matches_independent_stacked_sandwich():
    y, exposure, ids = _exact_cells()
    time = np.tile([0.5, 1., 1.5, 2.], 4)
    fitted = fit_poisson_ee(y, exposure, ids, q=None, exposure_time=time)
    conditional = fit_poisson(y, exposure, q=None, exposure_time=time)
    np.testing.assert_allclose(fitted.conditional_coefficients,
                               conditional.coefficients, atol=1e-11)
    assert fitted.psi == pytest.approx(sum(conditional.coefficients[1:3]), abs=1e-11)
    assert fitted.rate_ratio == pytest.approx(np.exp(fitted.psi))
    assert "Observation-time offset: supplied" in fitted.summary()
    assert "Rate ratio" in fitted.summary()
    np.testing.assert_array_equal(fitted.exposure_time, time)
    assert not fitted.exposure_time.flags.writeable

    n = len(y)
    x = np.column_stack((np.ones(n), exposure))
    levels = fitted.intervention_levels
    interventions = [np.column_stack((np.ones(n), np.full((n, 2), a)))
                     for a in levels]
    marginal_design = [np.tile([1., a], (n, 1)) for a in levels]
    def row_scores(parameters):
        beta, msm = parameters[:3], parameters[3:]
        conditional_mean = time * np.exp(x @ beta)
        cond = x * (y - conditional_mean)[:, None]
        marginal = sum(
            m * (time * np.exp(xi @ beta) - time * np.exp(m @ msm))[:, None]
            for xi, m in zip(interventions, marginal_design))
        return np.column_stack((cond, marginal))

    parameters = np.concatenate((fitted.conditional_coefficients,
                                 fitted.msm_coefficients))
    assert np.max(np.abs(row_scores(parameters).sum(axis=0))) < 1e-9
    step = 1e-5
    numerical_bread = np.column_stack([
        (row_scores(parameters + step*np.eye(5)[j]).sum(axis=0)
         - row_scores(parameters - step*np.eye(5)[j]).sum(axis=0)) / (2*step)
        for j in range(5)])
    np.testing.assert_allclose(fitted.bread, numerical_bread, rtol=1e-8, atol=1e-8)
    cluster_scores = np.array([row_scores(parameters)[ids == group].sum(axis=0)
                               for group in np.unique(ids)])
    independent_meat = cluster_scores.T @ cluster_scores
    np.testing.assert_allclose(fitted.meat, independent_meat, atol=1e-10)
    inverse_bread = np.linalg.inv(numerical_bread)
    independent_covariance = inverse_bread @ independent_meat @ inverse_bread.T
    np.testing.assert_allclose(fitted.full_covariance, independent_covariance,
                               rtol=1e-8, atol=1e-10)


def test_unit_time_matches_unoffset_fit_and_invalid_times_fail():
    y, exposure, ids = _exact_cells()
    ordinary = fit_poisson_ee(y, exposure, ids, q=None)
    timed = fit_poisson_ee(y, exposure, ids, q=None,
                           exposure_time=np.ones(len(y)))
    for name in ("conditional_coefficients", "msm_coefficients", "full_covariance",
                 "intervention_means"):
        np.testing.assert_allclose(getattr(timed, name), getattr(ordinary, name),
                                   atol=1e-11)
    assert ordinary.exposure_time is None
    for bad in (np.zeros(len(y)), -np.ones(len(y)),
                np.full(len(y), np.nan), np.full(len(y), np.inf),
                np.ones(len(y)-1)):
        with pytest.raises(ValueError, match="exposure_time"):
            fit_poisson_ee(y, exposure, ids, q=None, exposure_time=bad)


def test_weighted_unequal_time_matches_row_replication_and_time_scale():
    y, exposure, ids = _exact_cells()
    time = np.tile([0.5, 1., 1.5, 2.], 4)
    weights = np.tile([1, 2, 3, 2], 4)
    weighted = fit_poisson_ee(y, exposure, ids, q=None,
                               weights=weights, exposure_time=time)
    repeated = np.repeat(np.arange(len(y)), weights)
    duplicated = fit_poisson_ee(y[repeated], exposure[repeated], ids[repeated],
                                 q=None, exposure_time=time[repeated])
    np.testing.assert_allclose(weighted.conditional_coefficients,
                               duplicated.conditional_coefficients, atol=1e-11)
    np.testing.assert_allclose(weighted.msm_coefficients,
                               duplicated.msm_coefficients, atol=1e-11)
    np.testing.assert_allclose(weighted.full_covariance,
                               duplicated.full_covariance, atol=1e-10)
    rescaled = fit_poisson_ee(y, exposure, ids, q=None,
                              weights=weights, exposure_time=3*time)
    assert rescaled.psi == pytest.approx(weighted.psi, abs=1e-11)
    assert rescaled.var_psi == pytest.approx(weighted.var_psi, abs=1e-10)
    np.testing.assert_allclose(rescaled.intervention_means,
                               weighted.intervention_means, atol=1e-10)
    assert rescaled.conditional_coefficients[0] == pytest.approx(
        weighted.conditional_coefficients[0]-np.log(3), abs=1e-10)


def test_installed_r_2196_does_not_use_poisson_ee_offset():
    fixture = Path(__file__).with_name("fixtures") / "r_poisson_ee_offset_2_19_6.json"
    saved = json.loads(fixture.read_text())
    assert saved["qgcomp_version"] == "2.19.6"
    inputs = saved["inputs"]
    y = np.asarray(inputs["y"], dtype=float)
    exposure = np.column_stack((inputs["x1"], inputs["x2"]))
    ids = np.asarray(inputs["id"], dtype=float)
    time = np.asarray(inputs["obstime"], dtype=float)
    ordinary = fit_poisson_ee(y, exposure, ids, q=None)
    timed = fit_poisson_ee(y, exposure, ids, q=None, exposure_time=time)
    for name in ("psi", "var_psi"):
        assert saved["no_offset"][name] == pytest.approx(
            saved["supplied_offset"][name], abs=1e-12)
        assert getattr(ordinary, name) == pytest.approx(saved["no_offset"][name],
                                                       abs=1e-10)
    np.testing.assert_allclose(ordinary.conditional_coefficients,
                               saved["no_offset"]["coefficients"], atol=1e-10)
    assert abs(timed.psi - saved["supplied_offset"]["psi"]) > 0.1


@pytest.mark.parametrize("bad_y", [lambda y: y[:-1], lambda y: np.full_like(y, -1),
                                   lambda y: y+0.5, lambda y: np.zeros_like(y)])
def test_bad_counts(bad_y):
    y, exposure, ids = _exact_cells()
    with pytest.raises(ValueError):
        fit_poisson_ee(bad_y(y), exposure, ids, q=None)


@pytest.mark.parametrize("ids_transform", [lambda ids: ids[:-1],
                                           lambda ids: np.zeros_like(ids),
                                           lambda ids: ids+0.5])
def test_bad_clusters(ids_transform):
    y, exposure, ids = _exact_cells()
    with pytest.raises(ValueError, match="cluster_ids|clusters"):
        fit_poisson_ee(y, exposure, ids_transform(ids), q=None)


def test_bad_levels_rank_and_covariate_rows():
    y, exposure, ids = _exact_cells()
    with pytest.raises(ValueError, match="consecutive"):
        fit_poisson_ee(y, exposure+2, ids, q=None)
    with pytest.raises(ValueError, match="rank deficient"):
        fit_poisson_ee(y, np.column_stack((exposure[:, 0], exposure[:, 0])), ids, q=None)
    with pytest.raises(ValueError, match="same number of rows"):
        fit_poisson_ee(y, exposure, ids, np.ones(len(y)-1), q=None)


@pytest.mark.parametrize("digits", [0, 18, True, 2.5])
def test_bad_summary_digits(digits):
    y, exposure, ids = _exact_cells()
    with pytest.raises(ValueError, match="digits"):
        fit_poisson_ee(y, exposure, ids, q=None).summary(digits)
