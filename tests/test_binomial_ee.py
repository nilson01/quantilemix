"""Known-answer and failure checks for clustered binary estimating equations."""

import numpy as np
import pytest
from scipy import special, stats

from quantilemix import fit_binomial_ee


def _exact_cells():
    # Four cells have event probabilities 1/5, 1/3, 1/2 and 2/3.
    cells = ((0, 0), (1, 0), (0, 1), (1, 1))
    events = (6, 10, 15, 20)
    multiplier = (1, 7, 11, 13)
    offset = (0, 2, 4, 6)
    rows = [(a, b, int((cluster*multiplier[cell]+offset[cell]) % 30 < events[cell]), cluster)
            for cluster in range(30) for cell, (a, b) in enumerate(cells)]
    array = np.asarray(rows, dtype=float)
    return array[:, 2], array[:, :2], array[:, 3]


@pytest.mark.parametrize("rr,ratio", [(False, 8), (True, 10/3)])
def test_exact_event_cells_and_independent_cluster_sandwich(rr, ratio):
    y, exposure, ids = _exact_cells()
    fitted = fit_binomial_ee(y, exposure, ids, q=None, rr=rr)
    np.testing.assert_allclose(fitted.conditional_coefficients,
                               [special.logit(.2), np.log(2), np.log(4)], atol=1e-12)
    np.testing.assert_allclose(fitted.intervention_risks, [.2, 2/3], atol=1e-12)
    assert fitted.effect_ratio == pytest.approx(ratio, abs=1e-12)
    assert fitted.link == ("log" if rr else "logit")
    x = np.column_stack((np.ones(len(y)), exposure))
    eta = x @ np.array([special.logit(.2), np.log(2), np.log(4)])
    probability = special.expit(eta)
    cluster_scores = np.array([x[ids == cluster].T @
                               (y[ids == cluster] - probability[ids == cluster])
                               for cluster in np.unique(ids)])
    inverse_info = np.linalg.solve(
        x.T @ ((probability*(1-probability))[:, None] * x), np.eye(3))
    robust = inverse_info @ cluster_scores.T @ cluster_scores @ inverse_info
    gradient = (np.array([.2-2/3, 1-2/3, 1-2/3]) if rr
                else np.array([0., 1., 1.]))
    expected_var = float(gradient @ robust @ gradient)
    assert fitted.var_psi == pytest.approx(expected_var, abs=1e-12)
    critical = stats.norm.isf(.025)
    np.testing.assert_allclose(fitted.ci, [np.log(ratio)-critical*fitted.se,
                                           np.log(ratio)+critical*fitted.se], atol=1e-12)
    assert not fitted.msm_covariance.flags.writeable
    with pytest.raises(ValueError):
        fitted.msm_covariance[0, 0] = 0
    assert ("risk ratio" if rr else "odds ratio") in fitted.summary()


def test_quantized_fit_and_saved_cuts():
    y, exposure, ids = _exact_cells()
    fitted = fit_binomial_ee(y, exposure, ids, q=2)
    assert fitted.breaks is not None
    assert fitted.cluster_count == 30
    np.testing.assert_allclose(fitted.transform(exposure).scores, fitted.scores)
    assert fitted.effect_ratio_ci[0] < fitted.effect_ratio < fitted.effect_ratio_ci[1]


@pytest.mark.parametrize("bad_y", [lambda y: y[:-1], lambda y: y+2,
                                   lambda y: np.ones_like(y)])
def test_bad_outcomes(bad_y):
    y, exposure, ids = _exact_cells()
    with pytest.raises(ValueError):
        fit_binomial_ee(bad_y(y), exposure, ids, q=None)


@pytest.mark.parametrize("ids_transform", [lambda ids: ids[:-1],
                                           lambda ids: np.zeros_like(ids),
                                           lambda ids: ids+0.5])
def test_bad_clusters(ids_transform):
    y, exposure, ids = _exact_cells()
    with pytest.raises(ValueError, match="cluster_ids|clusters"):
        fit_binomial_ee(y, exposure, ids_transform(ids), q=None)


def test_bad_levels_rank_and_covariate_rows():
    y, exposure, ids = _exact_cells()
    with pytest.raises(ValueError, match="consecutive"):
        fit_binomial_ee(y, exposure+2, ids, q=None)
    with pytest.raises(ValueError, match="rank deficient"):
        fit_binomial_ee(y, np.column_stack((exposure[:, 0], exposure[:, 0])), ids, q=None)
    with pytest.raises(ValueError, match="same number of rows"):
        fit_binomial_ee(y, exposure, ids, np.ones(len(y)-1), q=None)


@pytest.mark.parametrize("rr", [0, 1, "risk", None])
def test_bad_rr(rr):
    y, exposure, ids = _exact_cells()
    with pytest.raises(ValueError, match="rr"):
        fit_binomial_ee(y, exposure, ids, q=None, rr=rr)


@pytest.mark.parametrize("digits", [0, 18, True, 2.5])
def test_bad_summary_digits(digits):
    y, exposure, ids = _exact_cells()
    with pytest.raises(ValueError, match="digits"):
        fit_binomial_ee(y, exposure, ids, q=None).summary(digits)
