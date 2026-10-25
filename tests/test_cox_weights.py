"""Independent arithmetic and installed-R fixtures for Cox case weights."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from quantilemix import fit_cox
from quantilemix._cox import _weighted_partial_likelihood, _weighted_score_residuals

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((ROOT / "tests/fixtures/r_cox_weights_2_19_6.json").read_text())
SPEC = importlib.util.spec_from_file_location(
    "check_cox_weights_r", ROOT / "validation/check_cox_weights_r.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def test_hand_calculated_weighted_untied_risk_sets_and_derivatives():
    # Three ordered times. The second and third rows have events.
    x = np.array([[0.], [1.], [2.]])
    event = np.array([0., 1., 1.])
    time = np.array([3., 2., 1.])
    weights = np.array([0.5, 1.5, 2.])
    loglik, score, information = _weighted_partial_likelihood(
        x, event, np.zeros(1), time, weights)
    assert loglik == pytest.approx(-1.5*np.log(2)-2*np.log(4))
    assert score[0] == pytest.approx(1.625)
    assert information[0, 0] == pytest.approx(1.25)
    step = 1e-5
    plus = _weighted_partial_likelihood(x, event, np.array([step]), time, weights)
    minus = _weighted_partial_likelihood(x, event, np.array([-step]), time, weights)
    assert (plus[0]-minus[0])/(2*step) == pytest.approx(score[0], abs=1e-9)
    assert -(plus[1][0]-minus[1][0])/(2*step) == pytest.approx(
        information[0, 0], abs=1e-9)


def test_hand_calculated_weighted_efron_and_score_residuals():
    x = np.array([[0.], [1.], [2.]])
    event = np.array([1., 1., 0.])
    time = np.array([2., 2., 2.])
    weights = np.array([0.5, 1.5, 2.])
    loglik, score, information = _weighted_partial_likelihood(
        x, event, np.zeros(1), time, weights)
    assert loglik == pytest.approx(-np.log(4)-np.log(3))
    assert score[0] == pytest.approx(1.5-1.375-4.75/3)
    step = 1e-5
    plus = _weighted_partial_likelihood(x, event, np.array([step]), time, weights)
    minus = _weighted_partial_likelihood(x, event, np.array([-step]), time, weights)
    assert -(plus[1][0]-minus[1][0])/(2*step) == pytest.approx(
        information[0, 0], abs=1e-9)
    residual = _weighted_score_residuals(x, event, time, np.zeros(1), weights)
    np.testing.assert_allclose(residual.sum(axis=0), score, atol=1e-14)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: c["name"])
def test_saved_installed_r_weighted_case(case):
    values, expected = case["inputs"], case["expected"]
    x = np.column_stack((values["x1"], values["x2"]))
    fitted = fit_cox(values["time"], values["event"], x, values["z"],
                     q=None, weights=values["weights"])
    for actual, reference in ((fitted.coefficients, expected["coefficients"]),
                              (fitted.covariance, expected["covariance"]),
                              (fitted.ci, expected["ci"]),
                              (fitted.timeline, expected["baseline_time"]),
                              (fitted.baseline_cumulative_hazard,
                               expected["baseline_cumulative_hazard"])):
        np.testing.assert_allclose(actual, reference, atol=2e-8, rtol=0)
    assert fitted.psi == pytest.approx(expected["psi"], abs=2e-8)
    assert fitted.var_psi == pytest.approx(expected["variance"], abs=2e-8)
    assert fitted.p_value == pytest.approx(expected["p"], abs=2e-8)
    assert fitted.robust_variance == (case["name"] != "integer_untied")
    assert not fitted.case_weights.flags.writeable
    assert "Case weights: supplied" in fitted.summary()
    if fitted.robust_variance:
        assert "robust score-sandwich SE" in fitted.summary()


def test_unit_weights_preserve_unweighted_fit_and_invalid_weights_fail():
    case = REFERENCE["cases"][0]["inputs"]
    x = np.column_stack((case["x1"], case["x2"]))
    ordinary = fit_cox(case["time"], case["event"], x, case["z"], q=None)
    unit = fit_cox(case["time"], case["event"], x, case["z"], q=None,
                   weights=np.ones(len(case["time"])))
    np.testing.assert_allclose(unit.coefficients, ordinary.coefficients, atol=1e-11)
    np.testing.assert_allclose(unit.covariance, ordinary.covariance, atol=1e-11)
    np.testing.assert_allclose(unit.baseline_cumulative_hazard,
                               ordinary.baseline_cumulative_hazard, atol=1e-11)
    assert not unit.robust_variance and ordinary.case_weights is None
    for bad in (np.zeros(len(case["time"])), -np.ones(len(case["time"])),
                np.full(len(case["time"]), np.nan),
                np.full(len(case["time"]), np.inf),
                np.ones(len(case["time"])-1)):
        with pytest.raises(ValueError, match="weights"):
            fit_cox(case["time"], case["event"], x, case["z"],
                    q=None, weights=bad)


def test_optional_checker_detects_changed_reference():
    assert CHECK.check(REFERENCE, REFERENCE)["status"] == "passed"
    changed = copy.deepcopy(REFERENCE)
    changed["cases"][0]["expected"]["psi"] += 0.01
    with pytest.raises(ValueError, match="psi"):
        CHECK.check(REFERENCE, changed)
    changed = copy.deepcopy(REFERENCE)
    changed["survival_version"] = "0.0.0"
    with pytest.raises(ValueError, match="survival_version"):
        CHECK.check(REFERENCE, changed)
