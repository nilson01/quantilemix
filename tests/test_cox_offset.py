"""Known-answer and installed-R checks for conditional Cox log offsets."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from quantilemix import fit_cox
from quantilemix._cox import _baseline_hazard, _partial_likelihood

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((ROOT / "tests/fixtures/r_cox_offset_2_19_6.json").read_text())
SPEC = importlib.util.spec_from_file_location(
    "check_cox_offset_r", ROOT / "validation/check_cox_offset_r.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def test_hand_calculated_offset_risk_sets_and_derivatives():
    # Ordered follow-up: one censor, then two distinct events.
    x = np.array([[0.], [1.], [2.]])
    event = np.array([0., 1., 1.])
    time = np.array([3., 2., 1.])
    offset = np.log([1., 2., 0.5])
    likelihood, score, information = _partial_likelihood(
        x, event, np.zeros(1), time, offset)
    assert likelihood == pytest.approx(-np.log(10.5))
    assert score[0] == pytest.approx(1-2/3+2-3/3.5)
    step = 1e-5
    plus = _partial_likelihood(x, event, np.array([step]), time, offset)
    minus = _partial_likelihood(x, event, np.array([-step]), time, offset)
    assert (plus[0]-minus[0])/(2*step) == pytest.approx(score[0], abs=1e-9)
    assert -(plus[1][0]-minus[1][0])/(2*step) == pytest.approx(
        information[0, 0], abs=1e-9)
    timeline, baseline = _baseline_hazard(x, event, time, np.zeros(1), offset=offset)
    np.testing.assert_array_equal(timeline, [1., 2., 3.])
    np.testing.assert_allclose(baseline, [2/7, 2/7+1/3, 2/7+1/3], atol=1e-14)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: c["name"])
def test_saved_installed_r_offset_case(case):
    inputs, expected = case["inputs"], case["expected"]
    x = np.column_stack((inputs["x1"], inputs["x2"]))
    fitted = fit_cox(inputs["time"], inputs["event"], x, inputs["z"],
                     q=None, weights=inputs["weights"], offset=inputs["offset"])
    profile = inputs["prediction"]
    times, survival = fitted.survival_curve(
        np.column_stack((profile["x1"], profile["x2"])), profile["z"],
        offset=profile["offset"])
    for actual, reference in ((fitted.coefficients, expected["coefficients"]),
                              (fitted.covariance, expected["covariance"]),
                              (fitted.ci, expected["ci"]),
                              (times, expected["prediction_time"]),
                              (survival, expected["prediction_survival"])):
        np.testing.assert_allclose(actual, reference, atol=2e-8, rtol=0)
    assert fitted.psi == pytest.approx(expected["psi"], abs=2e-8)
    assert fitted.var_psi == pytest.approx(expected["variance"], abs=2e-8)
    assert fitted.p_value == pytest.approx(expected["p"], abs=2e-8)
    mean_offset = np.average(inputs["offset"], weights=inputs["weights"])
    np.testing.assert_allclose(
        fitted.baseline_cumulative_hazard*np.exp(mean_offset),
        expected["baseline_cumulative_hazard"], atol=2e-8, rtol=0)
    assert not fitted.offset.flags.writeable
    assert "Known log-hazard offset: supplied" in fitted.summary()


def test_zero_and_constant_offsets_preserve_cox_contrast():
    inputs = REFERENCE["cases"][0]["inputs"]
    x = np.column_stack((inputs["x1"], inputs["x2"]))
    n = len(inputs["time"])
    ordinary = fit_cox(inputs["time"], inputs["event"], x, inputs["z"], q=None)
    zero = fit_cox(inputs["time"], inputs["event"], x, inputs["z"],
                   q=None, offset=np.zeros(n))
    shifted = fit_cox(inputs["time"], inputs["event"], x, inputs["z"],
                      q=None, offset=np.full(n, np.log(3.)))
    np.testing.assert_allclose(zero.coefficients, ordinary.coefficients, atol=1e-11)
    np.testing.assert_allclose(zero.covariance, ordinary.covariance, atol=1e-11)
    np.testing.assert_allclose(shifted.coefficients, ordinary.coefficients, atol=1e-11)
    np.testing.assert_allclose(shifted.baseline_cumulative_hazard,
                               ordinary.baseline_cumulative_hazard/3, atol=1e-11)
    ordinary_time, ordinary_curve = ordinary.survival_curve([[0, 0]], [[0]])
    shifted_time, shifted_curve = shifted.survival_curve(
        [[0, 0]], [[0]], offset=[np.log(3.)])
    np.testing.assert_array_equal(shifted_time, ordinary_time)
    np.testing.assert_allclose(shifted_curve, ordinary_curve, atol=1e-12)


def test_offset_validation_and_prediction_requirement():
    inputs = REFERENCE["cases"][0]["inputs"]
    x = np.column_stack((inputs["x1"], inputs["x2"]))
    n = len(inputs["time"])
    for bad in (np.full(n, np.nan), np.full(n, np.inf), np.zeros(n-1)):
        with pytest.raises(ValueError, match="offset"):
            fit_cox(inputs["time"], inputs["event"], x, inputs["z"],
                    q=None, offset=bad)
    fitted = fit_cox(inputs["time"], inputs["event"], x, inputs["z"],
                     q=None, offset=inputs["offset"])
    with pytest.raises(ValueError, match="offset is required"):
        fitted.survival_curve([[0, 0]], [[0]])
    with pytest.raises(ValueError, match="offset"):
        fitted.survival_curve([[0, 0]], [[0]], offset=[np.nan])
    with pytest.raises(ValueError, match="offset"):
        fitted.survival_curve([[0, 0]], [[0]], offset=[0, 1])


def test_optional_checker_detects_changed_reference():
    assert CHECK.check(REFERENCE, REFERENCE)["status"] == "passed"
    changed = copy.deepcopy(REFERENCE)
    changed["cases"][0]["expected"]["psi"] += 0.01
    with pytest.raises(ValueError, match="psi"):
        CHECK.check(REFERENCE, changed)
