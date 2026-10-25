"""Independent risk-set arithmetic and fixed installed-R Cox comparisons."""

import json
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from quantilemix import fit_cox
from quantilemix._cox import _baseline_hazard, _partial_likelihood

CHECKER = Path(__file__).resolve().parents[1] / "validation/check_cox_r.py"
SPEC = importlib.util.spec_from_file_location("check_cox_r", CHECKER)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


FIXTURE = Path(__file__).parent / "fixtures/r_cox_2_19_6.json"


def test_hand_calculated_risk_sets():
    # Risk sets at the second and third durations contain 2 and 3 people.
    x = np.array([[0.], [1.], [2.]])
    likelihood, score, information = _partial_likelihood(x, np.array([0., 1., 1.]), np.zeros(1))
    assert likelihood == pytest.approx(-np.log(2) - np.log(3))
    assert score[0] == pytest.approx(1.5)
    assert information[0, 0] == pytest.approx(0.25 + 2/3)


def test_hand_calculated_efron_risk_set_and_score():
    # At the shared event time, all three are at risk and two have events.
    # Efron removes half the two event risks in the second denominator.
    x = np.array([[0.], [1.], [2.]])
    likelihood, score, information = _partial_likelihood(
        x, np.array([1., 1., 0.]), np.zeros(1), np.array([2., 2., 2.]))
    assert likelihood == pytest.approx(-np.log(3)-np.log(2))
    assert score[0] == pytest.approx(-1.25)
    assert information[0, 0] == pytest.approx(2/3 + 11/16)


def test_hand_calculated_efron_baseline_with_censor_only_time():
    # Three people are at risk at time 2, including one censored there.
    # Two events give Efron hazard increments 1/3 and 1/2.
    time, cumulative = _baseline_hazard(
        np.zeros((4, 1)), np.array([1., 1., 0., 0.]),
        np.array([2., 2., 2., 1.]), np.zeros(1))
    np.testing.assert_array_equal(time, [1., 2.])
    np.testing.assert_allclose(cumulative, [0., 5/6], rtol=0, atol=1e-14)


def test_tied_event_and_censor_times_fit():
    x = np.array([(a, b) for a in range(3) for b in range(3)] * 8, dtype=float)
    n = len(x)
    durations = 1 + (np.arange(n)*7) % 19
    events = ((np.arange(n)*11) % 5 < 3).astype(float)
    result = fit_cox(durations, events, x, q=None)
    assert result.tied_time_groups > 0
    assert np.isfinite(result.psi) and result.var_psi > 0
    assert "Efron tie handling" in result.summary()
    times, curves = result.survival_curve([[0, 0], [1, 1]])
    assert curves.shape == (len(times), 2)
    assert np.all((curves >= 0) & (curves <= 1))
    assert np.all(np.diff(curves, axis=0) <= 1e-14)
    with pytest.raises(ValueError, match="without covariates"):
        result.survival_curve([[0, 0]], [[0]])


def test_optional_checker_accepts_pinned_cases_and_rejects_changed_reference():
    reference = json.loads(FIXTURE.read_text())
    assert CHECK.compare(reference)["status"] == "passed"
    CHECK.compare_r_exports(reference, reference)
    changed = copy.deepcopy(reference)
    changed["cases"][0]["expected"]["psi"] += 0.01
    with pytest.raises(ValueError, match="psi"):
        CHECK.compare(changed)
    with pytest.raises(ValueError, match="psi"):
        CHECK.compare_r_exports(reference, changed)


@pytest.mark.parametrize("case", json.loads(FIXTURE.read_text())["cases"], ids=lambda c: c["name"])
def test_saved_installed_r_case(case):
    x, expected = case["inputs"], case["expected"]
    result = fit_cox(x["time"], x["event"], np.column_stack((x["x1"], x["x2"])),
                     x["z"], q=x["q"])
    np.testing.assert_allclose(result.coefficients, expected["coefficients"], rtol=0, atol=1e-11)
    np.testing.assert_allclose(result.covariance, expected["covariance"], rtol=0, atol=1e-11)
    np.testing.assert_array_equal(result.scores, expected["scores"])
    for actual, saved in [(result.psi, expected["psi"]), (result.var_psi, expected["variance"]),
                          (result.ci, expected["ci"]), (result.p_value, expected["p"])]:
        np.testing.assert_allclose(actual, saved, rtol=0, atol=1e-11)
    assert result.event_count == expected["events"]
    assert result.hazard_ratio == pytest.approx(np.exp(result.psi))
    assert "Baseline curve is available" in result.summary()
    with pytest.raises(ValueError, match="digits"):
        result.summary(0)


@pytest.mark.parametrize("change,match", [
    ({"durations": [0., 2., 3., 4., 5., 6.]}, "positive"),
    ({"events": [0., 1., 2., 0., 1., 1.]}, "zero and one"),
    ({"events": [0., 0., 0., 0., 0., 0.]}, "more events"),
    ({"alpha": 1.}, "alpha"),
])
def test_restricted_input_errors(change, match):
    kwargs = dict(durations=[1., 2., 3., 4., 5., 6.], events=[0., 1., 0., 1., 1., 1.],
                  exposures=[0., 1., 2., 0., 1., 2.], q=None)
    kwargs.update(change)
    with pytest.raises(ValueError, match=match):
        fit_cox(**kwargs)
