"""Exact-cell multinomial arithmetic and fixed installed-R comparisons."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from quantilemix import QGCompWarning, fit_multinomial
from quantilemix._multinomial import _likelihood_score_information


FIXTURE = Path(__file__).parent / "fixtures/r_multinomial_2_19_6.json"
CHECKER = Path(__file__).resolve().parents[1] / "validation/check_multinomial_r.py"
SPEC = importlib.util.spec_from_file_location("check_multinomial_r", CHECKER)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def test_exact_category_cell_odds_and_probabilities():
    # At exposure 0: counts 8,4,2; at exposure 1: counts 4,8,4.
    x = np.repeat([0., 1.], [14, 16])
    y = (["low"]*8 + ["mid"]*4 + ["high"]*2
         + ["low"]*4 + ["mid"]*8 + ["high"]*4)
    result = fit_multinomial(y, x, levels=("low", "mid", "high"), q=None)
    np.testing.assert_allclose(result.coefficients[:, 0], [np.log(0.5), np.log(0.25)], atol=1e-11)
    np.testing.assert_allclose(result.psi, [np.log(4), np.log(4)], atol=1e-11)
    np.testing.assert_allclose(result.odds_ratios, [4, 4], atol=1e-11)
    np.testing.assert_allclose(result.predict_proba([0, 1]),
                               [[8/14, 4/14, 2/14], [4/16, 8/16, 4/16]], atol=1e-11)
    with pytest.warns(QGCompWarning, match="training ranges"):
        outside = result.predict_proba([2])
    np.testing.assert_allclose(outside.sum(axis=1), 1, atol=1e-14)
    assert result.homogeneity_test()["chi_square"] == pytest.approx(0, abs=1e-12)
    assert result.homogeneity_test()["p_value"] == pytest.approx(1)
    assert result.joint_test()["df"] == 2


def test_score_and_information_match_finite_differences():
    x = np.array([[1., 0.], [1., 1.], [1., 2.]])
    outcome = np.array([0, 1, 2])
    theta = np.array([0.2, -0.1, -0.3, 0.4])

    def independent_loglik(value):
        b = value.reshape(2, 2)
        logits = x @ b.T
        total = 0.
        for row, category in enumerate(outcome):
            denominator = 1 + np.exp(logits[row, 0]) + np.exp(logits[row, 1])
            total += (0 if category == 0 else logits[row, category-1]) - np.log(denominator)
        return total

    likelihood, score, information = _likelihood_score_information(x, outcome, theta.reshape(2, 2))
    assert likelihood == pytest.approx(independent_loglik(theta), abs=1e-13)
    step = 1e-5
    basis = np.eye(4)*step
    numeric_score = np.array([(independent_loglik(theta+e)-independent_loglik(theta-e))
                              /(2*step) for e in basis])
    np.testing.assert_allclose(score, numeric_score, rtol=0, atol=1e-10)
    numeric_info = np.empty((4, 4))
    for i, ei in enumerate(basis):
        for j, ej in enumerate(basis):
            numeric_info[i, j] = -(independent_loglik(theta+ei+ej)
                                   - independent_loglik(theta+ei-ej)
                                   - independent_loglik(theta-ei+ej)
                                   + independent_loglik(theta-ei-ej))/(4*step*step)
    np.testing.assert_allclose(information, numeric_info, rtol=0, atol=1e-5)


@pytest.mark.parametrize("case", json.loads(FIXTURE.read_text())["cases"], ids=lambda c: c["name"])
def test_saved_installed_r_case(case):
    x = case["inputs"]
    result = fit_multinomial(x["y"], np.column_stack((x["x1"], x["x2"])),
                             x["z"], levels=x["levels"], q=x["q"])
    actual = {"coefficients": result.coefficients, "covariance": result.covariance,
              "psi": result.psi, "psi_covariance": result.psi_covariance,
              "ci": result.ci, "p": result.p_value,
              "fitted_probabilities": result.fitted_probabilities}
    for field, value in actual.items():
        np.testing.assert_allclose(value, case["tighter_expected"][field], rtol=0, atol=1e-7)
        np.testing.assert_allclose(value, case["expected"][field], rtol=0, atol=1e-5)
    np.testing.assert_array_equal(result.scores, case["expected"]["scores"])
    np.testing.assert_allclose(result.fitted_probabilities.sum(axis=1), 1, atol=1e-14)
    np.testing.assert_allclose(result.predict_proba(np.column_stack((x["x1"], x["x2"])), x["z"]),
                               result.fitted_probabilities, atol=1e-14)
    assert "not one pooled effect" in result.summary()
    with pytest.raises(ValueError, match="digits"):
        result.summary(0)


def test_optional_checker_rejects_changed_reference():
    reference = json.loads(FIXTURE.read_text())
    assert CHECK.compare(reference)["status"] == "passed"
    CHECK.compare_r_exports(reference, reference)
    changed = copy.deepcopy(reference)
    changed["cases"][0]["tighter_expected"]["psi"][0] += 0.01
    with pytest.raises(ValueError, match="psi"):
        CHECK.compare(changed)
    with pytest.raises(ValueError, match="psi"):
        CHECK.compare_r_exports(reference, changed)


@pytest.mark.parametrize("change,match", [
    ({"levels": ("low", "mid")}, "at least three"),
    ({"levels": ("low", "mid", "mid")}, "distinct"),
    ({"y": ["low", "mid", "high", "other", "low", "mid"]}, "appear in levels"),
    ({"alpha": 0.}, "alpha"),
])
def test_restricted_input_errors(change, match):
    kwargs = dict(y=["low", "mid", "high", "low", "mid", "high"],
                  exposures=[0., 1., 2., 0., 1., 2.],
                  levels=("low", "mid", "high"), q=None)
    kwargs.update(change)
    with pytest.raises(ValueError, match=match):
        fit_multinomial(**kwargs)
