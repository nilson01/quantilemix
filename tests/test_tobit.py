"""Independent censored-normal likelihood checks and installed-R fixtures."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

from quantilemix import fit_tobit
from quantilemix._tobit import _likelihood_score_information


FIXTURE = Path(__file__).parent / "fixtures/r_tobit_2_19_6.json"
CHECKER = Path(__file__).resolve().parents[1] / "validation/check_tobit_r.py"
SPEC = importlib.util.spec_from_file_location("check_tobit_r", CHECKER)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def test_analytic_score_and_information_against_independent_likelihood_differences():
    # Three rows: one left censored, one observed exactly, one right censored.
    design = np.array([[1., 0.], [1., 1.], [1., 2.]])
    y = np.array([0., 1., 2.])
    theta = np.array([0.2, 0.1, np.log(1.2)])

    def independent_loglik(parameter):
        mean = design @ parameter[:2]
        sd = np.exp(parameter[-1])
        return (norm.logcdf((0-mean[0])/sd)
                + norm.logpdf(y[1], mean[1], sd)
                + norm.logsf((2-mean[2])/sd))

    likelihood, score, information = _likelihood_score_information(design, y, 0., 2., theta)
    assert likelihood == pytest.approx(independent_loglik(theta), abs=1e-13)
    step = 1e-5
    basis = np.eye(3)*step
    numeric_score = np.array([(independent_loglik(theta+e)-independent_loglik(theta-e))
                              /(2*step) for e in basis])
    np.testing.assert_allclose(score, numeric_score, rtol=0, atol=2e-10)
    numeric_information = np.empty((3, 3))
    for i, ei in enumerate(basis):
        for j, ej in enumerate(basis):
            numeric_information[i, j] = -(independent_loglik(theta+ei+ej)
                                          - independent_loglik(theta+ei-ej)
                                          - independent_loglik(theta-ei+ej)
                                          + independent_loglik(theta-ei-ej))/(4*step*step)
    np.testing.assert_allclose(information, numeric_information, rtol=0, atol=1e-5)


@pytest.mark.parametrize("case", json.loads(FIXTURE.read_text())["cases"], ids=lambda c: c["name"])
def test_saved_installed_r_case(case):
    x, expected = case["inputs"], case["expected"]
    fit = fit_tobit(x["y"], np.column_stack((x["x1"], x["x2"])), x["z"],
                    q=x["q"], left=x["left"], right=x["right"])
    np.testing.assert_allclose(fit.coefficients, expected["coefficients"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(fit.full_covariance, expected["covariance"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(fit.sigma, expected["sigma"], rtol=0, atol=1e-10)
    np.testing.assert_array_equal(fit.scores, expected["scores"])
    for actual, saved in [(fit.psi, expected["psi"]), (fit.var_psi, expected["variance"]),
                          (fit.ci, expected["ci"]), (fit.p_value, expected["p"])]:
        np.testing.assert_allclose(actual, saved, rtol=0, atol=1e-10)
    assert fit.left_count == expected["left_count"]
    assert fit.right_count == expected["right_count"]
    assert "latent mean" in fit.summary().lower()
    with pytest.raises(ValueError, match="digits"):
        fit.summary(0)


def test_optional_checker_rejects_changed_reference():
    reference = json.loads(FIXTURE.read_text())
    assert CHECK.compare(reference)["status"] == "passed"
    CHECK.compare_r_exports(reference, reference)
    changed = copy.deepcopy(reference)
    changed["cases"][0]["expected"]["psi"] += 0.01
    with pytest.raises(ValueError, match="psi"):
        CHECK.compare(changed)
    with pytest.raises(ValueError, match="psi"):
        CHECK.compare_r_exports(reference, changed)


@pytest.mark.parametrize("change,match", [
    ({"left": None, "right": None}, "at least one"),
    ({"left": 2., "right": 1.}, "below right"),
    ({"y": [-1., 0.2, 1., 1.4, 1.8, 2.1]}, "clipped at the left"),
    ({"right": 2., "y": [0., 0.2, 1., 1.4, 1.8, 2.1]}, "clipped at the right"),
    ({"alpha": 0.}, "alpha"),
])
def test_restricted_input_errors(change, match):
    kwargs = dict(y=[0., 0.2, 1., 1.4, 1.8, 2.1],
                  exposures=[0., 1., 2., 0., 1., 2.], q=None)
    kwargs.update(change)
    with pytest.raises(ValueError, match=match):
        fit_tobit(**kwargs)
