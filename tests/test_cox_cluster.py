"""Known-score and installed-R checks for conditional Cox cluster variance."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from quantilemix import fit_cox, fit_cox_formula, qgcomp
from quantilemix._cox import _weighted_score_residuals

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((ROOT / "tests/fixtures/r_cox_cluster_2_19_6.json").read_text())
SPEC = importlib.util.spec_from_file_location(
    "check_cox_cluster_r", ROOT / "validation/check_cox_cluster_r.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def test_hand_calculated_row_and_cluster_scores():
    # At beta=0, the two event risk-set means are 1/2 and 1. The score
    # contribution for each row includes its event term and risk-set term.
    x = np.array([[0.], [1.], [2.]])
    event = np.array([0., 1., 1.])
    time = np.array([3., 2., 1.])
    scores = _weighted_score_residuals(x, event, time, np.zeros(1), np.ones(3))
    np.testing.assert_allclose(scores[:, 0], [7/12, 1/4, 2/3], atol=1e-14)
    cluster_scores = np.array([scores[[0, 2]].sum(), scores[1].sum()])
    np.testing.assert_allclose(cluster_scores, [5/4, 1/4], atol=1e-14)
    assert cluster_scores.sum() == pytest.approx(3/2)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: c["name"])
def test_saved_installed_r_cluster_case(case):
    inputs, expected = case["inputs"], case["expected"]
    x = np.column_stack((inputs["x1"], inputs["x2"]))
    fitted = fit_cox(inputs["time"], inputs["event"], x, inputs["z"], q=None,
                     cluster_ids=inputs["cluster_ids"], weights=inputs["weights"],
                     offset=inputs["offset"])
    for actual, reference in ((fitted.coefficients, expected["coefficients"]),
                              (fitted.covariance, expected["covariance"]),
                              (fitted.ci, expected["ci"])):
        np.testing.assert_allclose(actual, reference, atol=2e-8, rtol=0)
    assert fitted.psi == pytest.approx(expected["psi"], abs=2e-8)
    assert fitted.var_psi == pytest.approx(expected["variance"], abs=2e-8)
    assert fitted.p_value == pytest.approx(expected["p"], abs=2e-8)
    assert fitted.event_count == expected["events"]
    assert fitted.cluster_count == 16
    assert fitted.robust_variance
    assert not fitted.cluster_ids.flags.writeable
    assert "cluster-robust SE" in fitted.summary()


def test_cluster_relabeling_preserves_fit_but_changes_model_based_uncertainty():
    inputs = REFERENCE["cases"][0]["inputs"]
    x = np.column_stack((inputs["x1"], inputs["x2"]))
    ids = np.asarray(inputs["cluster_ids"])
    clustered = fit_cox(inputs["time"], inputs["event"], x, inputs["z"],
                        q=None, cluster_ids=ids)
    relabeled = fit_cox(inputs["time"], inputs["event"], x, inputs["z"],
                        q=None, cluster_ids=100-ids)
    ordinary = fit_cox(inputs["time"], inputs["event"], x, inputs["z"], q=None)
    np.testing.assert_allclose(clustered.coefficients, ordinary.coefficients, atol=1e-11)
    np.testing.assert_allclose(clustered.covariance, relabeled.covariance, atol=1e-12)
    assert np.max(np.abs(clustered.covariance-ordinary.covariance)) > 1e-4


def test_invalid_cluster_ids_fail_clearly():
    inputs = REFERENCE["cases"][0]["inputs"]
    x = np.column_stack((inputs["x1"], inputs["x2"]))
    n = len(inputs["time"])
    bad = (np.ones(n), np.zeros(n-1), np.full(n, np.nan),
           np.arange(n)+0.5, np.full(n, 2**53), ["a"]*n)
    for ids in bad:
        with pytest.raises(ValueError, match="cluster_ids|clusters"):
            fit_cox(inputs["time"], inputs["event"], x, inputs["z"],
                    q=None, cluster_ids=ids)


def test_survival_formula_and_dispatch_reach_same_conditional_fit():
    inputs = REFERENCE["cases"][2]["inputs"]
    data = {"time": inputs["time"], "event": inputs["event"],
            "x1": inputs["x1"], "x2": inputs["x2"], "z": inputs["z"]}
    formula = "Surv(time,event) ~ x1 + x2 + z"
    options = {"q": None, "weights": inputs["weights"],
               "cluster_ids": inputs["cluster_ids"], "offset": inputs["offset"]}
    direct = fit_cox(inputs["time"], inputs["event"],
                     np.column_stack((inputs["x1"], inputs["x2"])), inputs["z"],
                     **options)
    named = fit_cox_formula(formula, data, ["x1", "x2"], **options)
    routed = qgcomp(formula, data, ["x1", "x2"], family="cox", **options)
    for fitted in (named, routed):
        np.testing.assert_allclose(fitted.coefficients, direct.coefficients, atol=1e-12)
        np.testing.assert_allclose(fitted.covariance, direct.covariance, atol=1e-12)
        assert fitted.coefficient_names == ("x1", "x2", "z")
    category = ["low" if value == 0 else "high" for value in inputs["z"]]
    data["group"] = category
    category_fit = fit_cox_formula("Surv(time,event) ~ x1 + x2 + C(group)",
                                   data, ["x1", "x2"], q=None)
    assert category_fit.coefficient_names == ("x1", "x2", "group[T.low]")


def test_survival_formula_rejects_unsupported_forms():
    inputs = REFERENCE["cases"][0]["inputs"]
    data = {"time": inputs["time"], "event": inputs["event"],
            "x1": inputs["x1"], "x2": inputs["x2"], "z": inputs["z"]}
    for formula in ("time ~ x1 + x2", "Surv(time,time) ~ x1 + x2",
                    "Surv(time,event) ~ x1:x2", "Surv(time,event) ~ time + x1"):
        with pytest.raises(ValueError):
            fit_cox_formula(formula, data, ["x1", "x2"], q=None)
    with pytest.raises(ValueError, match="missing='raise'"):
        qgcomp("Surv(time,event) ~ x1 + x2", data, ["x1", "x2"],
               family="cox", q=None, missing="drop")


def test_optional_checker_detects_changed_reference():
    assert CHECK.check(REFERENCE, REFERENCE)["status"] == "passed"
    changed = copy.deepcopy(REFERENCE)
    changed["cases"][0]["expected"]["psi"] += 0.01
    with pytest.raises(ValueError, match="psi"):
        CHECK.check(REFERENCE, changed)
    malformed = copy.deepcopy(REFERENCE)
    del malformed["cases"][0]["expected"]["covariance"]
    with pytest.raises(ValueError, match="fields changed"):
        CHECK.check(REFERENCE, malformed)


def test_optional_checker_reports_missing_rscript(tmp_path):
    output = tmp_path / "report.json"
    run = subprocess.run(
        [sys.executable, str(ROOT / "validation/check_cox_cluster_r.py"),
         "--rscript", str(tmp_path / "missing-rscript"),
         "--library", str(tmp_path), "--output", str(output)],
        capture_output=True, text=True, check=False)
    assert run.returncode != 0
    assert "missing-rscript" in run.stderr
    assert not output.exists()
