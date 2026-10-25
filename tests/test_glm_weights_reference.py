"""Fixed weighted GLM results exported from installed R qgcomp 2.19.6."""

import hashlib
import json
from pathlib import Path

from numpy.testing import assert_allclose, assert_array_equal
import pytest

from quantilemix import fit_binomial_formula, fit_poisson_formula


FIXTURE = Path(__file__).parent / "fixtures/r_glm_weights_2_19_6.json"
SHA256 = "cb51d6eb918089c528bf974b9de40954e1f5c92b4c5666320b11c6dabf0727b2"


@pytest.mark.parametrize("index", range(4))
def test_weighted_installed_r_case(index):
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == SHA256
    payload = json.loads(FIXTURE.read_text())
    assert payload["qgcomp_version"] == "2.19.6"
    case = payload["cases"][index]
    a, expected = case["inputs"], case["expected"]
    data = {name: a[name] for name in ("y", "x1", "x2")}
    formula = "y ~ x1 + x2"
    if a["z"] is not None:
        data["z"] = a["z"]
        formula += " + z"
    fitter = fit_binomial_formula if case["family"] == "binomial" else fit_poisson_formula
    result = fitter(formula, data, ("x1", "x2"), q=a["q"], weights=a["weights"])
    assert_array_equal(result.scores, expected["scores"])
    assert_array_equal(result.case_weights, expected["prior_weights"])
    if a["q"] is None:
        assert result.breaks is None and expected["breaks"] is None
    else:
        for actual, reference in zip(result.breaks, expected["breaks"]):
            assert_allclose(actual, reference, rtol=0, atol=0)
    # R glm uses an IRLS stopping rule; its final covariance can differ at
    # about 3e-7 from the converged weighted Hessian on these fixed cases.
    for actual, name in ((result.coefficients, "coefficients"),
                         (result.covariance, "covariance"),
                         (result.psi, "psi"), (result.var_psi, "variance"),
                         (result.ci, "ci"), (result.p_value, "p"),
                         (result.fitted_values, "fitted")):
        assert_allclose(actual, expected[name], rtol=0, atol=5e-7)
    assert result.residual_df == expected["residual_df"]
    new = {"x1": a["new_x1"], "x2": a["new_x2"]}
    if a["new_z"] is not None:
        new["z"] = a["new_z"]
    assert_allclose(result.predict_data(new), expected["predictions"], rtol=0, atol=5e-7)
