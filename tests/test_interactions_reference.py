"""Two fixed Gaussian interaction row-bootstrap cases from installed R qgcomp."""

import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from quantilemix._quadratic import _fit_quadratic_with_indices

FIXTURE = Path(__file__).parent / "fixtures/r_interactions_2_19_6.json"
SHA256 = "6fdebd26f8cf5a0b1f9aad1b410f17984e8626e57d5f46e99c196dd310cf051c"


@pytest.mark.parametrize("index", [0, 1])
def test_interaction_installed_r_case(index):
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == SHA256
    payload = json.loads(FIXTURE.read_text())
    assert payload["qgcomp_version"] == "2.19.6"
    case = payload["cases"][index]
    a, e = case["inputs"], case["expected"]
    x = np.column_stack((a["x1"], a["x2"]))
    result = _fit_quadratic_with_indices(a["y"], x, case["resample_indices"],
                                          a["z"], q=a["q"], pairwise_interactions=True)
    assert_array_equal(result.scores, e["scores"])
    assert_array_equal(result.intervention_levels, e["levels"])
    if a["q"] is None:
        # R stores synthetic boundaries for already assigned score levels.
        # The Python prequantized API keeps its established None contract.
        assert result.breaks is None
        assert len(e["breaks"]) == 2
    else:
        for cut, reference in zip(result.breaks, e["breaks"]):
            assert_allclose(cut, reference, rtol=5e-12, atol=5e-12)
    for actual, field in ((result.conditional_coefficients, "conditional_coefficients"),
                          (result.conditional_covariance, "conditional_covariance"),
                          (result.fitted_values, "fitted"),
                          (result.msm_coefficients, "msm_coefficients"),
                          (result.psi, "psi"), (result.covariance, "covariance"),
                          (result.se**2, "variance"), (result.se, "se"),
                          (result.ci, "ci"), (result.test_statistic, "statistic"),
                          (result.p_value, "p"),
                          (result.bootstrap_coefficients, "bootstrap_coefficients"),
                          (result.intervention_means, "means")):
        assert_allclose(actual, e[field], rtol=0, atol=1e-10)
    assert result.p_value_df == e["df"]
