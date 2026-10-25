"""Two fixed clustered Gaussian estimating-equation cases from installed R."""

import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy import stats

from quantilemix import fit_gaussian_ee

FIXTURE = Path(__file__).parent / "fixtures/r_gaussian_ee_2_19_6.json"
SHA256 = "a574c5c7544a8e56e494ef2908a69537463c5120301d9d37c646ad8c7a2fd0d0"


@pytest.mark.parametrize("index", [0, 1])
def test_gaussian_ee_installed_r_case(index):
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == SHA256
    payload = json.loads(FIXTURE.read_text())
    assert payload["qgcomp_version"] == "2.19.6"
    case = payload["cases"][index]
    a, e = case["inputs"], case["expected"]
    x = np.column_stack((a["x1"], a["x2"]))
    result = fit_gaussian_ee(a["y"], x, a["id"], a["z"], q=a["q"])
    assert_array_equal(result.scores, e["scores"])
    assert_array_equal(result.intervention_levels, e["levels"])
    if a["q"] is None:
        assert result.breaks is None  # R stores synthetic prequantized endpoints.
        assert len(e["breaks"]) == 2
    else:
        for actual, reference in zip(result.breaks, e["breaks"]):
            assert_allclose(actual, reference, rtol=0, atol=1e-10)
    for actual, name in (
        (result.conditional_coefficients, "conditional_coefficients"),
        (result.conditional_covariance, "conditional_covariance"),
        (result.msm_coefficients, "msm_coefficients"),
        (result.msm_covariance, "msm_covariance"),
        (result.full_covariance, "full_covariance"),
        (result.meat, "meat"),
        (result.psi, "psi"), (result.var_psi, "variance"),
        (result.ci, "ci"), (result.intervention_means, "means"),
    ):
        assert_allclose(actual, e[name], rtol=0, atol=2e-10)
    # R saves a numerical Richardson derivative; Python uses exact derivatives.
    assert_allclose(result.bread, e["bread"], rtol=0, atol=2e-8)
    t = result.msm_coefficients / np.sqrt(np.diag(result.msm_covariance))
    assert_allclose(t, e["statistic"], rtol=0, atol=2e-10)
    assert_allclose(2*stats.t.sf(abs(t), result.p_value_df), e["p"],
                    rtol=0, atol=2e-10)
    assert_allclose(result.test_statistic, e["statistic"][1], rtol=0, atol=2e-10)
    assert_allclose(result.p_value, e["p"][1], rtol=0, atol=2e-10)
    assert result.p_value_df == e["df"]
