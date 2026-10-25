"""Fixed exposure-covariate R values are exercised without R in CI."""

import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from quantilemix._quadratic import _fit_quadratic_with_indices


FIXTURE = Path(__file__).parent / "fixtures/r_covariate_interactions_2_19_6.json"


def test_two_fixed_covariate_interaction_cases():
    cases = json.loads(FIXTURE.read_text())["cases"]
    assert len(cases) == 2
    for case in cases:
        a, e = case["inputs"], case["expected"]
        result = _fit_quadratic_with_indices(
            a["y"], np.column_stack((a["x1"], a["x2"])),
            case["resample_indices"], a["z"], q=a["q"],
            pairwise_interactions=a["pairwise"], covariate_interactions=True)
        for actual, reference in (
            (result.msm_coefficients, e["msm_coefficients"]),
            (result.psi, e["psi"]), (result.covariance, e["covariance"]),
            (result.se, e["se"]), (result.ci, e["ci"]),
            (result.p_value, e["p"]), (result.p_value_df, e["df"]),
            (result.bootstrap_coefficients, e["bootstrap_coefficients"]),
            (result.intervention_levels, e["levels"]),
            (result.intervention_means, e["means"]),
            (result.scores, e["scores"]),
        ):
            assert_allclose(actual, reference, rtol=0, atol=1e-10)
        if a["q"] is None:
            assert result.breaks is None
        else:
            for cut, reference in zip(result.breaks, e["breaks"]):
                assert_allclose(cut, reference, rtol=0, atol=1e-10)
