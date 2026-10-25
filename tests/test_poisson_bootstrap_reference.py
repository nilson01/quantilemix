"""Fixed installed-R Poisson bootstrap values are checked without R in CI."""

import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from quantilemix._poisson_bootstrap import _from_indices


FIXTURE = Path(__file__).parent / "fixtures/r_poisson_bootstrap_2_19_6.json"


def test_fixed_poisson_bootstrap_cases():
    cases = json.loads(FIXTURE.read_text())["cases"]
    assert len(cases) == 2
    for case in cases:
        a, e = case["inputs"], case["expected"]
        x = np.column_stack((a["x1"], a["x2"]))
        z = None if a["z"] is None else np.asarray(a["z"])[:, None]
        result = _from_indices(a["y"], x, a["indices"], z, q=a["q"])
        for actual, reference in (
            (result.psi, e["psi"]), (result.se, e["se"]),
            (result.ci, e["ci"]), (result.p_value, e["p"]),
            (result.intervention_means, e["means"]),
            (result.bootstrap_effects, e["effects"]),
            (result.point_fit.scores, e["scores"]),
        ):
            assert_allclose(actual, reference, rtol=0, atol=2e-9)
        if a["q"] is None:
            assert result.point_fit.breaks is None
        else:
            for cut, reference in zip(result.point_fit.breaks, e["breaks"]):
                assert_allclose(cut, reference, rtol=0, atol=1e-10)
