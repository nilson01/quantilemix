"""Fixed installed-R binary risk-ratio fixture is exercised without R."""

import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from quantilemix._risk_ratio import _from_indices


FIXTURE = Path(__file__).parent / "fixtures/r_risk_ratio_2_19_6.json"


def test_fixed_risk_ratio_cases():
    cases = json.loads(FIXTURE.read_text())["cases"]
    assert len(cases) == 2
    for case in cases:
        a, e = case["inputs"], case["expected"]
        x = np.column_stack((a["x1"], a["x2"]))
        z = None if a["z"] is None else np.asarray(a["z"])[:, None]
        result = _from_indices(a["y"], x, a["indices"], z, q=None)
        for actual, reference, atol in (
            (result.psi, e["psi"], 2e-9),
            (result.se, e["se"], 2e-9),
            (result.ci, e["ci"], 2e-9),
            (result.p_value, e["p"], 2e-9),
            (result.intervention_risks, e["risks"], 2e-9),
            (result.bootstrap_effects, e["effects"], 2e-8),
        ):
            assert_allclose(actual, reference, rtol=0, atol=atol)
