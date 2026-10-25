"""Two fixed installed-R qgcomp 2.19.6 survival-curve tables."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantilemix import fit_cox_bootstrap, qgcomp_survcurve_boot


ROOT = Path(__file__).resolve().parents[1]
SOURCE = json.loads((ROOT / "tests/fixtures/r_cox_ties_2_19_6.json").read_text())
REFERENCE = json.loads((ROOT / "tests/fixtures/r_cox_survcurve_2_19_6.json").read_text())


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: f"degree_{c['degree']}")
def test_fixed_r_survival_curve_tables(case):
    x = SOURCE["cases"][1]["inputs"]
    fit = fit_cox_bootstrap(x["time"], x["event"],
                            np.column_stack((x["x1"], x["x2"])), x["z"],
                            q=4, degree=case["degree"], B=3, MCsize=128, seed=271828)
    tables = qgcomp_survcurve_boot(fit)
    assert_allclose(fit.psi, np.atleast_1d(case["psi"]), atol=2e-8, rtol=0)
    for name in ("mdfpop", "cdfpop", "mdfq", "cdfq"):
        for column, expected in case["curves"][name].items():
            assert_allclose(tables[name][column], expected, atol=2e-8, rtol=0)
