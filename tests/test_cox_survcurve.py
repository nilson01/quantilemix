"""Population averaging and R-named conditional/marginal survival tables."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from quantilemix import fit_cox_bootstrap, qgcomp_survcurve_boot


SOURCE = json.loads((Path(__file__).parent / "fixtures/r_cox_ties_2_19_6.json").read_text())
X = SOURCE["cases"][1]["inputs"]


def _fit(degree: int):
    return fit_cox_bootstrap(X["time"], X["event"],
                             np.column_stack((X["x1"], X["x2"])), X["z"],
                             q=4, degree=degree, B=3, MCsize=128, seed=271828)


def test_population_averages_and_first_adjustment_profile():
    fit = _fit(2)
    curves = qgcomp_survcurve_boot(fit)
    levels = fit.intervention_levels
    n = len(fit.conditional_fit.scores)
    assert_array_equal(curves.levels, [0, 1, 2, 3])
    assert_array_equal(curves.mdfq["q"], np.repeat([1, 2, 3, 4], len(fit.marginal_timeline)))
    assert_array_equal(curves.cdfq["q"], np.repeat([1, 2, 3, 4], len(fit.conditional_fit.timeline)))
    assert_array_equal(curves.mdfpop["time"], fit.marginal_timeline)
    assert_array_equal(curves.cdfpop["time"], fit.conditional_fit.timeline)

    # Independent conditional-model arithmetic: first row for each level,
    # every original adjustment row for the population mean.
    conditional = fit.conditional_fit
    beta_x = sum(conditional.coefficients[:2])
    beta_z = conditional.coefficients[2:]
    hazard = conditional.baseline_cumulative_hazard
    expected_first = np.exp(-hazard[:, None] *
                            np.exp(levels[None, :]*beta_x +
                                   fit.training_covariates[0] @ beta_z))
    expected_pop = np.mean(np.exp(-hazard[:, None, None] *
                                  np.exp(levels[None, :, None]*beta_x +
                                         fit.training_covariates[None, :, :] @ beta_z)),
                           axis=(1, 2))
    assert_allclose(curves.cdfq["surv"].reshape(4, -1).T, expected_first)
    assert_allclose(curves.cdfpop["surv"], expected_pop)
    assert_allclose(curves.mdfpop["surv"],
                    curves.mdfq["surv"].reshape(4, -1).mean(axis=0))
    assert len(curves.cdfq["surv"]) == 4*len(hazard)
    assert len(curves.mdfq["surv"]) == 4*len(fit.marginal_timeline)
    assert fit.training_covariates.shape == (n, 1)
    assert not fit.training_covariates.flags.writeable
    with pytest.raises(ValueError):
        curves.mdfpop["surv"][0] = .5
    with pytest.raises(TypeError):
        curves.mdfpop["new"] = np.array([1.])


def test_degree_one_and_wrong_result_type():
    curves = qgcomp_survcurve_boot(_fit(1))
    assert np.all((0 <= curves.mdfpop["surv"]) & (curves.mdfpop["surv"] <= 1))
    assert np.all(np.diff(curves.mdfpop["surv"]) <= 1e-14)
    with pytest.raises(TypeError, match="CoxBootstrapResult"):
        qgcomp_survcurve_boot(object())
    with pytest.raises(KeyError):
        _ = curves["unknown"]
