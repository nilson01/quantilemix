"""Independent standardized-risk arithmetic and bounded API checks."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy import special

from quantilemix import fit_binomial_risk_ratio
from quantilemix._risk_ratio import _from_indices


def cells():
    i = np.arange(120)
    x = np.column_stack((i % 2, (i // 2) % 2)).astype(float)
    z = ((i*7) % 13 / 12)[:, None]
    y = (((i*19+3) % 29) / 29 < .22 + .1*x[:, 0] + .09*x[:, 1] + .11*z[:, 0]).astype(float)
    return y, x, z


def test_standardized_probability_arithmetic_and_bootstrap_inference():
    y, x, z = cells()
    result = fit_binomial_risk_ratio(y, x, z, q=None, B=24, seed=7)
    beta = result.point_fit.coefficients
    eta = beta[0] + z[:, 0]*beta[3]
    risks = [np.mean(special.expit(eta + level*(beta[1]+beta[2]))) for level in (0, 1)]
    assert_allclose(result.intervention_risks, risks, atol=1e-13)
    assert_allclose(result.psi, np.log(risks[1]/risks[0]), atol=1e-13)
    assert_allclose(result.risk_ratio, risks[1]/risks[0], atol=1e-13)
    assert_allclose(result.se, np.std(result.bootstrap_effects, ddof=1), atol=1e-13)
    assert_allclose(result.ci, result.psi + np.array([-1, 1])*special.ndtri(.975)*result.se,
                    atol=1e-13)
    assert_allclose(result.risk_ratio_ci, np.exp(result.ci), atol=1e-13)
    assert_allclose(result.p_value, 2*special.ndtr(-abs(result.psi/result.se)), atol=1e-13)
    assert "marginal binary risk ratio" in result.summary()
    assert not result.bootstrap_effects.flags.writeable
    assert not result.intervention_risks.flags.writeable
    assert_array_equal(result.intervention_levels, [0, 1])


def test_reproducibility_and_unadjusted_fit():
    y, x, _ = cells()
    a = fit_binomial_risk_ratio(y, x, q=None, B=12, seed=17)
    b = fit_binomial_risk_ratio(y, x, q=None, B=12, seed=17)
    assert_array_equal(a.bootstrap_effects, b.bootstrap_effects)
    beta = a.point_fit.coefficients
    assert_allclose(a.intervention_risks,
                    special.expit([beta[0], beta[0]+beta[1]+beta[2]]), atol=1e-12)
    q2 = fit_binomial_risk_ratio(y, x, q=2, B=12, seed=17)
    assert_allclose(a.psi, q2.psi, atol=1e-12)


@pytest.mark.parametrize("change,pattern", [
    ("B", "B must"), ("seed", "seed must"), ("levels", "contiguous score levels"),
    ("schedule", "B-by-n"), ("replicate", "bootstrap replicate 1 failed"),
    ("summary", "digits"),
])
def test_invalid_inputs_are_explicit(change, pattern):
    y, x, z = cells()
    with pytest.raises(ValueError, match=pattern):
        if change == "B":
            fit_binomial_risk_ratio(y, x, z, q=None, B=1)
        elif change == "seed":
            fit_binomial_risk_ratio(y, x, z, q=None, seed=-1)
        elif change == "levels":
            fit_binomial_risk_ratio(y, x*2, z, q=None, B=5)
        elif change == "schedule":
            _from_indices(y, x, np.ones((2, 3)), z, q=None)
        elif change == "replicate":
            schedule = np.tile(np.where(y == 1)[0][0], (2, len(y)))
            _from_indices(y, x, schedule, z, q=None)
        else:
            fit_binomial_risk_ratio(y, x, z, q=None, B=5, seed=3).summary(0)
