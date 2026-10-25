"""Public dispatcher route, effect-measure and failure checks."""

import numpy as np
import pytest

from quantilemix import (BinomialResult, BinomialRiskRatioResult, PoissonResult,
                    QGCompResult, fit_formula, fit_poisson_formula, qgcomp)


def _binary_data():
    cells = ((0, 0, 8), (1, 0, 13), (0, 1, 12), (1, 1, 17))
    x1, x2, y = [], [], []
    for a, b, count in cells:
        x1.extend([a]*24)
        x2.extend([b]*24)
        y.extend([1]*count + [0]*(24-count))
    return {"x1": x1, "x2": x2, "y": y}


def test_default_gaussian_route_keeps_existing_estimator():
    data = {"x1": [0, 0, 1, 1]*4, "x2": [0, 1, 0, 1]*4,
            "z": list(range(16))}
    data["y"] = 2 + np.asarray(data["x1"]) - .4*np.asarray(data["x2"]) + .2*np.asarray(data["z"])
    data["y"] += .03*(np.arange(16) % 3)
    dispatched = qgcomp("y ~ x1 + x2 + z", data, ("x1", "x2"), q=2)
    direct = fit_formula("y ~ x1 + x2 + z", data, ("x1", "x2"), q=2)
    assert isinstance(dispatched, QGCompResult)
    np.testing.assert_allclose(dispatched.coefficients, direct.coefficients)
    np.testing.assert_allclose(dispatched.covariance, direct.covariance)


def test_binary_default_is_marginal_rr_and_false_is_conditional_or():
    data = _binary_data()
    ratio = qgcomp("y ~ x1 + x2", data, ("x1", "x2"),
                   family="binomial", q=2, B=12, seed=29)
    odds = qgcomp("y ~ x1 + x2", data, ("x1", "x2"),
                  family="binomial", rr=False, q=2)
    assert isinstance(ratio, BinomialRiskRatioResult)
    assert isinstance(odds, BinomialResult)
    assert ratio.B == 12
    np.testing.assert_allclose(np.exp(ratio.psi), ratio.risk_ratio)
    np.testing.assert_allclose(np.exp(odds.psi), odds.odds_ratio)
    assert not np.isclose(ratio.psi, odds.psi)


def test_poisson_offset_stays_conditional():
    x1 = np.tile([0, 0, 1, 1], 8)
    x2 = np.tile([0, 1, 0, 1], 8)
    time = 1 + np.arange(32) % 3
    data = {"x1": x1, "x2": x2, "time": time,
            "y": (1+x1+2*x2)*time + np.arange(32) % 2}
    dispatched = qgcomp("y ~ x1 + x2", data, ("x1", "x2"), family="poisson",
                         q=2, exposure_time="time")
    direct = fit_poisson_formula("y ~ x1 + x2", data, ("x1", "x2"), q=2,
                                 exposure_time="time")
    assert isinstance(dispatched, PoissonResult)
    np.testing.assert_allclose(dispatched.psi, direct.psi)
    np.testing.assert_allclose(dispatched.se, direct.se)


def test_conditional_weights_missing_and_categories_pass_through():
    data = {"x1": [0, 0, 1, 1]*4, "x2": [0, 1, 0, 1]*4,
            "z": (["low"]*4 + ["high"]*4)*2}
    data["y"] = [1+2*a-b+.2*(z=="high")+.04*(i%3)
                 for i, (a,b,z) in enumerate(zip(data["x1"],data["x2"],data["z"]))]
    data["y"][2] = None
    weights = np.arange(1, 17, dtype=float)
    dispatched = qgcomp("y ~ x1 + x2 + C(z)", data, ("x1", "x2"),
                         q=2, weights=weights, missing="drop")
    direct = fit_formula("y ~ x1 + x2 + C(z)", data, ("x1", "x2"),
                         q=2, weights=weights, missing="drop")
    np.testing.assert_allclose(dispatched.psi, direct.psi)
    np.testing.assert_allclose(dispatched.se, direct.se)


@pytest.mark.parametrize("kwargs,pattern", [
    ({"family": "cox"}, "survival formula"),
    ({"family": "binomial", "q": 2, "weights": [1]*96}, "weights"),
    ({"family": "binomial", "q": 2, "missing": "drop"}, "missing"),
    ({"family": "binomial", "q": 2, "rr": False, "B": 12}, "B and seed"),
    ({"family": "poisson", "B": 12}, "B and seed"),
    ({"family": "gaussian", "seed": 17}, "B and seed"),
    ({"family": "gaussian", "exposure_time": "time"}, "Poisson"),
    ({"family": "binomial", "q": 2, "rr": "yes"}, "rr"),
])
def test_unsupported_routes_rejected(kwargs, pattern):
    with pytest.raises(ValueError, match=pattern):
        qgcomp("y ~ x1 + x2", _binary_data(), ("x1", "x2"), **kwargs)


@pytest.mark.parametrize("formula", [
    "y ~ x1 + x2 + x1:x2", "y ~ x1 + I(x2^2)",
    "Surv(time, event) ~ x1 + x2", "y ~ x1 + x2 | z",
    "y ~ x1 + x2 + offset(log(time))",
])
def test_unsupported_formula_routes_rejected(formula):
    with pytest.raises(ValueError):
        qgcomp(formula, _binary_data(), ("x1", "x2"))
