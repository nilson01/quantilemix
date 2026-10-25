"""Known event-time draws and public marginal survival bootstrap checks."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import norm

from quantilemix import fit_cox_bootstrap
import quantilemix._cox_bootstrap as cox_bootstrap
from quantilemix._cox_bootstrap import _simulate_survival


CASES = json.loads((Path(__file__).parent / "fixtures/r_cox_ties_2_19_6.json").read_text())["cases"]


def test_inverse_cumulative_hazard_known_times():
    point = SimpleNamespace(timeline=np.array([1., 2.]),
                            baseline_cumulative_hazard=np.array([.2, .7]),
                            coefficients=np.array([np.log(2.)]),
                            exposure_names=("x1",))
    time, event, level = _simulate_survival(
        point, np.array([0., 1.]), np.empty((3, 0)),
        np.array([[.1, .5, .8], [.1, .5, .8]]))
    assert_array_equal(time, [1, 2, 2, 1, 2, 2])
    assert_array_equal(event, [1, 1, 0, 1, 1, 1])
    assert_array_equal(level, [0, 0, 0, 1, 1, 1])
    assert not time.flags.writeable


def test_marginal_fit_and_bootstrap_repeat_with_seed():
    x = CASES[1]["inputs"]
    args = (x["time"], x["event"], np.column_stack((x["x1"], x["x2"])), x["z"])
    one = fit_cox_bootstrap(*args, q=4, degree=2, B=3, MCsize=128, seed=314159)
    two = fit_cox_bootstrap(*args, q=4, degree=2, B=3, MCsize=128, seed=314159)
    assert_array_equal(one.simulated_time, two.simulated_time)
    assert_array_equal(one.simulated_event, two.simulated_event)
    assert_allclose(one.bootstrap_effects, two.bootstrap_effects, rtol=0, atol=0)
    assert_allclose(one.covariance, np.cov(one.bootstrap_effects, rowvar=False, ddof=1))
    assert_allclose(one.se, np.std(one.bootstrap_effects, axis=0, ddof=1))
    assert_allclose(one.ci[:, 0], one.psi - norm.ppf(.975)*one.se)
    assert_allclose(one.ci[:, 1], one.psi + norm.ppf(.975)*one.se)
    assert_allclose(one.effect_at(2), one.psi[0] + 5*one.psi[1])
    assert one.simulated_time.shape == (4*128,)
    assert one.bootstrap_effects.shape == (3, 2)
    assert "marginal Cox" in one.summary()
    assert not one.psi.flags.writeable


@pytest.mark.parametrize("option,message", [
    ({"degree": 3}, "degree"), ({"B": 1}, "B"),
    ({"MCsize": 127}, "MCsize"), ({"seed": -1}, "seed"),
])
def test_marginal_input_errors(option, message):
    x = CASES[1]["inputs"]
    settings = dict(q=4, B=2, MCsize=128, seed=13)
    settings.update(option)
    with pytest.raises(ValueError, match=message):
        fit_cox_bootstrap(x["time"], x["event"],
                          np.column_stack((x["x1"], x["x2"])), x["z"],
                          **settings)


def test_prequantized_intervention_levels_use_pooled_quartiles():
    x = CASES[0]["inputs"]
    result = fit_cox_bootstrap(x["time"], x["event"],
                               np.column_stack((x["x1"], x["x2"])), x["z"],
                               q=None, B=2, MCsize=128, seed=314159)
    expected = np.quantile(np.column_stack((x["x1"], x["x2"])).ravel(), [.25, .5, .75])
    assert_allclose(result.intervention_levels, expected)
    assert result.psi.shape == (1,)


def test_failed_bootstrap_retains_original_effect_and_completed_draw(monkeypatch):
    point = SimpleNamespace(
        scores=np.array([[0., 1.], [1., 2.], [2., 3.], [3., 0.]]),
        _quantization=SimpleNamespace(mode="quantized", category_counts=(4, 4)),
    )
    monkeypatch.setattr(cox_bootstrap, "fit_cox", lambda *args, **kwargs: point)
    calls = 0

    def fit_then_fail(*args):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise ValueError("Cox score did not converge")
        effect = np.array([.3 if calls == 1 else .1])
        return point, effect, np.array([1.]), np.array([1.]), np.array([0.]), np.array([1.]), np.array([.2])

    monkeypatch.setattr(cox_bootstrap, "_one_fit", fit_then_fail)
    with pytest.raises(cox_bootstrap.CoxBootstrapFitError,
                       match="bootstrap replicate 2 failed") as caught:
        fit_cox_bootstrap([1., 2., 3., 4.], [1., 1., 1., 1.],
                          point.scores, q=4, degree=1, B=2, MCsize=4, seed=314159)
    error = caught.value
    assert error.draw == 2
    assert error.estimate == (.3,)
    assert error.effects == ((.1,),)
    assert error.levels == (0., 1., 2., 3.)
