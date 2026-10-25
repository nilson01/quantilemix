"""Distribution-free invariants for the public quantized example generator."""

import numpy as np
import pytest

from quantilemix import QGCompWarning, simdata_quantized


def test_balanced_scores_seed_and_editable_copy():
    with pytest.warns(QGCompWarning, match="increased"):
        first = simdata_quantized(n=101, q=4, coef=[0.2, -0.1], seed=71)
    with pytest.warns(QGCompWarning, match="increased"):
        second = simdata_quantized(n=101, q=4, coef=[0.2, -0.1], seed=71)
    assert first.n == 104 and first.exposure_names == ("x1", "x2")
    np.testing.assert_array_equal(first["x1"], second["x1"])
    np.testing.assert_array_equal(first["y"], second["y"])
    np.testing.assert_array_equal(np.bincount(first["x1"]), [26]*4)
    assert not first["x1"].flags.writeable
    changed = first.to_dict()
    changed["x1"][0] = -1
    assert first["x1"][0] >= 0
    np.testing.assert_array_equal(first.truecoefs["coef"], [0.2, -0.1])


def test_copy_and_reverse_correlation_endpoints():
    sample = simdata_quantized(n=80, q=4, coef=[0, 0, 0], corr=[1, -1], seed=5)
    np.testing.assert_array_equal(sample["x2"], sample["x1"])
    np.testing.assert_array_equal(sample["x3"], 3-sample["x1"])


def test_binary_survival_and_multinomial_shapes():
    binary = simdata_quantized("logistic", n=80, coef=[0.2, -0.1], seed=1)
    assert set(np.unique(binary["y"])).issubset({0, 1})
    early = simdata_quantized("survival", n=80, coef=[0.2, -0.1],
                              scale0=2, censtime=100, seed=1)
    late = simdata_quantized("survival", n=80, coef=[0.2, -0.1],
                             scale0=4, censtime=100, seed=1)
    np.testing.assert_allclose(late["time"], 2*early["time"], atol=1e-12)
    assert set(np.unique(early["d"])).issubset({0, 1})
    multi = simdata_quantized("multinomial", n=80,
                              coef=[[0.1, -0.2], [0.3, 0.1]], seed=1)
    assert set(np.unique(multi["y"])).issubset({1, 2, 3})
    assert multi.true_coefficients.shape == (2, 2)


@pytest.mark.parametrize("kwargs", [
    {"n": 0}, {"q": 1}, {"coef": []}, {"corr": [2]},
    {"outcometype": "unknown"}, {"outcometype": "multinomial", "coef": [1, 2]},
    {"outcometype": "survival", "shape0": 0}, {"seed": -1},
])
def test_invalid_generator_inputs(kwargs):
    with pytest.raises(ValueError):
        simdata_quantized(**kwargs)
