"""Known generator answers from literal inputs; no random draws or model fits."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "validation" / "calibration.py"
SPEC = importlib.util.spec_from_file_location("calibration_generator_under_test", SCRIPT)
cal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cal)


# Columns are Z1, Z2, U, eta1, eta2, eta3, eta4, epsilon. Repeating these
# four literal rows meets the frozen n=100 cell without drawing any samples.
NORMAL_ROWS = [
    [0, 0, 0, -1, -.7, 0, .7, 1],
    [2, -4, 0, -4, -2, 0, 2, 1.5],
    [0, 2, 1, -2, -1, 0, 1, -.5],
    [-2, 1, -1, 2, 3, 4, 5, .25],
]

# Derived directly from the protocol: the quartiles are about +/-0.7541
# and zero. At rho=0 the first latent row is [-1, -.7, 0, .7], so -.7/0/.7
# cover below/at/above zero and distinguish scaled from unscaled quartiles.
# At rho=.75 the remaining latent rows are [-1, 0, 1, 2],
# [-.133975, .366025, .866025, 1.366025], and
# [-.866025, -.366025, .133975, .633975]. These distinguish the Z1 loading,
# shared U term, and exposure-specific noise columns and coefficients.
SCORES_BY_RHO = {
    0: [[0, 1, 1, 2], [0, 0, 3, 3], [0, 0, 1, 3], [3, 3, 3, 3]],
    .75: [[1, 1, 1, 2], [0, 1, 3, 3], [1, 2, 3, 3], [0, 1, 2, 2]],
}


class LiteralRandom:
    """Check the seed/shape contract and supply only the fixed normal matrix."""

    def __init__(self):
        self.calls = []
        self.seed_sequence = object()
        self.bit_generator = object()

    def SeedSequence(self, seed):
        self.calls.append(("SeedSequence", list(seed)))
        return self.seed_sequence

    def PCG64(self, seed_sequence):
        assert seed_sequence is self.seed_sequence
        self.calls.append(("PCG64",))
        return self.bit_generator

    def Generator(self, bit_generator):
        assert bit_generator is self.bit_generator
        self.calls.append(("Generator",))
        return self

    def standard_normal(self, size):
        assert size == (100, 8)
        self.calls.append(("standard_normal", size))
        return np.asarray(NORMAL_ROWS * 25, dtype=float)


class LocalNumpy:
    """Replace cal.np.random without modifying the shared NumPy module."""

    def __init__(self, random):
        self.random = random

    def __getattr__(self, name):
        return getattr(np, name)


@pytest.mark.parametrize("rho,beta_name,cell_index,expected_y", [
    (0, "null", 0, [2, 4.5, 0, 0]),
    (0, "cancellation", 1, [1.75, 4.5, 0, 0]),
    (0, "positive", 2, [2.15, 4.65, .05, .75]),
    (.75, "null", 3, [2, 4.5, 0, 0]),
    (.75, "cancellation", 4, [2, 4.25, -.25, -.25]),
    (.75, "positive", 5, [2.25, 4.75, .45, .2]),
])
def test_generate_case_matches_protocol_known_answers(
        monkeypatch, rho, beta_name, cell_index, expected_y):
    config = cal.load_json(SCRIPT.with_name("calibration-config.json"))
    cell = cal.make_plan(config)["cells"][cell_index]
    assert (cell["n"], cell["rho"], cell["beta_name"]) == (100, rho, beta_name)
    literal_random = LiteralRandom()
    original_random = np.random
    original_generator = np.random.Generator
    monkeypatch.setattr(cal, "np", LocalNumpy(literal_random))

    def forbidden(*args, **kwargs):
        pytest.fail("the generator check must not fit models or execute a study")

    for name in ("fit", "qr_result", "_load_adapter", "execute"):
        monkeypatch.setattr(cal, name, forbidden)

    actual = cal.generate_case(config, "study", cell, 7)

    expected_seed = [20261022, 2, cell_index, 7]
    assert actual["seed"] == expected_seed
    assert actual["repetition"] == 7
    assert actual["cell"] == cell["id"]
    assert actual["name"] == f"study-{cell['id']}-r00007"
    assert literal_random.calls == [
        ("SeedSequence", expected_seed), ("PCG64",), ("Generator",),
        ("standard_normal", (100, 8)),
    ]
    assert np.random is original_random
    assert np.random.Generator is original_generator

    inputs = actual["inputs"]
    assert np.asarray(inputs["exposures"]).shape == (100, 4)
    assert np.asarray(inputs["covariates"]).shape == (100, 2)
    assert np.asarray(inputs["y"]).shape == (100,)
    np.testing.assert_array_equal(inputs["exposures"], SCORES_BY_RHO[rho] * 25)
    np.testing.assert_array_equal(inputs["covariates"], [[0, 0], [2, -4], [0, 2], [-2, 1]] * 25)
    # Intercept + covariate contribution + epsilon is [2, 4.5, 0, 0].
    # Literal outcomes above then add the specified score coefficients;
    # neither the generator nor a fitted model supplies expected values.
    np.testing.assert_allclose(inputs["y"], expected_y * 25, rtol=0, atol=1e-14)
    assert inputs["q"] is None and inputs["alpha"] == .05
