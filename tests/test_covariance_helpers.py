"""Named linear-contrast covariance checks."""

import json
from pathlib import Path

import numpy as np
import pytest

from quantilemix import se_comb, vc_comb


REFERENCE = json.loads((Path(__file__).parent / "fixtures/r_covariance_helpers_2_19_6.json").read_text())


def test_installed_r_defaults_and_gradients():
    names, covariance = REFERENCE["names"], REFERENCE["covariance"]
    assert se_comb(["x1", "x2"], covariance, names=names) == pytest.approx(REFERENCE["se_default"], abs=1e-14)
    assert se_comb(["x1", "x2"], covariance, names=names, grad=.5) == pytest.approx(REFERENCE["se_scalar"], abs=1e-14)
    assert se_comb(["x1", "x2"], covariance, names=names, grad=[0, .3, .1]) == pytest.approx(REFERENCE["se_full"], abs=1e-14)
    np.testing.assert_allclose(vc_comb(["x1", "x2"], covariance, names=names),
                               REFERENCE["vc_default"], atol=1e-14)


def test_weighted_cross_term_uses_gradient():
    names, covariance = REFERENCE["names"], REFERENCE["covariance"]
    default = vc_comb(["x1", "x2"], covariance, names=names)
    weighted = vc_comb(["x1", "x2"], covariance, names=names, grad=.5)
    np.testing.assert_allclose(weighted[0, 1], .5*default[0, 1], atol=1e-14)
    np.testing.assert_allclose(weighted[1, 1], .25*default[1, 1], atol=1e-14)
    assert weighted[0, 1] != pytest.approx(REFERENCE["vc_scalar_r"][0][1])
    assert not weighted.flags.writeable


@pytest.mark.parametrize("names,covariance,exposures", [
    (["a", "a"], [[1, 0], [0, 1]], ["a"]),
    (["a", "b"], [[1, 2], [2, 1]], ["a"]),
    (["a", "b"], [[1, .1], [.2, 1]], ["a"]),
    (["a", "b"], [[1, 0], [0, 1]], ["c"]),
])
def test_reject_invalid_covariance_or_names(names, covariance, exposures):
    with pytest.raises(ValueError):
        se_comb(exposures, covariance, names=names)


def test_reject_bad_gradient_length():
    with pytest.raises(ValueError, match="grad"):
        vc_comb(["x1"], np.eye(2), names=["(Intercept)", "x1"], grad=[1])
