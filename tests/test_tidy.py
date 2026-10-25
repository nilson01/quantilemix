"""Joint-effect table contracts and repaired reference display conventions."""

import numpy as np
import pytest

from quantilemix import fit, fit_binomial, fit_cox, fit_poisson, tidy


def _rows():
    x = np.array([(a, b) for a in range(4) for b in range(4)]*8, dtype=float)
    i = np.arange(len(x))
    return x, i


def test_gaussian_tidy_has_distinct_intercept_and_joint_inference():
    x, i = _rows()
    y = 2+0.3*x[:, 0]-0.1*x[:, 1]+0.2*np.sin(i*7)
    fitted = fit(y, x, q=None)
    table = tidy(fitted)
    assert tuple(table["term"]) == ("(Intercept)", "psi1")
    assert table["estimate"][1] == pytest.approx(fitted.psi)
    assert table["p_value"][1] == pytest.approx(fitted.p_value)
    assert table["p_value"][0] < table["p_value"][1]/1e10
    assert np.all(table["lower_ci"] <= table["estimate"])
    assert np.all(table["estimate"] <= table["upper_ci"])
    assert not table["estimate"].flags.writeable
    assert tidy(fitted, quick=True).columns == ("term", "estimate")
    with pytest.raises(ValueError, match="Gaussian"):
        tidy(fitted, exponentiate=True)


def test_log_link_and_cox_exponentiation_uses_upper_limit():
    x, i = _rows()
    outcomes = (((i*7) % 11 < 5).astype(float), ((i*13) % 7).astype(float))
    for fitted in (fit_binomial(outcomes[0], x, q=None),
                   fit_poisson(outcomes[1], x, q=None),
                   fit_cox(1+(i*7)%29, outcomes[0], x, q=None)):
        plain = tidy(fitted)
        ratio = tidy(fitted, exponentiate=True)
        np.testing.assert_allclose(ratio["estimate"], np.exp(plain["estimate"]))
        np.testing.assert_allclose(ratio["upper_ci"], np.exp(plain["upper_ci"]))
        np.testing.assert_allclose(ratio["lower_ci"], np.exp(plain["lower_ci"]))
        assert np.all(ratio["upper_ci"] > ratio["lower_ci"])
        assert ratio["p_value"][-1] == pytest.approx(fitted.p_value)


def test_invalid_tidy_request():
    x, i = _rows()
    fitted = fit(2+x[:, 0]+0.2*np.sin(i), x, q=None)
    with pytest.raises(ValueError, match="conf_level"):
        tidy(fitted, conf_level=1)
    with pytest.raises(TypeError, match="supported"):
        tidy(object())
