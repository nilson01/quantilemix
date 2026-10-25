"""Bounded interface checks for left-censored lognormal imputation."""

import numpy as np
import pytest

from quantilemix import impute_leftcenslognorm, impute_tobit


def _example():
    i = np.arange(80)
    x = np.linspace(-1, 1, 80)
    truth = np.exp(0.2+0.4*x+0.25*np.sin(i*7))
    observed = truth >= 1
    y = truth.copy()
    y[~observed] = np.nan
    return y, observed, x


def test_seeded_draws_obey_each_detection_limit():
    y, observed, x = _example()
    first = impute_leftcenslognorm(y, observed, x[:, None], lod=1., seed=11)
    second = impute_tobit(y, observed, x[:, None], lod=1., seed=11)
    np.testing.assert_array_equal(first, second)
    assert len(first) == np.count_nonzero(~observed)
    assert np.all((first > 0) & (first <= 1))
    varying = np.linspace(0.9, 1.1, len(y))
    drawn = impute_leftcenslognorm(y, observed, x[:, None], lod=varying, seed=11)
    assert np.all((drawn > 0) & (drawn <= varying[~observed]))


def test_empty_selection_returns_empty_values():
    y, observed, x = _example()
    y[~observed] = 1.
    np.testing.assert_array_equal(
        impute_leftcenslognorm(y, np.ones(len(y), dtype=bool), x[:, None]), [])


def test_intercept_only_uses_inferred_detection_limit():
    y, observed, _ = _example()
    drawn = impute_leftcenslognorm(y, observed, seed=23)
    assert len(drawn) == np.count_nonzero(~observed)
    assert np.all((drawn > 0) & (drawn <= np.nanmin(y[observed])))


@pytest.mark.parametrize("change,pattern", [
    (lambda y, r, x: (y, r.astype(int), x, 1.), "ry"),
    (lambda y, r, x: (y, r, x[:-1], 1.), "x"),
    (lambda y, r, x: (y, r, x, 0.), "lod"),
    (lambda y, r, x: (np.where(r, -1., y), r, x, 1.), "observed y"),
])
def test_invalid_inputs_fail(change, pattern):
    y, observed, x = _example()
    bad_y, bad_ry, bad_x, bad_lod = change(y, observed, x)
    with pytest.raises(ValueError, match=pattern):
        impute_leftcenslognorm(bad_y, bad_ry, bad_x[:, None], lod=bad_lod, seed=11)
