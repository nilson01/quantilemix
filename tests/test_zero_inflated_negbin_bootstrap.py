"""Focused checks for marginal zero-inflated negative-binomial bootstrap."""

import numpy as np
import pytest
from scipy import special, stats

from quantilemix import fit_zero_inflated_negative_binomial_bootstrap
import quantilemix._zero_inflated_negbin_bootstrap as nb_bootstrap


def _case():
    i = np.arange(720)
    x = np.column_stack((i % 4, (i // 4) % 4))
    z = np.sin(i*.17)
    mu = np.exp(.3+.18*x[:, 0]-.11*x[:, 1]+.12*z)
    pi = special.expit(-.8+.15*x[:, 0]-.12*x[:, 1]+.1*z)
    u0 = ((i*37) % 719+.5)/719
    u1 = ((i*113) % 719+.5)/719
    y = np.where(u0 < pi, 0, stats.nbinom.ppf(u1, 1.2, 1.2/(1.2+mu)))
    return y, x, z


def test_seeded_marginal_count_and_zero_effects():
    y, x, z = _case()
    options = dict(q=None, B=3, MCsize=1000, seed=314)
    first = fit_zero_inflated_negative_binomial_bootstrap(y, x, z, **options)
    second = fit_zero_inflated_negative_binomial_bootstrap(y, x, z, **options)
    np.testing.assert_array_equal(first.bootstrap_effects, second.bootstrap_effects)
    assert first.psi_count == second.psi_count and first.psi_zero == second.psi_zero
    assert first.intervention_levels.tolist() == [0, 1, 2, 3]
    assert first.bootstrap_effects.shape == (3, 2)
    assert first.bootstrap_means.shape == (3, 4)
    assert first.point_fit.theta > 0 and first.marginal_fit.theta > 0
    np.testing.assert_allclose(first.covariance,
                               np.cov(first.bootstrap_effects, rowvar=False, ddof=1))
    np.testing.assert_allclose(first.se,
                               np.std(first.bootstrap_effects, axis=0, ddof=1))
    point = np.array([first.psi_count, first.psi_zero])
    np.testing.assert_allclose(first.ci[:, 0], point-stats.norm.ppf(.975)*first.se)
    np.testing.assert_allclose(first.ci[:, 1], point+stats.norm.ppf(.975)*first.se)
    assert "dispersion" in first.summary()
    with pytest.raises(ValueError, match="digits"):
        first.summary(0)
    with pytest.raises(ValueError):
        first.bootstrap_effects[0, 0] = 0


def test_standardized_expected_means_use_fitted_components():
    y, x, z = _case()
    result = fit_zero_inflated_negative_binomial_bootstrap(
        y, x, z, q=None, B=2, MCsize=1000, seed=314)
    selected = np.random.default_rng(314).integers(0, len(y), size=1000)
    for j, score in enumerate(result.intervention_levels):
        proposed = np.full((1000, x.shape[1]), score)
        mu, pi = result.point_fit.predict_components(proposed, z[selected])
        assert result.intervention_means[j] == pytest.approx(
            np.mean((1-pi)*mu), abs=1e-12)


@pytest.mark.parametrize("change,match", [
    ({"B": 1}, "B"), ({"B": True}, "B"),
    ({"MCsize": 1}, "MCsize"), ({"MCsize": False}, "MCsize"),
    ({"seed": -1}, "seed"), ({"seed": True}, "seed"),
])
def test_invalid_controls(change, match):
    y, x, z = _case()
    options = dict(q=None, B=2, MCsize=1000, seed=314)
    options.update(change)
    with pytest.raises(ValueError, match=match):
        fit_zero_inflated_negative_binomial_bootstrap(y, x, z, **options)


def test_failed_resample_is_named_and_not_replaced(monkeypatch):
    y, x, z = _case()
    original = nb_bootstrap._negbin_marginal_stage
    calls = 0
    def fail_first_resample(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("injected stage failure")
        return original(*args)
    monkeypatch.setattr(nb_bootstrap, "_negbin_marginal_stage", fail_first_resample)
    with pytest.raises(ValueError, match="bootstrap replicate 1 failed: injected stage failure"):
        fit_zero_inflated_negative_binomial_bootstrap(
            y, x, z, q=None, B=2, MCsize=1000, seed=314)
    assert calls == 2
