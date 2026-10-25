"""Focused checks for marginal negative-binomial hurdle bootstrap."""

import numpy as np
import pytest
from scipy import special, stats

from quantilemix import fit_hurdle_negative_binomial, fit_hurdle_negative_binomial_bootstrap
import quantilemix._hurdle_negbin_bootstrap as nb_bootstrap


def _case():
    i = np.arange(480)
    x = np.column_stack((i % 4, (i // 4) % 4))
    z = np.sin(i*.17)
    mu = np.exp(.3+.15*x[:, 0]-.1*x[:, 1]+.2*z)
    chance = special.expit(.2+.2*x[:, 0]-.15*x[:, 1]-.1*z)
    u0 = ((i*37) % 479+.5)/479
    u1 = ((i*113) % 479+.5)/479
    p0 = stats.nbinom.pmf(0, 1.3, 1.3/(1.3+mu))
    y = (u0 < chance)*np.maximum(
        1, stats.nbinom.ppf(np.minimum(p0+u1*(1-p0), np.nextafter(1., 0.)),
                             1.3, 1.3/(1.3+mu)))
    return y, x, z


def test_seeded_marginal_components_and_intervals():
    y, x, z = _case()
    options = dict(q=None, B=3, MCsize=1000, seed=314)
    first = fit_hurdle_negative_binomial_bootstrap(y, x, z, **options)
    second = fit_hurdle_negative_binomial_bootstrap(y, x, z, **options)
    np.testing.assert_array_equal(first.bootstrap_effects, second.bootstrap_effects)
    assert first.psi_count == second.psi_count and first.psi_nonzero == second.psi_nonzero
    assert first.intervention_levels.tolist() == [0, 1, 2, 3]
    assert first.bootstrap_effects.shape == (3, 2)
    assert first.bootstrap_means.shape == (3, 4)
    assert first.point_fit.theta > 0 and first.marginal_fit.theta > 0
    np.testing.assert_allclose(first.covariance,
                               np.cov(first.bootstrap_effects, rowvar=False, ddof=1))
    np.testing.assert_allclose(first.se,
                               np.std(first.bootstrap_effects, axis=0, ddof=1))
    point = np.array([first.psi_count, first.psi_nonzero])
    np.testing.assert_allclose(first.ci[:, 0], point-stats.norm.ppf(.975)*first.se)
    np.testing.assert_allclose(first.ci[:, 1], point+stats.norm.ppf(.975)*first.se)
    assert "dispersion" in first.summary()
    with pytest.raises(ValueError, match="digits"):
        first.summary(0)
    with pytest.raises(ValueError):
        first.bootstrap_effects[0, 0] = 0


def test_standardized_means_match_public_hurdle_prediction():
    y, x, z = _case()
    fit = fit_hurdle_negative_binomial_bootstrap(y, x, z, q=None,
                                                  B=2, MCsize=1000, seed=314)
    selected = np.random.default_rng(314).integers(0, len(y), size=1000)
    for j, score in enumerate(fit.intervention_levels):
        proposed = np.full((1000, x.shape[1]), score)
        assert fit.intervention_means[j] == pytest.approx(
            np.mean(fit.point_fit.predict(proposed, z[selected])), abs=1e-12)


def test_positive_count_sampler_removes_nb_zeros(monkeypatch):
    y, x, z = _case()
    point = fit_hurdle_negative_binomial(y, x, z, q=None)
    captured = []
    def capture(outcomes, *_args, **_kwargs):
        captured.append(outcomes.copy())
        return object()
    monkeypatch.setattr(nb_bootstrap, "fit_hurdle_negative_binomial", capture)
    class FixedRNG:
        def integers(self, _low, _high, size, **_kwargs):
            return np.zeros(size, dtype=int)
        def binomial(self, _n, _p):
            return np.ones(4, dtype=int)
        def random(self, _size):
            return np.array([0., .25, .5, .75])
    nb_bootstrap._negbin_hurdle_marginal_stage(
        point, z[:, None], np.array([0., 1.]), 4, FixedRNG())
    assert len(captured) == 1 and np.all(captured[0] > 0)


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
        fit_hurdle_negative_binomial_bootstrap(y, x, z, **options)


def test_failed_resample_is_named_and_not_replaced(monkeypatch):
    y, x, z = _case()
    original = nb_bootstrap._negbin_hurdle_marginal_stage
    calls = 0
    def fail_first_resample(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("injected stage failure")
        return original(*args)
    monkeypatch.setattr(nb_bootstrap, "_negbin_hurdle_marginal_stage", fail_first_resample)
    with pytest.raises(ValueError, match="bootstrap replicate 1 failed: injected stage failure"):
        fit_hurdle_negative_binomial_bootstrap(
            y, x, z, q=None, B=2, MCsize=1000, seed=314)
    assert calls == 2
