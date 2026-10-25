"""Focused checks for the simulated marginal multinomial bootstrap."""

import numpy as np
import pytest
from scipy import stats

from quantilemix import fit_multinomial_bootstrap
from quantilemix._multinomial_bootstrap import _marginal_stage


def _cells(repeats=10):
    # At score 0: categories 8, 4, 2; at score 1: 4, 8, 4.
    x = np.tile(np.repeat([0., 1.], [14, 16]), repeats)
    y = (["low"]*8 + ["mid"]*4 + ["high"]*2
         + ["low"]*4 + ["mid"]*8 + ["high"]*4)*repeats
    return y, x


def test_probabilities_and_bootstrap_arithmetic_are_reproducible():
    y, x = _cells()
    arguments = dict(levels=("low", "mid", "high"), q=None, B=8, seed=19)
    first = fit_multinomial_bootstrap(y, x, **arguments)
    second = fit_multinomial_bootstrap(y, x, **arguments)
    np.testing.assert_array_equal(first.psi, second.psi)
    np.testing.assert_array_equal(first.bootstrap_effects, second.bootstrap_effects)
    assert first.intervention_probabilities.shape == (2, 3)
    np.testing.assert_allclose(first.intervention_probabilities,
                               [[8/14, 4/14, 2/14], [4/16, 8/16, 4/16]], atol=1e-12)
    np.testing.assert_allclose(first.intervention_probabilities.sum(axis=1), 1, atol=1e-14)
    np.testing.assert_allclose(first.covariance,
                               np.cov(first.bootstrap_effects, rowvar=False, ddof=1), atol=1e-14)
    np.testing.assert_allclose(first.se, np.std(first.bootstrap_effects, axis=0, ddof=1))
    np.testing.assert_allclose(first.ci[:, 0], first.psi-stats.norm.ppf(.975)*first.se)
    np.testing.assert_allclose(first.ci[:, 1], first.psi+stats.norm.ppf(.975)*first.se)
    assert first.bootstrap_effects.shape == (8, 2)
    assert first.psi.shape == (2,)
    assert "marginal multinomial" in first.summary()
    with pytest.raises(ValueError, match="digits"):
        first.summary(0)
    with pytest.raises(ValueError):
        first.psi[0] = 0


def test_marginal_stage_uses_probability_draws_and_score_order():
    y, x = _cells()
    from quantilemix import fit_multinomial
    point = fit_multinomial(y, x, levels=("low", "mid", "high"), q=None)
    slope, probabilities = _marginal_stage(point, np.empty((len(x), 0)),
                                           np.array([0., 1.]), np.random.default_rng(7))
    assert slope.shape == (2,)
    np.testing.assert_allclose(probabilities,
                               [[8/14, 4/14, 2/14], [4/16, 8/16, 4/16]], atol=1e-12)


def test_exact_cell_marginal_slope_from_controlled_category_draws():
    y, x = _cells(56)
    from quantilemix import fit_multinomial
    point = fit_multinomial(y, x, levels=("low", "mid", "high"), q=None)

    class FixedDraws:
        calls = 0

        def random(self, count):
            self.calls += 1
            # Uniform midpoints produce exact counts for probabilities with
            # denominators 14 and 16 because 1680 is a common multiple.
            return (np.arange(count)+0.5)/count

    draw = FixedDraws()
    slope, _ = _marginal_stage(point, np.empty((len(x), 0)),
                               np.array([0., 1.]), draw)
    assert draw.calls == 2
    np.testing.assert_allclose(slope, [np.log(4), np.log(4)], atol=1e-10)


def test_four_level_joint_interventions_with_numeric_adjustment():
    row = np.arange(360)
    x = np.column_stack((np.sin(row*.31), np.cos(row*.17)))
    z = (row % 5).astype(float)
    labels = ("low", "mid", "high")
    y = [labels[int((i*17+i//7) % 3)] for i in row]
    result = fit_multinomial_bootstrap(y, x, z, levels=labels, q=4, B=3, seed=23)
    np.testing.assert_array_equal(result.intervention_levels, [0, 1, 2, 3])
    assert result.intervention_probabilities.shape == (4, 3)
    assert result.bootstrap_effects.shape == (3, 2)
    assert np.all(np.isfinite(result.ci))


@pytest.mark.parametrize("change,match", [
    ({"B": 1}, "B"), ({"B": True}, "B"), ({"seed": -1}, "seed"),
    ({"seed": True}, "seed"),
])
def test_invalid_controls(change, match):
    y, x = _cells()
    options = dict(levels=("low", "mid", "high"), q=None, B=3, seed=5)
    options.update(change)
    with pytest.raises(ValueError, match=match):
        fit_multinomial_bootstrap(y, x, **options)
