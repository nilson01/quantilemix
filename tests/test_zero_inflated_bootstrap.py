"""Focused checks for simulated marginal zero-inflated Poisson effects."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from quantilemix import fit_zero_inflated_poisson_bootstrap
import quantilemix._zero_inflated_bootstrap as zip_bootstrap


ROWS = json.loads((Path(__file__).parent / "fixtures/r_zero_inflated_2_19_6.json").read_text())["cases"][0]["data"]
Y = np.array([r["y"] for r in ROWS])
X = np.array([[r["x1"], r["x2"]] for r in ROWS])
Z = np.array([r["z"] for r in ROWS])


def test_seeded_marginal_components_and_uncertainty():
    options = dict(q=None, B=5, MCsize=1000, seed=19)
    original_rng = np.random.get_state()
    first = fit_zero_inflated_poisson_bootstrap(Y, X, Z, **options)
    second = fit_zero_inflated_poisson_bootstrap(Y, X, Z, **options)
    after_rng = np.random.get_state()
    assert all(np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
               for a, b in zip(original_rng, after_rng))
    np.testing.assert_array_equal(first.bootstrap_effects, second.bootstrap_effects)
    assert first.psi_count == second.psi_count
    assert first.psi_zero == second.psi_zero
    assert first.intervention_levels.tolist() == [0, 1, 2, 3]
    assert first.intervention_means.shape == (4,)
    assert first.bootstrap_means.shape == (5, 4)
    assert first.bootstrap_effects.shape == (5, 2)
    np.testing.assert_allclose(first.covariance,
                               np.cov(first.bootstrap_effects, rowvar=False, ddof=1))
    np.testing.assert_allclose(first.se,
                               np.std(first.bootstrap_effects, axis=0, ddof=1))
    point = np.array([first.psi_count, first.psi_zero])
    np.testing.assert_allclose(first.ci[:, 0], point-stats.norm.ppf(.975)*first.se)
    np.testing.assert_allclose(first.ci[:, 1], point+stats.norm.ppf(.975)*first.se)
    assert first.point_fit.family == "zero_inflated_poisson"
    assert first.marginal_fit.exposure_names == ("x1",)
    assert "not an observed-count mean ratio" in first.summary()
    with pytest.raises(ValueError, match="digits"):
        first.summary(0)
    with pytest.raises(ValueError):
        first.bootstrap_effects[0, 0] = 0


def test_standardized_expected_means_agree_with_independent_component_prediction():
    result = fit_zero_inflated_poisson_bootstrap(Y, X, Z, q=None,
                                                  B=2, MCsize=600, seed=11)
    # The point stage samples adjustment rows first. Reconstruct that one
    # schedule from the seed, then use the public conditional predictor.
    selected = np.random.default_rng(11).integers(0, len(Y), size=600)
    for j, score in enumerate(result.intervention_levels):
        proposed = np.full((600, X.shape[1]), score)
        mu, pi = result.point_fit.predict_components(proposed, Z[selected])
        assert result.intervention_means[j] == pytest.approx(np.mean((1-pi)*mu), abs=1e-12)


def test_four_level_quantization_with_adjustment():
    result = fit_zero_inflated_poisson_bootstrap(Y, X+0.1, Z,
                                                  q=4, B=2, MCsize=800, seed=23)
    np.testing.assert_array_equal(result.intervention_levels, [0, 1, 2, 3])
    assert np.all(np.isfinite(result.ci))


@pytest.mark.parametrize("change,match", [
    ({"B": 1}, "B"), ({"B": True}, "B"),
    ({"MCsize": 1}, "MCsize"), ({"MCsize": False}, "MCsize"),
    ({"seed": -1}, "seed"), ({"seed": True}, "seed"),
])
def test_invalid_controls(change, match):
    options = dict(q=None, B=2, MCsize=500, seed=7)
    options.update(change)
    with pytest.raises(ValueError, match=match):
        fit_zero_inflated_poisson_bootstrap(Y, X, Z, **options)


def test_failed_resample_is_named_and_not_replaced(monkeypatch):
    original = zip_bootstrap._marginal_stage
    calls = 0

    def fail_first_resample(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("injected stage failure")
        return original(*args)

    monkeypatch.setattr(zip_bootstrap, "_marginal_stage", fail_first_resample)
    with pytest.raises(ValueError, match="bootstrap replicate 1 failed: injected stage failure"):
        fit_zero_inflated_poisson_bootstrap(Y, X, Z, q=None,
                                            B=2, MCsize=500, seed=7)
    assert calls == 2
