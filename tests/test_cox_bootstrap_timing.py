"""Fixed profile settings and timed runner's result/failure records."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from quantilemix import fit_cox_bootstrap


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cox_bootstrap_timing", ROOT / "validation/cox_bootstrap_timing.py")
timing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(timing)


def test_profiles_fixed_and_seeds_distinct():
    assert timing.plan("medium")["MCsize"] == 1000
    assert timing.plan("default_mc")["MCsize"] == 10000
    assert timing.plan("medium")["B"] == timing.plan("default_mc")["B"] == 20
    assert timing.seed("medium") != timing.seed("default_mc")
    with pytest.raises(ValueError, match="profile"):
        timing.plan("unknown")


def test_timing_runner_records_settings_and_refuses_overwrite(tmp_path, monkeypatch):
    def fake_fit(duration, event, x, *, q, degree, B, MCsize, seed):
        assert (q, degree, B, MCsize, seed) == (None, 1, 20, 1000, timing.seed("medium"))
        return SimpleNamespace(psi=np.array([.4]), se=np.array([.2]),
                               bootstrap_effects=np.full((20, 1), .4),
                               simulated_event=np.ones(100))
    monkeypatch.setattr(timing, "fit_cox_bootstrap", fake_fit)
    output = tmp_path / "timing.json"
    report = timing.run("medium", output)
    assert report["status"] == "passed"
    assert len(json.loads(output.read_text())["bootstrap_effects"]) == 20
    with pytest.raises(ValueError, match="new path"):
        timing.run("medium", output)


def test_timing_runner_saves_failure(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("deliberate timing fit failure")
    monkeypatch.setattr(timing, "fit_cox_bootstrap", fail)
    output = tmp_path / "failure.json"
    report = timing.run("default_mc", output)
    assert report["status"] == "failed"
    assert json.loads(output.read_text())["error_class"] == "ValueError"


def test_default_size_marginal_cox_converges_through_replicate_six():
    """The fixed profile exposed a likelihood-roundoff rejection at replicate 6."""
    x, duration, event = timing.preflight.generate()
    result = fit_cox_bootstrap(
        duration, event, x, q=None, degree=1, B=6, MCsize=10000,
        seed=timing.seed("default_mc"))
    assert np.all(np.isfinite(result.bootstrap_effects))
    assert result.bootstrap_effects.shape == (6, 1)
