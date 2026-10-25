"""Analytic truth, deterministic data and actual preflight control flow."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_array_equal


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cox_bootstrap_preflight", ROOT / "validation/cox_bootstrap_preflight.py")
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


def test_known_joint_hazard_ratio_and_deterministic_inputs():
    zero = np.zeros((1, 4), dtype=int)
    one = np.ones((1, 4), dtype=int)
    ratio = preflight.conditional_hazard(one)[0]/preflight.conditional_hazard(zero)[0]
    np.testing.assert_allclose(np.log(ratio), preflight.TRUE_LOG_HR, atol=1e-15)
    assert preflight.design()["coverage_result"] is False
    first = preflight.generate()
    second = preflight.generate()
    for a, b in zip(first, second):
        assert_array_equal(a, b)
    scores, duration, event = first
    assert scores.shape == (300, 4)
    assert np.all((0 < duration) & (duration <= preflight.ADMIN_END))
    assert set(np.unique(event)).issubset({0., 1.})
    assert preflight.digest(*first) == preflight.digest(*second)
    assert preflight.fit_seed() == preflight.fit_seed()


def test_preflight_writes_success_and_refuses_overwrite(tmp_path, monkeypatch):
    def fake_fit(duration, event, scores, *, q, degree, B, MCsize, seed):
        assert len(duration) == len(event) == len(scores) == 300
        assert (q, degree, B, MCsize, seed) == (None, 1, 2, 300, preflight.fit_seed())
        return SimpleNamespace(psi=np.array([.3]), se=np.array([.1]),
                               bootstrap_effects=np.array([[.2], [.4]]),
                               intervention_levels=np.array([.75, 1.5, 2.25]),
                               simulated_event=np.array([1., 0., 1.]))
    monkeypatch.setattr(preflight, "fit_cox_bootstrap", fake_fit)
    output = tmp_path / "preflight.json"
    report = preflight.run(output)
    assert report["status"] == "passed"
    assert json.loads(output.read_text())["bootstrap_effects"] == [.2, .4]
    with pytest.raises(ValueError, match="new path"):
        preflight.run(output)


def test_preflight_preserves_failure_record(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("deliberate fitted-stage failure")
    monkeypatch.setattr(preflight, "fit_cox_bootstrap", fail)
    output = tmp_path / "failure.json"
    report = preflight.run(output)
    assert report["status"] == "failed"
    assert report["error_class"] == "ValueError"
    assert "deliberate" in json.loads(output.read_text())["error_message"]
