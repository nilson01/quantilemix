# SPDX-License-Identifier: GPL-2.0-or-later
"""Independent design checks for the paper's Gaussian simulation translation."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    "paper_reproduction", Path(__file__).resolve().parents[1] / "validation/paper_reproduction.py")
paper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(paper)


@pytest.mark.parametrize("scenario", list(paper.SCENARIOS))
def test_balanced_exposures_and_replay(scenario):
    x, y = paper.generate(scenario, 0)
    assert x.shape == (500, 4)
    assert y.shape == (500,)
    for column in x.T:
        assert np.array_equal(np.bincount(column), [125, 125, 125, 125])
    again_x, again_y = paper.generate(scenario, 0)
    assert np.array_equal(x, again_x)
    assert np.array_equal(y, again_y)
    assert paper.input_hash(x, y) == paper.input_hash(again_x, again_y)


def test_seed_spaces_and_truths():
    hashes = set()
    for mode in ("pilot", "study"):
        for scenario in paper.SCENARIOS:
            x, y = paper.generate(scenario, 0, mode=mode)
            hashes.add(paper.input_hash(x, y))
    assert len(hashes) == 8
    assert sum(paper.SCENARIOS["null"]) == 0
    assert sum(paper.SCENARIOS["cancellation"]) == 0
    assert sum(paper.SCENARIOS["single"]) == .25
    assert sum(paper.SCENARIOS["distributed"]) == .25


def test_invalid_identity():
    with pytest.raises(ValueError):
        paper.generate("missing", 0)
    with pytest.raises(ValueError):
        paper.generate("null", 5)
    with pytest.raises(ValueError):
        paper.generate("null", True)


def test_fit_against_independent_least_squares():
    from quantilemix import fit

    x, y = paper.generate("cancellation", 0)
    result = fit(y, x, q=None)
    design = np.column_stack((np.ones(len(y)), x))
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ beta
    contrast = np.array([0, 1, 1, 1, 1])
    covariance = (residual @ residual / (len(y) - 5)) * np.linalg.inv(design.T @ design)
    assert result.psi == pytest.approx(float(contrast @ beta), abs=1e-10)
    assert result.var_psi == pytest.approx(float(contrast @ covariance @ contrast), abs=1e-10)


def _checker():
    spec = importlib.util.spec_from_file_location(
        "check_paper_results", Path(__file__).resolve().parents[1] /
        "validation/check_paper_results.py")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    return checker


def test_saved_pilot_checker_rejects_tampered_input_and_fit(tmp_path):
    checker = _checker()
    path = tmp_path / "pilot.json"
    result = paper.run("pilot")
    path.write_text(json.dumps(result))
    report = checker.check(path)
    assert report["verified_cases"] == 20
    assert report["all_integrity_checks_pass"]
    result["records"][0]["input_sha256"] = "0" * 64
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="input hash"):
        checker.check(path)
    result["records"][0]["input_sha256"] = paper.input_hash(*paper.generate("null", 0))
    result["records"][0]["estimate"] += .01
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="independent estimate"):
        checker.check(path)


def test_interruption_preserves_completed_case_journal(tmp_path, monkeypatch):
    original = paper.fit
    count = 0

    def interrupted(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 3:
            raise KeyboardInterrupt
        return original(*args, **kwargs)

    monkeypatch.setattr(paper, "fit", interrupted)
    journal = tmp_path / "cases.jsonl"
    with pytest.raises(KeyboardInterrupt):
        paper.run("pilot", journal=journal)
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [(r["scenario"], r["repetition"]) for r in records] == [
        ("null", 0), ("null", 1)]


def test_paper_comparison_includes_both_monte_carlo_errors():
    checker = _checker()
    summaries = {}
    for name, row in checker.PUBLISHED.items():
        summaries[name] = {"successful": 1000, "bias": row["bias"],
                           "bias_mcse": row["empirical_sd"] / np.sqrt(1000),
                           "coverage": row["coverage"],
                           "rejection_rate": row["rejection_rate"]}
    result = checker.paper_comparison(summaries)
    expected = np.sqrt(2 * .94 * .06 / 1000)
    assert result["null"]["checks"]["coverage"]["combined_mcse"] == pytest.approx(expected)
    assert not result["null"]["checks"]["coverage"]["descriptive_divergence_flag"]
    summaries["null"]["coverage"] = .80
    assert checker.paper_comparison(summaries)["null"]["checks"]["coverage"]["descriptive_divergence_flag"]
