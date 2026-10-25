# SPDX-License-Identifier: GPL-2.0-or-later
"""Focused source-translation and independent-algebra checks for scenario 5."""

import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest


HERE = Path(__file__).resolve().parents[1] / "validation"


def _module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


paper = _module("paper_scenario5")
checker = _module("check_paper_scenario5")
cross_checker = _module("check_paper_scenario5_cross_platform")


def test_design_and_balanced_replay():
    assert paper.design("pilot")["total_fits"] == 90
    assert paper.design("study")["total_fits"] == 18_000
    hashes = set()
    for variant in paper.VARIANTS:
        for cell in (0, 1, 2, 8):
            x, y = paper.generate(variant, cell, 0)
            assert x.shape == (500, 9)
            assert y.shape == (500,)
            for column in x.T:
                assert np.array_equal(np.bincount(column), [125]*4)
            x2, y2 = paper.generate(variant, cell, 0)
            assert np.array_equal(x, x2)
            assert np.array_equal(y, y2)
            hashes.add(paper.input_hash(x, y))
    assert len(hashes) == 8
    assert [round(c["truth"], 2) for c in paper.cells()[::3]] == [.05, .15, .20]


def test_correlation_variant_changes_third_column():
    released_x, _ = paper.generate("released_ncor2", 8, 0)
    two_causal_x, _ = paper.generate("two_causal_ncor1", 8, 0)
    assert abs(np.corrcoef(released_x[:, 0], released_x[:, 2])[0, 1]) > .4
    assert abs(np.corrcoef(two_causal_x[:, 0], two_causal_x[:, 2])[0, 1]) < .2


def test_fit_against_independent_qr():
    from quantilemix import fit

    x, y = paper.generate("released_ncor2", 8, 0)
    fitted = fit(y, x, q=None)
    expected = checker.independent_gaussian(x, y)
    assert fitted.psi == pytest.approx(expected["estimate"], abs=1e-10)
    assert fitted.var_psi == pytest.approx(expected["variance"], abs=1e-10)
    assert list(fitted.ci) == pytest.approx(expected["ci"], abs=1e-10)


def test_reject_invalid_case_keys():
    with pytest.raises(ValueError):
        paper.generate("other", 0, 0)
    with pytest.raises(ValueError):
        paper.generate("released_ncor2", True, 0)
    with pytest.raises(ValueError):
        paper.generate("released_ncor2", 0, 5)


def test_pilot_checker_and_corruption(tmp_path):
    path = tmp_path / "pilot.json"
    journal = tmp_path / "pilot.json.jsonl"
    result = paper.run("pilot", journal=journal)
    path.write_text(json.dumps(result))
    checked = checker.check(path)
    assert checked["verified_cases"] == 90
    assert checked["independent_gaussian_checks"] == 90
    journal.write_text(journal.read_text() + "\n")
    with pytest.raises(ValueError, match="journal hash"):
        checker.check(path)
    result["journal_sha256"] = None
    result["records"][0]["estimate"] += .01
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="independent estimate"):
        checker.check(path)


def test_study_rejects_unknown_git_status(monkeypatch):
    def fake_run(command, **_kwargs):
        if command[1] == "rev-parse":
            return subprocess.CompletedProcess(command, 0, "a" * 40 + "\n", "")
        return subprocess.CompletedProcess(command, 1, "", "status unavailable")

    monkeypatch.setattr(paper.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="clean, committed"):
        paper.provenance("study")


def test_checker_rejects_study_provenance_faults(tmp_path, monkeypatch):
    path = tmp_path / "pilot.json"
    journal = tmp_path / "pilot.json.jsonl"
    result = paper.run("pilot", journal=journal)
    result["plan"]["mode"] = "study"
    result["source_identity"]["git_commit"] = "a" * 40
    result["source_identity"]["source_revision"] = "a" * 40
    result["source_identity"]["git_dirty"] = None
    path.write_text(json.dumps(result))
    monkeypatch.setattr(checker.paper, "design", lambda mode: result["plan"])
    with pytest.raises(ValueError, match="clean committed provenance"):
        checker.check(path)


@pytest.mark.parametrize("fault, expected", [
    ("input", "input hash"),
    ("failed_count", "summary denominators"),
    ("duplicate", "duplicate case identity"),
])
def test_checker_rejects_retained_fault_classes(tmp_path, fault, expected):
    path = tmp_path / "pilot.json"
    result = paper.run("pilot")
    if fault == "input":
        result["records"][0]["input_sha256"] = "0" * 64
    elif fault == "failed_count":
        result["summaries"]["released_ncor2:cell0"]["failed"] = 1
    else:
        result["records"][1]["repetition"] = 0
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match=expected):
        checker.check(path)


def test_cross_platform_recheck_requires_exact_saved_input(tmp_path, monkeypatch):
    result_path = tmp_path / "pilot.json"
    supplement_path = tmp_path / "supplement.json"
    result = paper.run("pilot")
    result_path.write_text(json.dumps(result))
    key = ("released_ncor2", 0, 0)
    x, y = paper.generate(*key)
    row = {"identity": key, "input_sha256": paper.input_hash(x, y),
           "x_sha256": hashlib.sha256(x.astype("<i8").tobytes()).hexdigest(),
           "y_sha256": hashlib.sha256(y.astype("<f8").tobytes()).hexdigest(),
           "y": y.tolist()}
    supplement_path.write_text(json.dumps([row]))
    original = cross_checker.checker.paper.generate

    def locally_rounded(variant, cell, repetition, *, mode="pilot"):
        generated_x, generated_y = original(variant, cell, repetition, mode=mode)
        if (variant, cell, repetition) == key:
            generated_y[0] = np.nextafter(generated_y[0], np.inf)
        return generated_x, generated_y

    monkeypatch.setattr(cross_checker.checker.paper, "generate", locally_rounded)
    assert cross_checker.check(result_path, supplement_path)["verified_cases"] == 90
    row["y"][0] += 0.01
    supplement_path.write_text(json.dumps([row]))
    with pytest.raises(ValueError, match="supplementary outcome hash"):
        cross_checker.check(result_path, supplement_path)
