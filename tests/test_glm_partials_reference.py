"""Installed-R GLM partial-effect fixture and optional checker failures."""

import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))
import check_glm_partials_r  # noqa: E402


def test_saved_reference_matches_checker_contract():
    assert hashlib.sha256(check_glm_partials_r.FIXTURE.read_bytes()).hexdigest() == check_glm_partials_r.SHA256
    saved = json.loads(check_glm_partials_r.FIXTURE.read_text())
    assert check_glm_partials_r.compare(saved, saved) < 2e-6


def test_checker_rejects_changed_reference_input():
    saved = json.loads(check_glm_partials_r.FIXTURE.read_text())
    changed = json.loads(check_glm_partials_r.FIXTURE.read_text())
    changed["cases"][0]["valid"]["y"][0] = 1 - changed["cases"][0]["valid"]["y"][0]
    with pytest.raises(AssertionError):
        check_glm_partials_r.compare(saved, changed)


def test_checker_rejects_changed_reference_result():
    saved = json.loads(check_glm_partials_r.FIXTURE.read_text())
    changed = json.loads(check_glm_partials_r.FIXTURE.read_text())
    changed["cases"][0]["positive_fit"]["psi"] += .01
    with pytest.raises(AssertionError):
        check_glm_partials_r.compare(saved, changed)


def test_optional_checker_reports_missing_rscript(tmp_path, monkeypatch):
    missing = tmp_path / "missing-Rscript"
    monkeypatch.setattr(sys, "argv", ["check_glm_partials_r.py", "--rscript", str(missing),
                                    "--library", str(tmp_path)])
    with pytest.raises(FileNotFoundError):
        check_glm_partials_r.main()
