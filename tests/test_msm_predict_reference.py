"""Pinned installed-R marginal prediction check and checker failure cases."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))
import check_msm_predict_r  # noqa: E402


def test_pinned_installed_r_reference():
    fixture = check_msm_predict_r.FIXTURE
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == check_msm_predict_r.SHA256
    saved = json.loads(fixture.read_text())
    assert check_msm_predict_r.compare(saved, saved) < 5e-8


def test_changed_input_rejected():
    saved = json.loads(check_msm_predict_r.FIXTURE.read_text())
    live = json.loads(check_msm_predict_r.FIXTURE.read_text())
    live["cases"][0]["inputs"]["new_scores"][1] += 1
    with pytest.raises(AssertionError):
        check_msm_predict_r.compare(saved, live)


def test_changed_r_prediction_rejected():
    saved = json.loads(check_msm_predict_r.FIXTURE.read_text())
    live = json.loads(check_msm_predict_r.FIXTURE.read_text())
    live["cases"][3]["expected"]["supplied"][2] += 1
    with pytest.raises(AssertionError):
        check_msm_predict_r.compare(saved, live)


def test_checker_reports_missing_r_executable(tmp_path):
    missing = tmp_path / "missing-Rscript"
    result = subprocess.run(
        [sys.executable, str(ROOT / "validation/check_msm_predict_r.py"),
         "--rscript", str(missing), "--library", str(tmp_path)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "FileNotFoundError" in result.stderr
