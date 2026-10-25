"""Pinned installed-R dispatcher fixture and checker failure cases."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))
import check_qgcomp_dispatch_r  # noqa: E402


def test_pinned_installed_r_reference():
    fixture = check_qgcomp_dispatch_r.FIXTURE
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == check_qgcomp_dispatch_r.SHA256
    saved = json.loads(fixture.read_text())
    assert check_qgcomp_dispatch_r.compare(saved, saved) < 2e-6


def test_changed_route_rejected():
    saved = json.loads(check_qgcomp_dispatch_r.FIXTURE.read_text())
    live = json.loads(check_qgcomp_dispatch_r.FIXTURE.read_text())
    live["cases"][3]["expected"]["bootstrap"] = False
    with pytest.raises(ValueError, match="route"):
        check_qgcomp_dispatch_r.compare(saved, live)


def test_changed_input_rejected():
    saved = json.loads(check_qgcomp_dispatch_r.FIXTURE.read_text())
    live = json.loads(check_qgcomp_dispatch_r.FIXTURE.read_text())
    live["cases"][0]["inputs"]["x1"][0] += 1
    with pytest.raises(AssertionError):
        check_qgcomp_dispatch_r.compare(saved, live)


def test_checker_reports_missing_r_executable(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "validation/check_qgcomp_dispatch_r.py"),
         "--rscript", str(tmp_path / "missing-Rscript"), "--library", str(tmp_path)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "FileNotFoundError" in result.stderr
