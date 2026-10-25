"""Pinned installed-R model-bound fixture and optional-checker fault tests."""

import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))
import check_modelbound_boot_r  # noqa: E402

FIXTURE = check_modelbound_boot_r.FIXTURE
SHA256 = check_modelbound_boot_r.SHA256
compare = check_modelbound_boot_r.compare


def test_pinned_installed_r_reference():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == SHA256
    saved = json.loads(FIXTURE.read_text())
    assert compare(saved, saved) < 2e-8


def test_changed_row_schedule_rejected():
    saved = json.loads(FIXTURE.read_text())
    live = json.loads(FIXTURE.read_text())
    live["cases"][0]["inputs"]["indices"][0][0] += 1
    with pytest.raises(AssertionError):
        compare(saved, live)


def test_changed_simultaneous_limit_rejected():
    saved = json.loads(FIXTURE.read_text())
    live = json.loads(FIXTURE.read_text())
    live["cases"][1]["expected"]["tables"][1]["table"]["ll.simul"][1] += .01
    with pytest.raises(AssertionError):
        compare(saved, live)


def test_wrong_installed_r_version_rejected():
    saved = json.loads(FIXTURE.read_text())
    live = json.loads(FIXTURE.read_text())
    live["qgcomp_version"] = "2.19.5"
    with pytest.raises(ValueError, match="2.19.6"):
        compare(saved, live)


def test_missing_r_executable_is_a_failure(monkeypatch):
    saved = json.loads(FIXTURE.read_text())
    monkeypatch.setattr(sys, "argv",
                        ["check_modelbound_boot_r.py", "--rscript", "/missing/Rscript",
                         "--library", "/missing/library"])
    monkeypatch.setattr(check_modelbound_boot_r, "FIXTURE", FIXTURE)
    assert compare(saved, saved) < 2e-8
    with pytest.raises(FileNotFoundError):
        check_modelbound_boot_r.main()
