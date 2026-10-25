"""Pinned installed-R pointwise fixture and optional-checker failure tests."""

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))
import check_pointwise_noboot_r  # noqa: E402


SAVED = json.loads(check_pointwise_noboot_r.FIXTURE.read_text())


def test_saved_pointwise_fixture_agrees_with_python():
    assert hashlib.sha256(check_pointwise_noboot_r.FIXTURE.read_bytes()).hexdigest() == check_pointwise_noboot_r.SHA256
    assert check_pointwise_noboot_r.compare(SAVED, SAVED) < 2e-6


def test_checker_rejects_changed_input():
    live = copy.deepcopy(SAVED)
    live["cases"][0]["inputs"]["y"][0] += 1
    with pytest.raises(AssertionError):
        check_pointwise_noboot_r.compare(SAVED, live)


def test_checker_rejects_changed_reference_output():
    live = copy.deepcopy(SAVED)
    live["cases"][1]["expected"]["table"]["ll.diff"][0] += .01
    with pytest.raises(AssertionError):
        check_pointwise_noboot_r.compare(SAVED, live)


def test_optional_checker_reports_missing_r_executable(tmp_path, monkeypatch):
    missing = tmp_path / "missing-Rscript"
    monkeypatch.setattr(sys, "argv", ["check_pointwise_noboot_r.py", "--rscript", str(missing),
                                    "--library", str(tmp_path)])
    with pytest.raises(FileNotFoundError):
        check_pointwise_noboot_r.main()
