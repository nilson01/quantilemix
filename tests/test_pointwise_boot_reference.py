"""Pinned installed-R pointwise bootstrap fixture checks."""

import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))
import check_pointwise_boot_r  # noqa: E402

FIXTURE = check_pointwise_boot_r.FIXTURE
SHA256 = check_pointwise_boot_r.SHA256
compare = check_pointwise_boot_r.compare


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


def test_changed_r_table_rejected():
    saved = json.loads(FIXTURE.read_text())
    live = json.loads(FIXTURE.read_text())
    live["cases"][3]["expected"]["table"]["rr"][1] += .01
    with pytest.raises(AssertionError):
        compare(saved, live)
