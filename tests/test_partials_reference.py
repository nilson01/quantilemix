"""Saved installed-R partial-effect evidence and checker failures."""

import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))
import check_partials_r  # noqa: E402


def test_saved_reference_and_hash():
    fixture = ROOT / "tests/fixtures/r_partials_2_19_6.json"
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == check_partials_r.SHA256
    saved = json.loads(fixture.read_text())
    check_partials_r.compare(saved, saved)


def test_checker_rejects_changed_reference_value():
    saved = json.loads(check_partials_r.FIXTURE.read_text())
    changed = json.loads(check_partials_r.FIXTURE.read_text())
    changed["positive_fit"]["psi"] += .01
    with pytest.raises(AssertionError):
        check_partials_r.compare(changed, saved)


def test_checker_rejects_changed_cluster_behavior():
    saved = json.loads(check_partials_r.FIXTURE.read_text())
    changed = json.loads(check_partials_r.FIXTURE.read_text())
    changed["cluster_split"]["overlapping_ids"] = []
    with pytest.raises(ValueError, match="cluster split"):
        check_partials_r.compare(changed, saved)
