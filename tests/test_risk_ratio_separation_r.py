"""Fault checks for the optional installed-R separated-draw comparison."""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "validation"
sys.path.insert(0, str(VALIDATION))
spec = importlib.util.spec_from_file_location(
    "check_risk_ratio_separation_r", VALIDATION / "check_risk_ratio_separation_r.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
SAVED = json.loads((VALIDATION / "risk-ratio-separation-r-results.json").read_text())


def test_exact_retained_draw_and_saved_installed_r_result(tmp_path):
    x, z, y, rows = checker.exact_draw()
    path = tmp_path / "draw.csv"
    checker.write_draw(path, x, z, y, rows)
    assert checker.sha256(path) == SAVED["draw_csv_sha256"]
    assert checker.sha256(checker.DRIVER) == SAVED["r_driver_sha256"]
    with path.open(newline="") as handle:
        observed = list(csv.DictReader(handle))
    assert len(observed) == 500
    assert sum(int(row["y"]) for row in observed if row["z"] == "0") == 0
    checker.validate_r(SAVED["R"])
    assert SAVED["python_public_failure"].startswith("bootstrap replicate 83 failed:")


@pytest.mark.parametrize("field,value", [
    ("qgcomp_version", "2.19.5"),
    ("draw", {"rows": 499}),
    ("internal_warnings", ["glm did not converge"]),
    ("public_psi", 0.4),
    ("public_ci", [0.0]),
])
def test_changed_or_partial_installed_r_result_is_rejected(field, value):
    payload = copy.deepcopy(SAVED["R"])
    if field == "internal_warnings":
        payload["internal"]["warnings"] = value
    elif field == "public_psi":
        payload["public"]["psi"] = value
    elif field == "public_ci":
        payload["public"]["ci"] = value
    else:
        payload[field] = value
    with pytest.raises(ValueError):
        checker.validate_r(payload)


def test_missing_r_executable_does_not_publish_report(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["check_risk_ratio_separation_r.py",
                                   "--rscript", str(tmp_path / "missing-rscript"),
                                   "--library", str(tmp_path / "unused-library"),
                                   "--output", str(output)])
    with pytest.raises(FileNotFoundError):
        checker.main()
    assert not output.exists()


def test_existing_output_is_protected(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    output.write_text("previous report\n")
    monkeypatch.setattr(sys, "argv", ["check_risk_ratio_separation_r.py",
                                   "--rscript", "unused", "--library", "unused",
                                   "--output", str(output)])
    with pytest.raises(ValueError, match="already exists"):
        checker.main()
    assert output.read_text() == "previous report\n"
