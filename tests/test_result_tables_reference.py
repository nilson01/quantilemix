"""Saved installed-R tables stay distinct from live optional R execution."""

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "validation/check_result_tables_r.py"
SPEC = importlib.util.spec_from_file_location("check_result_tables_r", SCRIPT)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def test_saved_result_tables_agree_with_python():
    saved = json.loads(CHECK.SAVED.read_text())
    report = CHECK.check(saved)
    assert report["status"] == "passed"
    assert len(report["cases"]) == 7
    assert {case["family"] for case in report["cases"]} == {
        "gaussian", "binomial", "poisson", "cox"}


def test_checker_rejects_changed_r_table():
    saved = json.loads(CHECK.SAVED.read_text())
    changed = copy.deepcopy(saved)
    changed["cases"][0]["tidy"][1]["estimate"] += 0.1
    with pytest.raises(ValueError, match="saved_R"):
        CHECK.check(changed)


def test_missing_rscript_does_not_write_success(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["check_result_tables_r.py",
                        "--rscript", str(tmp_path / "missing-Rscript"),
                        "--library", str(tmp_path / "missing-library"),
                        "--output", str(output)])
    with pytest.raises(FileNotFoundError):
        CHECK.main()
    assert not output.exists()
