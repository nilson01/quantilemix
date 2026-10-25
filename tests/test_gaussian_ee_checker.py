"""The optional Gaussian EE checker rejects changed cases without R in CI."""

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "validation/check_gaussian_ee_r.py"
SPEC = importlib.util.spec_from_file_location("check_gaussian_ee_r", SCRIPT)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


@pytest.mark.parametrize("change", ["input", "id", "r_output", "version"])
def test_gaussian_ee_checker_rejects_changed_live_case(change):
    saved = json.loads(CHECK.FIXTURE.read_text())
    live = copy.deepcopy(saved)
    if change == "input":
        live["cases"][0]["inputs"]["x1"][0] += 1
    elif change == "id":
        live["cases"][0]["inputs"]["id"][0] += 1
    elif change == "r_output":
        live["cases"][0]["expected"]["variance"] += .1
    else:
        live["qgcomp_version"] = "2.19.5"
    with pytest.raises(ValueError):
        CHECK.check(saved, live)


def test_gaussian_ee_checker_accepts_unchanged_saved_case():
    saved = json.loads(CHECK.FIXTURE.read_text())
    result = CHECK.check(saved, copy.deepcopy(saved))
    assert result["status"] == "passed"
    assert len(result["cases"]) == 2


def test_gaussian_ee_checker_rejects_nonfinite_reference_value():
    with pytest.raises(ValueError, match="finiteness"):
        CHECK.compare_value([1.], [float("nan")], "invalid", atol=1e-10)


def test_gaussian_ee_checker_refuses_existing_output(tmp_path, monkeypatch):
    target = tmp_path / "existing.json"
    target.write_text("preserve me")
    monkeypatch.setattr(sys, "argv", ["check_gaussian_ee_r.py", "--rscript", "missing-r",
                                     "--library", "missing-library", "--output", str(target)])
    with pytest.raises(ValueError, match="already exists"):
        CHECK.main()
    assert target.read_text() == "preserve me"


def test_gaussian_ee_checker_reports_missing_r_runtime(tmp_path, monkeypatch):
    target = tmp_path / "new.json"
    monkeypatch.setattr(sys, "argv", ["check_gaussian_ee_r.py", "--rscript",
                                     str(tmp_path / "missing-r"), "--library", "missing-library",
                                     "--output", str(target)])
    with pytest.raises(FileNotFoundError):
        CHECK.main()
    assert not target.exists()
