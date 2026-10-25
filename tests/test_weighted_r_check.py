"""Optional weighted installed-R checker boundary and failure behavior."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "validation/check_weighted_r.py"
SPEC = importlib.util.spec_from_file_location("weighted_r_check", SCRIPT)
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)
SAVED = CHECKER.load_json(CHECKER.FIXTURE)


def test_saved_weighted_cases_compare_against_the_public_python_fit():
    cases = CHECKER.cases_by_name(SAVED)
    for name in sorted(CHECKER.CASES):
        assert CHECKER.compare_case(cases[name], copy.deepcopy(cases[name]))["status"] == "passed"


def test_changed_reference_value_or_input_fails():
    saved = SAVED["cases"][0]
    changed = copy.deepcopy(saved)
    changed["expected"]["covariance"][0][0] += 0.1
    with pytest.raises(CHECKER.CheckError, match="covariance"):
        CHECKER.compare_case(saved, changed)
    changed = copy.deepcopy(saved)
    changed["inputs"]["weights"][0] *= 2
    with pytest.raises(CHECKER.CheckError, match="inputs changed"):
        CHECKER.compare_case(saved, changed)


def test_missing_case_and_duplicate_json_key_fail(tmp_path):
    with pytest.raises(CHECKER.CheckError, match="exactly the two"):
        CHECKER.cases_by_name({"cases": [SAVED["cases"][0]]})
    malformed = tmp_path / "duplicate.json"
    malformed.write_text('{"status":"ok","status":"bad"}')
    with pytest.raises(CHECKER.CheckError, match="duplicate JSON key"):
        CHECKER.load_json(malformed)


def test_nonfinite_or_nonpositive_weights_fail_even_if_reference_matches():
    saved = copy.deepcopy(SAVED["cases"][0])
    saved["inputs"]["weights"][0] = 0
    with pytest.raises(CHECKER.CheckError, match="positive finite"):
        CHECKER.compare_case(saved, copy.deepcopy(saved))


def test_missing_library_and_existing_output_fail_before_export(tmp_path):
    args = SimpleNamespace(fixture=str(CHECKER.FIXTURE), r_library=str(tmp_path / "missing"),
                           output=str(tmp_path / "result.json"), rscript="Rscript", timeout=60)
    with pytest.raises(CHECKER.CheckError, match="unavailable"):
        CHECKER.check(args)
    library = tmp_path / "r-lib"
    (library / "qgcomp").mkdir(parents=True)
    (library / "qgcomp/DESCRIPTION").write_text("Package: qgcomp\n")
    args.r_library = str(library)
    Path(args.output).write_text("owned")
    with pytest.raises(CHECKER.CheckError, match="already exists"):
        CHECKER.check(args)


def test_public_checker_accepts_valid_export_and_writes_once(tmp_path, monkeypatch):
    library = tmp_path / "r-lib"
    (library / "qgcomp").mkdir(parents=True)
    (library / "qgcomp/DESCRIPTION").write_text("Package: qgcomp\n")
    args = SimpleNamespace(fixture=str(CHECKER.FIXTURE), r_library=str(library),
                           output=str(tmp_path / "report.json"), rscript="Rscript", timeout=60)

    def fake_run(command, *, capture_output, text, timeout, check):
        assert command[1] == "--vanilla"
        assert timeout == 60 and capture_output and text and check is False
        payload = {"qgcomp_version": "2.19.6", "library": str(library),
                   "qgcomp_origin": str(library / "qgcomp"),
                   "R_version": "R test double", "cases": SAVED["cases"]}
        Path(command[-1]).write_text(json.dumps(payload))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(CHECKER, "run_export", fake_run)
    report = CHECKER.check(args)
    assert report["case_count"] == 2
    assert json.loads(Path(args.output).read_text())["status"] == "passed"
    with pytest.raises(CHECKER.CheckError, match="already exists"):
        CHECKER.check(args)


def test_export_failure_leaves_no_pass_report(tmp_path, monkeypatch):
    library = tmp_path / "r-lib"
    (library / "qgcomp").mkdir(parents=True)
    (library / "qgcomp/DESCRIPTION").write_text("Package: qgcomp\n")
    args = SimpleNamespace(fixture=str(CHECKER.FIXTURE), r_library=str(library),
                           output=str(tmp_path / "report.json"), rscript="Rscript", timeout=60)
    monkeypatch.setattr(CHECKER, "run_export", lambda *a, **k: SimpleNamespace(returncode=9))
    with pytest.raises(CHECKER.CheckError, match="exit code 9"):
        CHECKER.check(args)
    assert not Path(args.output).exists()
