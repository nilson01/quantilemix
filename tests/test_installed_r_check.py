"""Checker contract tests with synthetic payloads; these do not execute R."""

from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/r_gaussian_2_19_6.json"


def input_column_names(case):
    def column_count(values):
        if values is None:
            return 0
        return len(values[0]) if isinstance(values[0], list) else 1

    inputs = case["inputs"]
    exposures = [f"x{i + 1}" for i in range(column_count(inputs["exposures"]))]
    covariates = [f"z{i + 1}" for i in range(column_count(inputs["covariates"]))]
    return {"coefficients": ["(Intercept)", *exposures, *covariates], "exposures": exposures}


def synthetic_summary(case):
    """Model the exporter protocol using fixture values, without executing R."""
    expected = case["expected"]
    intercept = expected["coefficients"][0]
    intercept_se = math.sqrt(expected["covariance"][0][0])
    critical = stats.norm.isf(case["inputs"]["alpha"] / 2)
    intercept_statistic = intercept / intercept_se
    return {
        "status": "ok",
        "row_names": ["(Intercept)", "psi1"],
        "column_names": ["Estimate", "Std. Error", "Lower CI", "Upper CI", "t value", "Pr(>|t|)"],
        "values": [
            [intercept, intercept_se, intercept - critical * intercept_se,
             intercept + critical * intercept_se, intercept_statistic,
             float(2 * stats.t.sf(abs(intercept_statistic), expected["df_pvalue"]))],
            [expected["psi"], expected["se"], *expected["ci"],
             expected["statistic"], expected["p_value"]],
        ],
        "output": ["Mixture slope parameters (delta method CI):", ""],
        "warnings": [],
    }


@pytest.fixture(scope="module")
def checker():
    path = ROOT / "validation/check_installed_r.py"
    name = "_qgcomp_installed_r_checker_tests"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


@pytest.fixture
def reference_fixtures():
    return json.loads(FIXTURES.read_text())


@pytest.fixture
def installed_payload(reference_fixtures):
    # Fixture values model the exporter protocol, not a new installed-R result.
    return {
        "reference": {
            "package": "qgcomp",
            "version": "2.19.6",
            "mode": "installed package",
            "library_verified": True,
            "R_version": "R version 4.5.1 (synthetic test metadata)",
            "platform": "aarch64-apple-darwin20",
            "package_versions": {"qgcomp": "2.19.6", "jsonlite": "2.0.0"},
            "installed_package_md5": {
                name: "a" * 32
                for name in ("DESCRIPTION", "NAMESPACE", "R/qgcomp.rdb", "R/qgcomp.rdx")
            },
            "r_runtime_md5": {"bin/exec/R": "b" * 32},
        },
        "cases": [
            {
                "name": case["name"],
                "expected": deepcopy(case["expected"]),
                "column_names": input_column_names(case),
                "wrapper_prediction": {
                    "status": "ok",
                    "values": deepcopy(case["expected"]["predictions"]),
                },
                "registered_summary": synthetic_summary(case),
            }
            for case in reference_fixtures["cases"]
        ],
    }


def comparison(report, case, kind):
    rows = [
        row for row in report["comparisons"]
        if row["case"] == case and row["comparison"] == kind
    ]
    assert len(rows) == 1
    return rows[0]


def test_valid_payload_checks_every_case_against_both_oracles(
    checker, installed_payload, reference_fixtures
):
    report = checker.compare_reference(installed_payload, reference_fixtures)
    assert report["status"] == "passed"
    assert report["case_count"] == len(reference_fixtures["cases"]) == 9
    assert not report["errors"]
    for case in reference_fixtures["cases"]:
        for kind in ("stored_fixture", "native_python", "registered_prediction", "registered_summary"):
            row = comparison(report, case["name"], kind)
            assert row["status"] == "passed"
            assert not row["errors"]


@pytest.mark.parametrize(
    "text",
    [
        "{",
        '{"version":"2.19.6","version":"2.20.0"}',
        '{"outer":{"value":1,"value":2}}',
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"value":-Infinity}',
        '{"value":1e309}',
    ],
)
def test_json_loader_rejects_ambiguous_or_nonfinite_json(checker, tmp_path, text):
    path = tmp_path / "invalid.json"
    path.write_text(text)
    with pytest.raises(checker.CheckError):
        checker.load_json(path)


def test_json_loader_preserves_valid_metadata_and_numbers(checker, tmp_path):
    path = tmp_path / "valid.json"
    value = {"library_verified": True, "values": [0, -1.25, 2.5e-10], "breaks": None}
    path.write_text(json.dumps(value))
    assert checker.load_json(path) == value


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("package", "different-package"),
        ("version", "2.19.5"),
        ("mode", "unmodified source functions; not installed package"),
        ("library_verified", False),
        ("library_verified", 1),
        ("R_version", ""),
        ("platform", ""),
        ("package_versions", []),
    ],
)
def test_false_reference_identity_is_rejected(
    checker, installed_payload, reference_fixtures, key, value
):
    installed_payload["reference"][key] = value
    with pytest.raises(checker.CheckError):
        checker.compare_reference(installed_payload, reference_fixtures)


def test_required_reference_identity_cannot_be_omitted(
    checker, installed_payload, reference_fixtures
):
    del installed_payload["reference"]["library_verified"]
    with pytest.raises(checker.CheckError):
        checker.compare_reference(installed_payload, reference_fixtures)


@pytest.mark.parametrize("field", ["installed_package_md5", "r_runtime_md5"])
def test_installed_package_and_runtime_hashes_are_required(
    checker, installed_payload, reference_fixtures, field
):
    del installed_payload["reference"][field]
    with pytest.raises(checker.CheckError):
        checker.compare_reference(installed_payload, reference_fixtures)


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_installed_package_hash_inventory_is_exact(
    checker, installed_payload, reference_fixtures, change
):
    hashes = installed_payload["reference"]["installed_package_md5"]
    if change == "missing":
        del hashes["R/qgcomp.rdb"]
    else:
        hashes["unexpected"] = "c" * 32
    with pytest.raises(checker.CheckError):
        checker.compare_reference(installed_payload, reference_fixtures)


@pytest.mark.parametrize("field", ["installed_package_md5", "r_runtime_md5"])
@pytest.mark.parametrize("value", ["a" * 31, "A" * 32, "g" * 32, None])
def test_identity_hashes_are_lowercase_md5_hex(
    checker, installed_payload, reference_fixtures, field, value
):
    hashes = installed_payload["reference"][field]
    hashes[next(iter(hashes))] = value
    with pytest.raises(checker.CheckError):
        checker.compare_reference(installed_payload, reference_fixtures)


def test_runtime_hash_inventory_cannot_be_empty(
    checker, installed_payload, reference_fixtures
):
    installed_payload["reference"]["r_runtime_md5"] = {}
    with pytest.raises(checker.CheckError):
        checker.compare_reference(installed_payload, reference_fixtures)


@pytest.mark.parametrize("path", ["", "/bin/exec/R", "../exec/R", "bin/../exec/R"])
def test_runtime_hash_paths_are_safe_and_relative(
    checker, installed_payload, reference_fixtures, path
):
    installed_payload["reference"]["r_runtime_md5"] = {path: "b" * 32}
    with pytest.raises(checker.CheckError):
        checker.compare_reference(installed_payload, reference_fixtures)


@pytest.mark.parametrize("field", ["column_names", "coefficients", "exposures"])
def test_case_column_identity_is_required(
    checker, installed_payload, reference_fixtures, field
):
    case = installed_payload["cases"][0]
    if field == "column_names":
        del case[field]
    else:
        del case["column_names"][field]
    with pytest.raises(checker.CheckError):
        checker.compare_reference(installed_payload, reference_fixtures)


@pytest.mark.parametrize("group", ["exposures", "exposure_coefficients", "covariate_coefficients"])
def test_case_column_identity_preserves_input_order(
    checker, installed_payload, reference_fixtures, group
):
    case = next(c for c in installed_payload["cases"] if c["name"] == "adjusted_positive")
    if group == "exposures":
        names = case["column_names"]["exposures"]
        names[0], names[1] = names[1], names[0]
    else:
        names = case["column_names"]["coefficients"]
        left, right = (1, 2) if group == "exposure_coefficients" else (-2, -1)
        names[left], names[right] = names[right], names[left]
    with pytest.raises(checker.CheckError):
        checker.compare_reference(installed_payload, reference_fixtures)


@pytest.mark.parametrize("change", ["empty", "duplicate", "missing", "extra"])
def test_case_inventory_must_match_exactly(
    checker, installed_payload, reference_fixtures, change
):
    cases = installed_payload["cases"]
    if change == "empty":
        cases.clear()
    elif change == "duplicate":
        cases[-1] = deepcopy(cases[0])
    elif change == "missing":
        cases.pop()
    else:
        extra = deepcopy(cases[0])
        extra["name"] = "unexpected_case"
        cases.append(extra)
    with pytest.raises(checker.CheckError):
        checker.compare_reference(installed_payload, reference_fixtures)


@pytest.mark.parametrize("field", ["coefficients", "covariance", "psi", "predictions"])
def test_required_numerical_outputs_cannot_be_omitted(
    checker, installed_payload, reference_fixtures, field
):
    del installed_payload["cases"][0]["expected"][field]
    with pytest.raises(checker.CheckError):
        checker.compare_reference(installed_payload, reference_fixtures)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("psi", True),
        ("psi", None),
        ("psi", float("nan")),
        ("variance", float("inf")),
        ("coefficients", [True, 0.3]),
        ("coefficients", [1.0]),
        ("covariance", [[1.0, 0.0]]),
        ("scores", [0.0, 1.0]),
        ("predictions", [[1.0] * 7]),
        ("positive_weights", {"x1": True}),
    ],
)
def test_numerical_outputs_require_finite_numbers_and_exact_shapes(
    checker, installed_payload, reference_fixtures, field, value
):
    installed_payload["cases"][0]["expected"][field] = value
    with pytest.raises(checker.CheckError):
        checker.compare_reference(installed_payload, reference_fixtures)


def test_installed_result_disagreement_fails_both_independent_comparisons(
    checker, installed_payload, reference_fixtures
):
    case = installed_payload["cases"][0]
    case["expected"]["psi"] += 1.0
    report = checker.compare_reference(installed_payload, reference_fixtures)
    assert report["status"] == "failed"
    assert report["errors"]
    for kind in ("stored_fixture", "native_python"):
        row = comparison(report, case["name"], kind)
        assert row["status"] == "failed"
        assert row["errors"]


def test_agreeing_modified_fixture_does_not_replace_native_python_comparison(
    checker, installed_payload, reference_fixtures
):
    # A shared change to R and the saved oracle must not hide a Python mismatch.
    case = installed_payload["cases"][0]
    case["expected"]["psi"] += 1.0
    reference_fixtures["cases"][0]["expected"]["psi"] += 1.0
    report = checker.compare_reference(installed_payload, reference_fixtures)
    assert report["status"] == "failed"
    assert comparison(report, case["name"], "stored_fixture")["status"] == "passed"
    assert comparison(report, case["name"], "native_python")["status"] == "failed"


def test_registered_prediction_is_checked_separately(
    checker, installed_payload, reference_fixtures
):
    case = installed_payload["cases"][0]
    case["wrapper_prediction"]["values"][0] += 1.0
    report = checker.compare_reference(installed_payload, reference_fixtures)
    assert report["status"] == "failed"
    assert comparison(report, case["name"], "stored_fixture")["status"] == "passed"
    assert comparison(report, case["name"], "native_python")["status"] == "passed"
    assert comparison(report, case["name"], "registered_prediction")["status"] == "failed"


def test_known_prequantized_wrapper_error_does_not_skip_numerical_checks(
    checker, installed_payload, reference_fixtures
):
    case = next(c for c in installed_payload["cases"] if c["name"] == "prequantized")
    case["wrapper_prediction"] = {
        "status": "error", "error": "'by' must be of length 1", "error_class": "simpleError"
    }
    report = checker.compare_reference(installed_payload, reference_fixtures)
    assert report["status"] == "passed"
    for kind in ("stored_fixture", "native_python"):
        assert comparison(report, "prequantized", kind)["status"] == "passed"
    case["expected"]["predictions"][0] += 1.0
    report = checker.compare_reference(installed_payload, reference_fixtures)
    assert report["status"] == "failed"
    for kind in ("stored_fixture", "native_python"):
        assert comparison(report, "prequantized", kind)["status"] == "failed"


@pytest.mark.parametrize("change", ["other_case", "other_message", "other_class"])
def test_wrapper_error_allowance_is_narrow(
    checker, installed_payload, reference_fixtures, change
):
    case = next(c for c in installed_payload["cases"] if c["name"] == "prequantized")
    if change == "other_case":
        case = installed_payload["cases"][0]
    error = {"status": "error", "error": "'by' must be of length 1", "error_class": "simpleError"}
    if change == "other_message":
        error["error"] = "unrelated prediction failure"
    if change == "other_class":
        error["error_class"] = "differentError"
    case["wrapper_prediction"] = error
    report = checker.compare_reference(installed_payload, reference_fixtures)
    assert report["status"] == "failed"
    assert comparison(report, case["name"], "registered_prediction")["status"] == "failed"


def test_registered_summary_omitted_t_column_is_explicit_and_values_still_checked(
    checker, installed_payload, reference_fixtures
):
    case = installed_payload["cases"][0]
    outcome = case["registered_summary"]
    outcome["column_names"].pop(4)
    for row in outcome["values"]:
        row.pop(4)
    report = checker.compare_reference(installed_payload, reference_fixtures)
    assert report["status"] == "passed"
    row = comparison(report, case["name"], "registered_summary")
    assert row["status"] == "known_reference_limitation"
    assert row["observed"] == outcome
    assert report["known_limitations"][0]["missing_columns"] == ["t value"]
    outcome["values"][1][0] += 1
    report = checker.compare_reference(installed_payload, reference_fixtures)
    assert report["status"] == "failed"
    assert comparison(report, case["name"], "registered_summary")["status"] == "failed"
    assert comparison(report, case["name"], "native_python")["status"] == "passed"


@pytest.mark.parametrize("row", [0, 1])
@pytest.mark.parametrize("column", range(6))
def test_registered_summary_values_are_compared_independently(
    checker, installed_payload, reference_fixtures, row, column
):
    case = installed_payload["cases"][0]
    case["registered_summary"]["values"][row][column] += 0.1
    report = checker.compare_reference(installed_payload, reference_fixtures)
    assert report["status"] == "failed"
    summary = comparison(report, case["name"], "registered_summary")
    assert summary["status"] == "failed"
    assert any("stored_fixture" in error for error in summary["errors"])
    assert any("native_python" in error for error in summary["errors"])
    assert comparison(report, case["name"], "stored_fixture")["status"] == "passed"


@pytest.mark.parametrize("field", [None, "status", "row_names", "column_names", "values", "output", "warnings"])
def test_registered_summary_required_evidence_cannot_be_omitted(
    checker, installed_payload, reference_fixtures, field
):
    case = installed_payload["cases"][0]
    if field is None:
        del case["registered_summary"]
    else:
        del case["registered_summary"][field]
    with pytest.raises(checker.CheckError):
        checker.compare_reference(installed_payload, reference_fixtures)


@pytest.mark.parametrize(
    ("field", "value"),
    [("row_names", ["psi1", "(Intercept)"]),
     ("column_names", ["Estimate", "Std. Error", "Upper CI", "Lower CI", "t value", "Pr(>|t|)"]),
     ("column_names", ["Estimate", "Std. Error", "Lower CI", "Upper CI"]),
     ("values", []), ("values", [[1, 2, 3, 4, 5, 6]]),
     ("output", "Mixture slope parameters (delta method CI):"),
     ("warnings", None), ("warnings", [False])],
)
def test_registered_summary_malformed_evidence_is_rejected(
    checker, installed_payload, reference_fixtures, field, value
):
    installed_payload["cases"][0]["registered_summary"][field] = value
    with pytest.raises(checker.CheckError):
        checker.compare_reference(installed_payload, reference_fixtures)


@pytest.mark.parametrize("value", [None, True, float("nan"), float("inf")])
def test_registered_summary_values_require_finite_numbers(
    checker, installed_payload, reference_fixtures, value
):
    installed_payload["cases"][0]["registered_summary"]["values"][1][0] = value
    with pytest.raises(checker.CheckError):
        checker.compare_reference(installed_payload, reference_fixtures)


@pytest.mark.parametrize("change", ["error", "warning", "wrong_header"])
def test_registered_summary_runtime_problems_are_retained_and_fail(
    checker, installed_payload, reference_fixtures, change
):
    case = installed_payload["cases"][0]
    outcome = case["registered_summary"]
    if change == "error":
        outcome = {"status": "error", "error": "synthetic summary failure",
                   "error_class": "simpleError", "output": outcome["output"], "warnings": []}
        case["registered_summary"] = outcome
    elif change == "warning":
        outcome["warnings"] = ["synthetic summary warning"]
    else:
        outcome["output"] = ["Mixture slope parameters (bootstrap CI):", ""]
    report = checker.compare_reference(installed_payload, reference_fixtures)
    assert report["status"] == "failed"
    summary = comparison(report, case["name"], "registered_summary")
    assert summary["status"] == "failed"
    assert summary["observed"] == outcome
    assert comparison(report, case["name"], "native_python")["status"] == "passed"


def cli_arguments(tmp_path):
    library = tmp_path / "r-library"
    library.mkdir()
    fixtures = tmp_path / "fixtures.json"
    fixtures.write_bytes(FIXTURES.read_bytes())
    output = tmp_path / "report.json"
    return [
        "--r-library", str(library), "--rscript", "synthetic-Rscript",
        "--fixtures", str(fixtures), "--output", str(output), "--timeout", "1",
    ], output


def test_cli_success_uses_export_file_and_writes_report(
    checker, installed_payload, monkeypatch, tmp_path
):
    args, output = cli_arguments(tmp_path)
    calls = []

    def run(command, **kwargs):
        assert command[0] == "synthetic-Rscript"
        calls.append(command)
        Path(command[-1]).write_text(json.dumps(installed_payload))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(checker, "run_export", run)
    assert checker.main(args) == 0
    assert len(calls) == 1
    report = json.loads(output.read_text())
    assert report["status"] == "passed"
    assert report["case_count"] == 9


def assert_error_paths_absent(report, private_paths):
    errors_text = "\n".join(report["errors"])
    for path in private_paths:
        assert path not in errors_text
    return errors_text


def test_path_redaction_guard_rejects_windows_paths():
    private_path = r"C:\Users\runner\private-library"
    leaked = json.loads(json.dumps({"errors": [f"failed at {private_path}"]}))
    with pytest.raises(AssertionError):
        assert_error_paths_absent(leaked, {private_path})
    assert_error_paths_absent({"errors": ["failed at <local-path>"]}, {private_path})


@pytest.mark.parametrize("failure", ["missing_executable", "nonzero", "malformed", "timeout"])
def test_cli_execution_failures_write_failed_reports(
    checker, monkeypatch, tmp_path, failure
):
    args, output = cli_arguments(tmp_path)
    if failure == "timeout":
        args[args.index("--rscript") + 1] = "/private/rscript"
    private_paths = {
        "/private/rscript", str(tmp_path),
        args[args.index("--r-library") + 1], args[args.index("--fixtures") + 1],
    }

    def run(command, **kwargs):
        private_paths.add(str(Path(command[-1]).parent))
        if failure == "missing_executable":
            raise FileNotFoundError(2, "synthetic Rscript missing", "/private/rscript")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 1, output=command[-1], stderr=command[-3])
        if failure == "nonzero":
            detail = f"synthetic execution failed: {command[-3]} {command[-2]} {command[-1]}"
            return subprocess.CompletedProcess(command, 1, stdout="", stderr=detail)
        assert command[0] == "synthetic-Rscript"
        Path(command[-1]).write_text('{"invalid":')
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(checker, "run_export", run)
    assert checker.main(args) != 0
    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert report["errors"]
    errors_text = assert_error_paths_absent(report, private_paths)
    if failure == "missing_executable":
        assert "FileNotFoundError" in errors_text
        assert "synthetic Rscript missing" not in errors_text
    if failure == "nonzero":
        assert "synthetic execution failed" in errors_text


@pytest.mark.parametrize("protected", ["fixtures", "source", "previous_report"])
def test_cli_never_overwrites_existing_evidence_or_source(
    checker, monkeypatch, tmp_path, protected
):
    args, _ = cli_arguments(tmp_path)
    if protected == "fixtures":
        target = Path(args[args.index("--fixtures") + 1])
    else:
        target = tmp_path / ("check_installed_r.py" if protected == "source" else "previous_report.json")
        target.write_text("preserve these exact bytes\n")
    before = target.read_bytes()
    args[args.index("--output") + 1] = str(target)

    def forbidden_run(*args, **kwargs):
        pytest.fail("protected output must be rejected before starting R")

    monkeypatch.setattr(checker, "run_export", forbidden_run)
    assert checker.main(args) == 2
    assert target.read_bytes() == before


def test_cli_refuses_dangling_output_symlink_without_creating_target(
    checker, monkeypatch, tmp_path
):
    args, output = cli_arguments(tmp_path)
    target = tmp_path / "must-not-be-created.json"
    try:
        output.symlink_to(target)
    except (OSError, NotImplementedError) as exc:
        if sys.platform == "win32":
            pytest.skip(f"symlink creation is unavailable: {exc}")
        raise

    def forbidden_run(*args, **kwargs):
        pytest.fail("dangling output symlink must be rejected before starting R")

    monkeypatch.setattr(checker, "run_export", forbidden_run)
    assert checker.main(args) == 2
    assert output.is_symlink()
    assert not target.exists()
