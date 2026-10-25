"""Public adapter checks with a fake process; these do not execute R."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def adapter():
    name = "_qgcomp_calibration_reference_tests"
    spec = importlib.util.spec_from_file_location(name, ROOT / "validation/calibration_reference.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


@pytest.fixture
def library(tmp_path):
    path = tmp_path / "isolated-library"
    path.mkdir()
    return path


@pytest.fixture
def cases():
    # These values exercise serialization only, not statistical accuracy.
    return [{
        "name": "fixed-input",
        "inputs": {
            "exposures": [[0], [1], [2], [3], [0]],
            "covariates": [[0], [1], [0], [1], [2]],
            "y": [1, 2, 4, 7, 3], "q": None, "alpha": 0.05,
        },
    }]


@pytest.fixture
def reference():
    return {
        "package": "qgcomp", "version": "2.19.6", "mode": "installed package",
        "R_version": "R version 4.5.1 (synthetic metadata)", "platform": "synthetic-test",
        "library_verified": True,
        "installed_packages": {
            "qgcomp": {"version": "2.19.6", "Built": "R 4.5.1; synthetic-test"},
            "jsonlite": {"version": "2.0.0", "Built": None},
        },
        "installed_package_md5": {
            label: "a" * 32
            for label in ("DESCRIPTION", "NAMESPACE", "R/qgcomp.rdb", "R/qgcomp.rdx")
        },
        "r_runtime_md5": {"bin/exec/R": "b" * 32},
        "package_versions": {"qgcomp": "2.19.6", "jsonlite": "2.0.0", "base": "4.5.1"},
        "package_library_origins": {
            "qgcomp": "isolated", "jsonlite": "isolated", "base": "base-recommended",
        },
        "unexpected_loaded_namespaces": [],
    }


def success(name="fixed-input"):
    return {
        "name": name, "status": "ok", "warnings": [],
        "result": {
            "psi": 2.0, "variance": 0.25, "se": 0.5, "ci": [1.02, 2.98],
            "p_value": 0.1, "df_pvalue": 3, "df_residual": 2,
        },
    }


def fake_export(monkeypatch, adapter, payload, visited):
    def run(command, **kwargs):
        visited.append(Path(command[-2]).parent)
        Path(command[-1]).write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
    monkeypatch.setattr(adapter, "run_r", run)


def test_public_success_uses_private_files_isolation_and_one_process(
    adapter, monkeypatch, library, cases, reference
):
    cases.append({**deepcopy(cases[0]), "name": "second-input"})
    original = deepcopy(cases)
    payload = {"reference": reference, "cases": [success("second-input"), success()]}
    payload["cases"][1]["warnings"] = ["Reference warning retained"]
    visited = []
    system_run = subprocess.run

    def run(command, **kwargs):
        assert command[1] == "--vanilla"
        assert Path(command[2]) == adapter.DRIVER.resolve()
        assert Path(command[3]) == library.resolve()
        input_file, output_file = map(Path, command[-2:])
        assert input_file.parent == output_file.parent == kwargs["cwd"]
        assert not output_file.exists()
        assert json.loads(input_file.read_text()) == {"cases": original}
        assert kwargs["stdin"] == subprocess.DEVNULL
        assert kwargs["timeout"] == 15
        assert kwargs["capture_output"] is True and kwargs["check"] is False
        for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                         "VECLIB_MAXIMUM_THREADS"):
            assert kwargs["env"][variable] == "1"
        for variable in ("R_LIBS", "R_LIBS_USER", "R_LIBS_SITE"):
            assert kwargs["env"][variable] == ""
        visited.append(input_file.parent)
        output_file.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(adapter, "run_r", run)
    result = adapter.evaluate_r_batch(cases, sys.executable, library, timeout=15)
    assert result["reference"] == reference
    assert [case["name"] for case in result["cases"]] == ["fixed-input", "second-input"]
    assert result["cases"][0]["warnings"] == ["Reference warning retained"]
    assert cases == original
    assert len(visited) == 1 and not visited[0].exists()
    assert subprocess.run is system_run  # No global subprocess patch.


def test_empty_batch_returns_identity_only(adapter, monkeypatch, library, reference):
    visited = []
    fake_export(monkeypatch, adapter, {"reference": reference, "cases": []}, visited)
    assert adapter.evaluate_r_batch([], sys.executable, library) == {
        "reference": reference, "cases": [],
    }
    assert not visited[0].exists()


def test_twenty_cases_are_allowed(adapter, monkeypatch, library, reference, cases):
    batch = [{**deepcopy(cases[0]), "name": f"case-{index}"} for index in range(20)]
    payload = {"reference": reference, "cases": [success(case["name"]) for case in batch]}
    visited = []
    fake_export(monkeypatch, adapter, payload, visited)
    assert len(adapter.evaluate_r_batch(batch, sys.executable, library)["cases"]) == 20


def test_finite_unexpected_df_is_retained_with_valid_sibling(adapter, monkeypatch, library, reference, cases):
    cases.append({**deepcopy(cases[0]), "name": "second"})
    rows = [success(), success("second")]
    rows[1]["result"]["df_pvalue"] = 2
    fake_export(monkeypatch, adapter, {"reference": reference, "cases": rows}, [])
    result = adapter.evaluate_r_batch(cases, sys.executable, library)
    assert result["cases"] == rows


def test_unadjusted_case_has_matching_degrees_of_freedom(
    adapter, monkeypatch, library, reference, cases
):
    cases[0]["inputs"]["covariates"] = None
    record = success()
    record["result"]["df_residual"] = 3
    visited = []
    fake_export(monkeypatch, adapter, {"reference": reference, "cases": [record]}, visited)
    assert adapter.evaluate_r_batch(cases, sys.executable, library)["cases"] == [record]


def test_fit_failure_and_prior_warnings_are_preserved(
    adapter, monkeypatch, library, reference, cases
):
    failure = {
        "name": "fixed-input", "status": "error", "error": "The reference design is not full rank",
        "error_class": "simpleError", "warnings": ["Prior warning"],
    }
    visited = []
    fake_export(monkeypatch, adapter, {"reference": reference, "cases": [failure]}, visited)
    assert adapter.evaluate_r_batch(cases, sys.executable, library)["cases"] == [failure]
    assert not visited[0].exists()


@pytest.mark.parametrize("malformation", [
    "not-list", "too-many", "duplicate-name", "non-string-name", "extra-input",
    "missing-q", "quantization", "ragged", "boolean", "infinite", "fractional-score",
    "negative-score", "large-score", "huge-integer", "wrong-y", "wrong-covariates",
    "no-residual-df", "alpha",
])
def test_malformed_inputs_fail_before_launch(adapter, monkeypatch, library, cases, malformation):
    inputs = cases[0]["inputs"]
    if malformation == "not-list":
        cases = tuple(cases)
    elif malformation == "too-many":
        cases *= 21
    elif malformation == "duplicate-name":
        cases *= 2
    elif malformation == "non-string-name":
        cases[0]["name"] = []
    elif malformation == "extra-input":
        inputs["weights"] = [1] * 5
    elif malformation == "missing-q":
        del inputs["q"]
    elif malformation == "quantization":
        inputs["q"] = 4
    elif malformation == "ragged":
        inputs["exposures"][1].append(2)
    elif malformation == "boolean":
        inputs["exposures"][0][0] = True
    elif malformation == "infinite":
        inputs["y"][0] = float("inf")
    elif malformation == "fractional-score":
        inputs["exposures"][0][0] = 0.5
    elif malformation == "negative-score":
        inputs["exposures"][0][0] = -1
    elif malformation == "large-score":
        inputs["exposures"][0][0] = 2**53
    elif malformation == "huge-integer":
        inputs["exposures"][0][0] = 10**1000
    elif malformation == "wrong-y":
        inputs["y"].pop()
    elif malformation == "wrong-covariates":
        inputs["covariates"].pop()
    elif malformation == "no-residual-df":
        for key in ("exposures", "covariates", "y"):
            inputs[key] = inputs[key][:3]
    else:
        inputs["alpha"] = 0
    monkeypatch.setattr(adapter, "run_r", lambda *a, **k: pytest.fail("Must not launch R"))
    with pytest.raises(adapter.RReferenceError):
        adapter.evaluate_r_batch(cases, sys.executable, library)


def test_missing_executable_is_a_clear_dependency_failure(adapter, library):
    with pytest.raises(adapter.RReferenceError, match="Rscript executable was not found") as error:
        adapter.evaluate_r_batch([], library / "missing-rscript", library)
    assert error.value.code == "r_missing_executable"


def test_missing_library_fails_before_launch(adapter, monkeypatch, library):
    monkeypatch.setattr(adapter, "run_r", lambda *a, **k: pytest.fail("Must not launch R"))
    with pytest.raises(adapter.RReferenceError, match="configuration is invalid") as error:
        adapter.evaluate_r_batch([], sys.executable, library / "missing")
    assert error.value.code == "r_invalid_configuration"


@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf")])
def test_invalid_timeout_is_rejected(adapter, library, timeout):
    with pytest.raises(adapter.RReferenceError, match="timeout"):
        adapter.evaluate_r_batch([], sys.executable, library, timeout=timeout)


@pytest.mark.parametrize("failure", ["timeout", "launch", "dependency", "missing-output"])
def test_process_failures_are_clean_and_remove_temporary_files(
    adapter, monkeypatch, library, cases, failure
):
    visited = []

    def run(command, **kwargs):
        visited.append(Path(command[-2]).parent)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"], stderr="private details")
        if failure == "launch":
            raise OSError(f"Cannot launch {library}/private-file")
        if failure == "dependency":
            return subprocess.CompletedProcess(
                command, 1, stdout="", stderr=f"jsonlite is required in {library}; no packages were installed"
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(adapter, "run_r", run)
    with pytest.raises(adapter.RReferenceError) as error:
        adapter.evaluate_r_batch(cases, sys.executable, library)
    assert str(library) not in str(error.value)
    assert error.value.code == {"timeout": "r_timeout", "launch": "r_io_error",
                                "dependency": "r_startup_failure", "missing-output": "r_invalid_response"}[failure]
    assert len(visited) == 1 and not visited[0].exists()
    if failure == "dependency":
        assert "jsonlite is required" in str(error.value)


@pytest.mark.parametrize("text", [
    "{", '{"cases":[],"cases":[]}', '{"value":NaN}', '{"value":1e309}',
])
def test_invalid_json_is_rejected_and_cleaned(adapter, monkeypatch, library, text):
    visited = []

    def run(command, **kwargs):
        visited.append(Path(command[-2]).parent)
        Path(command[-1]).write_text(text, encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(adapter, "run_r", run)
    with pytest.raises(adapter.RReferenceError):
        adapter.evaluate_r_batch([], sys.executable, library)
    assert not visited[0].exists()


@pytest.mark.parametrize("malformation", [
    "missing-case", "extra-case", "duplicate-case", "wrong-name", "bad-status",
    "missing-field", "boolean-scalar", "null-scalar", "infinite-scalar", "ci-shape",
    "ci-order", "wrong-df", "negative-variance", "p-range", "warnings-shape",
    "wrong-version", "unverified-library", "missing-fingerprint", "bad-runtime-hash",
    "outside-namespace", "dependency-mismatch", "metadata-path", "case-error-path",
])
def test_malformed_reference_response_is_rejected(
    adapter, monkeypatch, library, cases, reference, malformation
):
    payload = {"reference": reference, "cases": [success()]}
    record = payload["cases"][0]
    result = record["result"]
    if malformation == "missing-case":
        payload["cases"] = []
    elif malformation == "extra-case":
        payload["cases"].append(success("extra"))
    elif malformation == "duplicate-case":
        cases.append({**deepcopy(cases[0]), "name": "second"})
        payload["cases"].append(success())
    elif malformation == "wrong-name":
        record["name"] = "unknown"
    elif malformation == "bad-status":
        record["status"] = "unknown"
    elif malformation == "missing-field":
        del result["psi"]
    elif malformation == "boolean-scalar":
        result["psi"] = True
    elif malformation == "null-scalar":
        result["psi"] = None
    elif malformation == "infinite-scalar":
        result["psi"] = float("inf")
    elif malformation == "ci-shape":
        result["ci"] = [[1.02, 2.98]]
    elif malformation == "ci-order":
        result["ci"].reverse()
    elif malformation == "wrong-df":
        result["df_pvalue"] = 0
    elif malformation == "negative-variance":
        result["variance"] = -1
    elif malformation == "p-range":
        result["p_value"] = 1.1
    elif malformation == "warnings-shape":
        record["warnings"] = "warning"
    elif malformation == "wrong-version":
        reference["version"] = "2.19.5"
    elif malformation == "unverified-library":
        reference["library_verified"] = 1
    elif malformation == "missing-fingerprint":
        del reference["installed_package_md5"]["NAMESPACE"]
    elif malformation == "bad-runtime-hash":
        reference["r_runtime_md5"]["bin/exec/R"] = "b" * 31
    elif malformation == "outside-namespace":
        reference["unexpected_loaded_namespaces"] = ["qgcomp"]
    elif malformation == "dependency-mismatch":
        reference["installed_packages"]["qgcomp"]["version"] = "2.19.5"
    elif malformation == "metadata-path":
        reference["installed_packages"]["qgcomp"]["Built"] = "/private/build/path"
    else:
        payload["cases"][0] = {
            "name": "fixed-input", "status": "error", "error": "/private/case/path failed",
            "error_class": "simpleError", "warnings": [],
        }
    visited = []
    fake_export(monkeypatch, adapter, payload, visited)
    with pytest.raises(adapter.RReferenceError):
        adapter.evaluate_r_batch(cases, sys.executable, library)
    assert not visited[0].exists()
