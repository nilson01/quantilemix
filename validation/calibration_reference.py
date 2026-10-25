"""Bounded installed-R adapter for prequantized Gaussian reference checks.

R is optional validation software, not a dependency of the Python package.
An empty batch establishes reference identity only; it is not fit evidence.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


DRIVER = Path(__file__).with_name("calibration_r.R")
MAX_CASES = 20
# A module-local process seam lets tests exercise the public adapter without R.
run_r = subprocess.run


class RReferenceError(RuntimeError):
    """The requested R batch could not produce a valid reference response."""

    def __init__(self, message, *, code="r_invalid_response"):
        super().__init__(message)
        self.code = code


def _number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _matrix(value, label):
    if not isinstance(value, list) or not value:
        raise RReferenceError(f"{label} must be a nonempty array of rows")
    if not isinstance(value[0], list) or not value[0]:
        raise RReferenceError(f"{label} rows must be nonempty arrays")
    width = len(value[0])
    for row in value:
        if not isinstance(row, list) or len(row) != width or not all(map(_number, row)):
            raise RReferenceError(f"{label} must be rectangular and finite numeric")
    return len(value), width


def _validate_inputs(cases):
    if not isinstance(cases, list) or len(cases) > MAX_CASES:
        raise RReferenceError(f"cases must be a list with at most {MAX_CASES} entries")
    dimensions = {}
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"name", "inputs"}:
            raise RReferenceError("Each case must contain exactly name and inputs")
        name = case["name"]
        if not _safe_text(name) or not name or name in dimensions:
            raise RReferenceError("Case names must be nonempty unique strings without local paths")
        inputs = case["inputs"]
        keys = {"exposures", "covariates", "y", "q", "alpha"}
        if not isinstance(inputs, dict) or set(inputs) != keys:
            raise RReferenceError("Each input must contain exposures, covariates, y, q, alpha")
        if inputs["q"] is not None:
            raise RReferenceError("Only prequantized inputs with q=None are supported")
        alpha = inputs["alpha"]
        if not _number(alpha) or not 0 < alpha < 1:
            raise RReferenceError("alpha must be finite and strictly between zero and one")
        n, d = _matrix(inputs["exposures"], "exposures")
        if any(
            value < 0 or value >= 2**53 or value != int(value)
            for row in inputs["exposures"] for value in row
        ):
            raise RReferenceError("Exposure scores must be nonnegative integers below 2**53")
        y = inputs["y"]
        if not isinstance(y, list) or len(y) != n or not all(map(_number, y)):
            raise RReferenceError("y must be a finite numeric array matching exposure rows")
        c = 0
        if inputs["covariates"] is not None:
            rows, c = _matrix(inputs["covariates"], "covariates")
            if rows != n:
                raise RReferenceError("Covariate and exposure row counts must agree")
        if n <= 1 + d + c:
            raise RReferenceError("Inputs must have positive residual degrees of freedom")
        dimensions[name] = (n, d, c)
    return dimensions


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RReferenceError("R output contains duplicate JSON keys")
        result[key] = value
    return result


def _invalid_constant(_value):
    raise RReferenceError("R output contains nonfinite JSON numbers")


def _parse_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise RReferenceError("R output contains nonfinite JSON numbers")
    return result


def _load_json(path):
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_invalid_constant,
            parse_float=_parse_float,
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise RReferenceError("R output is missing, unreadable, or invalid JSON") from error


def _safe_text(value):
    # Reports carry labels and diagnostics, never absolute local paths.
    return (
        isinstance(value, str)
        and "\x00" not in value
        and re.search(r"(?:[A-Za-z]:[\\/]|(?<![A-Za-z0-9.])/(?:[^\s/]+))", value) is None
    )


def _validate_reference(reference):
    keys = {
        "package", "version", "mode", "R_version", "platform", "library_verified",
        "installed_packages", "installed_package_md5", "r_runtime_md5",
        "package_versions", "package_library_origins", "unexpected_loaded_namespaces",
    }
    if not isinstance(reference, dict) or set(reference) != keys:
        raise RReferenceError("R reference metadata has an unexpected schema")
    if (
        reference["package"] != "qgcomp"
        or reference["version"] != "2.19.6"
        or reference["mode"] != "installed package"
        or reference["library_verified"] is not True
        or reference["unexpected_loaded_namespaces"] != []
    ):
        raise RReferenceError("R reference package version or library isolation is unverified")
    for key in ("R_version", "platform"):
        if not _safe_text(reference[key]) or not reference[key]:
            raise RReferenceError("R runtime metadata is missing or contains a local path")
    required_files = {"DESCRIPTION", "NAMESPACE", "R/qgcomp.rdb", "R/qgcomp.rdx"}
    runtime_files = {
        "bin/exec/R", "bin/x64/Rterm.exe", "bin/Rterm.exe",
        "bin/x64/R.exe", "bin/R.exe", "bin/R",
    }
    for key, labels in (
        ("installed_package_md5", required_files), ("r_runtime_md5", runtime_files)
    ):
        hashes = reference[key]
        if not isinstance(hashes, dict) or not hashes:
            raise RReferenceError("R package and runtime identity checksums are required")
        valid_labels = (
            set(hashes) == labels if key == "installed_package_md5"
            else len(hashes) == 1 and set(hashes) <= labels
        )
        if not valid_labels or any(
            not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{32}", value) is None
            for value in hashes.values()
        ):
            raise RReferenceError("R package or runtime identity checksums are malformed")
    installed = reference["installed_packages"]
    versions = reference["package_versions"]
    origins = reference["package_library_origins"]
    if (
        not isinstance(installed, dict) or not installed
        or not isinstance(versions, dict) or not versions
        or not isinstance(origins, dict) or set(origins) != set(versions)
    ):
        raise RReferenceError("R dependency metadata is missing or inconsistent")
    for package, record in installed.items():
        if (
            not isinstance(package, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9.]*", package) is None
            or not isinstance(record, dict) or set(record) != {"version", "Built"}
            or not _safe_text(record["version"]) or not record["version"]
            or (record["Built"] is not None and not _safe_text(record["Built"]))
        ):
            raise RReferenceError("R installed dependency snapshot is malformed")
    for package, version in versions.items():
        if (
            not isinstance(package, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9.]*", package) is None
            or not _safe_text(version) or not version
            or not isinstance(origins[package], str)
            or origins[package] not in {"isolated", "base-recommended"}
            or (origins[package] == "isolated" and (
                package not in installed or installed[package]["version"] != version
            ))
        ):
            raise RReferenceError("R loaded dependency metadata is malformed")
    if (
        versions.get("qgcomp") != "2.19.6"
        or "jsonlite" not in versions
        or any(origins.get(name) != "isolated" for name in ("qgcomp", "jsonlite"))
    ):
        raise RReferenceError("R qgcomp and jsonlite must load from the isolated library")


def _validate_response(payload, dimensions):
    if not isinstance(payload, dict) or set(payload) != {"reference", "cases"}:
        raise RReferenceError("R response must contain exactly reference and cases")
    _validate_reference(payload["reference"])
    records = payload["cases"]
    if not isinstance(records, list) or len(records) != len(dimensions):
        raise RReferenceError("R response has a missing or extra case")
    seen = set()
    fields = {"psi", "variance", "se", "ci", "p_value", "df_pvalue", "df_residual"}
    for record in records:
        if not isinstance(record, dict):
            raise RReferenceError("R case response must be an object")
        name = record.get("name")
        if not isinstance(name, str) or name not in dimensions or name in seen:
            raise RReferenceError("R response has a missing, extra, or duplicate case name")
        seen.add(name)
        warnings = record.get("warnings")
        if not isinstance(warnings, list) or not all(_safe_text(item) for item in warnings):
            raise RReferenceError("R case warnings must be an array of sanitized strings")
        if record.get("status") == "error":
            if (
                set(record) != {"name", "status", "error", "error_class", "warnings"}
                or not _safe_text(record["error"]) or not record["error"]
                or not _safe_text(record["error_class"]) or not record["error_class"]
            ):
                raise RReferenceError("R case error is malformed or contains a local path")
            continue
        if record.get("status") != "ok" or set(record) != {"name", "status", "result", "warnings"}:
            raise RReferenceError("R case status or response schema is invalid")
        result = record["result"]
        if not isinstance(result, dict) or set(result) != fields:
            raise RReferenceError("R successful case result has an unexpected schema")
        if any(not _number(result[key]) for key in fields - {"ci"}):
            raise RReferenceError("R successful case contains a nonfinite or nonnumeric scalar")
        ci = result["ci"]
        if not isinstance(ci, list) or len(ci) != 2 or not all(map(_number, ci)) or ci[0] > ci[1]:
            raise RReferenceError("R confidence limits must be two ordered finite numbers")
        # Finite unexpected degrees of freedom are comparison evidence. Keep
        # them and valid sibling cases; the runner records the exact mismatch.
        if any(type(result[key]) is not int or result[key] <= 0
               for key in ("df_pvalue", "df_residual")):
            raise RReferenceError("R degrees of freedom must be positive integers")
        if result["variance"] < 0 or result["se"] < 0 or not 0 <= result["p_value"] <= 1:
            raise RReferenceError("R variance, standard error, or p-value is outside its valid range")
    # Preserve caller order even if a process returns the same cases reordered.
    lookup = {record["name"]: record for record in records}
    return {"reference": payload["reference"], "cases": [lookup[name] for name in dimensions]}


def evaluate_r_batch(cases, rscript, library, timeout=180):
    """Evaluate up to twenty fixed prequantized Gaussian inputs in installed R.

    Parameters
    ----------
    cases : list of dict
        Unique ``name`` and ``inputs`` records. Inputs contain finite numeric
        ``exposures`` (n by d integer scores), ``covariates`` (n by c, or None),
        ``y`` (length n), ``q=None``, and ``alpha`` strictly between zero and one.
        An empty list performs identity preflight without fitting any model.
    rscript, library : str or path-like
        Rscript executable and isolated library containing qgcomp 2.19.6 and
        jsonlite. No packages are installed and no user/site fallback is used.
    timeout : float, default 180
        Maximum seconds for the single R process, including startup.

    Returns
    -------
    dict
        Stable installed-reference metadata and ordered per-case results or
        errors, with captured warnings. Checksums identify files, not authenticity.

    Raises
    ------
    RReferenceError
        Invalid input, missing dependencies, startup/isolation/version failure,
        timeout, or malformed/incomplete reference output. Individual fit errors
        remain case records. All owned temporary files are removed on exit.
    """
    dimensions = _validate_inputs(cases)
    if not _number(timeout) or timeout <= 0:
        raise RReferenceError("timeout must be a positive finite number")
    try:
        executable = shutil.which(os.fspath(rscript))
        library_path = Path(library).resolve(strict=True)
    except (OSError, TypeError, ValueError) as error:
        raise RReferenceError("Rscript or isolated library configuration is invalid", code="r_invalid_configuration") from error
    if executable is None:
        raise RReferenceError("Rscript executable was not found; no packages were installed", code="r_missing_executable")
    if not library_path.is_dir():
        raise RReferenceError("The isolated R library must be an existing directory", code="r_invalid_library")
    if not DRIVER.is_file():
        raise RReferenceError("The R calibration driver is missing", code="r_missing_driver")
    env = os.environ.copy()
    env.update({
        "R_LIBS_USER": "", "R_LIBS_SITE": "", "R_LIBS": "",
        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
    })
    try:
        with tempfile.TemporaryDirectory(prefix="qgcomp-r-batch-") as directory:
            temporary = Path(directory)
            input_path = temporary / "inputs.json"
            output_path = temporary / "results.json"
            with input_path.open("x", encoding="utf-8") as handle:
                json.dump({"cases": cases}, handle, allow_nan=False)
            command = [str(Path(executable).resolve()), "--vanilla", str(DRIVER.resolve()),
                       str(library_path), str(input_path), str(output_path)]
            completed = run_r(
                command, cwd=temporary, env=env, stdin=subprocess.DEVNULL,
                capture_output=True, text=True, timeout=timeout, check=False,
            )
            if completed.returncode != 0:
                # Detailed errors belong to case JSON. Startup stderr can contain
                # dependency search paths, so expose a bounded sanitized diagnostic.
                diagnostic = completed.stderr.strip().replace(str(library_path), "<local-path>")
                diagnostic = diagnostic.replace(str(temporary), "<local-path>")
                diagnostic = diagnostic.replace(str(DRIVER.resolve()), "<local-path>")
                if not _safe_text(diagnostic):
                    diagnostic = "R startup or dependency loading failed (diagnostic contained a local path)"
                raise RReferenceError(f"R reference process failed: {diagnostic[:1200]}", code="r_startup_failure")
            if output_path.is_symlink():
                raise RReferenceError("R output must be a regular owned file")
            return _validate_response(_load_json(output_path), dimensions)
    except subprocess.TimeoutExpired as error:
        # subprocess.run terminates and waits for its direct child on timeout.
        raise RReferenceError("R reference process exceeded its time limit", code="r_timeout") from error
    except OSError as error:
        raise RReferenceError("R process or temporary file operation failed", code="r_io_error") from error
