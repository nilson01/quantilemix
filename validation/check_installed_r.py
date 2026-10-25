"""Optionally compare the fixed Gaussian cases with an installed R reference.

This development command never installs R/packages or changes source fixtures.
The native Python package and its ordinary tests do not require R.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import warnings

import numpy as np
import scipy
import quantilemix


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/r_gaussian_2_19_6.json"
DRIVER = Path(__file__).with_name("export_installed_r.R")
FIXTURE_SHA256 = "ff1c111c1eea044959e698e89344acaeba1eea4eb7e3e91e506a93f66ccf577e"
CASE_NAMES = {
    "single_exposure", "quantile_rounding_boundaries", "prediction_boundaries",
    "adjusted_positive", "adjusted_mixed", "adjusted_cancellation",
    "tied_endpoints", "explicit_breaks", "prequantized",
}
FIELDS = {
    "coefficients", "covariance", "scores", "breaks", "psi", "variance", "se",
    "ci", "p_value", "statistic", "df_pvalue", "df_residual", "positive_psi",
    "negative_psi", "positive_weights", "negative_weights", "fitted", "predictions",
}
EXACT_FIELDS = {"scores", "df_pvalue", "df_residual"}

# Keep test doubles local to this command, not the shared subprocess module.
run_export = subprocess.run


class CheckError(ValueError):
    """The comparison could not establish valid evidence."""


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CheckError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise CheckError(f"nonfinite JSON constant: {value}")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise CheckError("nonfinite JSON number")
    return number


def load_json(path):
    try:
        return json.loads(Path(path).read_text(), object_pairs_hook=_unique_object,
                          parse_constant=_invalid_constant, parse_float=_finite_float)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckError(f"cannot read valid JSON: {type(exc).__name__}") from exc


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_fingerprints(reference):
    for field in ("installed_package_md5", "r_runtime_md5"):
        hashes = reference.get(field)
        if not isinstance(hashes, dict) or not hashes:
            raise CheckError(f"reference.{field} is required")
        if field == "installed_package_md5" and set(hashes) != {
            "DESCRIPTION", "NAMESPACE", "R/qgcomp.rdb", "R/qgcomp.rdx"
        }:
            raise CheckError("installed package fingerprint files differ")
        for name, digest in hashes.items():
            if (not isinstance(name, str) or not name or name.startswith("/")
                    or ".." in name.split("/") or "\\" in name or ":" in name
                    or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{32}", digest)):
                raise CheckError(f"reference.{field}: invalid file name or fingerprint")


def _case_map(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise CheckError("cases must be a nonempty list")
    cases = payload["cases"]
    if any(not isinstance(case, dict) or not isinstance(case.get("name"), str) for case in cases):
        raise CheckError("every case must have a string name")
    names = [case["name"] for case in cases]
    if len(names) != 9 or len(set(names)) != 9 or set(names) != CASE_NAMES:
        raise CheckError("exactly the nine unique fixed Gaussian cases are required")
    return {case["name"]: case for case in cases}


def _validate_value(actual, expected, path):
    """Require the same container shapes and finite numeric leaves."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise CheckError(f"{path}: mapping fields or names differ")
        for key in expected:
            _validate_value(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise CheckError(f"{path}: array shape differs")
        for index, (a, e) in enumerate(zip(actual, expected)):
            _validate_value(a, e, f"{path}[{index}]")
    elif expected is None:
        if actual is not None:
            raise CheckError(f"{path}: expected null")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if (not isinstance(actual, (int, float)) or isinstance(actual, bool)
                or not math.isfinite(actual)):
            raise CheckError(f"{path}: expected a finite number")
    else:
        raise CheckError(f"{path}: unsupported reference value")


def _differences(actual, expected, path, *, exact=False, rtol=1e-8, atol=1e-10):
    if isinstance(expected, dict):
        return [error for key in expected for error in
                _differences(actual[key], expected[key], f"{path}.{key}",
                             exact=exact, rtol=rtol, atol=atol)]
    if isinstance(expected, list):
        return [error for i, (a, e) in enumerate(zip(actual, expected)) for error in
                _differences(a, e, f"{path}[{i}]", exact=exact, rtol=rtol, atol=atol)]
    if expected is None:
        return []
    agrees = actual == expected if exact else abs(actual - expected) <= atol + rtol * abs(expected)
    return [] if agrees else [f"{path}: {actual!r} differs from {expected!r}"]


def _native_expected(inputs, column_names):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", quantilemix.QGCompWarning)
        fit = quantilemix.fit(inputs["y"], inputs["exposures"], inputs["covariates"],
                         q=inputs["q"], breaks=inputs["breaks"], alpha=inputs["alpha"])
        prediction = fit.predict(inputs["new_exposures"], inputs["new_covariates"])
    if {"coefficients": list(fit.coefficient_names),
            "exposures": list(fit.exposure_names)} != column_names:
        raise CheckError("native Python column names or order differ")
    return {
        "coefficients": fit.coefficients.tolist(), "covariance": fit.covariance.tolist(),
        "scores": fit.scores.tolist(),
        "breaks": None if fit.breaks is None else [cut.tolist() for cut in fit.breaks],
        "psi": fit.psi, "variance": fit.var_psi, "se": fit.se, "ci": list(fit.ci),
        "p_value": fit.p_value, "statistic": fit.test_statistic,
        "df_pvalue": fit.p_value_df, "df_residual": fit.residual_df,
        "positive_psi": fit.pos_psi, "negative_psi": fit.neg_psi,
        "positive_weights": dict(fit.pos_weights), "negative_weights": dict(fit.neg_weights),
        "fitted": fit.fitted_values.tolist(), "predictions": prediction.tolist(),
    }


def _summary_table(target, alpha, columns):
    """Project existing fit values onto the Gaussian mixture summary rows."""
    intercept = target["coefficients"][0]
    intercept_se = math.sqrt(target["covariance"][0][0])
    critical = -float(scipy.special.ndtri_exp(math.log(alpha) - math.log(2.0)))
    intercept_statistic = intercept / intercept_se
    intercept_p = float(2 * scipy.special.stdtr(target["df_pvalue"], -abs(intercept_statistic)))
    rows = [
        [intercept, intercept_se, intercept - critical * intercept_se,
         intercept + critical * intercept_se, intercept_statistic, intercept_p],
        [target["psi"], target["se"], *target["ci"], target["statistic"], target["p_value"]],
    ]
    indices = [0, 1, 2, 3, 4, 5] if "t value" in columns else [0, 1, 2, 3, 5]
    return [[row[index] for index in indices] for row in rows]


def _compare_summary(outcome, name, inputs, expected, native):
    if not isinstance(outcome, dict) or outcome.get("status") not in ("ok", "error"):
        raise CheckError(f"{name}: registered summary outcome is required")
    for field in ("output", "warnings"):
        if (not isinstance(outcome.get(field), list)
                or any(not isinstance(value, str) for value in outcome[field])):
            raise CheckError(f"{name}: registered summary {field} must be a string array")
    found = [f"{name}: registered summary warning: {warning}" for warning in outcome["warnings"]]
    limitation = None
    if outcome["status"] == "error":
        if any(not isinstance(outcome.get(field), str) or not outcome[field]
               for field in ("error", "error_class")):
            raise CheckError(f"{name}: registered summary error details are required")
        found.append(f"{name}: unexpected registered summary error: {outcome['error']}")
    else:
        if outcome.get("row_names") != ["(Intercept)", "psi1"]:
            raise CheckError(f"{name}: registered summary row names or order differ")
        columns = outcome.get("column_names")
        full_columns = ["Estimate", "Std. Error", "Lower CI", "Upper CI", "t value", "Pr(>|t|)"]
        missing_t_columns = [*full_columns[:4], full_columns[-1]]
        if columns not in (full_columns, missing_t_columns):
            raise CheckError(f"{name}: registered summary columns differ")
        if outcome["output"] != ["Mixture slope parameters (delta method CI):", ""]:
            found.append(f"{name}: registered summary printed header differs")
        for origin, target in (("stored_fixture", expected), ("native_python", native)):
            table = _summary_table(target, inputs["alpha"], columns)
            path = f"{name}.registered_summary.{origin}"
            _validate_value(outcome.get("values"), table, path)
            found.extend(_differences(outcome["values"], table, path))
        if columns == missing_t_columns:
            limitation = {
                "case": name, "method": "summary.qgcompfit", "missing_columns": ["t value"],
                "meaning": "Registered Gaussian summary omits the t statistic; common returned values are compared."
            }
    comparison = {"case": name, "comparison": "registered_summary",
                  "status": "failed" if found else ("known_reference_limitation" if limitation else "passed"),
                  "errors": found, "observed": outcome}
    return comparison, limitation


def compare_reference(payload, fixtures):
    installed = _case_map(payload)
    baseline = _case_map(fixtures)
    reference = payload.get("reference")
    if not isinstance(reference, dict):
        raise CheckError("installed reference metadata is required")
    for key, expected in {"package": "qgcomp", "version": "2.19.6",
                          "mode": "installed package"}.items():
        if reference.get(key) != expected:
            raise CheckError(f"reference.{key} does not identify the pinned installed package")
    if reference.get("library_verified") is not True:
        raise CheckError("the loaded qgcomp library was not verified")
    for key in ("R_version", "platform"):
        if not isinstance(reference.get(key), str) or not reference[key]:
            raise CheckError(f"reference.{key} is required")
    versions = reference.get("package_versions")
    if (not isinstance(versions, dict) or versions.get("qgcomp") != "2.19.6"
            or any(not isinstance(v, str) or not v for v in versions.values())):
        raise CheckError("loaded package versions are missing or invalid")
    if reference.get("unexpected_loaded_namespaces", []) != []:
        raise CheckError("dependencies were loaded outside the allowed libraries")
    _validate_fingerprints(reference)

    comparisons, errors, limitations = [], [], []
    for name in sorted(CASE_NAMES):
        case, source = installed[name], baseline[name]
        inputs = source["inputs"]
        exposures = [f"x{i + 1}" for i in range(len(inputs["exposures"][0]))]
        covariates = [] if inputs["covariates"] is None else [
            f"z{i + 1}" for i in range(len(inputs["covariates"][0]))
        ]
        column_names = {"coefficients": ["(Intercept)"] + exposures + covariates,
                        "exposures": exposures}
        if case.get("column_names") != column_names:
            raise CheckError(f"{name}: column names or order differ")
        actual = case.get("expected")
        expected = source.get("expected")
        if not isinstance(expected, dict) or set(expected) != FIELDS:
            raise CheckError(f"{name}: stored reference fields differ")
        if not isinstance(actual, dict) or set(actual) != FIELDS:
            raise CheckError(f"{name}: installed reference fields differ")
        _validate_value(actual, expected, name)
        native = _native_expected(source["inputs"], column_names)
        _validate_value(native, expected, f"{name}.native_python")
        for comparison, target in (("stored_fixture", expected), ("native_python", native)):
            found = []
            for field in sorted(FIELDS):
                tolerances = {"rtol": 1e-12, "atol": 1e-12} if field == "breaks" else {}
                found.extend(_differences(actual[field], target[field], f"{name}.{field}",
                                          exact=field in EXACT_FIELDS, **tolerances))
            comparisons.append({"case": name, "comparison": comparison,
                                "status": "failed" if found else "passed", "errors": found})
            errors.extend(f"{comparison}: {error}" for error in found)

        wrapper = case.get("wrapper_prediction")
        if not isinstance(wrapper, dict) or wrapper.get("status") not in ("ok", "error"):
            raise CheckError(f"{name}: registered prediction outcome is required")
        known_error = (name == "prequantized" and source["inputs"]["q"] is None
                       and source["inputs"]["breaks"] is None
                       and wrapper.get("status") == "error"
                       and wrapper.get("error_class") == "simpleError"
                       and wrapper.get("error") == "'by' must be of length 1")
        if known_error:
            limitations.append({"case": name, "method": "predict.qgcompfit",
                                "error": wrapper["error"], "error_class": wrapper["error_class"],
                                "meaning": "Known no-break wrapper limitation; GLM prediction is compared separately."})
            comparisons.append({"case": name, "comparison": "registered_prediction",
                                "status": "known_reference_limitation", "errors": []})
        else:
            if wrapper["status"] == "error":
                found = [f"{name}: unexpected registered prediction error: {wrapper.get('error', 'unspecified')}"]
            else:
                _validate_value(wrapper.get("values"), native["predictions"], f"{name}.wrapper_prediction")
                found = _differences(wrapper["values"], native["predictions"], f"{name}.wrapper_prediction")
            comparisons.append({"case": name, "comparison": "registered_prediction",
                                "status": "failed" if found else "passed", "errors": found})
            errors.extend(found)
        summary_comparison, summary_limitation = _compare_summary(
            case.get("registered_summary"), name, inputs, expected, native
        )
        comparisons.append(summary_comparison)
        errors.extend(summary_comparison["errors"])
        if summary_limitation:
            limitations.append(summary_limitation)
    return {"status": "failed" if errors else "passed", "case_count": len(installed),
            "comparisons": comparisons, "errors": errors, "known_limitations": limitations,
            "reference": reference,
            "tolerances": {"numerical": {"atol": 1e-10, "rtol": 1e-8},
                           "breaks": {"atol": 1e-12, "rtol": 1e-12},
                           "exact": sorted(EXACT_FIELDS)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r-library", required=True, type=Path)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--fixtures", type=Path, default=FIXTURES)
    parser.add_argument("--output", type=Path, required=True, help="New JSON report; existing files are never overwritten")
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if (args.output.is_symlink() or output.exists()
            or output in {args.fixtures.resolve(), DRIVER.resolve(), Path(__file__).resolve()}):
        print("Refusing to overwrite an existing file or validation input.", file=sys.stderr)
        return 2
    report = {"status": "failed", "case_count": 0, "errors": [], "comparisons": []}
    try:
        if not args.r_library.is_dir():
            raise CheckError("--r-library must be an existing isolated R library")
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise CheckError("--timeout must be finite and positive")
        fixtures = load_json(args.fixtures)
        if _sha256(args.fixtures) != FIXTURE_SHA256:
            raise CheckError("fixture SHA-256 differs from the nine pinned Gaussian cases")
        _case_map(fixtures)
        env = dict(os.environ, R_LIBS_USER=str(args.r_library.resolve()), R_LIBS_SITE="",
                   LC_ALL="C", LANG="C", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                   VECLIB_MAXIMUM_THREADS="1")
        with tempfile.TemporaryDirectory(prefix="qgcomp-installed-r-") as directory:
            exported = Path(directory) / "reference.json"
            command = [args.rscript, "--vanilla", str(DRIVER), str(args.r_library.resolve()),
                       str(args.fixtures.resolve()), str(exported)]
            completed = run_export(command, capture_output=True, text=True, env=env,
                                       timeout=args.timeout, check=False)
            if completed.returncode:
                detail = completed.stderr.strip()[-4000:]
                for path in (str(args.r_library.resolve()), str(args.fixtures.resolve()), str(ROOT), directory):
                    detail = detail.replace(path, "<local-path>")
                raise CheckError(f"R exporter exited with code {completed.returncode}: {detail}")
            report = compare_reference(load_json(exported), fixtures)
        report["identities"] = {
            "fixtures_sha256": _sha256(args.fixtures), "r_driver_sha256": _sha256(DRIVER),
            "checker_sha256": _sha256(__file__),
            "python_core_sha256": _sha256(Path(quantilemix.__file__).with_name("_core.py")),
        }
        report["python_environment"] = {"python": platform.python_version(), "numpy": np.__version__,
                                        "scipy": scipy.__version__, "platform": sys.platform,
                                        "machine": platform.machine()}
    except (CheckError, OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as exc:
        if isinstance(exc, subprocess.TimeoutExpired):
            detail = f"R exporter exceeded the {args.timeout:g}-second timeout"
        elif isinstance(exc, OSError):
            detail = f"R executable or validation file unavailable (errno {exc.errno})"
        else:
            detail = str(exc)
            for private_path in (str(args.r_library.resolve()), str(args.fixtures.resolve()),
                                 str(args.rscript), str(ROOT), str(Path.home()),
                                 str(sys.prefix), str(sys.base_prefix)):
                if private_path and private_path != ".":
                    detail = detail.replace(private_path, "<local-path>")
        report = {"status": "failed", "case_count": 0, "comparisons": [],
                  "errors": [f"{type(exc).__name__}: {detail}"]}
    report["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["scope"] = "Nine fixed additive Gaussian cases; not full R feature coverage or statistical calibration."
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except OSError as exc:
        print(f"Could not save comparison report: {type(exc).__name__}", file=sys.stderr)
        return 2
    print(f"Installed R comparison: {report['status']}; {report['case_count']} cases.")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
