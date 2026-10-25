"""Optional installed-R check for the two fixed positive-weight Gaussian cases.

The Python package never imports R. This command requires an existing isolated
R qgcomp 2.19.6 library and writes a report only to a new output path.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import quantilemix


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/r_weighted_gaussian_2_19_6.json"
DRIVER = Path(__file__).with_name("export_weighted_r.R")
FIXTURE_SHA256 = "9594e90d09ace4d5c7705498ffb1e6cfb503ad9e59b121ed0bd517aee7e013f2"
CASES = {"weighted_raw_adjusted", "weighted_prequantized_correlated"}
NUMERIC_FIELDS = {"coefficients", "covariance", "psi", "variance", "se", "ci",
                  "t", "p", "dispersion", "fitted", "predictions",
                  "positive_psi", "negative_psi"}
EXACT_FIELDS = {"scores", "p_df", "residual_df"}
run_export = subprocess.run


class CheckError(ValueError):
    """The optional comparison could not establish agreement."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise CheckError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def reject_constant(value):
    raise CheckError(f"nonfinite JSON constant: {value}")


def load_json(path: Path):
    try:
        return json.loads(path.read_text(), object_pairs_hook=unique_object,
                          parse_constant=reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckError(f"cannot read valid JSON: {type(exc).__name__}") from exc


def cases_by_name(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise CheckError("cases must be a list")
    cases = payload["cases"]
    names = [case.get("name") if isinstance(case, dict) else None for case in cases]
    if len(names) != 2 or set(names) != CASES or len(set(names)) != 2:
        raise CheckError("exactly the two fixed weighted cases are required")
    return {case["name"]: case for case in cases}


def numeric_equal(actual, expected, field):
    try:
        a = np.asarray(actual, dtype=float)
        b = np.asarray(expected, dtype=float)
    except (TypeError, ValueError) as exc:
        raise CheckError(f"{field}: numeric shape or type is invalid") from exc
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise CheckError(f"{field}: numeric shape or finiteness differs")
    if field in EXACT_FIELDS:
        if not np.array_equal(a, b):
            raise CheckError(f"{field}: exact values differ")
    elif not np.allclose(a, b, rtol=5e-12, atol=5e-12):
        raise CheckError(f"{field}: numerical values differ")


def compare_case(saved, installed):
    name = saved["name"]
    if set(saved) != {"name", "inputs", "expected"} or set(installed) != set(saved):
        raise CheckError(f"{name}: case structure differs")
    if json.dumps(saved["inputs"], sort_keys=True) != json.dumps(installed["inputs"], sort_keys=True):
        raise CheckError(f"{name}: reference inputs changed")
    inputs = saved["inputs"]
    if (not isinstance(inputs.get("weights"), list)
            or len(inputs["weights"]) != len(inputs["y"])
            or any(isinstance(w, bool) or not isinstance(w, (int, float))
                   or not math.isfinite(w) or w <= 0 for w in inputs["weights"])):
        raise CheckError(f"{name}: weights are not a positive finite row vector")
    expected, observed = saved["expected"], installed["expected"]
    fields = NUMERIC_FIELDS | EXACT_FIELDS | {"breaks", "positive_weights", "negative_weights"}
    if set(expected) != fields or set(observed) != fields:
        raise CheckError(f"{name}: expected field set differs")
    fit = quantilemix.fit(inputs["y"], inputs["exposures"], inputs["covariates"],
                     q=inputs["q"], weights=inputs["weights"])
    python_values = {
        "coefficients": fit.coefficients, "covariance": fit.covariance,
        "scores": fit.scores, "psi": fit.psi, "variance": fit.var_psi,
        "se": fit.se, "ci": fit.ci, "t": fit.test_statistic,
        "p": fit.p_value, "p_df": fit.p_value_df,
        "residual_df": fit.residual_df, "dispersion": fit.residual_variance,
        "fitted": fit.fitted_values,
        "predictions": fit.predict(inputs["new_exposures"], inputs["new_covariates"]),
        "positive_psi": fit.pos_psi, "negative_psi": fit.neg_psi,
    }
    for field in NUMERIC_FIELDS | EXACT_FIELDS:
        numeric_equal(observed[field], expected[field], f"{name}.{field}.saved_R")
        numeric_equal(python_values[field], observed[field], f"{name}.{field}.Python")
    for field, value in (("breaks", fit.breaks),):
        if (value is None) != (expected[field] is None) or (value is None) != (observed[field] is None):
            raise CheckError(f"{name}.{field}: missingness differs")
        if value is not None:
            if len(value) != len(expected[field]) or len(value) != len(observed[field]):
                raise CheckError(f"{name}.{field}: exposure count differs")
            for j, cuts in enumerate(value):
                numeric_equal(observed[field][j], expected[field][j], f"{name}.{field}[{j}].saved_R")
                numeric_equal(cuts, observed[field][j], f"{name}.{field}[{j}].Python")
    for field, value in (("positive_weights", fit.pos_weights),
                         ("negative_weights", fit.neg_weights)):
        if (set(value) != set(expected[field]) or set(value) != set(observed[field])):
            raise CheckError(f"{name}.{field}: exposure names differ")
        for label in value:
            numeric_equal(observed[field][label], expected[field][label],
                          f"{name}.{field}.{label}.saved_R")
            numeric_equal(value[label], observed[field][label],
                          f"{name}.{field}.{label}.Python")
    return {"name": name, "status": "passed", "numeric_fields": len(NUMERIC_FIELDS | EXACT_FIELDS)}


def check(args):
    fixture = Path(args.fixture)
    if sha256(fixture) != FIXTURE_SHA256:
        raise CheckError("the pinned weighted fixture SHA-256 differs")
    saved = load_json(fixture)
    reference = saved.get("reference", {})
    if (reference.get("version") != "2.19.6"
            or reference.get("exporter_sha256") != sha256(DRIVER)):
        raise CheckError("fixture version or exporter identity differs")
    saved_cases = cases_by_name(saved)
    library = Path(args.r_library)
    if not library.is_dir() or not (library / "qgcomp/DESCRIPTION").is_file():
        raise CheckError("the isolated R qgcomp library is unavailable")
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise CheckError("output already exists; choose a new path")
    if not output.parent.is_dir():
        raise CheckError("output parent directory does not exist")
    with tempfile.TemporaryDirectory(prefix="qgcomp-weighted-r-") as tmp:
        exported = Path(tmp) / "reference.json"
        try:
            process = run_export([args.rscript, "--vanilla", str(DRIVER),
                                  str(library), str(exported)],
                                 capture_output=True, text=True, timeout=args.timeout,
                                 check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CheckError(f"R exporter did not complete: {type(exc).__name__}") from exc
        if process.returncode != 0:
            raise CheckError(f"R exporter failed with exit code {process.returncode}")
        installed = load_json(exported)
    if (installed.get("qgcomp_version") != "2.19.6"
            or Path(installed.get("library", "")).resolve() != library.resolve()
            or Path(installed.get("qgcomp_origin", "")).resolve()
            != (library / "qgcomp").resolve()):
        raise CheckError("installed R package identity or origin differs")
    current_cases = cases_by_name(installed)
    results = [compare_case(saved_cases[name], current_cases[name]) for name in sorted(CASES)]
    report = {"status": "passed", "case_count": len(results), "comparisons": results,
              "reference": {"package": "qgcomp", "version": "2.19.6",
                            "R_version": installed.get("R_version")},
              "identities": {"fixture_sha256": sha256(fixture),
                             "exporter_sha256": sha256(DRIVER)},
              "scope": "two fixed positive-weight Gaussian cases; no simulations or R runtime dependency",
              "checked_at_utc": datetime.now(timezone.utc).isoformat()}
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", required=True)
    parser.add_argument("--r-library", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fixture", default=str(FIXTURE))
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    try:
        report = check(args)
    except (CheckError, OSError, ValueError) as exc:
        print(f"Weighted R check failed: {exc}", file=sys.stderr)
        return 1
    print(f"{report['case_count']} installed-R weighted cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
