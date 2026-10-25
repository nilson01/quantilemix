"""Optional Poisson exposure-time offset check against installed R qgcomp 2.19.6."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from quantilemix import fit_poisson_formula


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/r_poisson_offset_2_19_6.json"
DRIVER = Path(__file__).with_name("export_poisson_offset_r.R")
FIXTURE_SHA256 = "fbeb2ba9e7b52c6b8e83f182972d2d479e0a34f4bf483a29894118c56c4c1739"
CASES = {"offset_exact_scores", "offset_quantized_adjusted"}
FIELDS = {"coefficients", "covariance", "scores", "breaks", "psi", "variance",
          "se", "ci", "zstat", "p", "residual_df", "fitted", "predictions",
          "iterations", "coefficient_names"}
run_export = subprocess.run


class CheckError(ValueError):
    """Reference agreement could not be established."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CheckError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load(path: Path):
    try:
        return json.loads(path.read_text(), object_pairs_hook=_unique)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckError(f"invalid JSON: {type(exc).__name__}") from exc


def _cases(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise CheckError("cases must be a list")
    cases = payload["cases"]
    names = [case.get("name") if isinstance(case, dict) else None for case in cases]
    if len(cases) != 2 or len(set(names)) != 2 or set(names) != CASES:
        raise CheckError("exactly the two fixed poisson offset cases are required")
    return {case["name"]: case for case in cases}


def _equal(actual, expected, field, *, rtol=5e-12, atol=5e-12, exact=False):
    try:
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    except (TypeError, ValueError) as exc:
        raise CheckError(f"{field}: invalid numeric values") from exc
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise CheckError(f"{field}: shape or finiteness differs")
    agree = np.array_equal(a, b) if exact else np.allclose(a, b, rtol=rtol, atol=atol)
    if not agree:
        raise CheckError(f"{field}: values differ")


def compare_case(saved, installed):
    name = saved["name"]
    if set(saved) != {"name", "inputs", "expected"} or set(installed) != set(saved):
        raise CheckError(f"{name}: case structure differs")
    if json.dumps(saved["inputs"], sort_keys=True) != json.dumps(installed["inputs"], sort_keys=True):
        raise CheckError(f"{name}: inputs changed")
    old, current = saved["expected"], installed["expected"]
    if set(old) != FIELDS or set(current) != FIELDS:
        raise CheckError(f"{name}: expected fields differ")
    if old["coefficient_names"] != current["coefficient_names"]:
        raise CheckError(f"{name}: R coefficient names changed")
    for field in FIELDS - {"breaks", "coefficient_names"}:
        _equal(current[field], old[field], f"{name}.{field}.saved_R",
               exact=field in {"scores", "residual_df", "iterations"})
    if (old["breaks"] is None) != (current["breaks"] is None):
        raise CheckError(f"{name}.breaks: presence changed")
    if old["breaks"] is not None:
        if len(old["breaks"]) != len(current["breaks"]):
            raise CheckError(f"{name}.breaks: column count changed")
        for j, (a, b) in enumerate(zip(old["breaks"], current["breaks"])):
            _equal(a, b, f"{name}.breaks[{j}].saved_R")
    inputs = saved["inputs"]
    data = {"y": inputs["y"], "x1": inputs["x1"], "x2": inputs["x2"],
            "time": inputs["time"]}
    new = {"x1": inputs["new_x1"], "x2": inputs["new_x2"],
           "time": inputs["new_time"]}
    formula = "y ~ x1 + x2"
    if inputs["z"] is not None:
        data["z"], new["z"] = inputs["z"], inputs["new_z"]
        formula += " + z"
    fit = fit_poisson_formula(formula, data, ("x1", "x2"), q=inputs["q"],
                              exposure_time="time")
    if fit.coefficient_names != tuple(current["coefficient_names"]):
        raise CheckError(f"{name}: Python coefficient names differ")
    if fit.p_value_df is not None or fit.family != "poisson":
        raise CheckError(f"{name}: Python family inference differs")
    python = {
        "coefficients": fit.coefficients, "covariance": fit.covariance,
        "scores": fit.scores, "psi": fit.psi, "variance": fit.var_psi,
        "se": fit.se, "ci": fit.ci, "zstat": fit.test_statistic,
        "p": fit.p_value, "residual_df": fit.residual_df,
        "fitted": fit.fitted_values, "predictions": fit.predict_data(new),
    }
    max_absolute_difference = {}
    for field, value in python.items():
        uncertainty = field in {"covariance", "variance", "se", "ci", "zstat", "p"}
        _equal(value, current[field], f"{name}.{field}.Python",
               rtol=2e-6 if uncertainty else 5e-10,
               atol=2e-6 if uncertainty else 5e-10,
               exact=field in {"scores", "residual_df"})
        max_absolute_difference[field] = float(np.max(np.abs(
            np.asarray(value, dtype=float) - np.asarray(current[field], dtype=float)
        )))
    if (fit.breaks is None) != (current["breaks"] is None):
        raise CheckError(f"{name}.breaks: presence differs")
    if fit.breaks is not None:
        if len(fit.breaks) != len(current["breaks"]):
            raise CheckError(f"{name}.breaks: Python column count differs")
        for j, (a, b) in enumerate(zip(fit.breaks, current["breaks"])):
            _equal(a, b, f"{name}.breaks[{j}].Python")
    return {"name": name, "status": "passed",
            "max_absolute_difference": max_absolute_difference}


def check(args):
    if sha256(Path(args.fixture)) != FIXTURE_SHA256:
        raise CheckError("pinned poisson fixture SHA-256 differs")
    saved = _load(Path(args.fixture))
    if (saved.get("reference", {}).get("version") != "2.19.6"
            or saved["reference"].get("exporter_sha256") != sha256(DRIVER)):
        raise CheckError("reference version or exporter identity differs")
    library = Path(args.r_library)
    if not library.is_dir() or not (library / "qgcomp/DESCRIPTION").is_file():
        raise CheckError("isolated R qgcomp library is unavailable")
    output = Path(args.output)
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise CheckError("output must be a new path in an existing directory")
    with tempfile.TemporaryDirectory(prefix="qgcomp-poisson-r-") as tmp:
        exported = Path(tmp) / "reference.json"
        try:
            process = run_export([args.rscript, "--vanilla", str(DRIVER),
                                  str(library), str(exported)], capture_output=True,
                                 text=True, timeout=args.timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CheckError(f"R exporter did not complete: {type(exc).__name__}") from exc
        if process.returncode:
            raise CheckError(f"R exporter exited with {process.returncode}")
        installed = _load(exported)
    if (installed.get("qgcomp_version") != "2.19.6"
            or Path(installed.get("library", "")).resolve() != library.resolve()
            or Path(installed.get("qgcomp_origin", "")).resolve()
            != (library / "qgcomp").resolve()):
        raise CheckError("R package version or origin differs")
    saved_cases, current_cases = _cases(saved), _cases(installed)
    rows = [compare_case(saved_cases[name], current_cases[name]) for name in sorted(CASES)]
    report = {
        "status": "passed", "case_count": len(rows), "comparisons": rows,
        "reference": {"package": "qgcomp", "version": "2.19.6",
                      "R_version": installed.get("R_version")},
        "identities": {"fixture_sha256": sha256(Path(args.fixture)),
                       "exporter_sha256": sha256(DRIVER)},
        "scope": "two fixed additive Poisson-log exposure-time offset fits; not calibration",
        "python_vs_R_tolerance": {
            "coefficients_and_predictions": {"absolute": 5e-10, "relative": 5e-10},
            "uncertainty": {"absolute": 2e-6, "relative": 2e-6},
        },
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    }
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
        result = check(args)
    except (CheckError, OSError, ValueError) as exc:
        print(f"Poisson offset R check failed: {exc}", file=sys.stderr)
        return 1
    print(f"{result['case_count']} installed-R poisson offset cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
