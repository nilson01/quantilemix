"""Optional fixed Gaussian bootstrap check against installed R qgcomp 2.19.6."""

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

from quantilemix._bootstrap import _fit_bootstrap_with_indices


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/r_bootstrap_2_19_6.json"
DRIVER = Path(__file__).with_name("export_bootstrap_r.R")
FIXTURE_SHA256 = "3cb75d425742963c99d41a359cb94c4180df840c9c1887eb27a8cea3126cf062"
CASES = {"bootstrap_prequantized", "bootstrap_quantized_adjusted"}
FIELDS = {"psi", "variance", "se", "ci", "statistic", "p", "df",
          "bootstrap_effects", "intervention_levels", "intervention_means",
          "scores", "breaks"}
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
        raise CheckError("exactly the two fixed bootstrap cases are required")
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
    required = {"name", "inputs", "resample_indices", "expected"}
    if set(saved) != required or set(installed) != required:
        raise CheckError(f"{name}: case structure differs")
    if json.dumps(saved["inputs"], sort_keys=True) != json.dumps(installed["inputs"], sort_keys=True):
        raise CheckError(f"{name}: inputs changed")
    _equal(installed["resample_indices"], saved["resample_indices"],
           f"{name}.saved_R_indices", exact=True)
    old, current = saved["expected"], installed["expected"]
    if set(old) != FIELDS or set(current) != FIELDS:
        raise CheckError(f"{name}: expected fields differ")
    for field in FIELDS - {"breaks"}:
        _equal(current[field], old[field], f"{name}.{field}.saved_R",
               exact=field in {"scores", "df", "intervention_levels"})
    if len(old["breaks"]) != len(current["breaks"]):
        raise CheckError(f"{name}.breaks: column count changed")
    for j, (a, b) in enumerate(zip(old["breaks"], current["breaks"])):
        _equal(a, b, f"{name}.breaks[{j}].saved_R")
    inputs = saved["inputs"]
    x = np.column_stack((inputs["x1"], inputs["x2"]))
    result = _fit_bootstrap_with_indices(
        inputs["y"], x, saved["resample_indices"], inputs["z"], q=inputs["q"]
    )
    if result.B != inputs["B"] or result.p_value_df != current["df"]:
        raise CheckError(f"{name}: resample count or reference degrees of freedom differ")
    python = {
        "psi": result.psi, "variance": result.var_psi, "se": result.se,
        "ci": result.ci, "statistic": result.test_statistic,
        "p": result.p_value, "bootstrap_effects": result.bootstrap_effects,
        "intervention_levels": result.intervention_levels,
        "intervention_means": result.intervention_means,
        "scores": result.point_fit.scores,
    }
    differences = {}
    for field, value in python.items():
        _equal(value, current[field], f"{name}.{field}.Python",
               rtol=5e-10, atol=5e-10,
               exact=field in {"scores", "intervention_levels"})
        differences[field] = float(np.max(np.abs(
            np.asarray(value, dtype=float) - np.asarray(current[field], dtype=float)
        )))
    if inputs["q"] is not None:
        if len(result.point_fit.breaks) != len(current["breaks"]):
            raise CheckError(f"{name}.breaks: Python column count differs")
        for j, (a, b) in enumerate(zip(result.point_fit.breaks, current["breaks"])):
            _equal(a, b, f"{name}.breaks[{j}].Python")
    return {"name": name, "status": "passed", "max_absolute_difference": differences}


def check(args):
    if sha256(Path(args.fixture)) != FIXTURE_SHA256:
        raise CheckError("pinned bootstrap fixture SHA-256 differs")
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
    with tempfile.TemporaryDirectory(prefix="qgcomp-bootstrap-r-") as tmp:
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
        "scope": "fixed additive Gaussian row bootstrap with shared R resample schedules",
        "python_vs_R_tolerance": {"absolute": 5e-10, "relative": 5e-10},
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
        print(f"Bootstrap R check failed: {exc}", file=sys.stderr)
        return 1
    print(f"{result['case_count']} installed-R Gaussian bootstrap cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
