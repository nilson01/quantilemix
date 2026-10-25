"""Optional installed-R check for two weighted Gaussian bootstrap cases."""

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
FIXTURE = ROOT / "tests/fixtures/r_weighted_bootstrap_2_19_6.json"
DRIVER = Path(__file__).with_name("export_weighted_bootstrap_r.R")
FIXTURE_SHA256 = "31209cce2c2982966b1a4aa83c079653cd0e498a16ec99f3453ba74b6e845948"
NAMES = {"weighted_prequantized", "weighted_quantized"}
FIELDS = {"psi", "se", "ci", "bootstrap_effects", "intervention_means",
          "scores", "breaks"}


class CheckError(ValueError):
    """The optional installed-R comparison could not be established."""


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


def _compare(actual, expected, label, *, exact=False):
    try:
        a = np.asarray(actual, dtype=float)
        b = np.asarray(expected, dtype=float)
    except (TypeError, ValueError) as exc:
        raise CheckError(f"{label}: invalid numeric values") from exc
    if (a.shape != b.shape or a.size == 0 or not np.all(np.isfinite(a))
            or not np.all(np.isfinite(b))):
        raise CheckError(f"{label}: shape or finiteness differs")
    same = np.array_equal(a, b) if exact else np.allclose(a, b, rtol=5e-10, atol=5e-10)
    if not same:
        raise CheckError(f"{label}: values differ")
    return float(np.max(np.abs(a-b)))


def _cases(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise CheckError("cases must be a list")
    cases = payload["cases"]
    if len(cases) != 2 or any(not isinstance(case, dict) for case in cases):
        raise CheckError("exactly two named cases are required")
    names = [case.get("name") for case in cases]
    if any(not isinstance(name, str) for name in names) or set(names) != NAMES:
        raise CheckError("expected weighted bootstrap cases are missing")
    return {case["name"]: case for case in cases}


def compare_case(saved, installed):
    name = saved["name"]
    if set(saved) != {"name", "inputs", "resample_indices", "expected"} or set(installed) != set(saved):
        raise CheckError(f"{name}: case structure differs")
    if json.dumps(saved["inputs"], sort_keys=True) != json.dumps(installed["inputs"], sort_keys=True):
        raise CheckError(f"{name}: inputs differ from fixed fixture")
    _compare(installed["resample_indices"], saved["resample_indices"],
             f"{name}.resample_indices", exact=True)
    if set(saved["expected"]) != FIELDS or set(installed["expected"]) != FIELDS:
        raise CheckError(f"{name}: expected fields differ")
    for key in FIELDS - {"breaks"}:
        _compare(installed["expected"][key], saved["expected"][key],
                 f"{name}.saved_R.{key}", exact=key == "scores")
    old_breaks = saved["expected"]["breaks"]
    new_breaks = installed["expected"]["breaks"]
    if len(old_breaks) != len(new_breaks):
        raise CheckError(f"{name}: break count differs")
    for j, (old, new) in enumerate(zip(old_breaks, new_breaks)):
        _compare(old, new, f"{name}.saved_R.breaks[{j}]")
    inputs = saved["inputs"]
    x = np.column_stack((inputs["x1"], inputs["x2"]))
    result = _fit_bootstrap_with_indices(
        inputs["y"], x, saved["resample_indices"], inputs["z"],
        q=inputs["q"], weights=inputs["weights"],
    )
    if result.B != inputs["B"]:
        raise CheckError(f"{name}: bootstrap count differs")
    python = {"psi": result.psi, "se": result.se, "ci": result.ci,
              "bootstrap_effects": result.bootstrap_effects,
              "intervention_means": result.intervention_means,
              "scores": result.point_fit.scores}
    differences = {
        key: _compare(value, installed["expected"][key], f"{name}.Python.{key}",
                      exact=key == "scores") for key, value in python.items()
    }
    if inputs["q"] is not None:
        for j, (actual, expected) in enumerate(zip(result.point_fit.breaks, new_breaks)):
            _compare(actual, expected, f"{name}.Python.breaks[{j}]")
    return {"name": name, "status": "passed", "max_absolute_difference": differences}


def check(args):
    if sha256(FIXTURE) != FIXTURE_SHA256:
        raise CheckError("pinned fixture SHA-256 differs")
    saved = _load(FIXTURE)
    if (saved.get("reference", {}).get("version") != "2.19.6"
            or saved["reference"].get("exporter_sha256") != sha256(DRIVER)):
        raise CheckError("reference version or exporter identity differs")
    library = Path(args.r_library)
    if not library.is_dir() or not (library / "qgcomp/DESCRIPTION").is_file():
        raise CheckError("isolated R qgcomp library is unavailable")
    output = Path(args.output)
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise CheckError("output must be a new path in an existing directory")
    with tempfile.TemporaryDirectory(prefix="qgcomp-weighted-bootstrap-r-") as tmp:
        exported = Path(tmp) / "reference.json"
        try:
            process = subprocess.run(
                [args.rscript, "--vanilla", str(DRIVER), str(library), str(exported)],
                capture_output=True, text=True, timeout=args.timeout, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CheckError(f"R exporter did not complete: {type(exc).__name__}") from exc
        if process.returncode:
            raise CheckError(f"R exporter exited with {process.returncode}: {process.stderr[-500:]}")
        installed = _load(exported)
    if (installed.get("R_version") != saved["reference"].get("R_version")
            or installed.get("qgcomp_version") != "2.19.6"
            or Path(installed.get("library", "")).resolve() != library.resolve()
            or Path(installed.get("qgcomp_origin", "")).resolve()
            != (library / "qgcomp").resolve()):
        raise CheckError("R package version or origin differs")
    saved_cases, current_cases = _cases(saved), _cases(installed)
    rows = [compare_case(saved_cases[name], current_cases[name]) for name in sorted(NAMES)]
    report = {"status": "passed", "case_count": len(rows), "comparisons": rows,
              "reference": {"package": "qgcomp", "version": "2.19.6",
                            "R_version": installed.get("R_version")},
              "identities": {"fixture_sha256": sha256(FIXTURE),
                             "exporter_sha256": sha256(DRIVER)},
              "scope": "two weighted Gaussian fixed-cut row-bootstrap cases; matched R row schedules",
              "checked_at_utc": datetime.now(timezone.utc).isoformat()}
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", required=True)
    parser.add_argument("--r-library", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args(argv)
    try:
        report = check(args)
    except (CheckError, KeyError, TypeError) as exc:
        print(f"weighted bootstrap R check failed: {exc}", file=sys.stderr)
        return 1
    print(f"{report['case_count']} weighted bootstrap cases agree with installed R qgcomp 2.19.6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
