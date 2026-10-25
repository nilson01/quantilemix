"""Optional live check of two fixed formula/categorical cases against installed R."""

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

from quantilemix import fit_formula


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/r_formula_gaussian_2_19_6.json"
DRIVER = Path(__file__).with_name("export_formula_r.R")
FIXTURE_SHA256 = "6910f0bf2b30c9b3cf5fa6bee52ddc19aa78e5927fc8264f10bedf2c88b08619"
CASES = {"formula_categorical_quantized", "formula_categorical_prequantized_weighted"}
NUMERIC_FIELDS = {"coefficients", "covariance", "psi", "variance", "se",
                  "ci", "t", "p", "dispersion", "fitted", "predictions"}
EXACT_FIELDS = {"scores", "p_df", "residual_df"}
run_export = subprocess.run


class CheckError(ValueError):
    """Reference agreement could not be established."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise CheckError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


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
    if len(cases) != 2 or set(names) != CASES or len(set(names)) != 2:
        raise CheckError("exactly the two fixed formula cases are required")
    return {case["name"]: case for case in cases}


def _equal(actual, expected, field, *, exact=False):
    try:
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    except (TypeError, ValueError) as exc:
        raise CheckError(f"{field}: invalid numeric values") from exc
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise CheckError(f"{field}: shape or finiteness differs")
    agree = np.array_equal(a, b) if exact else np.allclose(a, b, rtol=5e-12, atol=5e-12)
    if not agree:
        raise CheckError(f"{field}: values differ")


def compare_case(saved, installed):
    name = saved["name"]
    if set(saved) != {"name", "inputs", "expected"} or set(installed) != set(saved):
        raise CheckError(f"{name}: case structure differs")
    if json.dumps(saved["inputs"], sort_keys=True) != json.dumps(installed["inputs"], sort_keys=True):
        raise CheckError(f"{name}: reference inputs changed")
    inputs = saved["inputs"]
    old, current = saved["expected"], installed["expected"]
    fields = NUMERIC_FIELDS | EXACT_FIELDS | {"coefficient_names", "breaks"}
    if set(old) != fields or set(current) != fields:
        raise CheckError(f"{name}: expected fields differ")
    if old["coefficient_names"] != current["coefficient_names"]:
        raise CheckError(f"{name}: R coefficient names differ")
    fit = fit_formula(inputs["formula"], inputs["data"], inputs["exposures"],
                      q=inputs["q"], weights=inputs["weights"])
    if fit.coefficient_names[:3] != tuple(current["coefficient_names"][:3]):
        raise CheckError(f"{name}: exposure coefficient names differ")
    if fit.coefficient_names[3:] != ("g[T.B]", "g[T.C]"):
        raise CheckError(f"{name}: categorical columns differ")
    python = {
        "coefficients": fit.coefficients, "covariance": fit.covariance,
        "scores": fit.scores, "psi": fit.psi, "variance": fit.var_psi,
        "se": fit.se, "ci": fit.ci, "t": fit.test_statistic,
        "p": fit.p_value, "p_df": fit.p_value_df,
        "residual_df": fit.residual_df, "dispersion": fit.residual_variance,
        "fitted": fit.fitted_values,
        "predictions": fit.predict_data(inputs["new_data"]),
    }
    for field in NUMERIC_FIELDS | EXACT_FIELDS:
        _equal(current[field], old[field], f"{name}.{field}.saved_R",
               exact=field in EXACT_FIELDS)
        _equal(python[field], current[field], f"{name}.{field}.Python",
               exact=field in EXACT_FIELDS)
    if (fit.breaks is None) != (old["breaks"] is None) or (
        fit.breaks is None
    ) != (current["breaks"] is None):
        raise CheckError(f"{name}.breaks: presence differs")
    if fit.breaks is not None:
        if len(fit.breaks) != len(old["breaks"]) or len(fit.breaks) != len(current["breaks"]):
            raise CheckError(f"{name}.breaks: column count differs")
        for j, cuts in enumerate(fit.breaks):
            _equal(current["breaks"][j], old["breaks"][j], f"{name}.breaks[{j}].saved_R")
            _equal(cuts, current["breaks"][j], f"{name}.breaks[{j}].Python")
    return {"name": name, "status": "passed"}


def check(args):
    if sha256(Path(args.fixture)) != FIXTURE_SHA256:
        raise CheckError("pinned formula fixture SHA-256 differs")
    saved = _load(Path(args.fixture))
    if (saved.get("reference", {}).get("version") != "2.19.6"
            or saved["reference"].get("exporter_sha256") != sha256(DRIVER)):
        raise CheckError("reference version or exporter identity differs")
    saved_cases = _cases(saved)
    library = Path(args.r_library)
    if not library.is_dir() or not (library / "qgcomp/DESCRIPTION").is_file():
        raise CheckError("isolated R qgcomp library is unavailable")
    output = Path(args.output)
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise CheckError("output must be a new path in an existing directory")
    with tempfile.TemporaryDirectory(prefix="qgcomp-formula-r-") as tmp:
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
    current_cases = _cases(installed)
    rows = [compare_case(saved_cases[name], current_cases[name]) for name in sorted(CASES)]
    report = {
        "status": "passed", "case_count": len(rows), "comparisons": rows,
        "reference": {"package": "qgcomp", "version": "2.19.6",
                      "R_version": installed.get("R_version")},
        "identities": {"fixture_sha256": sha256(Path(args.fixture)),
                       "exporter_sha256": sha256(DRIVER)},
        "scope": "fixed additive Gaussian formula/categorical fits; no missing-row removal",
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
        print(f"Formula R check failed: {exc}", file=sys.stderr)
        return 1
    print(f"{result['case_count']} installed-R formula cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
