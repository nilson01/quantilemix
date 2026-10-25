"""Optional live comparison with isolated installed R qgcomp 2.19.6.

Run from the repository root with a Python environment containing this package.
R is only needed for this optional development check, not for package use.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import warnings

import numpy as np

from quantilemix import fit_binomial_formula, fit_poisson_formula


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/r_glm_missing_2_19_6.json"
DRIVER = Path(__file__).with_name("export_glm_missing_r.R")
FIXTURE_SHA256 = "b2ffc8da33baaa7cb86837859f26da251adfcf190fea7f8b1a3311ad0f5de9d7"
CASES = {"binomial_quantized", "binomial_prequantized",
         "poisson_quantized", "poisson_prequantized"}


class CheckError(ValueError):
    """Installed-reference comparison could not be established."""


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
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list) or len(cases) != 4:
        raise CheckError("exactly four cases are required")
    names = [item.get("name") if isinstance(item, dict) else None for item in cases]
    if len(set(names)) != 4 or set(names) != CASES:
        raise CheckError("case identities differ")
    return {item["name"]: item for item in cases}


def _equal(actual, expected, label: str, *, tolerance: float = 1e-10, exact=False):
    try:
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    except (TypeError, ValueError) as exc:
        raise CheckError(f"{label}: invalid numeric values") from exc
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise CheckError(f"{label}: shape or finiteness differs")
    agree = np.array_equal(a, b) if exact else np.allclose(a, b, rtol=tolerance, atol=tolerance)
    if not agree:
        raise CheckError(f"{label}: values differ")


def check(args):
    output = Path(args.output)
    if output.exists():
        raise CheckError("output already exists")
    if sha256(FIXTURE) != FIXTURE_SHA256:
        raise CheckError("pinned fixture hash differs")
    saved = _load(FIXTURE)
    if saved["reference"]["version"] != "2.19.6" or sha256(DRIVER) != saved["reference"]["exporter_sha256"]:
        raise CheckError("installed-reference source identity differs")
    library = Path(args.r_library).resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="qgcomp-glm-missing-r-") as tmp:
        exported = Path(tmp) / "installed.json"
        try:
            completed = subprocess.run(
                [args.rscript, str(DRIVER), str(library), str(exported)],
                text=True, capture_output=True, timeout=args.timeout, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CheckError(f"R export failed: {type(exc).__name__}") from exc
        if completed.returncode != 0:
            raise CheckError(f"R export failed ({completed.returncode}): {completed.stderr[-500:]}")
        live = _load(exported)
        if live.get("qgcomp_version") != "2.19.6" or Path(live.get("qgcomp_origin", "")).resolve() != library / "qgcomp":
            raise CheckError("R package version or library origin differs")
        reference = _cases(saved)
        installed = _cases(live)
        for name in sorted(CASES):
            a, b = reference[name], installed[name]
            if a["inputs"] != b["inputs"]:
                raise CheckError(f"{name}: saved R inputs differ")
            expected, observed = a["expected"], b["expected"]
            if expected.keys() != observed.keys() or expected["coefficient_names"] != observed["coefficient_names"]:
                raise CheckError(f"{name}: R output structure differs")
            for key in expected:
                if key == "coefficient_names":
                    continue
                if expected[key] is None or observed[key] is None:
                    if expected[key] is not None or observed[key] is not None:
                        raise CheckError(f"{name}.{key}: null differs")
                    continue
                _equal(observed[key], expected[key], f"{name}.{key}.R",
                       exact=key in {"scores", "training_row_indices", "original_row_count", "residual_df"})
            inputs = a["inputs"]
            fit = fit_binomial_formula if inputs["family"] == "binomial" else fit_poisson_formula
            result = fit(inputs["formula"], inputs["data"], inputs["exposures"],
                         q=inputs["q"], missing="drop")
            if result.training_row_indices != tuple(expected["training_row_indices"]):
                raise CheckError(f"{name}: retained rows differ")
            _equal(result.scores, expected["scores"], f"{name}.scores", exact=True)
            for field, actual in (("coefficients", result.coefficients),
                                  ("covariance", result.covariance),
                                  ("psi", result.psi), ("variance", result.var_psi),
                                  ("se", result.se), ("ci", result.ci),
                                  ("zstat", result.test_statistic),
                                  ("p", result.p_value), ("fitted", result.fitted_values)):
                tolerance = 1e-7 if field in {"covariance", "variance", "se", "ci", "zstat", "p"} else 1e-10
                _equal(actual, expected[field], f"{name}.{field}.Python", tolerance=tolerance)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                predictions = result.predict_data(inputs["new_data"])
            _equal(predictions, expected["predictions"], f"{name}.predictions.Python")
    report = {
        "status": "passed", "case_count": len(CASES),
        "scope": "fixed binomial/Poisson complete-case formulas; no imputation or repeated-data calibration",
        "installed_r_version": live["R_version"], "installed_qgcomp_version": live["qgcomp_version"],
        "installed_origin": live["qgcomp_origin"],
        "fixture_sha256": sha256(FIXTURE), "exporter_sha256": sha256(DRIVER),
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", required=True)
    parser.add_argument("--r-library", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    try:
        report = check(args)
    except (CheckError, OSError, ValueError) as exc:
        print(f"GLM missing-row R check failed: {exc}", file=sys.stderr)
        return 1
    print(f"{report['case_count']} installed-R complete-case GLM cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
