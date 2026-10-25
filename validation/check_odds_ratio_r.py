"""Optional installed-R qgcomp 2.19.6 marginal odds-ratio comparison."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from quantilemix import fit_binomial_odds_ratio
from quantilemix._odds_ratio import _fit


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/r_odds_ratio_2_19_6.json"
DRIVER = Path(__file__).with_name("export_odds_ratio_r.R")
FIXTURE_SHA = "f8242b63c28bd5878ae227b48539564e5c4885a11614286032c81f5a4eed50e1"
DRIVER_SHA = "04d2fe88aa85b08536170fb96333ecd0cc1febce9473848d8d3fcfc3c4dac84d"
NAMES = ("four_levels_unadjusted", "four_levels_adjusted")
FIELDS = ("psi", "se", "ci", "p", "risks", "msm_coefficients",
          "msm_fitted_risks", "effects")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(actual, expected, label: str, *, atol: float) -> float:
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError(f"{label}: shape or finiteness differs")
    gap = float(np.max(np.abs(a-b)))
    if not np.allclose(a, b, atol=atol, rtol=0):
        raise ValueError(f"{label}: numerical difference {gap:.3g} exceeds {atol:.3g}")
    return gap


def check(saved: dict, live: dict) -> dict:
    if saved.get("qgcomp_version") != "2.19.6" or live.get("qgcomp_version") != "2.19.6":
        raise ValueError("installed R qgcomp 2.19.6 required")
    old_cases, new_cases = saved.get("cases"), live.get("cases")
    if (not isinstance(old_cases, list) or not isinstance(new_cases, list)
            or tuple(c["name"] for c in old_cases) != NAMES
            or tuple(c["name"] for c in new_cases) != NAMES):
        raise ValueError("two fixed four-level cases required")
    results = []
    for old, new in zip(old_cases, new_cases):
        name = old["name"]
        if old["inputs"] != new["inputs"]:
            raise ValueError(f"{name}: reference inputs changed")
        if set(old["expected"]) != set(FIELDS) or set(new["expected"]) != set(FIELDS):
            raise ValueError(f"{name}: reference fields changed")
        for field in FIELDS:
            compare(new["expected"][field], old["expected"][field],
                    f"{name}.{field}.saved_R", atol=2e-10)
        data, expected = new["inputs"], new["expected"]
        x = np.column_stack((data["x1"], data["x2"]))
        z = None if data["z"] is None else np.asarray(data["z"])[:, None]
        result = _fit(data["y"], x, z, q=4, breaks=None, alpha=.05,
                      rows=np.asarray(data["indices"], dtype=np.intp), seed=451)
        scores_gap = compare(result.point_fit.scores, data["scores"],
                             f"{name}.scores", atol=0)
        actual = {"psi": result.psi, "se": result.se, "ci": result.ci,
                  "p": result.p_value, "risks": result.intervention_risks,
                  "msm_coefficients": result.msm_coefficients,
                  "msm_fitted_risks": result.msm_fitted_risks,
                  "effects": result.bootstrap_effects}
        gaps = {field: compare(actual[field], expected[field],
                               f"{name}.{field}.Python", atol=2e-8)
                for field in FIELDS}
        public = fit_binomial_odds_ratio(data["y"], x, z, q=4, B=2, seed=17)
        public_point_gap = compare(public.psi, expected["psi"],
                                   f"{name}.public_psi", atol=2e-8)
        compare(public.intervention_risks, expected["risks"],
                f"{name}.public_risks", atol=2e-8)
        results.append({"name": name, "score_difference": scores_gap,
                        "max_absolute_differences": gaps,
                        "public_point_difference": public_point_gap})
    return {"status": "passed", "checked_utc": datetime.now(timezone.utc).isoformat(),
            "R_version": live.get("R_version"),
            "qgcomp_version": "2.19.6", "jsonlite_version": live.get("jsonlite_version"),
            "saved_fixture_sha256": FIXTURE_SHA, "r_exporter_sha256": DRIVER_SHA,
            "cases": results,
            "scope": "Two fixed four-level cases with shared bootstrap rows; not interval calibration."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise ValueError("output already exists")
    if sha(FIXTURE) != FIXTURE_SHA or sha(DRIVER) != DRIVER_SHA:
        raise ValueError("four-level fixture or exporter hash differs")
    with tempfile.TemporaryDirectory() as folder:
        live_path = Path(folder) / "live.json"
        subprocess.run([args.rscript, "--vanilla", str(DRIVER), args.library,
                        str(live_path)], check=True, timeout=120)
        report = check(json.loads(FIXTURE.read_text()), json.loads(live_path.read_text()))
    output.write_text(json.dumps(report, indent=2)+"\n")
    print("Two marginal odds-ratio cases agree with installed R qgcomp 2.19.6")


if __name__ == "__main__":
    main()
