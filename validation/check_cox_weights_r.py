"""Optional installed-R check for conditional Cox case weights."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from quantilemix import fit_cox
from quantilemix._cox import _weighted_partial_likelihood, _weighted_score_residuals

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/r_cox_weights_2_19_6.json"
DRIVER = Path(__file__).with_name("export_cox_weights_r.R")
FIXTURE_SHA = "c03a071387d60759fa4d691e7162741a636051e12bde1a4b910a5823e87f3c51"
DRIVER_SHA = "0ba7537f382564a1d3925756280a9667b349231f4c91479119fc6537403abc8e"
NAMES = ("fractional_untied", "fractional_tied", "integer_untied")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compare(actual, expected, label: str, tolerance: float = 2e-8) -> float:
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError(f"{label}: shape or finiteness differs")
    gap = float(np.max(np.abs(a-b)))
    if gap > tolerance:
        raise ValueError(f"{label}: numerical difference {gap:.3g} exceeds {tolerance:.3g}")
    return gap


def check(saved: dict, live: dict) -> dict:
    if saved.get("qgcomp_version") != "2.19.6" or live.get("qgcomp_version") != "2.19.6":
        raise ValueError("installed qgcomp 2.19.6 required")
    for key in ("R_version", "survival_version", "jsonlite_version"):
        if saved.get(key) != live.get(key):
            raise ValueError(f"installed reference {key} differs from pinned fixture")
    old_cases, cases = saved.get("cases"), live.get("cases")
    if (not isinstance(old_cases, list) or not isinstance(cases, list)
            or tuple(c.get("name") for c in old_cases) != NAMES
            or tuple(c.get("name") for c in cases) != NAMES):
        raise ValueError("three pinned weighted Cox cases required")
    details = []
    for old, current in zip(old_cases, cases):
        name = old["name"]
        if old["inputs"] != current["inputs"]:
            raise ValueError(f"{name}: inputs changed")
        expected, reference = old["expected"], current["expected"]
        if set(expected) != set(reference):
            raise ValueError(f"{name}: reference fields changed")
        for key, value in expected.items():
            if value is None or reference[key] is None:
                if value != reference[key]:
                    raise ValueError(f"{name}.{key}: missing reference differs")
            else:
                _compare(reference[key], value, f"{name}.{key}.saved_R", 1e-10)
        inputs = current["inputs"]
        x = np.column_stack((inputs["x1"], inputs["x2"]))
        fitted = fit_cox(inputs["time"], inputs["event"], x, inputs["z"],
                         q=None, weights=inputs["weights"])
        actual = {"coefficients": fitted.coefficients,
                  "covariance": fitted.covariance,
                  "psi": fitted.psi, "variance": fitted.var_psi,
                  "ci": fitted.ci, "p": fitted.p_value,
                  "baseline_time": fitted.timeline,
                  "baseline_cumulative_hazard": fitted.baseline_cumulative_hazard,
                  "events": fitted.event_count}
        gaps = {key: _compare(value, reference[key], f"{name}.{key}.Python")
                for key, value in actual.items()}
        design = np.column_stack((x, inputs["z"]))
        durations = np.asarray(inputs["time"], dtype=float)
        events = np.asarray(inputs["event"], dtype=float)
        weights = np.asarray(inputs["weights"], dtype=float)
        order = np.argsort(-durations, kind="stable")
        _, score, information = _weighted_partial_likelihood(
            design[order], events[order], fitted.coefficients,
            durations[order], weights[order])
        if np.max(np.abs(score)) > 1e-7:
            raise ValueError(f"{name}: weighted score did not converge")
        naive = np.linalg.inv(information)
        if reference["naive_covariance"] is not None:
            gaps["naive_covariance"] = _compare(
                naive, reference["naive_covariance"], f"{name}.naive_covariance.Python")
        if fitted.robust_variance:
            residual = _weighted_score_residuals(
                design[order], events[order], durations[order],
                fitted.coefficients, weights[order])
            unsorted = np.empty_like(residual)
            unsorted[order] = residual
            gaps["weighted_score_residuals"] = _compare(
                unsorted, np.asarray(reference["score_residuals"])*weights[:, None],
                f"{name}.weighted_score_residuals.Python")
        details.append({"name": name, "max_absolute_differences": gaps})
    return {"status": "passed", "checked_utc": datetime.now(timezone.utc).isoformat(),
            "qgcomp_version": "2.19.6", "saved_fixture_sha256": FIXTURE_SHA,
            "r_exporter_sha256": DRIVER_SHA, "cases": details}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise ValueError("output already exists")
    if _sha(FIXTURE) != FIXTURE_SHA or _sha(DRIVER) != DRIVER_SHA:
        raise ValueError("weighted Cox fixture or exporter hash differs")
    with tempfile.TemporaryDirectory() as folder:
        live_path = Path(folder) / "live.json"
        subprocess.run([args.rscript, str(DRIVER), args.library, str(live_path)], check=True)
        report = check(json.loads(FIXTURE.read_text()), json.loads(live_path.read_text()))
    output.write_text(json.dumps(report, indent=2)+"\n")
    print("Three case-weighted Cox fits agree with installed R qgcomp 2.19.6")


if __name__ == "__main__":
    main()
