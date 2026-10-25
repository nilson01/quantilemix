"""Optional installed-R check of conditional Cox log-hazard offsets."""

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

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/r_cox_offset_2_19_6.json"
DRIVER = Path(__file__).with_name("export_cox_offset_r.R")
FIXTURE_SHA = "d79c0b4f27b4a6d1f30b3ed4443ee441d307cfd59e73a5a4a4843cb6282f2b78"
DRIVER_SHA = "1675494a5e6b462b459eb612374c60e3412e1c132c26a4bcbea56295a752df03"
NAMES = ("offset_untied", "offset_tied", "offset_weighted_tied")


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
    for key in ("R_version", "qgcomp_version", "survival_version", "jsonlite_version"):
        if saved.get(key) != live.get(key):
            raise ValueError(f"installed reference {key} differs from pinned fixture")
    if live.get("qgcomp_version") != "2.19.6":
        raise ValueError("installed qgcomp 2.19.6 required")
    old_cases, cases = saved.get("cases"), live.get("cases")
    if (not isinstance(old_cases, list) or not isinstance(cases, list)
            or tuple(c.get("name") for c in old_cases) != NAMES
            or tuple(c.get("name") for c in cases) != NAMES):
        raise ValueError("three pinned Cox offset cases required")
    details = []
    for old, current in zip(old_cases, cases):
        name = old["name"]
        if old["inputs"] != current["inputs"]:
            raise ValueError(f"{name}: inputs changed")
        expected, reference = old["expected"], current["expected"]
        if set(expected) != set(reference):
            raise ValueError(f"{name}: reference fields changed")
        for key, value in expected.items():
            _compare(reference[key], value, f"{name}.{key}.saved_R", 1e-10)
        inputs = current["inputs"]
        x = np.column_stack((inputs["x1"], inputs["x2"]))
        fitted = fit_cox(inputs["time"], inputs["event"], x, inputs["z"],
                         q=None, weights=inputs["weights"], offset=inputs["offset"])
        profile = inputs["prediction"]
        times, survival = fitted.survival_curve(
            np.column_stack((profile["x1"], profile["x2"])), profile["z"],
            offset=profile["offset"])
        actual = {"coefficients": fitted.coefficients,
                  "covariance": fitted.covariance,
                  "psi": fitted.psi, "variance": fitted.var_psi,
                  "ci": fitted.ci, "p": fitted.p_value,
                  "baseline_time": fitted.timeline,
                  "prediction_time": times,
                  "prediction_survival": survival,
                  "events": fitted.event_count}
        gaps = {key: _compare(value, reference[key], f"{name}.{key}.Python")
                for key, value in actual.items()}
        # R's basehaz(centered=FALSE) retains the mean fitted offset.
        mean_offset = np.average(inputs["offset"], weights=inputs["weights"])
        gaps["R_centered_baseline"] = _compare(
            fitted.baseline_cumulative_hazard*np.exp(mean_offset),
            reference["baseline_cumulative_hazard"],
            f"{name}.R_centered_baseline")
        details.append({"name": name, "max_absolute_differences": gaps,
                        "R_baseline_offset_factor": float(np.exp(mean_offset))})
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
        raise ValueError("Cox offset fixture or exporter hash differs")
    with tempfile.TemporaryDirectory() as folder:
        live_path = Path(folder) / "live.json"
        subprocess.run([args.rscript, str(DRIVER), args.library, str(live_path)], check=True)
        report = check(json.loads(FIXTURE.read_text()), json.loads(live_path.read_text()))
    output.write_text(json.dumps(report, indent=2)+"\n")
    print("Three Cox offset cases agree with isolated installed R qgcomp 2.19.6")


if __name__ == "__main__":
    main()
