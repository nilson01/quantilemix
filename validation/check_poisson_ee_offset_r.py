"""Optional installed-R check of the documented Poisson EE offset difference."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from quantilemix import fit_poisson_ee

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/r_poisson_ee_offset_2_19_6.json"
DRIVER = Path(__file__).with_name("export_poisson_ee_offset_r.R")
FIXTURE_SHA = "ad93e525ab8cabcaa7548cda7f354843d94469f1c49aab50848969067529dc26"
DRIVER_SHA = "465b25a6e3a0b456c832618ef5890591df3eeb3766383d7c9c2ba5ce4c7bb155"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compare(actual, reference, label: str, tolerance: float = 1e-10) -> float:
    a, b = np.asarray(actual, dtype=float), np.asarray(reference, dtype=float)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError(f"{label}: shape or finiteness differs")
    gap = float(np.max(np.abs(a-b)))
    if gap > tolerance:
        raise ValueError(f"{label}: numerical difference {gap:.3g} exceeds {tolerance:.3g}")
    return gap


def check(saved: dict, live: dict) -> dict:
    if saved.get("qgcomp_version") != "2.19.6" or live.get("qgcomp_version") != "2.19.6":
        raise ValueError("installed qgcomp 2.19.6 required")
    if saved.get("inputs") != live.get("inputs"):
        raise ValueError("fixed Poisson EE offset inputs changed")
    for branch in ("no_offset", "supplied_offset"):
        if set(saved[branch]) != set(live[branch]):
            raise ValueError(f"{branch}: reference fields changed")
        for name, reference in saved[branch].items():
            _compare(live[branch][name], reference, f"{branch}.{name}.saved_R")
    for name in saved["no_offset"]:
        _compare(live["supplied_offset"][name], live["no_offset"][name],
                 f"installed_R_ignored_offset.{name}", tolerance=1e-12)
    inputs = live["inputs"]
    x = np.column_stack((inputs["x1"], inputs["x2"]))
    ordinary = fit_poisson_ee(inputs["y"], x, inputs["id"], q=None)
    timed = fit_poisson_ee(inputs["y"], x, inputs["id"], q=None,
                           exposure_time=inputs["obstime"])
    for name, value in (("psi", ordinary.psi), ("var_psi", ordinary.var_psi),
                        ("coefficients", ordinary.conditional_coefficients)):
        _compare(value, live["no_offset"][name], f"Python_no_offset.{name}")
    gap = abs(timed.psi - float(live["supplied_offset"]["psi"]))
    if gap <= 0.1:
        raise ValueError("Python unequal-time estimate did not separate from R's ignored offset")
    return {"status": "passed", "checked_utc": datetime.now(timezone.utc).isoformat(),
            "qgcomp_version": "2.19.6", "saved_fixture_sha256": FIXTURE_SHA,
            "r_exporter_sha256": DRIVER_SHA,
            "R_supplied_offset_psi": live["supplied_offset"]["psi"],
            "Python_unequal_time_psi": timed.psi,
            "absolute_effect_difference": gap,
            "interpretation": "Installed R ignores this offset; Python fits a time-adjusted rate model."}


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
        raise ValueError("Poisson EE offset fixture or exporter hash differs")
    with tempfile.TemporaryDirectory() as folder:
        live_path = Path(folder) / "live.json"
        subprocess.run([args.rscript, str(DRIVER), args.library, str(live_path)], check=True)
        report = check(json.loads(FIXTURE.read_text()), json.loads(live_path.read_text()))
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("Installed R's ignored Poisson EE offset and Python's time-adjusted fit verified")


if __name__ == "__main__":
    main()
