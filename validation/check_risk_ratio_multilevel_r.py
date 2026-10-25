"""Optional installed-R qgcomp 2.19.6 four-level risk-ratio comparison."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from quantilemix import qgcomp
from quantilemix._risk_ratio import _from_indices


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/r_risk_ratio_multilevel_2_19_6.json"
DRIVER = Path(__file__).with_name("export_risk_ratio_multilevel_r.R")
FIXTURE_SHA = "053728944a11ca6d91c180ae0f79772b9efa27389a047a5b5e8ee6e06dff99ed"
DRIVER_SHA = "2c25ebf0410f8ff4623a4915419d3dd0cd99db3ee6ef61fc54318cb10b00578a"
NAMES = ("four_levels_unadjusted", "four_levels_adjusted")


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
        if old["inputs"] != new["inputs"]:
            raise ValueError(f"{old['name']}: reference inputs changed")
        if set(old["expected"]) != set(new["expected"]):
            raise ValueError(f"{old['name']}: reference fields changed")
        for field, value in old["expected"].items():
            compare(new["expected"][field], value, f"{old['name']}.{field}.saved_R",
                    atol=2e-8 if field == "effects" else 2e-10)
        data, expected = new["inputs"], new["expected"]
        x = np.column_stack((data["x1"], data["x2"]))
        z = None if data["z"] is None else np.asarray(data["z"])[:, None]
        result = _from_indices(data["y"], x, data["indices"], z, q=4)
        actual = {"psi": result.psi, "se": result.se, "ci": result.ci,
                  "p": result.p_value, "risks": result.intervention_risks,
                  "msm_coefficients": result.msm_coefficients,
                  "msm_fitted_risks": result.msm_fitted_risks,
                  "effects": result.bootstrap_effects}
        gaps = {name: compare(value, expected[name], f"{old['name']}.{name}.Python",
                              atol=1e-7 if name == "effects" else 2e-8)
                for name, value in actual.items()}
        formula = "y ~ x1 + x2" + (" + z" if z is not None else "")
        public = qgcomp(formula, {key: value for key, value in data.items()
                                  if key in ("y", "x1", "x2", "z") and value is not None},
                        ("x1", "x2"), family="binomial", q=4, B=2, seed=17)
        compare(public.psi, expected["psi"], f"{old['name']}.public_psi", atol=2e-8)
        compare(public.intervention_risks, expected["risks"],
                f"{old['name']}.public_risks", atol=2e-8)
        results.append({"name": old["name"], "max_absolute_differences": gaps})
    return {"status": "passed", "checked_utc": datetime.now(timezone.utc).isoformat(),
            "qgcomp_version": "2.19.6", "saved_fixture_sha256": FIXTURE_SHA,
            "r_exporter_sha256": DRIVER_SHA, "cases": results,
            "scope": "Two fixed four-level cases; shared R bootstrap rows; not interval calibration."}


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
        subprocess.run([args.rscript, str(DRIVER), args.library, str(live_path)], check=True)
        report = check(json.loads(FIXTURE.read_text()), json.loads(live_path.read_text()))
    output.write_text(json.dumps(report, indent=2)+"\n")
    print("Two four-level marginal risk-ratio cases agree with installed R qgcomp 2.19.6")


if __name__ == "__main__":
    main()
