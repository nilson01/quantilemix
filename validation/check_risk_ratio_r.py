"""Optional exact-fixture check against isolated installed R qgcomp 2.19.6."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from quantilemix._risk_ratio import _from_indices


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/r_risk_ratio_2_19_6.json"
DRIVER = Path(__file__).with_name("export_risk_ratio_r.R")
FIXTURE_SHA = "e9e4dba9c9372480d9d700c0f7a1595e41c703a303d358a9ebf5ba59f2530e82"
DRIVER_SHA = "ef3d0ffa258840e42a747ab43ad3584b40085525e96fb233ab07699e840f1332"
NAMES = ("binary_rr_two_exposures", "binary_rr_adjusted")


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
        raise ValueError("two fixed risk-ratio cases required")
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
        result = _from_indices(data["y"], x, data["indices"], z, q=None)
        actual = {"psi": result.psi, "se": result.se, "ci": result.ci,
                  "p": result.p_value, "risks": result.intervention_risks,
                  "effects": result.bootstrap_effects}
        gaps = {name: compare(value, expected[name], f"{old['name']}.{name}.Python",
                              atol=2e-8 if name == "effects" else 2e-9)
                for name, value in actual.items()}
        results.append({"name": old["name"], "max_absolute_differences": gaps})
    return {"status": "passed", "checked_utc": datetime.now(timezone.utc).isoformat(),
            "qgcomp_version": "2.19.6", "saved_fixture_sha256": FIXTURE_SHA,
            "r_exporter_sha256": DRIVER_SHA, "cases": results,
            "scope": "Two fixed prequantized two-level cases; not interval calibration."}


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
        raise ValueError("risk-ratio fixture or exporter hash differs")
    with tempfile.TemporaryDirectory() as folder:
        live_path = Path(folder) / "live.json"
        subprocess.run([args.rscript, str(DRIVER), args.library, str(live_path)], check=True)
        report = check(json.loads(FIXTURE.read_text()), json.loads(live_path.read_text()))
    output.write_text(json.dumps(report, indent=2)+"\n")
    print("Two marginal risk-ratio cases agree with installed R qgcomp 2.19.6")


if __name__ == "__main__":
    main()
