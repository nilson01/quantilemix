"""Optional fixed-case comparison with installed R qgcomp 2.19.6."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from quantilemix._poisson_bootstrap import _from_indices


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/r_poisson_bootstrap_2_19_6.json"
DRIVER = Path(__file__).with_name("export_poisson_bootstrap_r.R")
FIXTURE_SHA = "21dd9c430e5e5b086da992263f6d38813ebd711cca4ad260874aa80324cf33e1"
DRIVER_SHA = "57e5109846503a0b212f1f24296805f9aac711be68486b94d1a5c7eb70c9151e"
NAMES = ("poisson_bootstrap_two_levels", "poisson_bootstrap_four_levels_adjusted")


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
        raise ValueError("two fixed Poisson bootstrap cases required")
    results = []
    for old, new in zip(old_cases, new_cases):
        if old["inputs"] != new["inputs"]:
            raise ValueError(f"{old['name']}: reference inputs changed")
        if set(old["expected"]) != set(new["expected"]):
            raise ValueError(f"{old['name']}: reference fields changed")
        for field, value in old["expected"].items():
            if field == "breaks":
                for i, (old_cut, new_cut) in enumerate(zip(value, new["expected"][field])):
                    compare(new_cut, old_cut, f"{old['name']}.breaks[{i}].saved_R", atol=1e-10)
            else:
                compare(new["expected"][field], value, f"{old['name']}.{field}.saved_R",
                        atol=2e-9)
        data, expected = new["inputs"], new["expected"]
        x = np.column_stack((data["x1"], data["x2"]))
        z = None if data["z"] is None else np.asarray(data["z"])[:, None]
        result = _from_indices(data["y"], x, data["indices"], z, q=data["q"])
        actual = {"psi": result.psi, "se": result.se, "ci": result.ci,
                  "p": result.p_value, "means": result.intervention_means,
                  "effects": result.bootstrap_effects, "scores": result.point_fit.scores}
        gaps = {name: compare(value, expected[name], f"{old['name']}.{name}.Python",
                              atol=2e-9)
                for name, value in actual.items()}
        if data["q"] is None:
            if result.point_fit.breaks is not None:
                raise ValueError("Python prequantized breaks contract changed")
        else:
            for i, (cut, ref) in enumerate(zip(result.point_fit.breaks, expected["breaks"])):
                gaps[f"breaks[{i}]"] = compare(cut, ref, f"{old['name']}.breaks[{i}].Python",
                                                atol=1e-10)
        results.append({"name": old["name"], "max_absolute_differences": gaps})
    return {"status": "passed", "checked_utc": datetime.now(timezone.utc).isoformat(),
            "qgcomp_version": "2.19.6", "saved_fixture_sha256": FIXTURE_SHA,
            "r_exporter_sha256": DRIVER_SHA, "cases": results,
            "scope": "One prequantized and one quantized Poisson case; not interval calibration.",
            "prequantized_breaks_difference": "R saves synthetic boundaries; Python retains breaks=None."}


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
        raise ValueError("Poisson fixture or exporter hash differs")
    with tempfile.TemporaryDirectory() as folder:
        live_path = Path(folder) / "live.json"
        subprocess.run([args.rscript, str(DRIVER), args.library, str(live_path)], check=True)
        report = check(json.loads(FIXTURE.read_text()), json.loads(live_path.read_text()))
    output.write_text(json.dumps(report, indent=2)+"\n")
    print("Two marginal Poisson cases agree with installed R qgcomp 2.19.6")


if __name__ == "__main__":
    main()
