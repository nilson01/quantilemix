"""Optional replay of conditional Cox curves against isolated installed R."""

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
INPUT = ROOT / "tests/fixtures/r_cox_ties_2_19_6.json"
FIXTURE = ROOT / "tests/fixtures/r_cox_curves_2_19_6.json"
DRIVER = Path(__file__).with_name("export_cox_curves_r.R")
INPUT_SHA = "c2cca8cb735dab35b299cbd485f5e88c86d2b3097cd0049331faa9e2ac5fa161"
FIXTURE_SHA = "25aebf6afb2dd9d4a46eb2959faf6e2d25042fd9b6045e824de71b47511f82ac"
DRIVER_SHA = "08813ce1a685dd3a19e07e360900a17c71906b5a953fc35e2d8a34ced7c7b2c7"
NAMES = ("cox_prequantized_efron_ties", "cox_quantized_efron_ties")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(actual, reference, label, *, atol=1e-9) -> float:
    a, b = np.asarray(actual, dtype=float), np.asarray(reference, dtype=float)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError(f"{label}: shape or finiteness differs")
    gap = float(np.max(np.abs(a-b)))
    if gap > atol:
        raise ValueError(f"{label}: difference {gap:.3g} exceeds {atol:.3g}")
    return gap


def check(saved: dict, live: dict, inputs: dict) -> dict:
    if saved.get("qgcomp_version") != "2.19.6" or live.get("qgcomp_version") != "2.19.6":
        raise ValueError("installed qgcomp 2.19.6 required")
    old_cases, new_cases, input_cases = saved.get("cases"), live.get("cases"), inputs.get("cases")
    if (not all(isinstance(cases, list) and len(cases) == 2
                for cases in (old_cases, new_cases, input_cases))
            or any(tuple(case.get("name") for case in cases) != NAMES
                   for cases in (old_cases, new_cases, input_cases))):
        raise ValueError("two named conditional Cox curve cases required")
    details = []
    for old, new, source in zip(old_cases, new_cases, input_cases):
        name = old["name"]
        if old["profile_rows"] != new["profile_rows"]:
            raise ValueError(f"{name}: profile rows differ")
        expected, current = old["expected"], new["expected"]
        if set(expected) != set(current):
            raise ValueError(f"{name}: reference fields differ")
        for key in expected:
            compare(current[key], expected[key], f"{name}.{key}.saved_R")
        x = source["inputs"]
        model = fit_cox(x["time"], x["event"],
                        np.column_stack((x["x1"], x["x2"])), x["z"], q=x["q"])
        rows = np.asarray(old["profile_rows"], dtype=int) - 1
        if len(rows) != 4 or np.any(rows < 0) or np.any(rows >= len(x["time"])):
            raise ValueError(f"{name}: invalid profile rows")
        raw = np.column_stack((x["x1"], x["x2"]))[rows]
        scores = model._quantization.transform(raw).scores
        timeline, survival = model.survival_curve(raw, np.asarray(x["z"])[rows])
        actual = {"time": timeline, "baseline_time": model.timeline,
                  "baseline_cumhaz": model.baseline_cumulative_hazard,
                  "survival": survival, "profile_scores": scores}
        gaps = {key: compare(value, expected[key], f"{name}.{key}.Python")
                for key, value in actual.items()}
        details.append({"name": name, "max_absolute_differences": gaps})
    return {"status": "passed", "checked_utc": datetime.now(timezone.utc).isoformat(),
            "qgcomp_version": "2.19.6", "input_sha256": INPUT_SHA,
            "saved_fixture_sha256": FIXTURE_SHA, "r_exporter_sha256": DRIVER_SHA,
            "cases": details}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise ValueError("output already exists")
    if sha(INPUT) != INPUT_SHA or sha(FIXTURE) != FIXTURE_SHA or sha(DRIVER) != DRIVER_SHA:
        raise ValueError("conditional Cox curve fixture, input or exporter hash differs")
    with tempfile.TemporaryDirectory() as folder:
        live_path = Path(folder) / "live.json"
        subprocess.run([args.rscript, str(DRIVER), args.library, str(INPUT), str(live_path)], check=True)
        report = check(json.loads(FIXTURE.read_text()), json.loads(live_path.read_text()),
                       json.loads(INPUT.read_text()))
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("Two conditional Cox curves agree with isolated installed R qgcomp 2.19.6")


if __name__ == "__main__":
    main()
