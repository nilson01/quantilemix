"""Optional comparison with an isolated installed R qgcomp 2.19.6."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from quantilemix import fit_cox


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/r_cox_2_19_6.json"
EXPORTER = Path(__file__).with_name("export_cox_r.R")
FIXTURE_SHA256 = "72d1a7c5dc170a35469ec53f3d32474f228e5ac0b4631fefccb972aac50932d1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(reference: dict) -> dict:
    cases = reference["cases"]
    names = {c["name"] for c in cases}
    if names != {"cox_prequantized_untied", "cox_quantized_untied"} or len(cases) != 2:
        raise ValueError("two distinct fixed Cox cases are required")
    result = []
    for case in cases:
        x, expected = case["inputs"], case["expected"]
        fit = fit_cox(x["time"], x["event"], np.column_stack((x["x1"], x["x2"])),
                      x["z"], q=x["q"])
        actual = {"coefficients": fit.coefficients, "covariance": fit.covariance,
                  "psi": fit.psi, "variance": fit.var_psi, "ci": fit.ci,
                  "p": fit.p_value, "scores": fit.scores, "events": fit.event_count}
        gaps = {}
        for key, value in actual.items():
            a, b = np.asarray(value, dtype=float), np.asarray(expected[key], dtype=float)
            if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
                raise ValueError(f"{case['name']}.{key}: shape or finiteness mismatch")
            gap = float(np.max(np.abs(a-b)))
            if key in {"scores", "events"}:
                if not np.array_equal(a, b):
                    raise ValueError(f"{case['name']}.{key}: mismatch")
            elif not np.allclose(a, b, rtol=1e-8, atol=1e-8):
                raise ValueError(f"{case['name']}.{key}: mismatch")
            gaps[key] = gap
        if (fit.breaks is None) != (expected["breaks"] is None):
            raise ValueError(f"{case['name']}.breaks: presence mismatch")
        if fit.breaks is not None:
            for a, b in zip(fit.breaks, expected["breaks"]):
                if not np.allclose(a, b, rtol=0, atol=1e-8):
                    raise ValueError(f"{case['name']}.breaks: mismatch")
        result.append({"name": case["name"], "max_absolute_difference": gaps})
    return {"status": "passed", "cases": result,
            "maximum_numeric_gap": max(v for item in result for k, v in item["max_absolute_difference"].items()
                                       if k not in {"scores", "events"})}


def compare_r_exports(saved: dict, installed: dict) -> None:
    if installed.get("qgcomp_version") != "2.19.6":
        raise ValueError("installed qgcomp version differs")
    if len(saved.get("cases", [])) != len(installed.get("cases", [])):
        raise ValueError("installed R case count differs")
    for old, new in zip(saved["cases"], installed["cases"]):
        if old["name"] != new["name"] or old["inputs"] != new["inputs"]:
            raise ValueError("installed R case identity or inputs differ")
        if old["expected"].keys() != new["expected"].keys():
            raise ValueError("installed R expected fields differ")
        for field, value in old["expected"].items():
            other = new["expected"][field]
            if value is None or other is None:
                if value is not None or other is not None:
                    raise ValueError(f"installed R {field} differs")
            else:
                a, b = np.asarray(value, dtype=float), np.asarray(other, dtype=float)
                if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
                    raise ValueError(f"installed R {field} shape or finiteness differs")
                if not np.allclose(a, b, rtol=1e-8, atol=1e-8):
                    raise ValueError(f"installed R {field} differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r-library", required=True)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        parser.error("output must be a new path in an existing directory")
    if sha256(FIXTURE) != FIXTURE_SHA256:
        parser.error("pinned fixture digest differs")
    saved = json.loads(FIXTURE.read_text())
    if saved["reference"] != {"version": "2.19.6", "exporter_sha256": sha256(EXPORTER),
                              "scope": "two fixed untied right-censored cases"}:
        parser.error("pinned exporter identity differs")
    library = Path(args.r_library)
    if not (library / "qgcomp/DESCRIPTION").is_file():
        parser.error("isolated R qgcomp library unavailable")
    with tempfile.TemporaryDirectory(prefix="qgcomp-cox-r-") as directory:
        fresh = Path(directory) / "export.json"
        completed = subprocess.run([args.rscript, str(EXPORTER), str(library), str(fresh)],
                                   capture_output=True, text=True, timeout=60)
        if completed.returncode:
            raise SystemExit(f"R exporter failed: {completed.stderr[-1000:]}")
        installed = json.loads(fresh.read_text())
    compare_r_exports(saved, installed)
    report = compare(installed)
    report.update({"R_version": installed["R_version"], "qgcomp_version": installed["qgcomp_version"],
                   "survival_version": installed["survival_version"],
                   "fixture_sha256": sha256(FIXTURE), "exporter_sha256": sha256(EXPORTER)})
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
