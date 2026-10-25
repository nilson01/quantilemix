"""Optional Gaussian Tobit check against isolated installed R qgcomp 2.19.6."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from quantilemix import fit_tobit


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/r_tobit_2_19_6.json"
EXPORTER = Path(__file__).with_name("export_tobit_r.R")
FIXTURE_SHA256 = "3c16007faf1c30fecef9129ee2da15134c53be56eebf402882485d20da41b978"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _equal(actual, expected, label: str, *, exact=False, rtol=1e-8, atol=1e-8) -> float:
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError(f"{label}: shape or finiteness differs")
    if exact:
        agree = np.array_equal(a, b)
    else:
        agree = np.allclose(a, b, rtol=rtol, atol=atol)
    if not agree:
        raise ValueError(f"{label}: values differ")
    return float(np.max(np.abs(a-b)))


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
                _equal(value, other, f"installed R {field}", exact=field in {"scores", "left_count", "right_count"},
                       rtol=1e-5 if field == "p" else 1e-8,
                       atol=1e-14 if field == "p" else 1e-8)


def compare(reference: dict) -> dict:
    cases = reference["cases"]
    names = {case["name"] for case in cases}
    if names != {"tobit_left_prequantized", "tobit_two_sided_quantized"} or len(cases) != 2:
        raise ValueError("two distinct fixed Tobit cases are required")
    results = []
    for case in cases:
        name, x, expected = case["name"], case["inputs"], case["expected"]
        fit = fit_tobit(x["y"], np.column_stack((x["x1"], x["x2"])), x["z"],
                        q=x["q"], left=x["left"], right=x["right"])
        fields = {"coefficients": fit.coefficients, "covariance": fit.full_covariance,
                  "sigma": fit.sigma, "psi": fit.psi, "variance": fit.var_psi,
                  "ci": fit.ci, "p": fit.p_value, "scores": fit.scores,
                  "left_count": fit.left_count, "right_count": fit.right_count}
        gaps = {}
        for field, actual in fields.items():
            gaps[field] = _equal(actual, expected[field], f"{name}.{field}",
                                 exact=field in {"scores", "left_count", "right_count"},
                                 rtol=1e-5 if field == "p" else 1e-8,
                                 atol=1e-14 if field == "p" else 1e-8)
        if (fit.breaks is None) != (expected["breaks"] is None):
            raise ValueError(f"{name}.breaks: presence differs")
        if fit.breaks is not None:
            if len(fit.breaks) != len(expected["breaks"]):
                raise ValueError(f"{name}.breaks: count differs")
            for j, (a, b) in enumerate(zip(fit.breaks, expected["breaks"])):
                _equal(a, b, f"{name}.breaks[{j}]")
        results.append({"name": name, "max_absolute_difference": gaps})
    return {"status": "passed", "cases": results,
            "maximum_numeric_gap": max(v for item in results for k, v in item["max_absolute_difference"].items()
                                       if k not in {"scores", "left_count", "right_count"})}


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
                              "scope": "two fixed Gaussian Tobit cases"}:
        parser.error("pinned exporter identity differs")
    library = Path(args.r_library)
    if not (library / "qgcomp/DESCRIPTION").is_file():
        parser.error("isolated R qgcomp library unavailable")
    with tempfile.TemporaryDirectory(prefix="qgcomp-tobit-r-") as directory:
        fresh = Path(directory) / "export.json"
        completed = subprocess.run([args.rscript, str(EXPORTER), str(library), str(fresh)],
                                   capture_output=True, text=True, timeout=60)
        if completed.returncode:
            raise SystemExit(f"R exporter failed: {completed.stderr[-1000:]}")
        installed = json.loads(fresh.read_text())
    compare_r_exports(saved, installed)
    report = compare(installed)
    report.update({"R_version": installed["R_version"], "qgcomp_version": installed["qgcomp_version"],
                   "AER_version": installed["AER_version"], "survival_version": installed["survival_version"],
                   "fixture_sha256": sha256(FIXTURE), "exporter_sha256": sha256(EXPORTER)})
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
