"""Optional conditional multinomial check against installed R qgcomp 2.19.6."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from quantilemix import fit_multinomial


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/r_multinomial_2_19_6.json"
EXPORTER = Path(__file__).with_name("export_multinomial_r.R")
FIXTURE_SHA256 = "c5d4b863ca0ef674e2feb1e4eca7d3498b8c44bbf146d6d08f6bd611f43fd4f8"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _equal(actual, expected, label: str, *, tolerance: float, exact=False) -> float:
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError(f"{label}: shape or finiteness differs")
    if exact:
        agree = np.array_equal(a, b)
    else:
        agree = np.allclose(a, b, rtol=0, atol=tolerance)
    if not agree:
        raise ValueError(f"{label}: values differ")
    return float(np.max(np.abs(a-b)))


def _fields(expected: dict):
    if set(expected) != {"coefficients", "covariance", "psi", "psi_covariance", "ci", "p",
                         "fitted_probabilities", "scores", "breaks", "joint_test", "homogeneity_test"}:
        raise ValueError("R expected fields differ")


def _compare_expected(actual: dict, expected: dict, name: str, mode: str) -> dict:
    _fields(actual)
    _fields(expected)
    base = 1e-5 if mode == "default" else 1e-7
    gaps = {}
    for key in ("coefficients", "covariance", "psi", "psi_covariance", "ci", "p",
                "fitted_probabilities", "scores"):
        gaps[key] = _equal(actual[key], expected[key], f"{name}.{mode}.{key}",
                           tolerance=base, exact=key == "scores")
    if (actual["breaks"] is None) != (expected["breaks"] is None):
        raise ValueError(f"{name}.{mode}.breaks: presence differs")
    if actual["breaks"] is not None:
        if len(actual["breaks"]) != len(expected["breaks"]):
            raise ValueError(f"{name}.{mode}.breaks: count differs")
        for j, (a, b) in enumerate(zip(actual["breaks"], expected["breaks"])):
            _equal(a, b, f"{name}.{mode}.breaks[{j}]", tolerance=1e-8)
    for test_name in ("joint_test", "homogeneity_test"):
        a, b = actual[test_name], expected[test_name]
        if a["df"] != b["df"]:
            raise ValueError(f"{name}.{mode}.{test_name}.df differs")
        limit = 3e-5 if mode == "default" else 1e-6
        gaps[test_name+"_chisq"] = _equal(a["chisq"], b["chisq"],
                                           f"{name}.{mode}.{test_name}.chisq", tolerance=limit)
        gaps[test_name+"_p"] = _equal(a["p"], b["p"],
                                       f"{name}.{mode}.{test_name}.p", tolerance=base)
    return gaps


def compare_r_exports(saved: dict, installed: dict) -> None:
    if installed.get("qgcomp_version") != "2.19.6":
        raise ValueError("installed qgcomp version differs")
    if len(saved.get("cases", [])) != len(installed.get("cases", [])):
        raise ValueError("installed R case count differs")
    for old, new in zip(saved["cases"], installed["cases"]):
        if old["name"] != new["name"] or old["inputs"] != new["inputs"]:
            raise ValueError("installed R case identity or inputs differ")
        _compare_expected(old["expected"], new["expected"], old["name"], "default")
        _compare_expected(old["tighter_expected"], new["tighter_expected"], old["name"], "tighter")


def compare(reference: dict) -> dict:
    cases = reference["cases"]
    names = {case["name"] for case in cases}
    if names != {"multinomial_prequantized", "multinomial_quantized"} or len(cases) != 2:
        raise ValueError("two distinct multinomial cases are required")
    results = []
    for case in cases:
        name, x = case["name"], case["inputs"]
        fit = fit_multinomial(x["y"], np.column_stack((x["x1"], x["x2"])),
                              x["z"], levels=x["levels"], q=x["q"])
        actual = {"coefficients": fit.coefficients, "covariance": fit.covariance,
                  "psi": fit.psi, "psi_covariance": fit.psi_covariance,
                  "ci": fit.ci, "p": fit.p_value,
                  "fitted_probabilities": fit.fitted_probabilities,
                  "scores": fit.scores, "breaks": fit.breaks,
                  "joint_test": {"chisq": fit.joint_test()["chi_square"],
                                 "df": fit.joint_test()["df"], "p": fit.joint_test()["p_value"]},
                  "homogeneity_test": {"chisq": fit.homogeneity_test()["chi_square"],
                                       "df": fit.homogeneity_test()["df"],
                                       "p": fit.homogeneity_test()["p_value"]}}
        results.append({"name": name,
                        "default_R_gaps": _compare_expected(actual, case["expected"], name, "default"),
                        "tighter_R_gaps": _compare_expected(actual, case["tighter_expected"], name, "tighter")})
    return {"status": "passed", "cases": results,
            "maximum_default_R_gap": max(v for item in results for v in item["default_R_gaps"].values()),
            "maximum_tighter_R_gap": max(v for item in results for v in item["tighter_R_gaps"].values())}


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
                              "scope": "two fixed conditional multinomial cases with default and tighter R optimizer settings",
                              "tighter_reltol": 1e-13, "tighter_maxit": 1000}:
        parser.error("pinned exporter identity differs")
    library = Path(args.r_library)
    if not (library / "qgcomp/DESCRIPTION").is_file():
        parser.error("isolated R qgcomp library unavailable")
    with tempfile.TemporaryDirectory(prefix="qgcomp-multinomial-r-") as directory:
        fresh = Path(directory) / "export.json"
        completed = subprocess.run([args.rscript, str(EXPORTER), str(library), str(fresh)],
                                   capture_output=True, text=True, timeout=60)
        if completed.returncode:
            raise SystemExit(f"R exporter failed: {completed.stderr[-1000:]}")
        installed = json.loads(fresh.read_text())
    compare_r_exports(saved, installed)
    report = compare(installed)
    report.update({"R_version": installed["R_version"], "qgcomp_version": installed["qgcomp_version"],
                   "nnet_version": installed["nnet_version"],
                   "fixture_sha256": sha256(FIXTURE), "exporter_sha256": sha256(EXPORTER)})
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
