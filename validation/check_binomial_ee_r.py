"""Optional installed-R comparison for clustered binary estimating equations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from scipy import special

from quantilemix import fit_binomial_ee

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/r_binomial_ee_2_19_6.json"
DRIVER = Path(__file__).with_name("export_binomial_ee_r.R")
FIXTURE_SHA = "17a7d0d0e379449ca22fafedc22db3b216aa05d731459dff88a24ce6da7a8e98"
DRIVER_SHA = "a7c1c59316e9deecaf9f7f285b21c127cb1bf2d2bb719bc305711979123d3ae0"
CASES = (("binomial_ee_q2_or", False, 2), ("binomial_ee_q2_rr", True, 2),
         ("binomial_ee_q4_or", False, 4), ("binomial_ee_q4_rr", True, 4))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_value(actual, reference, label: str, *, atol: float) -> float:
    a, b = np.asarray(actual, dtype=float), np.asarray(reference, dtype=float)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError(f"{label}: shape or finiteness differs")
    gap = float(np.max(np.abs(a-b)))
    if not np.allclose(a, b, rtol=0, atol=atol):
        raise ValueError(f"{label}: numerical difference {gap:.3g} exceeds {atol:.3g}")
    return gap


def check(saved: dict, live: dict) -> dict:
    if saved.get("qgcomp_version") != "2.19.6" or live.get("qgcomp_version") != "2.19.6":
        raise ValueError("installed qgcomp 2.19.6 required")
    old_cases, new_cases = saved.get("cases"), live.get("cases")
    if (not isinstance(old_cases, list) or not isinstance(new_cases, list)
            or len(old_cases) != 4 or len(new_cases) != 4):
        raise ValueError("four fixed binary EE cases required")
    if (tuple((case["name"], case["rr"], case["inputs"]["q"]) for case in old_cases) != CASES
            or tuple((case["name"], case["rr"], case["inputs"]["q"]) for case in new_cases) != CASES):
        raise ValueError("binary EE case identities differ")
    results = []
    for old, new in zip(old_cases, new_cases):
        if old["inputs"] != new["inputs"]:
            raise ValueError(f"{old['name']}: inputs changed")
        if set(old["expected"]) != set(new["expected"]):
            raise ValueError(f"{old['name']}: reference fields changed")
        for name, reference in old["expected"].items():
            compare_value(new["expected"][name], reference,
                          f"{old['name']}.{name}.saved_R", atol=1e-10)
        a, e = new["inputs"], new["expected"]
        x = np.column_stack((a["x1"], a["x2"]))
        result = fit_binomial_ee(a["y"], x, a["id"], a["z"], q=a["q"], rr=new["rr"])
        zstats = result.msm_coefficients / np.sqrt(np.diag(result.msm_covariance))
        comparisons = {
            "conditional_coefficients": result.conditional_coefficients,
            "conditional_covariance": result.conditional_covariance,
            "msm_coefficients": result.msm_coefficients,
            "msm_covariance": result.msm_covariance,
            "full_covariance": result.full_covariance,
            "bread": result.bread, "meat": result.meat,
            "psi": result.psi, "variance": result.var_psi,
            "ci": result.ci, "statistic": zstats,
            "p": 2*special.ndtr(-abs(zstats)),
            "scores": result.scores, "levels": result.intervention_levels,
            "risks": result.intervention_risks,
            "msm_fitted_risks": result.msm_fitted_risks,
        }
        gaps = {name: compare_value(value, e[name], f"{old['name']}.{name}.Python",
                                    atol=2e-7 if name == "bread" else
                                    5e-10 if name == "statistic" else 2e-10)
                for name, value in comparisons.items()}
        for i, (value, reference) in enumerate(zip(result.breaks, e["breaks"])):
            gaps[f"breaks[{i}]"] = compare_value(
                value, reference, f"{old['name']}.breaks[{i}].Python", atol=1e-10)
        results.append({"name": old["name"], "max_absolute_differences": gaps})
    return {"status": "passed", "checked_utc": datetime.now(timezone.utc).isoformat(),
            "qgcomp_version": "2.19.6", "saved_fixture_sha256": FIXTURE_SHA,
            "r_exporter_sha256": DRIVER_SHA, "cases": results,
            "derivative_difference": "R saves a Richardson numerical Jacobian; Python uses exact derivatives.",
            "marginal_links": "rr=False uses logit odds; rr=True uses log risk, both with a conditional logistic fit."}


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
        raise ValueError("binary EE fixture or exporter hash differs")
    with tempfile.TemporaryDirectory() as folder:
        live_path = Path(folder) / "live.json"
        subprocess.run([args.rscript, str(DRIVER), args.library, str(live_path)], check=True)
        report = check(json.loads(FIXTURE.read_text()), json.loads(live_path.read_text()))
    output.write_text(json.dumps(report, indent=2)+"\n")
    print("Four binary EE cases agree with isolated installed R qgcomp 2.19.6")


if __name__ == "__main__":
    main()
