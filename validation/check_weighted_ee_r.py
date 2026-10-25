"""Optional installed-R comparison for weighted Gaussian, Poisson and binary EE."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from scipy import special, stats

from quantilemix import fit_gaussian_ee, fit_poisson_ee, fit_binomial_ee

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/r_weighted_ee_2_19_6.json"
DRIVER = Path(__file__).with_name("export_weighted_ee_r.R")
FIXTURE_SHA = "61646422bf2d2a8ef99620135bbfe284dc64d420b1f0363e771289420177b450"
DRIVER_SHA = "3d9299c9b914310a57edb829a09b43deee9adbe680ca73ebd6271d7a30410d03"
CASES = (("weighted_gaussian_ee", "gaussian", True),
         ("weighted_poisson_ee", "poisson", True),
         ("weighted_binomial_ee_or", "binomial", False),
         ("weighted_binomial_ee_rr", "binomial", True))
FITS = {"gaussian": fit_gaussian_ee, "poisson": fit_poisson_ee,
        "binomial": fit_binomial_ee}


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
        raise ValueError("four fixed weighted EE cases required")
    def identity(cases):
        return tuple((case["name"], case["family"], case["rr"])
                     for case in cases)
    if identity(old_cases) != CASES or identity(new_cases) != CASES:
        raise ValueError("weighted EE case identities differ")
    results = []
    for old, new in zip(old_cases, new_cases):
        if old["inputs"] != new["inputs"]:
            raise ValueError(f"{old['name']}: inputs changed")
        if set(old["expected"]) != set(new["expected"]):
            raise ValueError(f"{old['name']}: reference fields changed")
        for name, reference in old["expected"].items():
            if reference is None:
                if new["expected"][name] is not None:
                    raise ValueError(f"{old['name']}.{name}: saved-R null differs")
            elif name == "breaks":
                for i, (value, previous) in enumerate(zip(new["expected"][name], reference)):
                    compare_value(value, previous, f"{old['name']}.breaks[{i}].saved_R", atol=1e-10)
            else:
                compare_value(new["expected"][name], reference,
                              f"{old['name']}.{name}.saved_R", atol=1e-10)
        a, e = new["inputs"], new["expected"]
        x = np.column_stack((a["x1"], a["x2"]))
        family = new["family"]
        options = {"rr": new["rr"]} if family == "binomial" else {}
        result = FITS[family](a["y"], x, a["id"], a["z"], q=a["q"],
                              weights=a["w"], **options)
        zstats = result.msm_coefficients / np.sqrt(np.diag(result.msm_covariance))
        p = (2*stats.t.sf(abs(zstats), result.p_value_df) if family == "gaussian"
             else 2*special.ndtr(-abs(zstats)))
        standardized = (result.intervention_risks if family == "binomial"
                        else result.intervention_means)
        comparisons = {
            "conditional_coefficients": result.conditional_coefficients,
            "conditional_covariance": result.conditional_covariance,
            "msm_coefficients": result.msm_coefficients,
            "msm_covariance": result.msm_covariance,
            "full_covariance": result.full_covariance,
            "bread": result.bread, "meat": result.meat,
            "psi": result.psi, "variance": result.var_psi,
            "ci": result.ci, "statistic": zstats, "p": p,
            "scores": result.scores,
            "levels": result.intervention_levels, "standardized": standardized,
        }
        gaps = {name: compare_value(value, e[name], f"{old['name']}.{name}.Python",
                                    atol=1e-6 if name == "bread" else
                                    1e-8 if name == "meat" else 2e-9)
                for name, value in comparisons.items()}
        if e["df"] is not None:
            gaps["df"] = compare_value(result.p_value_df, e["df"],
                                       f"{old['name']}.df.Python", atol=0)
        if len(result.breaks) != len(e["breaks"]):
            raise ValueError(f"{old['name']}: break count differs")
        for i, (value, reference) in enumerate(zip(result.breaks, e["breaks"])):
            gaps[f"breaks[{i}]"] = compare_value(
                value, reference, f"{old['name']}.breaks[{i}].Python", atol=1e-10)
        results.append({"name": old["name"], "max_absolute_differences": gaps})
    return {"status": "passed", "checked_utc": datetime.now(timezone.utc).isoformat(),
            "qgcomp_version": "2.19.6", "saved_fixture_sha256": FIXTURE_SHA,
            "r_exporter_sha256": DRIVER_SHA, "cases": results,
            "derivative_difference": "R saves a Richardson numerical Jacobian; Python uses exact derivatives."}


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
        raise ValueError("weighted EE fixture or exporter hash differs")
    with tempfile.TemporaryDirectory() as folder:
        live_path = Path(folder) / "live.json"
        subprocess.run([args.rscript, str(DRIVER), args.library, str(live_path)], check=True)
        report = check(json.loads(FIXTURE.read_text()), json.loads(live_path.read_text()))
    output.write_text(json.dumps(report, indent=2)+"\n")
    print("Four weighted EE cases agree with isolated installed R qgcomp 2.19.6")


if __name__ == "__main__":
    main()
