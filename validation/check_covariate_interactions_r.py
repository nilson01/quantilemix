"""Optional installed-R comparison for Gaussian exposure-covariate products."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from quantilemix._quadratic import _fit_quadratic_with_indices


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/r_covariate_interactions_2_19_6.json"
DRIVER = Path(__file__).with_name("export_covariate_interactions_r.R")
FIXTURE_SHA = "8bd0bb56f47391113c7fcf8e2901bb2de12775827de7f4032493d23354b8a6de"
DRIVER_SHA = "617acdf55a31d56a4e11d281efce938418db94cd1937c8621b11a52e68a4e84d"
NAMES = ("covariate_interactions_prequantized",
         "covariate_interactions_pairwise_quantized")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(actual, expected, label: str, *, atol: float) -> float:
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError(f"{label}: shape or finiteness differs")
    gap = float(np.max(np.abs(a-b)))
    if not np.allclose(a, b, rtol=0, atol=atol):
        raise ValueError(f"{label}: numerical difference {gap:.3g} exceeds {atol:.3g}")
    return gap


def check(saved: dict, live: dict) -> dict:
    if saved.get("qgcomp_version") != "2.19.6" or live.get("qgcomp_version") != "2.19.6":
        raise ValueError("installed R qgcomp 2.19.6 required")
    a, b = saved.get("cases"), live.get("cases")
    if (not isinstance(a, list) or not isinstance(b, list)
            or tuple(c["name"] for c in a) != NAMES
            or tuple(c["name"] for c in b) != NAMES):
        raise ValueError("two fixed exposure-covariate interaction cases required")
    results = []
    for old, new in zip(a, b):
        if old["inputs"] != new["inputs"] or old["resample_indices"] != new["resample_indices"]:
            raise ValueError(f"{old['name']}: inputs or bootstrap schedule changed")
        if set(old["expected"]) != set(new["expected"]):
            raise ValueError(f"{old['name']}: reference fields changed")
        for field, value in old["expected"].items():
            if field == "breaks":
                for i, (old_cut, new_cut) in enumerate(zip(value, new["expected"][field])):
                    compare(new_cut, old_cut, f"{old['name']}.breaks[{i}].saved_R", atol=1e-10)
            else:
                compare(new["expected"][field], value, f"{old['name']}.{field}.saved_R",
                        atol=1e-10)
        inp, expected = new["inputs"], new["expected"]
        x = np.column_stack((inp["x1"], inp["x2"]))
        fit = _fit_quadratic_with_indices(
            inp["y"], x, new["resample_indices"], inp["z"], q=inp["q"],
            pairwise_interactions=inp["pairwise"], covariate_interactions=True)
        actual = {"msm_coefficients": fit.msm_coefficients, "psi": fit.psi,
                  "covariance": fit.covariance, "se": fit.se, "ci": fit.ci,
                  "p": fit.p_value, "df": fit.p_value_df,
                  "bootstrap_coefficients": fit.bootstrap_coefficients,
                  "levels": fit.intervention_levels,
                  "means": fit.intervention_means, "scores": fit.scores}
        gaps = {field: compare(value, expected[field], f"{old['name']}.{field}.Python",
                               atol=1e-10)
                for field, value in actual.items()}
        if inp["q"] is None:
            if fit.breaks is not None:
                raise ValueError("Python prequantized breaks contract changed")
        else:
            for i, (cut, ref) in enumerate(zip(fit.breaks, expected["breaks"])):
                gaps[f"breaks[{i}]"] = compare(cut, ref,
                                                f"{old['name']}.breaks[{i}].Python",
                                                atol=1e-10)
        results.append({"name": old["name"], "max_absolute_differences": gaps})
    return {"status": "passed", "checked_utc": datetime.now(timezone.utc).isoformat(),
            "qgcomp_version": "2.19.6", "saved_fixture_sha256": FIXTURE_SHA,
            "r_exporter_sha256": DRIVER_SHA, "cases": results,
            "prequantized_breaks_difference":
                "R saves synthetic boundaries; Python retains breaks=None."}


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
        raise ValueError("interaction fixture or exporter hash differs")
    with tempfile.TemporaryDirectory() as folder:
        live_path = Path(folder) / "live.json"
        subprocess.run([args.rscript, str(DRIVER), args.library, str(live_path)], check=True)
        report = check(json.loads(FIXTURE.read_text()), json.loads(live_path.read_text()))
    output.write_text(json.dumps(report, indent=2)+"\n")
    print("Two exposure-covariate cases agree with isolated installed R qgcomp 2.19.6")


if __name__ == "__main__":
    main()
