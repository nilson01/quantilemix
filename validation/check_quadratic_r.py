"""Optional installed-R comparison for restricted quadratic Gaussian bootstrap."""

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
FIXTURE = ROOT / "tests/fixtures/r_quadratic_2_19_6.json"
DRIVER = Path(__file__).with_name("export_quadratic_r.R")
FIXTURE_SHA = "4a22a63abcc07d05223818b6b9614be09bbfffd34a05c1c71550cc0bced647ea"
DRIVER_SHA = "a41b95caf702cc263c4dcf01ec6a697eacae832d2c67173052aae264790c8934"
NAMES = ("quadratic_prequantized", "quadratic_quantized")


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
    a, b = saved.get("cases"), live.get("cases")
    if not isinstance(a, list) or not isinstance(b, list) or len(a) != 2 or len(b) != 2:
        raise ValueError("two fixed quadratic cases required")
    if tuple(case["name"] for case in a) != NAMES or tuple(case["name"] for case in b) != NAMES:
        raise ValueError("quadratic case identities differ")
    results = []
    for old, new in zip(a, b):
        if old["inputs"] != new["inputs"] or old["resample_indices"] != new["resample_indices"]:
            raise ValueError(f"{old['name']}: inputs or bootstrap schedule changed")
        if set(old["expected"]) != set(new["expected"]):
            raise ValueError(f"{old['name']}: reference fields changed")
        for field, old_value in old["expected"].items():
            value = new["expected"][field]
            if old_value is None or value is None:
                if old_value is not None or value is not None:
                    raise ValueError(f"{old['name']}.{field}: missing value differs")
                continue
            compare_value(value, old_value, f"{old['name']}.{field}.saved_R", atol=5e-10)
        inp, expected = new["inputs"], new["expected"]
        x = np.column_stack((inp["x1"], inp["x2"]))
        fit = _fit_quadratic_with_indices(inp["y"], x, new["resample_indices"],
                                           inp["z"], q=inp["q"])
        comparisons = {
            "conditional_coefficients": fit.conditional_coefficients,
            "conditional_covariance": fit.conditional_covariance,
            "fitted": fit.fitted_values,
            "msm_coefficients": fit.msm_coefficients,
            "psi": fit.psi, "covariance": fit.covariance,
            "variance": fit.se**2, "se": fit.se, "ci": fit.ci,
            "statistic": fit.test_statistic, "p": fit.p_value,
            "df": fit.p_value_df,
            "bootstrap_coefficients": fit.bootstrap_coefficients,
            "levels": fit.intervention_levels,
            "means": fit.intervention_means, "scores": fit.scores,
        }
        gaps = {field: compare_value(value, expected[field],
                                     f"{old['name']}.{field}.Python", atol=1e-10)
                for field, value in comparisons.items()}
        if inp["q"] is not None:
            for i, (value, reference) in enumerate(zip(fit.breaks, expected["breaks"])):
                gaps[f"breaks[{i}]"] = compare_value(value, reference,
                                                     f"{old['name']}.breaks[{i}].Python",
                                                     atol=5e-12)
        elif fit.breaks is not None:
            raise ValueError("Python prequantized breaks contract changed")
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
        raise ValueError("quadratic fixture or exporter hash differs")
    with tempfile.TemporaryDirectory() as folder:
        live_path = Path(folder) / "live.json"
        subprocess.run([args.rscript, str(DRIVER), args.library, str(live_path)], check=True)
        report = check(json.loads(FIXTURE.read_text()), json.loads(live_path.read_text()))
    output.write_text(json.dumps(report, indent=2)+"\n")
    print("Two quadratic Gaussian cases agree with isolated installed R qgcomp 2.19.6")


if __name__ == "__main__":
    main()
