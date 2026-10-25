"""Optional installed-R comparison for Gaussian EE marginal-curve bounds."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from quantilemix import fit_gaussian_ee, modelbound_ee
from quantilemix._modelbound_ee import _from_coefficient_draws

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/r_modelbound_ee_2_19_6.json"
DRIVER = Path(__file__).with_name("export_modelbound_ee_r.R")
FIXTURE_SHA = "7f8141e241ed84a7ac97dd5d0d598cb48d1c67a607688a28d3105897575934fc"
DRIVER_SHA = "9ab5dc29d13dbd790404de3aa350d938dd440715bf4c368951e81e09dbb6bd01"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_value(actual, reference, label: str, *, atol: float = 2e-9) -> float:
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
    if saved.get("inputs") != live.get("inputs"):
        raise ValueError("Gaussian EE input dataset changed")
    a = live["inputs"]
    x = np.column_stack((a["x1"], a["x2"]))
    fitted = fit_gaussian_ee(a["y"], x, a["id"], a["z"], q=a["q"])
    gaps = {
        "coefficients": compare_value(fitted.msm_coefficients, live["coefficients"], "coefficients"),
        "covariance": compare_value(fitted.msm_covariance, live["covariance"], "covariance"),
    }
    for name in ("coefficients", "covariance"):
        compare_value(live[name], saved[name], f"{name}.saved_R", atol=1e-12)
    if len(saved.get("cases", [])) != 2 or len(live.get("cases", [])) != 2:
        raise ValueError("two Gaussian EE bound tables required")
    cases = []
    for old, new, pwonly in zip(saved["cases"], live["cases"], (False, True)):
        if old["pwonly"] is not pwonly or new["pwonly"] is not pwonly:
            raise ValueError("Gaussian EE pointwise setting changed")
        if old["columns"] != new["columns"]:
            raise ValueError("Gaussian EE bound columns changed")
        draws = np.asarray(new["draws"], dtype=float)
        if draws.shape != (5000, 2):
            raise ValueError("5000 paired R coefficient draws required")
        gaps[f"draws_{pwonly}"] = compare_value(draws, old["draws"],
                                                "saved R draws", atol=1e-12)
        expected = dict(zip(new["columns"], new["table"]))
        original = dict(zip(old["columns"], old["table"]))
        calculated = _from_coefficient_draws(fitted, draws, alpha=.05, pwonly=pwonly)
        public = modelbound_ee(fitted, alpha=.05, pwonly=pwonly, seed=418)
        if calculated.columns != tuple(new["columns"]) or public.columns != calculated.columns:
            raise ValueError("Python EE bound columns differ from R")
        field_gaps = {}
        for name in calculated.columns:
            r_value = expected[name]
            old_value = original[name]
            if all(value is None for value in r_value):
                if not all(value is None for value in old_value):
                    raise ValueError(f"{name}: saved R missingness changed")
                if not (np.all(np.isnan(calculated[name])) and np.all(np.isnan(public[name]))):
                    raise ValueError(f"{name}: Python missingness differs")
                continue
            compare_value(r_value, old_value, f"{name}.saved_R", atol=1e-12)
            field_gaps[name] = compare_value(calculated[name], r_value, name)
            if not np.all(np.isfinite(public[name])):
                raise ValueError(f"{name}: public Python bounds are nonfinite")
        cases.append({"pwonly": pwonly, "max_absolute_differences": field_gaps})
    return {"status": "passed", "checked_utc": datetime.now(timezone.utc).isoformat(),
            "qgcomp_version": "2.19.6", "saved_fixture_sha256": FIXTURE_SHA,
            "r_exporter_sha256": DRIVER_SHA, "fit_max_absolute_differences": gaps,
            "cases": cases,
            "random_stream_difference": "Python and R use different random generators; exact tables agree when supplied the same R coefficient draws.",
            "prequantized_R_limit": "Installed R modelbound.ee errors for q=NULL at seq_len(x$q); the comparison uses q=4."}


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
        raise ValueError("Gaussian EE bounds fixture or exporter hash differs")
    with tempfile.TemporaryDirectory() as folder:
        live_path = Path(folder) / "live.json"
        subprocess.run([args.rscript, str(DRIVER), args.library, str(live_path)], check=True)
        report = check(json.loads(FIXTURE.read_text()), json.loads(live_path.read_text()))
    output.write_text(json.dumps(report, indent=2)+"\n")
    print("Two Gaussian EE bound tables agree with isolated installed R qgcomp 2.19.6")


if __name__ == "__main__":
    main()
