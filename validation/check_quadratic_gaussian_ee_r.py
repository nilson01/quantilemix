"""Replay installed-R degree-two Gaussian EE cases and audit its cluster meat."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from quantilemix import fit_quadratic_gaussian_ee

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/r_quadratic_gaussian_ee_2_19_6.json"
DRIVER = Path(__file__).with_name("export_quadratic_gaussian_ee_r.R")
FIXTURE_SHA = "dacf79833b4432584af5b7ca8335f47f6f0cfbd1a422e0f7694ac1d5ba53e8dd"
DRIVER_SHA = "aae15072046467e38b86ca2bfa44fa8074a85bd4c161cc43cc070d9f6e08e7fb"
NAMES = ("quadratic_square_ee", "quadratic_pairwise_ee",
         "quadratic_covariate_weighted_ee")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(actual, expected, label: str, *, atol: float = 1e-8) -> float:
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError(f"{label}: shape or finiteness differs")
    gap = float(np.max(np.abs(a-b)))
    if gap > atol:
        raise ValueError(f"{label}: difference {gap:.3g} exceeds {atol:.3g}")
    return gap


def check(saved: dict, live: dict) -> dict:
    if saved.get("qgcomp_version") != "2.19.6" or live.get("qgcomp_version") != "2.19.6":
        raise ValueError("installed qgcomp 2.19.6 required")
    old_cases, new_cases = saved.get("cases"), live.get("cases")
    if (not isinstance(old_cases, list) or not isinstance(new_cases, list)
            or len(old_cases) != len(new_cases) or len(new_cases) != 3
            or tuple(case.get("name") for case in old_cases) != NAMES
            or tuple(case.get("name") for case in new_cases) != NAMES):
        raise ValueError("three fixed quadratic Gaussian EE cases required")
    details = []
    for old, new in zip(old_cases, new_cases):
        if old["inputs"] != new["inputs"] or old["pairwise"] != new["pairwise"] or \
                old["covariate_interactions"] != new["covariate_interactions"] or \
                old["weighted"] != new["weighted"]:
            raise ValueError(f"{old['name']}: fixed inputs changed")
        e_old, e = old["expected"], new["expected"]
        if set(e_old) != set(e) or e_old["conditional_names"] != e["conditional_names"]:
            raise ValueError(f"{old['name']}: installed-R fields changed")
        for key in e_old:
            if key == "conditional_names":
                continue
            if key == "breaks":
                if len(e_old[key]) != len(e[key]):
                    raise ValueError(f"{old['name']}: breaks changed")
                for previous, current in zip(e_old[key], e[key]):
                    compare(current, previous, f"{old['name']}.breaks.saved_R", atol=1e-9)
            else:
                compare(e[key], e_old[key], f"{old['name']}.{key}.saved_R", atol=1e-8)
        a = new["inputs"]
        result = fit_quadratic_gaussian_ee(
            a["y"], np.column_stack((a["x1"], a["x2"])), a["id"], a["z"],
            q=a["q"], weights=a["w"],
            pairwise_interactions=new["pairwise"],
            covariate_interactions=new["covariate_interactions"],
        )
        direct = {
            "conditional_coefficients": result.conditional_coefficients,
            "msm_coefficients": result.msm_coefficients,
            "psi": result.psi, "scores": result.scores,
            "levels": result.intervention_levels,
            "standardized": result.intervention_means,
            "df": result.p_value_df,
            "bread": result.bread,
            "fresh_meat": result.meat,
            "fresh_full_covariance": result.full_covariance,
        }
        gaps = {key: compare(value, e[key], f"{old['name']}.{key}.Python",
                             atol=1e-6 if key == "bread" else 1e-8)
                for key, value in direct.items()}
        for actual, expected in zip(result.breaks, e["breaks"]):
            compare(actual, expected, f"{old['name']}.breaks.Python", atol=1e-9)
        compare(e["stale_rebuilt_meat"], e["meat"],
                f"{old['name']}.stale_meat_R", atol=1e-7)
        reference_meat_gap = float(np.max(np.abs(result.meat-np.asarray(e["meat"]))))
        if reference_meat_gap < 1.0:
            raise ValueError(f"{old['name']}: expected R stale-term meat difference vanished")
        details.append({"name": old["name"], "max_absolute_differences": gaps,
                        "r_stale_term_meat_difference": reference_meat_gap})
    return {"status": "passed", "checked_utc": datetime.now(timezone.utc).isoformat(),
            "qgcomp_version": "2.19.6", "saved_fixture_sha256": FIXTURE_SHA,
            "r_exporter_sha256": DRIVER_SHA, "cases": details,
            "difference": "R's cluster meat reuses formula-derived I(...) columns from model.frame; Python recomputes them at each intervention."}


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
        raise ValueError("quadratic Gaussian EE fixture or exporter hash differs")
    with tempfile.TemporaryDirectory() as folder:
        live_path = Path(folder) / "live.json"
        subprocess.run([args.rscript, str(DRIVER), args.library, str(live_path)], check=True)
        report = check(json.loads(FIXTURE.read_text()), json.loads(live_path.read_text()))
    output.write_text(json.dumps(report, indent=2)+"\n")
    print("Three quadratic Gaussian EE point fits agree with isolated R; corrected cluster covariance independently agrees")


if __name__ == "__main__":
    main()
