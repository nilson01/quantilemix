"""Optional fixed-case check against installed R qgcomp 2.19.6."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from quantilemix import fit_zero_inflated_poisson


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/r_zero_inflated_2_19_6.json"
EXPORTER = Path(__file__).with_name("export_zero_inflated_r.R")
FIXTURE_SHA256 = "2397821e3989df7ab0b39b098a673b2571c3d8853a9481a31bfc7e9b0b9290ef"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regenerate", action="store_true", help="run the installed-R exporter first")
    args = parser.parse_args()
    if args.regenerate:
        subprocess.run([os.environ.get("RSCRIPT", "Rscript"), str(EXPORTER)],
                       cwd=ROOT, check=True)
    digest = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    if not args.regenerate and digest != FIXTURE_SHA256:
        raise SystemExit("pinned installed-R fixture changed")
    payload = json.loads(FIXTURE.read_text())
    if payload["reference_version"] != "2.19.6" or len(payload["cases"]) != 2:
        raise SystemExit("unexpected R reference cases")
    gaps = {}
    for index, case in enumerate(payload["cases"], 1):
        rows = case["data"]
        result = fit_zero_inflated_poisson(
            [r["y"] for r in rows], [[r["x1"], r["x2"]] for r in rows],
            [r["z"] for r in rows], q=None)
        actual = [result.psi_count, result.psi_zero, result.se_count, result.se_zero]
        expected = [case["psi"]["count"], case["psi"]["zero"],
                    case["se"]["count"], case["se"]["zero"]]
        gap = float(np.max(np.abs(np.asarray(actual)-expected)))
        if gap > 3e-6:
            raise SystemExit(f"case {index} R/Python effect or SE gap {gap:.3g} > 3e-6")
        beta_gap = float(np.max(np.abs(result.coefficients -
                                       np.asarray(case["coefficients"]["count"]+
                                                  case["coefficients"]["zero"]))))
        covariance_gap = float(np.max(np.abs(result.covariance-np.asarray(case["covariance"]))))
        if beta_gap > 5e-6 or covariance_gap > 3e-6:
            raise SystemExit(f"case {index} R/Python coefficient or covariance gap exceeded tolerance")
        gaps[f"case_{index}"] = gap
    print(json.dumps({"reference_version": "2.19.6", "cases": 2,
                      "maximum_effect_or_se_gap": max(gaps.values()),
                      "fixture_sha256": digest, "gaps": gaps}, indent=2))


if __name__ == "__main__":
    main()
