"""Recheck selected public qgcomp routes against installed R qgcomp 2.19.6."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from quantilemix import BinomialRiskRatioResult, qgcomp
from quantilemix._risk_ratio import _from_indices as risk_ratio_from_indices


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/r_qgcomp_dispatch_2_19_6.json"
DRIVER = Path(__file__).with_name("export_qgcomp_dispatch_r.R")
SHA256 = "22706d0b1ed75c8dcaebb8188059b288cf67d365a8027678f045ae76265d2f22"
CASE_NAMES = ("gaussian_adjusted", "gaussian_category", "binary_odds",
              "binary_default_rr", "poisson_adjusted", "poisson_offset_conditional")


def _assert_same(a: object, b: object) -> None:
    if a is None or b is None:
        if a is not None or b is not None:
            raise ValueError("reference input changed")
    elif isinstance(a, list) and a and isinstance(a[0], str):
        np.testing.assert_array_equal(a, b)
    else:
        np.testing.assert_allclose(b, a, rtol=0, atol=1e-12)


def compare(saved: dict, live: dict) -> float:
    """Check route identity, R outputs and the Python public dispatcher."""
    for item in (saved, live):
        if item.get("qgcomp_version") != "2.19.6" or item.get("namespace_verified") is not True:
            raise ValueError("installed R qgcomp 2.19.6 namespace verification is required")
    old, new = saved.get("cases"), live.get("cases")
    if not isinstance(old, list) or not isinstance(new, list) or len(old) != 6 or len(new) != 6:
        raise ValueError("six dispatcher cases are required")
    if tuple(c["name"] for c in old) != CASE_NAMES or tuple(c["name"] for c in new) != CASE_NAMES:
        raise ValueError("dispatcher case names changed")
    largest = 0.0
    for a, b in zip(old, new):
        if a["family"] != b["family"]:
            raise ValueError("dispatcher family changed")
        inputs, fresh_inputs = a["inputs"], b["inputs"]
        for field in ("y", "x1", "x2", "z", "time", "q", "rr", "offset", "B", "seed", "indices"):
            _assert_same(inputs[field], fresh_inputs[field])
        expected, fresh = a["expected"], b["expected"]
        if expected["bootstrap"] is not fresh["bootstrap"]:
            raise ValueError("installed R dispatch route changed")
        for field in ("psi", "se", "ci", "p"):
            np.testing.assert_allclose(fresh[field], expected[field], rtol=0, atol=2e-8)

        data = {key: inputs[key] for key in ("y", "x1", "x2")}
        formula = "y ~ x1 + x2"
        if inputs["z"] is not None:
            data["z"] = inputs["z"]
            formula += " + " + ("C(z)" if a["name"] == "gaussian_category" else "z")
        if inputs["time"] is not None:
            data["time"] = inputs["time"]
        options = {"family": a["family"], "rr": inputs["rr"], "q": inputs["q"]}
        if inputs["time"] is not None:
            options["exposure_time"] = "time"
        if inputs["B"] is not None:
            options.update(B=inputs["B"], seed=inputs["seed"])
        result = qgcomp(formula, data, ("x1", "x2"), **options)
        if isinstance(result, BinomialRiskRatioResult) is not expected["bootstrap"]:
            raise ValueError("Python dispatcher selected the wrong route")
        for field, value in (("psi", result.psi),):
            target = expected[field]
            largest = max(largest, abs(value-target))
            np.testing.assert_allclose(value, target, rtol=0, atol=2e-6)

        comparison = result
        if expected["bootstrap"]:
            comparison = risk_ratio_from_indices(
                inputs["y"], np.column_stack((inputs["x1"], inputs["x2"])),
                inputs["indices"], q=inputs["q"])
        for field, value in (("se", comparison.se), ("p", comparison.p_value)):
            target = expected[field]
            largest = max(largest, abs(value-target))
            np.testing.assert_allclose(value, target, rtol=0, atol=2e-6)
        actual_ci = np.asarray(comparison.ci)
        target_ci = np.asarray(expected["ci"])
        largest = max(largest, float(np.max(np.abs(actual_ci-target_ci))))
        np.testing.assert_allclose(actual_ci, target_ci, rtol=0, atol=2e-6)
    return largest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", required=True)
    parser.add_argument("--library", required=True)
    args = parser.parse_args()
    if hashlib.sha256(FIXTURE.read_bytes()).hexdigest() != SHA256:
        raise ValueError("saved dispatcher fixture hash differs")
    saved = json.loads(FIXTURE.read_text())
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp) / "live.json"
        subprocess.run([args.rscript, str(DRIVER), args.library, str(output)], check=True)
        largest = compare(saved, json.loads(output.read_text()))
    print(f"Six dispatcher cases agree with installed R qgcomp 2.19.6; "
          f"largest absolute Python/R difference {largest:.3g}")


if __name__ == "__main__":
    main()
