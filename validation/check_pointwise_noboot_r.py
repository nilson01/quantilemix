"""Recheck fixed pointwise comparisons against installed R qgcomp 2.19.6."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from quantilemix import fit, fit_binomial, fit_poisson, pointwisebound_noboot


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/r_pointwise_noboot_2_19_6.json"
DRIVER = Path(__file__).with_name("export_pointwise_noboot_r.R")
SHA256 = "b5f3090b9e60df8eca29e2b49494874b4029bcbf675130b73b0fc87346d1084e"
CASE_NAMES = ("gaussian_ref1", "gaussian_ref3", "binomial_ref1",
              "binomial_ref3", "poisson_ref1", "poisson_ref3")


def compare(saved: dict, live: dict) -> float:
    """Check input identity, live R tables and the corresponding Python fit."""
    if saved.get("qgcomp_version") != "2.19.6" or live.get("qgcomp_version") != "2.19.6":
        raise ValueError("installed R qgcomp 2.19.6 is required")
    if saved.get("namespace_verified") is not True or live.get("namespace_verified") is not True:
        raise ValueError("installed R qgcomp namespace verification is required")
    old, new = saved.get("cases"), live.get("cases")
    if not isinstance(old, list) or not isinstance(new, list) or len(old) != 6 or len(new) != 6:
        raise ValueError("six pointwise cases are required")
    if tuple(c.get("name") for c in old) != CASE_NAMES or tuple(c.get("name") for c in new) != CASE_NAMES:
        raise ValueError("pointwise case names changed")
    largest = 0.0
    for a, b in zip(old, new):
        if a["family"] != b["family"] or a["inputs"]["q"] != 4 or b["inputs"]["q"] != 4:
            raise ValueError("pointwise family or quantile count changed")
        for field in ("pointwiseref", "alpha", "y", "x1", "x2", "z"):
            np.testing.assert_allclose(b["inputs"][field], a["inputs"][field], rtol=0, atol=1e-12)
        expected, fresh = a["expected"], b["expected"]
        if expected["columns"] != fresh["columns"] or expected["link"] != fresh["link"]:
            raise ValueError("installed R pointwise columns or link changed")
        if (tuple(expected["table"]) != tuple(expected["columns"]) or
                tuple(fresh["table"]) != tuple(fresh["columns"])):
            raise ValueError("pointwise table columns do not match their order")
        for field in ("fitted_coefficients", "fitted_covariance", "joint_coefficients"):
            np.testing.assert_allclose(fresh[field], expected[field], rtol=0, atol=5e-11)
        for field in expected["columns"]:
            np.testing.assert_allclose(fresh["table"][field], expected["table"][field], rtol=0, atol=5e-11)
        inputs = a["inputs"]
        fitter = {"gaussian": fit, "binomial": fit_binomial, "poisson": fit_poisson}[a["family"]]
        model = fitter(inputs["y"], np.column_stack((inputs["x1"], inputs["x2"])),
                       inputs["z"], q=inputs["q"])
        result = pointwisebound_noboot(model, pointwiseref=inputs["pointwiseref"],
                                       alpha=inputs["alpha"])
        if result.columns != tuple(expected["columns"]) or result.link != expected["link"]:
            raise ValueError("Python pointwise columns or link differ from installed R")
        for field in result.columns:
            target = np.asarray(expected["table"][field], dtype=float)
            actual = result[field]
            largest = max(largest, float(np.max(np.abs(actual-target))))
            np.testing.assert_allclose(actual, target, rtol=0, atol=2e-6,
                                       err_msg=f"{a['name']}.{field}")
    return largest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", required=True)
    parser.add_argument("--library", required=True)
    args = parser.parse_args()
    if hashlib.sha256(FIXTURE.read_bytes()).hexdigest() != SHA256:
        raise ValueError("saved pointwise fixture hash differs")
    saved = json.loads(FIXTURE.read_text())
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp) / "live.json"
        subprocess.run([args.rscript, str(DRIVER), args.library, str(output)], check=True)
        largest = compare(saved, json.loads(output.read_text()))
    print(f"Six pointwise cases agree with installed R qgcomp 2.19.6; "
          f"largest absolute Python/R difference {largest:.3g}")


if __name__ == "__main__":
    main()
