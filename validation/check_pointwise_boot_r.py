"""Recheck fixed bootstrap pointwise tables against installed R qgcomp 2.19.6."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from quantilemix import pointwisebound_boot
from quantilemix._bootstrap import _fit_bootstrap_with_indices
from quantilemix._poisson_bootstrap import _from_indices as poisson_from_indices


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/r_pointwise_boot_2_19_6.json"
DRIVER = Path(__file__).with_name("export_pointwise_boot_r.R")
SHA256 = "ce38439ebda62c27eefc4d56c2185e9ba39710bf1d8270347f2a69783fd1c710"
CASE_NAMES = ("gaussian_ref1", "gaussian_ref3", "poisson_ref1", "poisson_ref4",
              "offset_ref1", "offset_ref4")


def compare(saved: dict, live: dict) -> float:
    """Check pinned inputs, installed-R outputs and Python outputs."""
    for item in (saved, live):
        if item.get("qgcomp_version") != "2.19.6" or item.get("namespace_verified") is not True:
            raise ValueError("installed R qgcomp 2.19.6 namespace verification is required")
    old, new = saved.get("cases"), live.get("cases")
    if not isinstance(old, list) or not isinstance(new, list) or len(old) != 6 or len(new) != 6:
        raise ValueError("six pointwise bootstrap cases are required")
    if tuple(c["name"] for c in old) != CASE_NAMES or tuple(c["name"] for c in new) != CASE_NAMES:
        raise ValueError("pointwise bootstrap case names changed")
    largest = 0.0
    for a, b in zip(old, new):
        if a["family"] != b["family"]:
            raise ValueError("pointwise bootstrap family changed")
        for field in ("y", "x1", "x2", "z", "time", "q", "indices", "pointwiseref", "alpha"):
            va, vb = a["inputs"][field], b["inputs"][field]
            if va is None or vb is None:
                if va is not None or vb is not None:
                    raise ValueError(f"{a['name']}.{field} changed")
            else:
                np.testing.assert_allclose(vb, va, rtol=0, atol=1e-12)
        expected, fresh = a["expected"], b["expected"]
        if (expected["columns"] != fresh["columns"] or expected["link"] != fresh["link"]
                or tuple(expected["table"]) != tuple(expected["columns"])):
            raise ValueError("installed R pointwise bootstrap schema changed")
        for field in ("means", "covariance"):
            np.testing.assert_allclose(fresh[field], expected[field], rtol=0, atol=5e-10)
        for field in expected["columns"]:
            np.testing.assert_allclose(fresh["table"][field], expected["table"][field],
                                       rtol=0, atol=5e-10)
        inputs = a["inputs"]
        x = np.column_stack((inputs["x1"], inputs["x2"]))
        if a["family"] == "gaussian":
            model = _fit_bootstrap_with_indices(inputs["y"], x, inputs["indices"],
                                                inputs["z"], q=inputs["q"])
            profiles = model.bootstrap_means
        else:
            model = poisson_from_indices(inputs["y"], x, inputs["indices"],
                                         inputs["z"], q=inputs["q"],
                                         exposure_time=inputs["time"])
            profiles = np.log(model.bootstrap_means)
        np.testing.assert_allclose(model.intervention_means, expected["means"],
                                   rtol=0, atol=2e-8)
        np.testing.assert_allclose(np.cov(profiles, rowvar=False), expected["covariance"],
                                   rtol=0, atol=2e-8)
        result = pointwisebound_boot(model, pointwiseref=inputs["pointwiseref"],
                                     alpha=inputs["alpha"])
        if result.columns != tuple(expected["columns"]) or result.link != expected["link"]:
            raise ValueError("Python pointwise bootstrap schema differs from installed R")
        for field in result.columns:
            actual = result[field]
            target = np.asarray(expected["table"][field], dtype=float)
            largest = max(largest, float(np.max(np.abs(actual-target))))
            np.testing.assert_allclose(actual, target, rtol=0, atol=2e-8,
                                       err_msg=f"{a['name']}.{field}")
    return largest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", required=True)
    parser.add_argument("--library", required=True)
    args = parser.parse_args()
    if hashlib.sha256(FIXTURE.read_bytes()).hexdigest() != SHA256:
        raise ValueError("saved pointwise bootstrap fixture hash differs")
    saved = json.loads(FIXTURE.read_text())
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp) / "live.json"
        subprocess.run([args.rscript, str(DRIVER), args.library, str(output)], check=True)
        largest = compare(saved, json.loads(output.read_text()))
    print(f"Six pointwise bootstrap cases agree with installed R qgcomp 2.19.6; "
          f"largest absolute Python/R difference {largest:.3g}")


if __name__ == "__main__":
    main()
