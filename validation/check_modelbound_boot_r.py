"""Recheck marginal-curve bounds against isolated installed R qgcomp 2.19.6."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from quantilemix import modelbound_boot
from quantilemix._bootstrap import _fit_bootstrap_with_indices
from quantilemix._poisson_bootstrap import _from_indices as poisson_from_indices


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/r_modelbound_boot_2_19_6.json"
DRIVER = Path(__file__).with_name("export_modelbound_boot_r.R")
SHA256 = "c5d3c44f5ed633a331c71952f64785478d6f574f7410acd222784f74d92cbd15"
CASE_NAMES = ("gaussian", "poisson", "poisson_offset")


def compare(saved: dict, live: dict) -> float:
    """Compare fixed inputs, installed-R tables and Python public results."""
    for item in (saved, live):
        if item.get("qgcomp_version") != "2.19.6" or item.get("namespace_verified") is not True:
            raise ValueError("verified installed R qgcomp 2.19.6 is required")
    old, new = saved.get("cases"), live.get("cases")
    if not isinstance(old, list) or not isinstance(new, list) or len(old) != 3 or len(new) != 3:
        raise ValueError("three model-bound cases are required")
    if tuple(c["name"] for c in old) != CASE_NAMES or tuple(c["name"] for c in new) != CASE_NAMES:
        raise ValueError("model-bound case names changed")
    largest = 0.0
    for a, b in zip(old, new):
        if a["family"] != b["family"]:
            raise ValueError("model-bound family changed")
        for field in ("y", "x1", "x2", "z", "time", "q", "indices", "alpha"):
            va, vb = a["inputs"][field], b["inputs"][field]
            if va is None or vb is None:
                if va is not None or vb is not None:
                    raise ValueError(f"{a['name']}.{field} changed")
            else:
                np.testing.assert_allclose(vb, va, rtol=0, atol=1e-12)
        expected, fresh = a["expected"], b["expected"]
        for field in ("coef", "bootsamps", "covariance", "means"):
            np.testing.assert_allclose(fresh[field], expected[field], rtol=0, atol=5e-10)
        inputs = a["inputs"]
        x = np.column_stack((inputs["x1"], inputs["x2"]))
        if a["family"] == "gaussian":
            model = _fit_bootstrap_with_indices(inputs["y"], x, inputs["indices"],
                                                inputs["z"], q=inputs["q"])
            profiles = model.bootstrap_means
        else:
            model = poisson_from_indices(inputs["y"], x, inputs["indices"], inputs["z"],
                                         q=inputs["q"], exposure_time=inputs["time"])
            profiles = np.log(model.bootstrap_means)
        design = np.column_stack((np.ones(inputs["q"]), np.arange(inputs["q"])))
        coef = np.linalg.lstsq(design, profiles.T, rcond=None)[0].T
        main = np.linalg.lstsq(design, np.log(model.intervention_means)
                               if a["family"] != "gaussian" else model.intervention_means,
                               rcond=None)[0]
        np.testing.assert_allclose(main, expected["coef"], rtol=0, atol=2e-8)
        np.testing.assert_allclose(coef, expected["bootsamps"], rtol=0, atol=2e-8)
        np.testing.assert_allclose(np.cov(coef, rowvar=False), expected["covariance"],
                                   rtol=0, atol=2e-8)
        np.testing.assert_allclose(model.intervention_means, expected["means"],
                                   rtol=0, atol=2e-8)
        old_tables, new_tables = expected["tables"], fresh["tables"]
        if len(old_tables) != 2 or len(new_tables) != 2:
            raise ValueError("pointwise-only and simultaneous tables are required")
        for original, regenerated in zip(old_tables, new_tables):
            if (original["columns"] != regenerated["columns"]
                    or original["link"] != regenerated["link"]
                    or original["pwonly"] != regenerated["pwonly"]):
                raise ValueError("installed R model-bound schema changed")
            result = modelbound_boot(model, alpha=inputs["alpha"], pwonly=original["pwonly"])
            if result.columns != tuple(original["columns"]) or result.link != original["link"]:
                raise ValueError("Python model-bound schema differs from installed R")
            for field in result.columns:
                target = np.asarray(original["table"][field], dtype=float)
                reread = np.asarray(regenerated["table"][field], dtype=float)
                np.testing.assert_allclose(reread, target, rtol=0, atol=5e-10,
                                           equal_nan=True)
                actual = result[field]
                finite = np.isfinite(target)
                if np.any(finite):
                    largest = max(largest, float(np.max(np.abs(actual[finite]-target[finite]))))
                np.testing.assert_allclose(actual, target, rtol=0, atol=2e-8,
                                           equal_nan=True, err_msg=f"{a['name']}.{field}")
    return largest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", required=True)
    parser.add_argument("--library", required=True)
    args = parser.parse_args()
    if hashlib.sha256(FIXTURE.read_bytes()).hexdigest() != SHA256:
        raise ValueError("saved model-bound fixture hash differs")
    saved = json.loads(FIXTURE.read_text())
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp) / "live.json"
        subprocess.run([args.rscript, str(DRIVER), args.library, str(output)], check=True,
                       timeout=120)
        largest = compare(saved, json.loads(output.read_text()))
    print(f"Six model-bound tables from three cases agree with installed R qgcomp 2.19.6; "
          f"largest absolute Python/R difference {largest:.3g}")


if __name__ == "__main__":
    main()
