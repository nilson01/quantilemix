"""Recheck marginal predictions against installed R qgcomp 2.19.6."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from quantilemix import msm_predict
from quantilemix._bootstrap import _fit_bootstrap_with_indices
from quantilemix._poisson_bootstrap import _from_indices as poisson_from_indices


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/r_msm_predict_2_19_6.json"
DRIVER = Path(__file__).with_name("export_msm_predict_r.R")
SHA256 = "0759a9c0d79a239501bcd5fc85ab0f29a1693c69554d8286eb84ffb3cbdcb66a"
CASE_NAMES = ("gaussian_prequantized", "gaussian_adjusted", "poisson_prequantized",
              "poisson_adjusted", "poisson_offset")


def compare(saved: dict, live: dict) -> float:
    """Check pinned data, regenerated R values and Python public output."""
    for item in (saved, live):
        if item.get("qgcomp_version") != "2.19.6" or item.get("namespace_verified") is not True:
            raise ValueError("installed R qgcomp 2.19.6 namespace verification is required")
    old, new = saved.get("cases"), live.get("cases")
    if not isinstance(old, list) or not isinstance(new, list) or len(old) != 5 or len(new) != 5:
        raise ValueError("five marginal prediction cases are required")
    if tuple(c["name"] for c in old) != CASE_NAMES or tuple(c["name"] for c in new) != CASE_NAMES:
        raise ValueError("marginal prediction case names changed")
    largest = 0.0
    for a, b in zip(old, new):
        if a["family"] != b["family"]:
            raise ValueError("marginal prediction family changed")
        inputs, fresh_inputs = a["inputs"], b["inputs"]
        for field in ("y", "x1", "x2", "z", "time", "q", "new_scores", "indices"):
            va, vb = inputs[field], fresh_inputs[field]
            if va is None or vb is None:
                if va is not None or vb is not None:
                    raise ValueError(f"{a['name']}.{field} changed")
            else:
                np.testing.assert_allclose(vb, va, rtol=0, atol=1e-12)
        expected, fresh = a["expected"], b["expected"]
        for field in ("default", "supplied", "levels", "msm_coefficients", "means", "index"):
            np.testing.assert_allclose(fresh[field], expected[field], rtol=0, atol=5e-9)

        x = np.column_stack((inputs["x1"], inputs["x2"]))
        if a["family"] == "gaussian":
            model = _fit_bootstrap_with_indices(inputs["y"], x, inputs["indices"],
                                                inputs["z"], q=inputs["q"])
            coefficients = (model.intervention_means[0]
                            - model.intervention_levels[0]*model.psi, model.psi)
        else:
            model = poisson_from_indices(inputs["y"], x, inputs["indices"],
                                         inputs["z"], q=inputs["q"],
                                         exposure_time=inputs["time"])
            coefficients = (np.log(model.intervention_means[0])
                            - model.intervention_levels[0]*model.psi, model.psi)
        np.testing.assert_allclose(model.intervention_levels, expected["levels"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(model.intervention_means, expected["means"], rtol=0, atol=2e-8)
        np.testing.assert_allclose(coefficients, expected["msm_coefficients"], rtol=0, atol=2e-8)
        n = len(inputs["y"])
        np.testing.assert_allclose(expected["index"], np.repeat(expected["levels"], n),
                                   rtol=0, atol=0)
        outputs = {"default": msm_predict(model),
                   "supplied": msm_predict(model, {"psi": inputs["new_scores"]})}
        for field, actual in outputs.items():
            target = np.asarray(expected[field], dtype=float)
            if actual.shape != target.shape:
                raise ValueError(f"{a['name']}.{field} output shape changed")
            largest = max(largest, float(np.max(np.abs(actual-target))))
            np.testing.assert_allclose(actual, target, rtol=3e-11, atol=2e-8,
                                       err_msg=f"{a['name']}.{field}")
    return largest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", required=True)
    parser.add_argument("--library", required=True)
    args = parser.parse_args()
    if hashlib.sha256(FIXTURE.read_bytes()).hexdigest() != SHA256:
        raise ValueError("saved marginal prediction fixture hash differs")
    saved = json.loads(FIXTURE.read_text())
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp) / "live.json"
        subprocess.run([args.rscript, str(DRIVER), args.library, str(output)], check=True)
        largest = compare(saved, json.loads(output.read_text()))
    print(f"Five marginal prediction cases agree with installed R qgcomp 2.19.6; "
          f"largest absolute Python/R difference {largest:.3g}")


if __name__ == "__main__":
    main()
