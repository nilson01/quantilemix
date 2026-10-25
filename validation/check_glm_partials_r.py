"""Recheck fixed binary/count partial effects against installed R qgcomp."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from quantilemix import qgcomp_partials


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/r_glm_partials_2_19_6.json"
DRIVER = Path(__file__).with_name("export_glm_partials_r.R")
SHA256 = "cfe113409aeb665f6242508710e1658382db0070cdfdc5f15f950801244a25ee"
CASE_NAMES = ("binomial_scores", "binomial_raw", "poisson_scores", "poisson_raw")


def _group(value):
    return () if value == "none" else (value,) if isinstance(value, str) else tuple(value)


def compare(saved: dict, live: dict) -> float:
    """Compare input identity, live R outputs and Python estimates."""
    if saved.get("version") != "2.19.6" or live.get("version") != "2.19.6":
        raise ValueError("installed R qgcomp 2.19.6 is required")
    old, new = saved.get("cases"), live.get("cases")
    if not isinstance(old, list) or not isinstance(new, list) or len(old) != 4 or len(new) != 4:
        raise ValueError("four GLM partial-effect cases are required")
    if tuple(case.get("name") for case in old) != CASE_NAMES or tuple(case.get("name") for case in new) != CASE_NAMES:
        raise ValueError("GLM partial-effect case names changed")
    largest = 0.0
    for a, b in zip(old, new):
        if a["name"] != b["name"] or a["family"] != b["family"] or a["q"] != b["q"]:
            raise ValueError("GLM partial-effect case identity changed")
        for portion in ("train", "valid"):
            for name in ("y", "x1", "x2", "z"):
                np.testing.assert_allclose(b[portion][name], a[portion][name], rtol=0, atol=1e-12)
        if _group(a["positive"]) != _group(b["positive"]) or _group(a["negative"]) != _group(b["negative"]):
            raise ValueError("installed R training sign groups changed")
        for name in ("training", "positive_fit", "negative_fit"):
            for field in ("psi", "se", "ci"):
                np.testing.assert_allclose(b[name][field], a[name][field], rtol=0, atol=5e-12)
        train, valid = a["train"], a["valid"]
        result = qgcomp_partials(
            train["y"], np.column_stack((train["x1"], train["x2"])),
            valid["y"], np.column_stack((valid["x1"], valid["x2"])),
            train_covariates=train["z"], validation_covariates=valid["z"],
            q=a["q"], family=a["family"],
        )
        if result.positive_exposures != _group(a["positive"]) or result.negative_exposures != _group(a["negative"]):
            raise ValueError("Python training sign groups differ from installed R")
        for name in ("training", "positive_fit", "negative_fit"):
            fit = result.training_fit if name == "training" else getattr(result, name)
            for field, actual in (("psi", fit.psi), ("se", fit.se), ("ci", fit.ci)):
                target = np.asarray(a[name][field], dtype=float)
                delta = float(np.max(np.abs(np.asarray(actual) - target)))
                largest = max(largest, delta)
                np.testing.assert_allclose(actual, target, rtol=0, atol=2e-6,
                                           err_msg=f"{a['name']}.{name}.{field}")
    return largest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", required=True)
    parser.add_argument("--library", required=True)
    args = parser.parse_args()
    if hashlib.sha256(FIXTURE.read_bytes()).hexdigest() != SHA256:
        raise ValueError("saved GLM partial-effect fixture hash differs")
    saved = json.loads(FIXTURE.read_text())
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp) / "live.json"
        subprocess.run([args.rscript, str(DRIVER), args.library, str(output)], check=True)
        largest = compare(saved, json.loads(output.read_text()))
    print(f"Four GLM partial-effect cases agree with installed R qgcomp 2.19.6; "
          f"largest absolute Python/R difference {largest:.3g}")


if __name__ == "__main__":
    main()
