"""Optional live re-export of weighted GLM cases from installed R qgcomp."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/r_glm_weights_2_19_6.json"
DRIVER = Path(__file__).with_name("export_glm_weights_r.R")
SHA256 = "cb51d6eb918089c528bf974b9de40954e1f5c92b4c5666320b11c6dabf0727b2"
NAMES = {"binomial_scores", "binomial_quantized", "poisson_scores", "poisson_quantized"}


def compare(saved: dict, live: dict) -> None:
    if saved.get("qgcomp_version") != "2.19.6" or live.get("qgcomp_version") != "2.19.6":
        raise ValueError("installed R version differs")
    old, new = saved.get("cases"), live.get("cases")
    if not isinstance(old, list) or not isinstance(new, list) or len(old) != 4 or len(new) != 4:
        raise ValueError("four weighted reference cases required")
    if {case["name"] for case in old} != NAMES or {case["name"] for case in new} != NAMES:
        raise ValueError("weighted reference case identities differ")
    for a, b in zip(old, new):
        if a["name"] != b["name"] or a["family"] != b["family"] or a["inputs"] != b["inputs"]:
            raise ValueError("weighted reference inputs changed")
        if set(a["expected"]) != set(b["expected"]):
            raise ValueError("weighted reference fields changed")
        for field in a["expected"]:
            x, y = a["expected"][field], b["expected"][field]
            if x is None or y is None:
                if x is not None or y is not None:
                    raise ValueError(f"{a['name']}.{field} differs")
                continue
            xx, yy = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
            if xx.shape != yy.shape or not np.all(np.isfinite(xx)) or not np.all(np.isfinite(yy)):
                raise ValueError(f"{a['name']}.{field} shape or finiteness differs")
            if not np.allclose(xx, yy, rtol=5e-12, atol=5e-12):
                raise ValueError(f"{a['name']}.{field} differs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", required=True, help="Rscript matching the isolated library")
    parser.add_argument("--library", required=True, help="installed qgcomp 2.19.6 library")
    args = parser.parse_args()
    if hashlib.sha256(FIXTURE.read_bytes()).hexdigest() != SHA256:
        raise ValueError("saved weighted fixture hash differs")
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp) / "live.json"
        subprocess.run([args.rscript, str(DRIVER), args.library, str(output)], check=True)
        compare(json.loads(FIXTURE.read_text()), json.loads(output.read_text()))
    print("Four weighted GLM cases agree with isolated installed R qgcomp 2.19.6")


if __name__ == "__main__":
    main()
