"""Compare a fixed Gaussian partial-effect case with isolated installed R."""

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
FIXTURE = ROOT / "tests/fixtures/r_partials_2_19_6.json"
DRIVER = Path(__file__).with_name("export_partials_r.R")
SHA256 = "63024bdcb5d4932ac79725f2e3e79ddbb1684cc380bb055e54839e518de01149"


def compare(live: dict, saved: dict) -> None:
    if saved["version"] != live["version"] or live["version"] != "2.19.6":
        raise ValueError("R qgcomp version differs")
    for portion in ("train", "valid"):
        for name in ("y", "x1", "x2", "z"):
            np.testing.assert_allclose(saved[portion][name], live[portion][name], rtol=0, atol=1e-12)
    for name in ("positive", "negative"):
        if saved[name] != live[name]:
            raise ValueError(f"{name} training selection differs")
    for name in ("training", "positive_fit", "negative_fit"):
        for field in ("psi", "se", "ci"):
            np.testing.assert_allclose(saved[name][field], live[name][field], rtol=0, atol=1e-11)
    if saved["cluster_split"] != live["cluster_split"]:
        raise ValueError("installed R cluster split differs from the pinned case")
    cluster_split = live["cluster_split"]
    if sorted(cluster_split["trainidx"] + cluster_split["valididx"]) != list(range(1, 13)):
        raise ValueError("installed R split does not partition all rows")
    if cluster_split["overlapping_ids"] != [1, 2, 3]:
        raise ValueError("R's within-cluster sampling behavior changed")
    train, valid = saved["train"], saved["valid"]
    result = qgcomp_partials(train["y"], np.column_stack((train["x1"], train["x2"])),
                             valid["y"], np.column_stack((valid["x1"], valid["x2"])),
                             train_covariates=train["z"], validation_covariates=valid["z"], q=None)
    if result.positive_exposures != (saved["positive"],) or result.negative_exposures != (saved["negative"],):
        raise ValueError("Python sign selection differs from installed R")
    for name, fit in (("training", result.training_fit),
                      ("positive_fit", result.positive_fit),
                      ("negative_fit", result.negative_fit)):
        for field, actual in (("psi", fit.psi), ("se", fit.se), ("ci", fit.ci)):
            np.testing.assert_allclose(actual, saved[name][field], rtol=0, atol=1e-10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", required=True)
    parser.add_argument("--library", required=True)
    args = parser.parse_args()
    if hashlib.sha256(FIXTURE.read_bytes()).hexdigest() != SHA256:
        raise ValueError("saved R partial fixture hash differs")
    saved = json.loads(FIXTURE.read_text())
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp) / "live.json"
        subprocess.run([args.rscript, str(DRIVER), args.library, str(output)], check=True)
        compare(json.loads(output.read_text()), saved)
    print("Gaussian sample-split partial effects agree with isolated installed R qgcomp 2.19.6")


if __name__ == "__main__":
    main()
