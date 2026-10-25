"""Optional fixed-covariance check against installed R qgcomp 2.19.6."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from quantilemix import se_comb, vc_comb


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/r_covariance_helpers_2_19_6.json"
EXPORTER = Path(__file__).with_name("export_covariance_helpers_r.R")
FIXTURE_SHA256 = "813c30d98ae663aad2776f793fe3625d8331eaab8cd4472b77ed436d3ee2ea13"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regenerate", action="store_true", help="run the installed-R exporter first")
    args = parser.parse_args()
    if args.regenerate:
        subprocess.run([os.environ.get("RSCRIPT", "Rscript"), str(EXPORTER)],
                       cwd=ROOT, check=True)
    digest = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    if not args.regenerate and digest != FIXTURE_SHA256:
        raise SystemExit("pinned installed-R covariance fixture changed")
    data = json.loads(FIXTURE.read_text())
    if data["reference_version"] != "2.19.6":
        raise SystemExit("unexpected reference version")
    names, covariance = data["names"], data["covariance"]
    actual = [se_comb(["x1", "x2"], covariance, names=names),
              se_comb(["x1", "x2"], covariance, names=names, grad=.5),
              se_comb(["x1", "x2"], covariance, names=names, grad=[0, .3, .1])]
    expected = [data["se_default"], data["se_scalar"], data["se_full"]]
    if not np.allclose(actual, expected, rtol=0, atol=1e-14):
        raise SystemExit("standard-error helpers differ from installed R")
    vc = vc_comb(["x1", "x2"], covariance, names=names)
    if not np.allclose(vc, data["vc_default"], rtol=0, atol=1e-14):
        raise SystemExit("default covariance helper differs from installed R")
    weighted = vc_comb(["x1", "x2"], covariance, names=names, grad=.5)
    if not np.isclose(weighted[0, 1], .5*vc[0, 1], rtol=0, atol=1e-14):
        raise SystemExit("weighted covariance cross term is incorrect")
    if np.isclose(weighted[0, 1], data["vc_scalar_r"][0][1], rtol=0, atol=1e-14):
        raise SystemExit("the documented R weighted cross-term difference disappeared")
    print(json.dumps({"reference_version": "2.19.6", "fixture_sha256": digest,
                      "default_max_gap": float(max(np.max(np.abs(np.asarray(actual)-expected)),
                                                   np.max(np.abs(vc-np.asarray(data["vc_default"]))))),
                      "weighted_cross_term_corrected": True}, indent=2))


if __name__ == "__main__":
    main()
