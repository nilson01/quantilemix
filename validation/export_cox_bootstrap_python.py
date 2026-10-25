"""Generate fixed marginal-Cox stage inputs from two pinned survival cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quantilemix import fit_cox_bootstrap

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests/fixtures/r_cox_ties_2_19_6.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise ValueError("output already exists")
    cases = json.loads(SOURCE.read_text())["cases"]
    settings = ((1, 314159), (2, 271828))
    records = []
    for case, (degree, seed) in zip(cases, settings):
        x = case["inputs"]
        result = fit_cox_bootstrap(
            x["time"], x["event"], np.column_stack((x["x1"], x["x2"])),
            x["z"], q=x["q"], degree=degree, B=3, MCsize=128, seed=seed)
        records.append({"name": case["name"], "degree": degree, "seed": seed,
                        "B": 3, "MCsize": 128,
                        "simulated_time": result.simulated_time.tolist(),
                        "simulated_event": result.simulated_event.tolist(),
                        "simulated_level": result.simulated_level.tolist(),
                        "bootstrap_effects": result.bootstrap_effects.tolist()})
    output.write_text(json.dumps({"cases": records}, indent=2) + "\n")
    print("Two fixed marginal Cox stage inputs exported")


if __name__ == "__main__":
    main()
