"""Optional installed-R check for the marginal Cox model on identical outcomes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from quantilemix import fit_cox_bootstrap

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests/fixtures/r_cox_ties_2_19_6.json"
INPUT = ROOT / "tests/fixtures/r_cox_bootstrap_stage_inputs_2_19_6.json"
EXPECTED = ROOT / "tests/fixtures/r_cox_bootstrap_stage_expected_2_19_6.json"
PY_EXPORTER = Path(__file__).with_name("export_cox_bootstrap_python.py")
R_EXPORTER = Path(__file__).with_name("export_cox_bootstrap_stage_r.R")
HASHES = {
    SOURCE: "c2cca8cb735dab35b299cbd485f5e88c86d2b3097cd0049331faa9e2ac5fa161",
    INPUT: "ae5bbdbcbc9d6675f4a25baa816511187d347145364bdda06f8d7143a25a20d4",
    EXPECTED: "4732c26b760a36b92824d276e5e35493ca976d282f897d6d94f1b496b9de3601",
    # Updated Python import namespace; input/output fixture hashes stay fixed.
    PY_EXPORTER: "acb4dadb9b5f5ccb440e2f648e75984b312ecdd2061276d73b5a97e5c32a537d",
    R_EXPORTER: "4625408d6d4ad55f5261e3c5409472dbec0b19b5ef3f620a46b68dde2a22ab3e",
}
NAMES = ("cox_prequantized_efron_ties", "cox_quantized_efron_ties")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(actual, expected, label: str, *, atol=2e-8) -> float:
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError(f"{label}: shape or finiteness differs")
    gap = float(np.max(np.abs(a-b)))
    if gap > atol:
        raise ValueError(f"{label}: difference {gap:.3g} exceeds {atol:.3g}")
    return gap


def check(inputs: dict, saved_r: dict, live_r: dict, source: dict) -> dict:
    if (saved_r.get("qgcomp_version") != "2.19.6"
            or live_r.get("qgcomp_version") != "2.19.6"):
        raise ValueError("installed qgcomp 2.19.6 required")
    cases = [data.get("cases") for data in (inputs, saved_r, live_r, source)]
    if (not all(isinstance(group, list) and len(group) == 2 for group in cases)
            or any(tuple(case.get("name") for case in group) != NAMES for group in cases)):
        raise ValueError("two named marginal Cox cases required")
    details = []
    for supplied, old, new, original in zip(*cases):
        name = supplied["name"]
        if supplied["degree"] != old["degree"] or supplied["degree"] != new["degree"]:
            raise ValueError(f"{name}: degree differs")
        expected, current = old["expected"], new["expected"]
        if set(expected) != set(current):
            raise ValueError(f"{name}: R result fields differ")
        for key, value in expected.items():
            a, b = current[key], value
            if key in ("psi", "se", "p"):
                a, b = np.atleast_1d(a), np.atleast_1d(b)
            compare(a, b, f"{name}.{key}.saved_R", atol=1e-9)
        x = original["inputs"]
        result = fit_cox_bootstrap(
            x["time"], x["event"], np.column_stack((x["x1"], x["x2"])),
            x["z"], q=x["q"], degree=supplied["degree"],
            B=supplied["B"], MCsize=supplied["MCsize"], seed=supplied["seed"])
        for key, value in (("simulated_time", result.simulated_time),
                           ("simulated_event", result.simulated_event),
                           ("simulated_level", result.simulated_level),
                           ("bootstrap_effects", result.bootstrap_effects)):
            compare(value, supplied[key], f"{name}.{key}.Python_input", atol=1e-9)
        actual = {"psi": result.psi, "covariance": result.covariance,
                  "se": result.se, "ci": result.ci,
                  "p": result.p_value, "events": int(np.sum(result.simulated_event))}
        gaps = {}
        for key, value in actual.items():
            reference = expected[key]
            if key in ("psi", "se", "p"):
                reference = np.atleast_1d(reference)
            gaps[key] = compare(value, reference, f"{name}.{key}.Python_R")
        details.append({"name": name, "degree": supplied["degree"],
                        "max_absolute_differences": gaps})
    return {"status": "passed", "checked_utc": datetime.now(timezone.utc).isoformat(),
            "qgcomp_version": "2.19.6", "input_sha256": HASHES[INPUT],
            "r_reference_sha256": HASHES[EXPECTED],
            "python_exporter_sha256": HASHES[PY_EXPORTER],
            "r_exporter_sha256": HASHES[R_EXPORTER], "cases": details}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise ValueError("output already exists")
    if any(sha(path) != fingerprint for path, fingerprint in HASHES.items()):
        raise ValueError("marginal Cox input or exporter hash differs")
    with tempfile.TemporaryDirectory() as folder:
        live_path = Path(folder) / "live.json"
        subprocess.run([args.rscript, str(R_EXPORTER), args.library,
                        str(INPUT), str(live_path)], check=True)
        report = check(json.loads(INPUT.read_text()), json.loads(EXPECTED.read_text()),
                       json.loads(live_path.read_text()), json.loads(SOURCE.read_text()))
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("Two marginal Cox stages agree with isolated installed R on identical outcomes")


if __name__ == "__main__":
    main()
