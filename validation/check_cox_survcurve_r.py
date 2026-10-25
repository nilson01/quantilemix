"""Optional installed-R comparison for four marginal Cox survival-curve tables."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from quantilemix import fit_cox_bootstrap, qgcomp_survcurve_boot


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests/fixtures/r_cox_ties_2_19_6.json"
STAGE = ROOT / "tests/fixtures/r_cox_bootstrap_stage_inputs_2_19_6.json"
EXPECTED = ROOT / "tests/fixtures/r_cox_survcurve_2_19_6.json"
R_EXPORTER = Path(__file__).with_name("export_cox_survcurve_r.R")
HASHES = {
    SOURCE: "c2cca8cb735dab35b299cbd485f5e88c86d2b3097cd0049331faa9e2ac5fa161",
    STAGE: "ae5bbdbcbc9d6675f4a25baa816511187d347145364bdda06f8d7143a25a20d4",
    EXPECTED: "f98f4e41b8f39e01a4a22d0bffe05cbeab9a6a4b0a8df3139f184f4e0097ec33",
    R_EXPORTER: "92d4df6d59319d10cc452a43fbef291bbb91300f7550dada25861d3ddcdaf81a",
}
TABLES = ("mdfpop", "cdfpop", "mdfq", "cdfq")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(actual, expected, label: str, *, atol: float = 2e-8) -> float:
    a, b = np.atleast_1d(actual).astype(float), np.atleast_1d(expected).astype(float)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError(f"{label}: shape or finiteness differs")
    gap = float(np.max(np.abs(a-b)))
    if gap > atol:
        raise ValueError(f"{label}: difference {gap:.3g} exceeds {atol:.3g}")
    return gap


def check(saved: dict, live: dict, source: dict, stage: dict) -> dict:
    if saved.get("qgcomp_version") != "2.19.6" or live.get("qgcomp_version") != "2.19.6":
        raise ValueError("installed qgcomp 2.19.6 required")
    old, new = saved.get("cases"), live.get("cases")
    if (not isinstance(old, list) or not isinstance(new, list)
            or len(old) != 2 or len(new) != 2
            or [case.get("degree") for case in old] != [1, 2]
            or [case.get("degree") for case in new] != [1, 2]):
        raise ValueError("degree-one and degree-two survival curve cases required")
    if (stage["cases"][1]["name"] != "cox_quantized_efron_ties"
            or source["cases"][1]["name"] != "cox_quantized_efron_ties"):
        raise ValueError("wrong fixed survival source case")
    x = source["cases"][1]["inputs"]
    details = []
    for old_case, new_case in zip(old, new):
        degree = old_case["degree"]
        differences = {}
        for name in TABLES:
            old_table, new_table = old_case["curves"][name], new_case["curves"][name]
            if set(old_table) != set(new_table):
                raise ValueError(f"{name}: R columns differ")
            for column in old_table:
                compare(new_table[column], old_table[column],
                        f"degree_{degree}.{name}.{column}.saved_R", atol=1e-12)
        fit = fit_cox_bootstrap(
            x["time"], x["event"], np.column_stack((x["x1"], x["x2"])),
            x["z"], q=4, degree=degree, B=3, MCsize=128, seed=271828)
        compare(fit.simulated_time, stage["cases"][1]["simulated_time"],
                f"degree_{degree}.simulated_time.saved_stage", atol=0)
        compare(fit.simulated_event, stage["cases"][1]["simulated_event"],
                f"degree_{degree}.simulated_event.saved_stage", atol=0)
        differences["psi"] = compare(fit.psi, old_case["psi"], f"degree_{degree}.psi.Python_R")
        curves = qgcomp_survcurve_boot(fit)
        for name in TABLES:
            for column, expected in old_case["curves"][name].items():
                key = f"{name}.{column}"
                differences[key] = compare(curves[name][column], expected,
                                           f"degree_{degree}.{key}.Python_R")
        details.append({"degree": degree, "maximum_absolute_differences": differences})
    return {"status": "passed", "checked_utc": datetime.now(timezone.utc).isoformat(),
            "qgcomp_version": "2.19.6", "source_sha256": HASHES[SOURCE],
            "stage_sha256": HASHES[STAGE], "r_fixture_sha256": HASHES[EXPECTED],
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
        raise ValueError("survival-curve reference input or exporter hash differs")
    with tempfile.TemporaryDirectory() as folder:
        live_path = Path(folder) / "live.json"
        subprocess.run([args.rscript, str(R_EXPORTER), args.library,
                        str(SOURCE), str(STAGE), str(live_path)], check=True)
        report = check(json.loads(EXPECTED.read_text()), json.loads(live_path.read_text()),
                       json.loads(SOURCE.read_text()), json.loads(STAGE.read_text()))
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("Eight survival-curve tables agree with isolated installed R")


if __name__ == "__main__":
    main()
