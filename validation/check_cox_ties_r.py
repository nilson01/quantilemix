"""Optional replay of installed-R Cox fits with shared event and censor times."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from quantilemix import fit_cox

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/r_cox_ties_2_19_6.json"
DRIVER = Path(__file__).with_name("export_cox_ties_r.R")
FIXTURE_SHA = "c2cca8cb735dab35b299cbd485f5e88c86d2b3097cd0049331faa9e2ac5fa161"
DRIVER_SHA = "aa6211f6662de76627d8e5c54dbfacf7aacdf095898faa94ccb29288dc07e887"
NAMES = ("cox_prequantized_efron_ties", "cox_quantized_efron_ties")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(actual, expected, label: str, *, atol: float = 1e-9) -> float:
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError(f"{label}: shape or finiteness differs")
    gap = float(np.max(np.abs(a-b)))
    if gap > atol:
        raise ValueError(f"{label}: difference {gap:.3g} exceeds {atol:.3g}")
    return gap


def check(saved: dict, live: dict) -> dict:
    if saved.get("qgcomp_version") != "2.19.6" or live.get("qgcomp_version") != "2.19.6":
        raise ValueError("installed qgcomp 2.19.6 required")
    old_cases, new_cases = saved.get("cases"), live.get("cases")
    if (not isinstance(old_cases, list) or not isinstance(new_cases, list)
            or len(old_cases) != len(new_cases) or len(new_cases) != 2
            or tuple(case.get("name") for case in old_cases) != NAMES
            or tuple(case.get("name") for case in new_cases) != NAMES):
        raise ValueError("two fixed Efron Cox cases required")
    details = []
    for old, new in zip(old_cases, new_cases):
        if old["inputs"] != new["inputs"]:
            raise ValueError(f"{old['name']}: fixed inputs changed")
        e_old, e = old["expected"], new["expected"]
        if (set(e_old) != set(e) or e_old["tie_method"] != "efron"
                or e["tie_method"] != "efron"):
            raise ValueError(f"{old['name']}: reference fields or tie method changed")
        for key in e_old:
            if key == "tie_method":
                continue
            if key == "breaks":
                if e_old[key] is None or e[key] is None:
                    if e_old[key] != e[key]:
                        raise ValueError(f"{old['name']}.breaks.saved_R differs")
                else:
                    if len(e_old[key]) != len(e[key]):
                        raise ValueError(f"{old['name']}.break count differs")
                    for previous, current in zip(e_old[key], e[key]):
                        compare(current, previous, f"{old['name']}.breaks.saved_R")
            else:
                compare(e[key], e_old[key], f"{old['name']}.{key}.saved_R")
        a = new["inputs"]
        result = fit_cox(a["time"], a["event"],
                         np.column_stack((a["x1"], a["x2"])), a["z"], q=a["q"])
        actual = {"coefficients": result.coefficients,
                  "covariance": result.covariance,
                  "psi": result.psi, "variance": result.var_psi,
                  "ci": result.ci, "p": result.p_value,
                  "scores": result.scores, "events": result.event_count,
                  "log_likelihood": result.log_likelihood,
                  "tied_time_groups": result.tied_time_groups}
        gaps = {key: compare(value, e[key], f"{old['name']}.{key}.Python")
                for key, value in actual.items()}
        if result.breaks is None or e["breaks"] is None:
            if result.breaks is not None or e["breaks"] is not None:
                raise ValueError(f"{old['name']}.breaks.Python differs")
        else:
            if len(result.breaks) != len(e["breaks"]):
                raise ValueError(f"{old['name']}.break count differs")
            for value, reference in zip(result.breaks, e["breaks"]):
                compare(value, reference, f"{old['name']}.breaks.Python")
        details.append({"name": old["name"], "max_absolute_differences": gaps})
    return {"status": "passed", "checked_utc": datetime.now(timezone.utc).isoformat(),
            "qgcomp_version": "2.19.6", "saved_fixture_sha256": FIXTURE_SHA,
            "r_exporter_sha256": DRIVER_SHA, "cases": details}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise ValueError("output already exists")
    if sha(FIXTURE) != FIXTURE_SHA or sha(DRIVER) != DRIVER_SHA:
        raise ValueError("Cox tie fixture or exporter hash differs")
    with tempfile.TemporaryDirectory() as folder:
        live_path = Path(folder) / "live.json"
        subprocess.run([args.rscript, str(DRIVER), args.library, str(live_path)], check=True)
        report = check(json.loads(FIXTURE.read_text()), json.loads(live_path.read_text()))
    output.write_text(json.dumps(report, indent=2)+"\n")
    print("Two Cox Efron-tie cases agree with isolated installed R qgcomp 2.19.6")


if __name__ == "__main__":
    main()
