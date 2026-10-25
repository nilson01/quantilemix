# SPDX-License-Identifier: GPL-2.0-or-later
"""Optional same-input installed-R audit of the 18 fixed scenario-5 pilot cases."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np


HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


paper = _load("paper_scenario5")
checker = _load("check_paper_scenario5")


def check(rscript: Path, library: Path, pilot: Path, *, r_version: str = "2.19.6",
          dependency_library: Path | None = None) -> dict:
    if r_version not in {"1.3.0", "2.19.6"}:
        raise ValueError("R qgcomp version must be 1.3.0 or 2.19.6")
    dependency_library = dependency_library or library
    if not rscript.is_file() or not library.is_dir() or not dependency_library.is_dir():
        raise ValueError("Rscript, package library and dependency library must exist")
    audited = checker.check(pilot)
    if audited["mode"] != "pilot" or audited["verified_cases"] != 90:
        raise ValueError("expected complete fixed pilot")
    saved = json.loads(pilot.read_text(encoding="utf-8"))
    first = {(r["variant"], r["cell"]): r for r in saved["records"] if r["repetition"] == 0}
    cases = []
    for variant in paper.VARIANTS:
        for specification in paper.cells():
            cell = specification["cell"]
            row = first[(variant, cell)]
            if row["failure"] is not None:
                raise ValueError(f"fixed R input failed in Python: {(variant, cell)}")
            x, y = paper.generate(variant, cell, 0)
            cases.append({"variant": variant, "cell": cell, "repetition": 0,
                          "exposures": x.tolist(), "y": y.tolist()})
    with tempfile.TemporaryDirectory(prefix="qgcomp-scenario5-r-") as temporary:
        inputs = Path(temporary) / "inputs.json"
        output = Path(temporary) / "r-results.json"
        inputs.write_text(json.dumps({"cases": cases}, allow_nan=False), encoding="utf-8")
        subprocess.run([str(rscript), "--vanilla", str(HERE / "export_paper_scenario5_r.R"),
                        str(library), str(dependency_library), r_version,
                        str(inputs), str(output)], check=True, capture_output=True,
                       text=True, timeout=120)
        reference = json.loads(output.read_text(encoding="utf-8"))
    if reference.get("reference") != f"installed R qgcomp {r_version}" or not reference.get(
            "package_origin_verified"):
        raise ValueError("R package identity was not verified")
    returned = reference.get("cases")
    if not isinstance(returned, list) or len(returned) != len(cases):
        raise ValueError("R result count differs from input count")
    comparisons = []
    differences = []
    for expected, observed in zip(cases, returned, strict=True):
        identity = (expected["variant"], expected["cell"], expected["repetition"])
        if (observed.get("variant"), observed.get("cell"), observed.get("repetition")) != identity:
            raise ValueError("R result identity differs from input identity")
        row = first[identity[:2]]
        compared = {"variant": identity[0], "cell": identity[1], "repetition": identity[2],
                    "input_sha256": row["input_sha256"]}
        for field, r_field in (("estimate", "psi"), ("variance", "variance")):
            value = observed.get(r_field)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("R returned a nonfinite value")
            difference = abs(row[field]-value)
            differences.append(difference)
            if not np.isclose(row[field], value, atol=1e-10, rtol=1e-8):
                raise ValueError(f"Python/R {field} differs for {identity}")
            compared[field] = {"python": row[field], "r": value,
                               "absolute_difference": difference}
        comparisons.append(compared)
    md5 = reference.get("installed_package_md5")
    if not isinstance(md5, dict) or set(md5) != {
            "DESCRIPTION", "NAMESPACE", "R/qgcomp.rdb", "R/qgcomp.rdx"
    } or not all(isinstance(value, str) and len(value) == 32 for value in md5.values()):
        raise ValueError("installed package MD5 map missing")
    dependencies = reference.get("dependency_versions")
    if not isinstance(dependencies, dict) or set(dependencies) != {
            "arm", "future", "future.apply", "ggplot2", "gridExtra", "jsonlite"
    } or not all(isinstance(value, str) and value for value in dependencies.values()):
        raise ValueError("R dependency versions missing")
    return {"reference": reference["reference"], "R_version": reference["R_version"],
            "installed_package_md5": md5, "dependency_versions": dependencies,
            "cases": len(cases), "all_agree": True,
            "max_absolute_difference": max(differences),
            "tolerance": {"atol": 1e-10, "rtol": 1e-8},
            "comparisons": comparisons}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", type=Path, required=True)
    parser.add_argument("--r-library", type=Path, required=True)
    parser.add_argument("--dependency-library", type=Path)
    parser.add_argument("--r-version", choices=("1.3.0", "2.19.6"), default="2.19.6")
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir():
        parser.error("output must be a fresh path in an existing directory")
    report = check(args.rscript, args.r_library, args.pilot, r_version=args.r_version,
                   dependency_library=args.dependency_library)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "cases": report["cases"],
                      "max_absolute_difference": report["max_absolute_difference"]}, indent=2))


if __name__ == "__main__":
    main()
