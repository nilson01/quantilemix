# SPDX-License-Identifier: GPL-2.0-or-later
"""Optional same-input check against isolated installed R qgcomp."""

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
SPEC = importlib.util.spec_from_file_location("paper_reproduction", HERE / "paper_reproduction.py")
paper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(paper)


def check(rscript: Path, library: Path, *, r_version: str = "2.19.6",
          dependency_library: Path | None = None) -> dict:
    if r_version not in {"1.3.0", "2.19.6"}:
        raise ValueError("R qgcomp version must be 1.3.0 or 2.19.6")
    dependency_library = dependency_library or library
    if not rscript.is_file() or not library.is_dir() or not dependency_library.is_dir():
        raise ValueError("Rscript, package library and dependency library must exist")
    cases, native = [], {}
    for scenario in paper.SCENARIOS:
        for repetition in range(5):
            x, y = paper.generate(scenario, repetition)
            fitted = paper.fit(y, x, q=None)
            cases.append({"scenario": scenario, "repetition": repetition,
                          "exposures": x.tolist(), "y": y.tolist()})
            native[(scenario, repetition)] = (fitted.psi, fitted.var_psi,
                                              paper.input_hash(x, y))
    with tempfile.TemporaryDirectory(prefix="qgcomp-paper-r-") as temporary:
        inputs = Path(temporary) / "inputs.json"
        output = Path(temporary) / "r-results.json"
        inputs.write_text(json.dumps({"cases": cases}, allow_nan=False), encoding="utf-8")
        subprocess.run([str(rscript), "--vanilla", str(HERE / "export_paper_r.R"),
                        str(library), str(dependency_library), r_version,
                        str(inputs), str(output)], check=True,
                       capture_output=True, text=True, timeout=120)
        reference = json.loads(output.read_text(encoding="utf-8"))
    if (reference.get("reference") != f"installed R qgcomp {r_version}" or
            not reference.get("package_origin_verified")):
        raise ValueError("R package identity was not verified")
    returned = reference.get("cases")
    if not isinstance(returned, list) or len(returned) != len(cases):
        raise ValueError("R result count differs from input count")
    differences = []
    audited = []
    for expected, observed in zip(cases, returned, strict=True):
        key = (expected["scenario"], expected["repetition"])
        if (observed.get("scenario"), observed.get("repetition")) != key:
            raise ValueError("R result identity differs from input identity")
        compared = {"scenario": key[0], "repetition": key[1],
                    "input_sha256": native[key][2]}
        for name, value in zip(("psi", "variance"), native[key][:2], strict=True):
            r_value = observed.get(name)
            if not isinstance(r_value, (int, float)) or not math.isfinite(r_value):
                raise ValueError("R returned a nonfinite result")
            difference = abs(value - r_value)
            differences.append(difference)
            compared[name] = {"python": value, "r": r_value,
                              "absolute_difference": difference}
            if not np.isclose(value, r_value, atol=1e-10, rtol=1e-8):
                raise ValueError(f"Python/R {name} mismatch for {key}")
        audited.append(compared)
    package_md5 = reference.get("installed_package_md5")
    if not isinstance(package_md5, dict) or set(package_md5) != {
        "DESCRIPTION", "NAMESPACE", "R/qgcomp.rdb", "R/qgcomp.rdx"
    } or not all(isinstance(v, str) and len(v) == 32 for v in package_md5.values()):
        raise ValueError("R package artifact identity missing")
    dependency_versions = reference.get("dependency_versions")
    if not isinstance(dependency_versions, dict) or set(dependency_versions) != {
        "arm", "future", "future.apply", "ggplot2", "gridExtra", "jsonlite"
    } or not all(isinstance(v, str) and v for v in dependency_versions.values()):
        raise ValueError("R dependency versions missing")
    return {"reference": reference["reference"], "R_version": reference["R_version"],
            "installed_package_md5": package_md5, "dependency_versions": dependency_versions,
            "cases": len(cases), "all_agree": True, "max_absolute_difference": max(differences),
            "tolerance": {"atol": 1e-10, "rtol": 1e-8}, "comparisons": audited}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", type=Path, required=True)
    parser.add_argument("--r-library", type=Path, required=True)
    parser.add_argument("--r-version", choices=("1.3.0", "2.19.6"), default="2.19.6")
    parser.add_argument("--dependency-library", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir():
        parser.error("output must be a fresh path in an existing directory")
    report = check(args.rscript, args.r_library, r_version=args.r_version,
                   dependency_library=args.dependency_library)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
