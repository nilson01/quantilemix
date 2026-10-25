#!/usr/bin/env python3
"""Compare two saved optional paper-pilot R audits without refitting."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def compare(old_path: Path, new_path: Path) -> dict:
    old = json.loads(old_path.read_text(encoding="utf-8"))
    new = json.loads(new_path.read_text(encoding="utf-8"))
    if old.get("reference") != "installed R qgcomp 1.3.0":
        raise ValueError("first report must use installed qgcomp 1.3.0")
    if new.get("reference") != "installed R qgcomp 2.19.6":
        raise ValueError("second report must use installed qgcomp 2.19.6")
    if not isinstance(old.get("R_version"), str) or not old["R_version"]:
        raise ValueError("first report lacks an R version")
    if old["R_version"] != new.get("R_version"):
        raise ValueError("R versions differ")
    required_dependencies = {"arm", "future", "future.apply", "ggplot2",
                             "gridExtra", "jsonlite"}
    dependencies = old.get("dependency_versions")
    if (not isinstance(dependencies, dict)
            or set(dependencies) != required_dependencies
            or not all(isinstance(value, str) and value for value in dependencies.values())
            or dependencies != new.get("dependency_versions")):
        raise ValueError("dependency versions are missing or differ")
    md5_maps = []
    for report in (old, new):
        checksums = report.get("installed_package_md5")
        if (not isinstance(checksums, dict)
                or not {"DESCRIPTION", "NAMESPACE", "R/qgcomp.rdb", "R/qgcomp.rdx"}
                <= checksums.keys()
                or not all(isinstance(value, str) and len(value) == 32
                           and all(char in "0123456789abcdef" for char in value)
                           for value in checksums.values())):
            raise ValueError("installed package MD5 map is missing or malformed")
        md5_maps.append(checksums)
    if md5_maps[0] == md5_maps[1]:
        raise ValueError("installed package MD5 maps must differ")
    expected = [(name, i) for name in ("null", "cancellation", "single", "distributed")
                for i in range(5)]
    if old.get("cases") != 20 or new.get("cases") != 20:
        raise ValueError("expected 20 cases in each report")
    if not old.get("all_agree") or not new.get("all_agree"):
        raise ValueError("underlying Python/R check did not pass")
    differences = []
    for report in (old, new):
        rows = report.get("comparisons")
        if not isinstance(rows, list) or [(r.get("scenario"), r.get("repetition"))
                                           for r in rows] != expected:
            raise ValueError("case identities differ from the fixed pilot")
    for a, b in zip(old["comparisons"], new["comparisons"], strict=True):
        if a.get("input_sha256") != b.get("input_sha256"):
            raise ValueError("input hashes differ")
        for field in ("psi", "variance"):
            if a[field]["python"] != b[field]["python"]:
                raise ValueError("Python comparison values differ")
            values = (a[field]["r"], b[field]["r"])
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
                raise ValueError("nonfinite R result")
            differences.append(abs(values[0] - values[1]))
    return {"cases": 20, "fields_per_case": 2,
            "old_report_sha256": hashlib.sha256(old_path.read_bytes()).hexdigest(),
            "new_report_sha256": hashlib.sha256(new_path.read_bytes()).hexdigest(),
            "max_absolute_r_version_difference": max(differences),
            "all_40_saved_r_values_equal": all(d == 0 for d in differences)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old_report", type=Path)
    parser.add_argument("new_report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.old_report, args.new_report)
    if args.output:
        if args.output.exists() or not args.output.parent.is_dir():
            parser.error("output must be a fresh path in an existing directory")
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
