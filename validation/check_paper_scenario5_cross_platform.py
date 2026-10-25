# SPDX-License-Identifier: GPL-2.0-or-later
"""Read-only cross-platform check using exact saved inputs for rare BLAS rounding differences."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("check_paper_scenario5", HERE / "check_paper_scenario5.py")
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def check(result_path: Path, supplement_path: Path) -> dict:
    saved = json.loads(result_path.read_text())
    rows = json.loads(supplement_path.read_text())
    if not isinstance(rows, list) or len(rows) > 20:
        raise ValueError("unexpected supplementary input count")
    records = {(r["variant"], r["cell"], r["repetition"]): r
               for r in saved["records"]}
    exact = {}
    ordinary_generate = checker.paper.generate
    for row in rows:
        key = tuple(row["identity"])
        if len(key) != 3 or key not in records or key in exact:
            raise ValueError("invalid supplementary identity")
        x, generated_y = ordinary_generate(*key, mode=saved["plan"]["mode"])
        if hashlib.sha256(x.astype("<i8").tobytes()).hexdigest() != row["x_sha256"]:
            raise ValueError("supplementary exposure hash differs")
        y = np.asarray(row["y"], dtype="<f8")
        if y.shape != generated_y.shape or not np.all(np.isfinite(y)):
            raise ValueError("supplementary outcome malformed")
        if hashlib.sha256(y.tobytes()).hexdigest() != row["y_sha256"]:
            raise ValueError("supplementary outcome hash differs")
        if checker.paper.input_hash(x, y) != row["input_sha256"] or row[
                "input_sha256"] != records[key]["input_sha256"]:
            raise ValueError("supplementary input differs from result")
        changed = np.flatnonzero(y != generated_y)
        if len(changed) != 1 or np.max(np.abs(y-generated_y)) > 1e-14:
            raise ValueError("supplementary input is not a one-value rounding difference")
        exact[key] = (x, y)

    def generate(variant: str, cell: int, repetition: int, *, mode: str = "pilot"):
        return exact.get((variant, cell, repetition)) or ordinary_generate(
            variant, cell, repetition, mode=mode)

    checker.paper.generate = generate
    try:
        report = checker.check(result_path)
    finally:
        checker.paper.generate = ordinary_generate
    report["cross_platform_supplement_sha256"] = hashlib.sha256(
        supplement_path.read_bytes()).hexdigest()
    report["cross_platform_corrected_cases"] = len(exact)
    report["cross_platform_case_identities"] = [list(key) for key in exact]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("supplement", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.result, args.supplement), indent=2))


if __name__ == "__main__":
    main()
