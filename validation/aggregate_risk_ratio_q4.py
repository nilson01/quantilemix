# SPDX-License-Identifier: GPL-2.0-or-later
"""Recalculate four-level study summaries from raw records without qgcomp/NumPy."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist, mean, stdev


def wilson(yes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = NormalDist().inv_cdf(.975)
    p = yes / total
    den = 1 + z*z/total
    mid = (p + z*z/(2*total))/den
    half = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/den
    return [mid-half, mid+half]


def aggregate(payload: dict) -> dict:
    plan = payload["plan"]
    if plan.get("protocol") != "binary-risk-ratio-four-level-known-scores-v1":
        raise ValueError("unexpected protocol")
    records = payload["records"]
    n = plan["n"]
    count = plan["repetitions_per_cell"]
    if len(records) != plan["total_datasets"]:
        raise ValueError("raw record count differs from plan")
    summaries = {}
    seen = set()
    for spec in plan["cells"]:
        cell = spec["cell"]
        rows = [row for row in records if row["cell"] == cell]
        if len(rows) != count:
            raise ValueError(f"cell {cell} record count differs")
        for row in rows:
            key = (row["cell"], row["repetition"])
            if key in seen or not 0 <= row["repetition"] < count:
                raise ValueError("duplicate or invalid case identity")
            seen.add(key)
        good = [row for row in rows if row["fit"]["status"] == "ok"]
        m = len(good)
        truth = spec["true_log_risk_ratio"]
        covered = sum(row["fit"]["ci"][0] <= truth <= row["fit"]["ci"][1]
                      for row in good)
        rejected = sum(row["fit"]["p_value"] < .05 for row in good)
        item = {"planned": count, "attempted": len(rows), "missing": count-len(rows),
                "successful": m, "failed": len(rows)-m,
                "warnings": sum(bool(row["fit"]["warnings"]) for row in rows),
                "observed_event_rate": sum(row["events"] for row in rows)/(n*len(rows)),
                "covered": covered, "rejected": rejected,
                "coverage_all_planned_bounds": [covered/count, (covered+count-m)/count]}
        if m:
            errors = [row["fit"]["estimate"]-truth for row in good]
            standard_errors = [row["fit"]["se"] for row in good]
            empirical_sd = stdev(errors) if m > 1 else None
            item.update({"bias": mean(errors),
                         "bias_mcse": empirical_sd/math.sqrt(m) if m > 1 else None,
                         "empirical_sd": empirical_sd,
                         "root_mean_variance": math.sqrt(mean(se*se for se in standard_errors)),
                         "coverage": covered/m, "coverage_wilson_95": wilson(covered,m),
                         "rejection_rate": rejected/m,
                         "rejection_wilson_95": wilson(rejected,m)})
        summaries[str(cell)] = item
    if len(seen) != plan["total_datasets"]:
        raise ValueError("case identity set incomplete")
    return summaries


def compare(saved: dict, fresh: dict) -> list[str]:
    differences = []
    for cell, values in fresh.items():
        candidate = saved.get(cell)
        if not isinstance(candidate, dict) or set(candidate) != set(values):
            differences.append(f"cell {cell}: fields differ")
            continue
        for name, expected in values.items():
            actual = candidate[name]
            if isinstance(expected, list):
                equal = (isinstance(actual, list) and len(actual) == len(expected)
                         and all(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
                                 for a,b in zip(actual,expected,strict=True)))
            elif isinstance(expected, float):
                equal = isinstance(actual,(int,float)) and math.isclose(
                    actual,expected,rel_tol=1e-12,abs_tol=1e-12)
            else:
                equal = actual == expected
            if not equal:
                differences.append(f"cell {cell}: {name} differs")
    return differences


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("output must be a new path in an existing directory")
    raw = args.input.read_bytes()
    payload = json.loads(raw)
    fresh = aggregate(payload)
    differences = compare(payload["summaries"],fresh)
    report = {"status":"passed" if not differences else "failed",
              "input_sha256":hashlib.sha256(raw).hexdigest(),
              "run_id":payload["run_id"],"summaries":fresh,"differences":differences}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(f"{len(payload['records'])} raw records; {len(differences)} summary differences")
    return 0 if not differences else 1


if __name__ == "__main__":
    raise SystemExit(main())
