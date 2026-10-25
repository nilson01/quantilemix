# SPDX-License-Identifier: GPL-2.0-or-later
"""Aggregate raw unequal-time Poisson bootstrap cases without importing the runner."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scipy.stats import norm

TRUTHS = (0.0, 0.4, 0.0, 0.4)


def wilson(yes: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = float(norm.ppf(0.975))
    fraction = yes / total
    denominator = 1 + z*z/total
    center = (fraction + z*z/(2*total)) / denominator
    half = z*math.sqrt(fraction*(1-fraction)/total + z*z/(4*total*total)) / denominator
    return [center-half, center+half]


def aggregate(path: Path) -> dict:
    payload = json.loads(path.read_text())
    plan = payload["plan"]
    if plan["protocol"] != "poisson-bootstrap-offset-known-scores-v1":
        raise ValueError("unexpected protocol")
    repetitions = plan["repetitions_per_cell"]
    records = payload["records"]
    seen = set()
    output = {}
    for cell, truth in enumerate(TRUTHS):
        rows = [row for row in records if row["cell"] == cell]
        if len(rows) != repetitions:
            raise ValueError(f"cell {cell}: incomplete records")
        for row in rows:
            identity = (row["cell"], row["repetition"])
            if identity in seen or not 0 <= row["repetition"] < repetitions:
                raise ValueError("duplicate or invalid identity")
            seen.add(identity)
        methods = {}
        for name in ("bootstrap", "model_based"):
            good = [row[name] for row in rows if row[name]["status"] == "ok"]
            count = len(good)
            covered = sum(fit["ci"][0] <= truth <= fit["ci"][1] for fit in good)
            rejected = sum(fit["p_value"] < 0.05 for fit in good)
            errors = [fit["estimate"]-truth for fit in good]
            bias = sum(errors)/count if count else None
            empirical_variance = (sum((error-bias)**2 for error in errors)/(count-1)
                                  if count > 1 else None)
            methods[name] = {
                "successful": count, "failed": repetitions-count,
                "covered": covered, "rejected": rejected,
                "coverage": covered/count if count else None,
                "coverage_wilson_95": wilson(covered,count),
                "coverage_all_planned_bounds": [covered/repetitions,
                    (covered+repetitions-count)/repetitions],
                "rejection_rate": rejected/count if count else None,
                "rejection_wilson_95": wilson(rejected,count),
                "bias": bias,
                "bias_mcse": math.sqrt(empirical_variance/count) if count > 1 else None,
                "empirical_sd": math.sqrt(empirical_variance) if count > 1 else None,
                "root_mean_variance": (math.sqrt(sum(fit["se"]**2 for fit in good)/count)
                                       if count else None),
                "warnings": sum(bool(row[name]["warnings"]) for row in rows),
            }
        output[str(cell)] = {
            "planned": repetitions, "attempted": len(rows), "missing": 0,
            "observed_mean": sum(row["count_total"] for row in rows)/(500*repetitions),
            "mean_observation_time": sum(row["time_total"] for row in rows)/(500*repetitions),
            "zero_fraction": sum(row["zero_count"] for row in rows)/(500*repetitions),
            **methods,
        }
    if len(seen) != len(records) or len(records) != plan["total_datasets"]:
        raise ValueError("unexpected total record count")
    for cell, independently_calculated in output.items():
        saved = payload["summaries"][cell]
        for key, value in independently_calculated.items():
            if isinstance(value, dict):
                for metric, number in value.items():
                    other = saved[key][metric]
                    if not _equal(number,other):
                        raise ValueError(f"summary differs: {cell}.{key}.{metric}")
            elif not _equal(value,saved[key]):
                raise ValueError(f"summary differs: {cell}.{key}")
    return output


def _equal(a, b):
    if isinstance(a,list):
        return isinstance(b,list) and len(a)==len(b) and all(_equal(x,y) for x,y in zip(a,b))
    if isinstance(a,float):
        return isinstance(b,(float,int)) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
    return a==b


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result",type=Path)
    args=parser.parse_args()
    print(json.dumps(aggregate(args.result),indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
