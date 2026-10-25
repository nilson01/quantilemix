# SPDX-License-Identifier: GPL-2.0-or-later
"""Read-only integrity check of a Keil Gaussian pilot or study result file."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm, t
import quantilemix._core as core


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("paper_reproduction", HERE / "paper_reproduction.py")
paper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(paper)
PUBLISHED = {
    "null": {"bias": 0., "empirical_sd": .08, "root_mean_variance": .08,
             "coverage": .94, "rejection_rate": .06},
    "cancellation": {"bias": 0., "empirical_sd": .09, "root_mean_variance": .09,
                     "coverage": .95, "rejection_rate": .05},
    "single": {"bias": 0., "empirical_sd": .08, "root_mean_variance": .08,
               "coverage": .94, "rejection_rate": .88},
    "distributed": {"bias": 0., "empirical_sd": .08, "root_mean_variance": .08,
                    "coverage": .95, "rejection_rate": .86},
}


def wilson(success: int, total: int) -> list[float]:
    if not 0 <= success <= total or total == 0:
        raise ValueError("invalid binomial count")
    z = float(norm.ppf(.975))
    p = success / total
    center = (p + z*z/(2*total)) / (1 + z*z/total)
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / (1 + z*z/total)
    return [center-half, center+half]


def independent_gaussian(x: np.ndarray, y: np.ndarray) -> dict:
    """Full-model QR contrast, without calling qgcomp's fitting function."""
    design = np.column_stack((np.ones(len(y)), x))
    q_matrix, r_matrix = np.linalg.qr(design, mode="reduced")
    if np.linalg.matrix_rank(r_matrix) != 5:
        raise ValueError("independent design is rank deficient")
    beta = np.linalg.solve(r_matrix, q_matrix.T @ y)
    residuals = y - design @ beta
    contrast = np.array([0., 1., 1., 1., 1.])
    psi = float(contrast @ beta)
    factor = np.linalg.solve(r_matrix.T, contrast)
    variance = float((residuals @ residuals) / (len(y)-5) * (factor @ factor))
    se = math.sqrt(variance)
    critical = float(norm.ppf(.975))
    return {"estimate": psi, "variance": variance,
            "ci": [psi-critical*se, psi+critical*se],
            "p_value": float(2*t.sf(abs(psi/se), len(y)-5))}


def paper_comparison(summaries: dict) -> dict:
    """Descriptive flags include uncertainty in both 1,000-case studies."""
    result = {}
    for scenario, published in PUBLISHED.items():
        current = summaries[scenario]
        n = current["successful"]
        if n < 2:
            result[scenario] = {"available": False}
            continue
        checks = {}
        for label in ("bias", "coverage", "rejection_rate"):
            observed, target = current[label], published[label]
            if label == "bias":
                se = math.sqrt(current["bias_mcse"]**2 + published["empirical_sd"]**2/1000)
            else:
                se = math.sqrt(observed*(1-observed)/n + target*(1-target)/1000)
            difference = observed-target
            checks[label] = {"observed": observed, "published_rounded": target,
                             "difference": difference, "combined_mcse": se,
                             "descriptive_divergence_flag": abs(difference) > 3*se+.005}
        result[scenario] = {"available": True, "checks": checks,
                            "published_empirical_sd": published["empirical_sd"],
                            "published_root_mean_variance": published["root_mean_variance"],
                            "not_equivalence_test": True}
    return result


def check(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    plan = data.get("plan")
    if plan != paper.design(plan.get("mode") if isinstance(plan, dict) else ""):
        raise ValueError("plan differs from fixed design")
    records = data.get("records")
    if not isinstance(records, list) or len(records) != plan["total_fits"]:
        raise ValueError("record count differs from plan")
    source = data.get("source_identity")
    if source is not None:
        if source.get("runner_sha256") != hashlib.sha256((HERE / "paper_reproduction.py").read_bytes()).hexdigest():
            raise ValueError("runner source hash differs from result")
        if source.get("core_sha256") != hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest():
            raise ValueError("fitting source hash differs from result")
        if plan["mode"] == "study" and not source.get("source_revision"):
            raise ValueError("full study lacks source revision")
    elif plan["mode"] == "study":
        raise ValueError("full study lacks source identity")
    journal = path.with_suffix(path.suffix + ".jsonl")
    if data.get("journal_sha256") is not None:
        if hashlib.sha256(journal.read_bytes()).hexdigest() != data["journal_sha256"]:
            raise ValueError("journal hash differs from result")
        journal_records = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
        if journal_records != records:
            raise ValueError("journal records differ from final result")
    elif plan["mode"] == "study":
        raise ValueError("full study lacks case journal")
    seen = set()
    verified = {}
    largest_fit_difference = 0.0
    for record in records:
        scenario, repetition = record.get("scenario"), record.get("repetition")
        if scenario not in paper.SCENARIOS or type(repetition) is not int:
            raise ValueError("invalid case identity")
        key = (scenario, repetition)
        if key in seen:
            raise ValueError("duplicate case identity")
        seen.add(key)
        x, y = paper.generate(scenario, repetition, mode=plan["mode"])
        if record.get("input_sha256") != paper.input_hash(x, y):
            raise ValueError("input hash differs from generated case")
        if record.get("truth") != sum(paper.SCENARIOS[scenario]):
            raise ValueError("truth differs from design")
        failure = record.get("failure")
        if failure is None:
            estimate, variance, pvalue = (record.get(k) for k in ("estimate", "variance", "p_value"))
            ci = record.get("ci")
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in
                       (estimate, variance, pvalue)) or variance <= 0 or not 0 <= pvalue <= 1:
                raise ValueError("invalid successful result")
            if not isinstance(ci, list) or len(ci) != 2 or not all(
                isinstance(v, (int, float)) and math.isfinite(v) for v in ci
            ) or not ci[0] <= estimate <= ci[1]:
                raise ValueError("invalid confidence interval")
            independent = independent_gaussian(x, y)
            for label in ("estimate", "variance", "p_value"):
                difference = abs(record[label] - independent[label])
                largest_fit_difference = max(largest_fit_difference, difference)
                if not np.isclose(record[label], independent[label], atol=1e-10, rtol=1e-8):
                    raise ValueError(f"independent {label} differs for {key}")
            for recorded, expected_limit in zip(ci, independent["ci"], strict=True):
                difference = abs(recorded - expected_limit)
                largest_fit_difference = max(largest_fit_difference, difference)
                if not np.isclose(recorded, expected_limit, atol=1e-10, rtol=1e-8):
                    raise ValueError(f"independent interval differs for {key}")
        elif not isinstance(failure, str) or not failure:
            raise ValueError("invalid failure record")
        verified.setdefault(scenario, []).append(record)
    expected = {(scenario, repetition) for scenario in paper.SCENARIOS
                for repetition in range(plan["repetitions_per_scenario"])}
    if seen != expected:
        raise ValueError("case identities do not cover the plan")
    summaries = data.get("summaries")
    if not isinstance(summaries, dict) or set(summaries) != set(paper.SCENARIOS):
        raise ValueError("summary scenarios differ from plan")
    intervals = {}
    for scenario, rows in verified.items():
        good = [r for r in rows if r["failure"] is None]
        summary = summaries[scenario]
        if summary["planned"] != len(rows) or summary["successful"] != len(good) or summary["failed"] != len(rows)-len(good):
            raise ValueError("summary denominators differ from records")
        if not good:
            if (summary.get("coverage_all_planned_bounds") != [0.0, 1.0] or
                    summary.get("rejection_all_planned_bounds") != [0.0, 1.0]):
                raise ValueError("all-failed bounds differ from records")
            continue
        truth = sum(paper.SCENARIOS[scenario])
        estimates = np.array([r["estimate"] for r in good])
        variances = np.array([r["variance"] for r in good])
        covered = sum(r["ci"][0] <= truth <= r["ci"][1] for r in good)
        rejected = sum(r["p_value"] < .05 for r in good)
        expected_values = {"bias": float(estimates.mean()-truth),
                           "empirical_sd": float(estimates.std(ddof=1)) if len(good) > 1 else None,
                           "root_mean_variance": float(np.sqrt(variances.mean())),
                           "coverage": covered/len(good), "rejection_rate": rejected/len(good),
                           "bias_mcse": float(estimates.std(ddof=1)/np.sqrt(len(good))) if len(good) > 1 else None}
        for label, value in expected_values.items():
            recorded = summary.get(label)
            if value is None:
                if recorded is not None:
                    raise ValueError(f"{scenario} {label} differs from records")
            elif not isinstance(recorded, (float, int)) or not math.isclose(recorded, value, abs_tol=1e-12):
                raise ValueError(f"{scenario} {label} differs from records")
        bounds = {"coverage_all_planned_bounds": [covered/len(rows),
                   (covered+len(rows)-len(good))/len(rows)],
                  "rejection_all_planned_bounds": [rejected/len(rows),
                   (rejected+len(rows)-len(good))/len(rows)]}
        for label, expected_bounds in bounds.items():
            if summary.get(label) != expected_bounds:
                raise ValueError(f"{scenario} {label} differs from records")
        intervals[scenario] = {"coverage_wilson_95": wilson(covered, len(good)),
                               "rejection_wilson_95": wilson(rejected, len(good))}
    return {"protocol": plan["protocol"], "mode": plan["mode"],
            "verified_cases": len(seen), "all_integrity_checks_pass": True,
            "independent_gaussian_checks": sum(r["failure"] is None for r in records),
            "largest_independent_fit_difference": largest_fit_difference,
            "intervals": intervals,
            "paper_comparison": paper_comparison(summaries) if plan["mode"] == "study" else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.result), indent=2))


if __name__ == "__main__":
    main()
