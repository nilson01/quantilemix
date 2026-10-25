# SPDX-License-Identifier: GPL-2.0-or-later
"""Read-only independent check of the saved scenario-5 Gaussian pilot or study."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from scipy.stats import norm, t
import quantilemix._core as core


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("paper_scenario5", HERE / "paper_scenario5.py")
paper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(paper)


def independent_gaussian(x: np.ndarray, y: np.ndarray) -> dict:
    matrix = np.column_stack((np.ones(len(y)), x))
    q_matrix, r_matrix = np.linalg.qr(matrix, mode="reduced")
    if np.linalg.matrix_rank(r_matrix) != x.shape[1] + 1:
        raise ValueError("independent design is rank deficient")
    coefficients = np.linalg.solve(r_matrix, q_matrix.T @ y)
    residual = y - matrix @ coefficients
    contrast = np.r_[0., np.ones(x.shape[1])]
    estimate = float(contrast @ coefficients)
    factor = np.linalg.solve(r_matrix.T, contrast)
    variance = float((residual @ residual) / (len(y) - matrix.shape[1]) * (factor @ factor))
    se = math.sqrt(variance)
    z = float(norm.ppf(.975))
    return {"estimate": estimate, "variance": variance,
            "ci": [estimate-z*se, estimate+z*se],
            "p_value": float(2*t.sf(abs(estimate/se), len(y)-matrix.shape[1]))}


def wilson(success: int, total: int) -> list[float]:
    if not 0 <= success <= total or total == 0:
        raise ValueError("invalid binomial count")
    z = float(norm.ppf(.975))
    p = success / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total))/denominator
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total))/denominator
    return [center-half, center+half]


def _close(recorded: object, expected: float | None, label: str) -> None:
    if expected is None:
        if recorded is not None:
            raise ValueError(f"{label} differs from records")
    elif (not isinstance(recorded, (float, int)) or not math.isfinite(recorded)
          or not math.isclose(recorded, expected, rel_tol=1e-9, abs_tol=1e-12)):
        raise ValueError(f"{label} differs from records")


def check(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    plan = data.get("plan")
    if plan != paper.design(plan.get("mode") if isinstance(plan, dict) else ""):
        raise ValueError("plan differs from fixed design")
    records = data.get("records")
    if not isinstance(records, list) or len(records) != plan["total_fits"]:
        raise ValueError("record count differs from plan")
    source = data.get("source_identity")
    if not isinstance(source, dict) or source.get("runner_sha256") != hashlib.sha256(
            (HERE / "paper_scenario5.py").read_bytes()).hexdigest():
        raise ValueError("runner source hash differs from result")
    if source.get("core_sha256") != hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest():
        raise ValueError("fitting source hash differs from result")
    if plan["mode"] == "study":
        commit = source.get("git_commit")
        if (not isinstance(commit, str) or len(commit) != 40
                or source.get("source_revision") != commit or source.get("git_dirty") is not False):
            raise ValueError("full study lacks clean committed provenance")
        root = HERE.parent
        for local_path in (HERE / "paper_scenario5.py", Path(core.__file__).resolve()):
            try:
                relative = local_path.resolve().relative_to(root)
            except ValueError as exc:
                raise ValueError("study source is outside committed checkout") from exc
            committed = subprocess.run(["git", "show", f"{commit}:{relative.as_posix()}"],
                                       cwd=root, capture_output=True, check=False)
            if committed.returncode or committed.stdout != local_path.read_bytes():
                raise ValueError(f"study source differs from commit: {relative}")
    journal = path.with_suffix(path.suffix + ".jsonl")
    if data.get("journal_sha256") is None:
        if plan["mode"] == "study":
            raise ValueError("full study lacks case journal")
    else:
        if not journal.is_file() or hashlib.sha256(journal.read_bytes()).hexdigest() != data["journal_sha256"]:
            raise ValueError("journal hash differs from result")
        if [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()] != records:
            raise ValueError("journal records differ from final result")
    seen = set()
    by_cell = {}
    failed_case_diagnostics = []
    largest = 0.
    for record in records:
        variant, cell, repetition = (record.get(k) for k in ("variant", "cell", "repetition"))
        if variant not in paper.VARIANTS or type(cell) is not int or type(repetition) is not int:
            raise ValueError("invalid case identity")
        if not 0 <= cell < len(plan["cells"]) or not 0 <= repetition < plan["repetitions_per_cell"]:
            raise ValueError("case identity outside plan")
        key = (variant, cell, repetition)
        if key in seen:
            raise ValueError("duplicate case identity")
        seen.add(key)
        x, y = paper.generate(variant, cell, repetition, mode=plan["mode"])
        if record.get("input_sha256") != paper.input_hash(x, y):
            raise ValueError("input hash differs from generated case")
        if any(not np.array_equal(np.bincount(column, minlength=4), [125]*4) for column in x.T):
            raise ValueError("exposure scores are not balanced")
        if record.get("truth") != plan["cells"][cell]["truth"]:
            raise ValueError("truth differs from design")
        expected_corr = paper.correlations(x)
        observed_corr = record.get("correlations")
        if not isinstance(observed_corr, list) or len(observed_corr) != 3:
            raise ValueError("correlations missing")
        for actual, expected in zip(observed_corr, expected_corr, strict=True):
            _close(actual, expected, "realized correlation")
        warnings = record.get("warnings")
        if not isinstance(warnings, list) or not all(isinstance(w, str) for w in warnings):
            raise ValueError("warning record malformed")
        failure = record.get("failure")
        if failure is None:
            independent = independent_gaussian(x, y)
            for label in ("estimate", "variance", "p_value"):
                actual, expected = record.get(label), independent[label]
                if not isinstance(actual, (float, int)) or not math.isfinite(actual):
                    raise ValueError(f"nonfinite {label}")
                largest = max(largest, abs(actual-expected))
                if not np.isclose(actual, expected, atol=1e-10, rtol=1e-8):
                    raise ValueError(f"independent {label} differs for {key}")
            ci = record.get("ci")
            if not isinstance(ci, list) or len(ci) != 2:
                raise ValueError("interval malformed")
            for actual, expected in zip(ci, independent["ci"], strict=True):
                if not isinstance(actual, (float, int)) or not math.isfinite(actual):
                    raise ValueError("interval nonfinite")
                largest = max(largest, abs(actual-expected))
                if not np.isclose(actual, expected, atol=1e-10, rtol=1e-8):
                    raise ValueError(f"independent interval differs for {key}")
        elif (not isinstance(failure, str) or not failure
              or any(record.get(label) is not None for label in ("estimate", "variance", "ci", "p_value"))):
            raise ValueError("failure record malformed")
        elif failure is not None:
            matrix = np.column_stack((np.ones(len(y)), x))
            failed_case_diagnostics.append({"variant": variant, "cell": cell,
                                            "repetition": repetition, "exception": failure,
                                            "correlations": expected_corr,
                                            "design_rank": int(np.linalg.matrix_rank(matrix)),
                                            "condition_number": float(np.linalg.cond(matrix))})
        by_cell.setdefault((variant, cell), []).append(record)
    expected_ids = {(variant, cell["cell"], repetition)
                    for variant in paper.VARIANTS for cell in plan["cells"]
                    for repetition in range(plan["repetitions_per_cell"])}
    if seen != expected_ids:
        raise ValueError("case identities do not cover the plan")
    summaries = data.get("summaries")
    expected_keys = {f"{variant}:cell{cell['cell']}"
                     for variant in paper.VARIANTS for cell in plan["cells"]}
    if not isinstance(summaries, dict) or set(summaries) != expected_keys:
        raise ValueError("summary cells differ from plan")
    results = {}
    for (variant, cell), rows in by_cell.items():
        summary = summaries[f"{variant}:cell{cell}"]
        good = [r for r in rows if r["failure"] is None]
        if (summary.get("planned"), summary.get("successful"), summary.get("failed")) != (
                len(rows), len(good), len(rows)-len(good)):
            raise ValueError("summary denominators differ from records")
        if summary.get("warnings") != sum(bool(r["warnings"]) for r in rows):
            raise ValueError("summary warning count differs from records")
        expected_corr = np.mean([r["correlations"] for r in rows], axis=0)
        observed_corr = summary.get("mean_correlations")
        if not isinstance(observed_corr, list) or len(observed_corr) != 3:
            raise ValueError("summary correlations missing")
        for actual, expected in zip(observed_corr, expected_corr, strict=True):
            _close(actual, float(expected), "summary correlation")
        expected_sd = np.std([r["correlations"] for r in rows], axis=0, ddof=1) if len(rows) > 1 else None
        observed_sd = summary.get("sd_correlations")
        if expected_sd is None:
            if observed_sd is not None:
                raise ValueError("summary correlation spread differs from records")
        elif not isinstance(observed_sd, list) or len(observed_sd) != 3:
            raise ValueError("summary correlation spread missing")
        else:
            for actual, expected in zip(observed_sd, expected_sd, strict=True):
                _close(actual, float(expected), "summary correlation spread")
        if not good:
            if summary.get("coverage_all_planned_bounds") != [0., 1.] or summary.get(
                    "rejection_all_planned_bounds") != [0., 1.]:
                raise ValueError("all-failed bounds differ from records")
            results[f"{variant}:cell{cell}"] = {"available": False}
            continue
        deviations = np.array([r["estimate"] - r["truth"] for r in good])
        variance = np.array([r["variance"] for r in good])
        covered = sum(r["ci"][0] <= r["truth"] <= r["ci"][1] for r in good)
        rejected = sum(r["p_value"] < .05 for r in good)
        expected = {"bias": float(deviations.mean()),
                    "bias_mcse": float(deviations.std(ddof=1)/math.sqrt(len(good))) if len(good) > 1 else None,
                    "empirical_sd": float(deviations.std(ddof=1)) if len(good) > 1 else None,
                    "root_mean_variance": float(np.sqrt(variance.mean())),
                    "coverage": covered/len(good), "rejection_rate": rejected/len(good)}
        for label, value in expected.items():
            _close(summary.get(label), value, label)
        quartiles = summary.get("bias_quartiles")
        if not isinstance(quartiles, list) or len(quartiles) != 3:
            raise ValueError("bias quartiles missing")
        for actual, expected in zip(quartiles, np.quantile(deviations, [.25, .5, .75]), strict=True):
            _close(actual, float(expected), "bias quartile")
        bounds = {"coverage_all_planned_bounds": [covered/len(rows),
                   (covered+len(rows)-len(good))/len(rows)],
                  "rejection_all_planned_bounds": [rejected/len(rows),
                   (rejected+len(rows)-len(good))/len(rows)]}
        for label, expected in bounds.items():
            if summary.get(label) != expected:
                raise ValueError(f"{label} differs from records")
        results[f"{variant}:cell{cell}"] = {
            "available": True, "coverage_wilson_95": wilson(covered, len(good)),
            "rejection_wilson_95": wilson(rejected, len(good)),
            "bias_three_mcse_flag": len(good) > 1 and abs(float(deviations.mean())) > 3*float(
                deviations.std(ddof=1)/math.sqrt(len(good))),
            "bias_mc_interval_95": [float(deviations.mean()-1.96*deviations.std(ddof=1)/math.sqrt(len(good))),
                                    float(deviations.mean()+1.96*deviations.std(ddof=1)/math.sqrt(len(good)))]
            if len(good) > 1 else None,
            "coverage_three_mcse_flag": abs(covered/len(good)-.95) > 3*math.sqrt(.95*.05/len(good)),
        }
    variant_differences = {}
    for cell in range(len(plan["cells"])):
        first = summaries[f"released_ncor2:cell{cell}"]
        second = summaries[f"two_causal_ncor1:cell{cell}"]
        if first.get("bias_mcse") is not None and second.get("bias_mcse") is not None:
            difference = first["bias"] - second["bias"]
            mcse = math.hypot(first["bias_mcse"], second["bias_mcse"])
            variant_differences[f"cell{cell}"] = {"bias_difference": difference,
                                                    "mcse": mcse,
                                                    "mc_interval_95": [difference-1.96*mcse,
                                                                       difference+1.96*mcse]}
    return {"protocol": plan["protocol"], "mode": plan["mode"],
            "verified_cases": len(seen), "independent_gaussian_checks": sum(r["failure"] is None for r in records),
            "largest_independent_fit_difference": largest, "all_integrity_checks_pass": True,
            "cells": results, "variant_bias_differences": variant_differences,
            "failed_case_diagnostics": failed_case_diagnostics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.result), indent=2))


if __name__ == "__main__":
    main()
