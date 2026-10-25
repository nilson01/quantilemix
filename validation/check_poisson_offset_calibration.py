# SPDX-License-Identifier: GPL-2.0-or-later
"""Recompute saved Poisson-log cases and summaries independently."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy import optimize
from scipy.stats import norm
import quantilemix._poisson as implementation
from quantilemix import fit_poisson
import scipy


HERE = Path(__file__).resolve().parent
PROTOCOL = "poisson-offset-known-scores-v1"
CELLS = tuple({"cell": 2 * i + j, "center_mean": mean, "truth": effect}
              for i, mean in enumerate((0.20, 2.00))
              for j, effect in enumerate((0.0, 0.40)))


def expected_plan(mode: str) -> dict:
    if mode not in ("pilot", "study"):
        raise ValueError("invalid mode")
    repetitions = 5 if mode == "pilot" else 1000
    return {"protocol": PROTOCOL, "mode": mode, "n": 500,
            "cells": CELLS, "repetitions_per_cell": repetitions,
            "total_fits": 4 * repetitions, "seed": 20261024}


def expected_input(mode: str, cell: int, repetition: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay the written seed and data model without importing the runner."""
    entropy = [20261024, {"pilot": 91, "study": 92}[mode], cell, repetition]
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))
    scores = rng.integers(low=0, high=4, size=(500, 4), dtype=np.int64)
    duration = np.array((0.25, 0.5, 1.0, 2.0), dtype=float)[
        rng.integers(low=0, high=4, size=500, dtype=np.int64)]
    spec = CELLS[cell]
    rate = spec["center_mean"] * np.exp(
        spec["truth"] * (scores.sum(axis=1) - 6) / 4)
    counts = rng.poisson(duration * rate).astype(np.int64)
    return scores, duration, counts


def expected_hash(scores: np.ndarray, duration: np.ndarray, counts: np.ndarray) -> str:
    return hashlib.sha256(scores.astype("<i8").tobytes() +
                          duration.astype("<f8").tobytes() +
                          counts.astype("<i8").tobytes()).hexdigest()


def audit_summary(records: list[dict], plan: dict, summaries: dict) -> None:
    """Check reported counts and rates against saved case rows."""
    for spec in CELLS:
        rows = [r for r in records if r["cell"] == spec["cell"]]
        successes = [r for r in rows if r["failure"] is None]
        n = len(successes)
        covered = sum(r["ci"][0] <= spec["truth"] <= r["ci"][1] for r in successes)
        rejected = sum(r["p_value"] < .05 for r in successes)
        actual = summaries[str(spec["cell"])]
        expected = {"planned": plan["repetitions_per_cell"], "attempted": len(rows),
                    "successful": n, "failed": len(rows)-n,
                    "missing": plan["repetitions_per_cell"]-len(rows),
                    "covered": covered, "rejected": rejected,
                    "warnings": sum(bool(r["warnings"]) for r in rows),
                    "coverage_all_planned_bounds": [covered/plan["repetitions_per_cell"],
                        (covered+plan["repetitions_per_cell"]-n)/plan["repetitions_per_cell"]]}
        if n:
            expected["coverage"] = covered/n
            expected["rejection_rate"] = rejected/n
        for name, value in expected.items():
            if actual.get(name) != value:
                raise ValueError(f"summary differs from records: cell {spec['cell']} {name}")


def independent_fit(x: np.ndarray, exposure_time: np.ndarray, y: np.ndarray) -> dict:
    design = np.column_stack((np.ones(len(y)), x.astype(float)))

    def equations(beta: np.ndarray) -> np.ndarray:
        return design.T @ (y - exposure_time * np.exp(design @ beta))

    def derivative(beta: np.ndarray) -> np.ndarray:
        mu = exposure_time * np.exp(design @ beta)
        return -(design.T @ (mu[:, None] * design))

    observed = float(y.sum() / exposure_time.sum())
    if observed <= 0:
        raise ValueError("independent Poisson fit has no observed events")
    initial = np.zeros(5)
    initial[0] = math.log(observed)
    result = optimize.root(equations, initial, jac=derivative, tol=1e-10)
    residual = equations(result.x)
    if (not np.all(np.isfinite(result.x)) or not np.all(np.isfinite(residual))
            or np.max(np.abs(residual)) > 1e-7):
        raise ValueError(f"independent score residual unacceptable: {np.max(np.abs(residual)):.6g}")
    information = -derivative(result.x)
    if not np.all(np.isfinite(information)) or np.linalg.eigvalsh(information)[0] <= 0:
        raise ValueError("independent Poisson information is not positive definite")
    contrast = np.array([0., 1., 1., 1., 1.])
    variance = float(contrast @ np.linalg.solve(information, contrast))
    estimate = float(contrast @ result.x)
    se = math.sqrt(variance)
    critical = float(norm.ppf(.975))
    return {"estimate": estimate, "variance": variance,
            "ci": [estimate-critical*se, estimate+critical*se],
            "p_value": float(2*norm.sf(abs(estimate/se))),
            "score_max_abs": float(np.max(np.abs(residual))),
            "solver_success": bool(result.success),
            "information_min_eigenvalue": float(np.linalg.eigvalsh(information)[0])}


def close(actual: object, expected: float) -> bool:
    return (type(actual) in (int, float) and math.isfinite(actual)
            and math.isclose(actual, expected, abs_tol=1e-7, rel_tol=1e-6))


def check_partial(path: Path) -> dict:
    """Audit an interrupted journal without calling it a complete result."""
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    if not entries or not isinstance(entries[0], dict):
        raise ValueError("partial journal header missing")
    header = entries[0]
    plan = header.get("plan")
    expected = expected_plan(plan.get("mode")) if isinstance(plan, dict) else None
    if expected is None or json.dumps(plan, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError("partial journal plan differs")
    source = header.get("source_identity")
    if (not isinstance(source, dict) or source.get("runner_sha256") !=
            hashlib.sha256((HERE / "poisson_offset_calibration.py").read_bytes()).hexdigest()):
        raise ValueError("partial journal runner differs")
    if source.get("implementation_sha256") != hashlib.sha256(
            Path(implementation.__file__).read_bytes()).hexdigest():
        raise ValueError("partial journal implementation differs")
    if len(entries)-1 > plan["total_fits"]:
        raise ValueError("too many partial case records")
    completed = []
    failures = []
    for index, row in enumerate(entries[1:]):
        if not isinstance(row, dict):
            raise ValueError("partial case row malformed")
        identity = (index // plan["repetitions_per_cell"],
                    index % plan["repetitions_per_cell"])
        if (row.get("cell"), row.get("repetition")) != identity:
            raise ValueError("partial case identities out of order")
        scores, duration, counts = expected_input(plan["mode"], *identity)
        if row.get("input_sha256") != expected_hash(scores, duration, counts):
            raise ValueError("partial case input differs")
        completed.append(list(identity))
        if row.get("failure") is not None:
            failures.append(list(identity))
    missing = [[index // plan["repetitions_per_cell"],
                index % plan["repetitions_per_cell"]]
               for index in range(len(completed), plan["total_fits"])]
    return {"status": "partial", "complete_study": False,
            "completed": completed, "failed": failures, "missing": missing,
            "journal_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def check(path: Path) -> dict:
    started = time.monotonic()
    payload = json.loads(path.read_text())
    plan = payload.get("plan")
    expected = expected_plan(plan.get("mode")) if isinstance(plan, dict) else None
    if expected is None or json.dumps(plan, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError("plan differs from fixed design")
    source = payload.get("source_identity")
    if not isinstance(source, dict):
        raise ValueError("source identity absent")
    for label, source_path in (("runner_sha256", HERE / "poisson_offset_calibration.py"),
                               ("implementation_sha256", Path(implementation.__file__))):
        if source.get(label) != hashlib.sha256(source_path.read_bytes()).hexdigest():
            raise ValueError(f"{label} differs")
    if plan["mode"] == "study":
        commit = source.get("git_commit")
        if (not isinstance(commit, str) or len(commit) != 40
                or source.get("git_dirty") is not False):
            raise ValueError("study lacks clean committed provenance")
        root = HERE.parent
        for source_path, rel in ((HERE / "poisson_offset_calibration.py", Path("validation/poisson_offset_calibration.py")),
                                 (Path(implementation.__file__), Path("src/quantilemix/_poisson.py"))):
            committed = subprocess.run(["git", "show", f"{commit}:{rel.as_posix()}"],
                                       cwd=root, capture_output=True, check=False)
            if committed.returncode or committed.stdout != source_path.read_bytes():
                raise ValueError("study source differs from commit")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != plan["total_fits"]:
        raise ValueError("record count differs from plan")
    seen = set()
    maximum_difference = 0.0
    paired = 0
    failures = 0
    independent_failures = 0
    issues = []
    case_audit = []
    per_cell = {str(spec["cell"]): {"paired": 0, "package_failures": 0,
                                    "independent_failures": 0, "mismatches": 0,
                                    "independent_successful": 0,
                                    "independent_covered": 0,
                                    "independent_rejected": 0}
                for spec in CELLS}
    independent_errors = {str(spec["cell"]): [] for spec in CELLS}
    for row in records:
        cell, repetition = row.get("cell"), row.get("repetition")
        if (type(cell) is not int or type(repetition) is not int
                or not 0 <= cell < len(CELLS)
                or not 0 <= repetition < plan["repetitions_per_cell"]
                or (cell, repetition) in seen):
            raise ValueError("case identity invalid or repeated")
        seen.add((cell, repetition))
        x, exposure_time, y = expected_input(plan["mode"], cell, repetition)
        if row.get("input_sha256") != expected_hash(x, exposure_time, y):
            raise ValueError("case input differs")
        if (row.get("count_total") != int(y.sum())
                or row.get("zero_count") != int(np.count_nonzero(y == 0))
                or row.get("exposure_time_total") != float(exposure_time.sum())):
            raise ValueError("count total or zero count differs")
        if not isinstance(row.get("warnings"), list) or any(
            not isinstance(w, dict) or set(w) != {"category", "message"}
            or not isinstance(w["category"], str) or not isinstance(w["message"], str)
            for w in row["warnings"]
        ):
            raise ValueError("warning list invalid")
        if row["warnings"]:
            issues.append({"cell": cell, "repetition": repetition,
                           "code": "package_warning", "warnings": row["warnings"]})
        failure = row.get("failure")
        if failure is not None:
            failures += 1
            per_cell[str(cell)]["package_failures"] += 1
            if not isinstance(failure, str) or not failure:
                raise ValueError("failure code invalid")
            if not isinstance(row.get("failure_detail"), str):
                raise ValueError("failure detail missing")
            if any(row.get(k) is not None for k in ("estimate", "variance", "ci", "p_value")):
                raise ValueError("failed case contains result")
            try:
                fit_poisson(y, x, q=None, exposure_time=exposure_time)
            except Exception as exc:
                if type(exc).__name__ != failure or str(exc)[:200] != row["failure_detail"]:
                    issues.append({"cell": cell, "repetition": repetition,
                                   "code": "package_failure_replay_differs",
                                   "recorded": [failure, row["failure_detail"]],
                                   "replayed": [type(exc).__name__, str(exc)[:200]]})
            else:
                issues.append({"cell": cell, "repetition": repetition,
                               "code": "fabricated_package_failure"})
        audit = {"cell": cell, "repetition": repetition,
                 "package_status": "failed" if failure else "succeeded"}
        try:
            other = independent_fit(x, exposure_time, y)
        except Exception as exc:
            independent_failures += 1
            per_cell[str(cell)]["independent_failures"] += 1
            audit.update({"independent_status": "failed", "independent_failure": type(exc).__name__,
                          "independent_failure_detail": str(exc)[:200]})
            case_audit.append(audit)
            issues.append({"cell": cell, "repetition": repetition,
                           "code": "independent_fit_failed",
                           "detail": type(exc).__name__, "reason": str(exc)[:200],
                           "package_failure": failure})
            continue
        per_cell[str(cell)]["independent_successful"] += 1
        truth = CELLS[cell]["truth"]
        independent_errors[str(cell)].append(other["estimate"]-truth)
        per_cell[str(cell)]["independent_covered"] += int(other["ci"][0] <= truth <= other["ci"][1])
        per_cell[str(cell)]["independent_rejected"] += int(other["p_value"] < .05)
        audit.update({"independent_status": "succeeded",
                      "independent_estimate": other["estimate"],
                      "independent_variance": other["variance"],
                      "score_max_abs": other["score_max_abs"],
                      "solver_success": other["solver_success"],
                      "information_min_eigenvalue": other["information_min_eigenvalue"]})
        if failure is not None:
            issues.append({"cell": cell, "repetition": repetition,
                           "code": "package_fit_failed", "detail": failure,
                           "reason": row["failure_detail"],
                           "independent_score_max_abs": other["score_max_abs"],
                           "independent_information_min_eigenvalue": other["information_min_eigenvalue"]})
            case_audit.append(audit)
            continue
        if row.get("failure_detail") is not None:
            raise ValueError("successful case has failure detail")
        mismatched = []
        for label in ("estimate", "variance", "p_value"):
            if type(row.get(label)) in (int, float) and math.isfinite(row[label]):
                maximum_difference = max(maximum_difference, abs(row[label]-other[label]))
            if not close(row.get(label), other[label]):
                mismatched.append(label)
        if not isinstance(row.get("ci"), list) or len(row["ci"]) != 2:
            raise ValueError("confidence interval malformed")
        for actual, expected in zip(row["ci"], other["ci"], strict=True):
            if type(actual) in (int, float) and math.isfinite(actual):
                maximum_difference = max(maximum_difference, abs(actual-expected))
            if not close(actual, expected):
                mismatched.append("ci")
        if mismatched:
            per_cell[str(cell)]["mismatches"] += 1
            issues.append({"cell": cell, "repetition": repetition,
                           "code": "numerical_mismatch", "fields": sorted(set(mismatched)),
                           "recorded": {k: row[k] for k in ("estimate", "variance", "ci", "p_value")},
                           "independent": other})
        audit["mismatched_fields"] = sorted(set(mismatched))
        audit["absolute_differences"] = {
            label: abs(row[label]-other[label]) if type(row[label]) in (int, float) else None
            for label in ("estimate", "variance", "p_value")}
        audit["absolute_differences"]["ci"] = [
            abs(a-b) if type(a) in (int, float) else None
            for a, b in zip(row["ci"], other["ci"], strict=True)]
        case_audit.append(audit)
        paired += 1
        per_cell[str(cell)]["paired"] += 1
    if len(seen) != plan["total_fits"]:
        raise ValueError("planned case identities missing")
    if not isinstance(payload.get("summaries"), dict):
        raise ValueError("summary absent")
    audit_summary(records, plan, payload["summaries"])
    for key, counts in per_cell.items():
        n = counts["independent_successful"]
        counts["independent_bias"] = float(np.mean(independent_errors[key])) if n else None
        counts["independent_coverage"] = counts["independent_covered"]/n if n else None
        counts["independent_rejection_rate"] = counts["independent_rejected"]/n if n else None
    return {"status": "passed" if not issues else "needs_review",
            "cases": len(records), "paired": paired,
            "fit_failures": failures, "independent_failures": independent_failures,
            "per_cell": per_cell, "case_audit": case_audit, "issues": issues,
            "maximum_absolute_difference": maximum_difference,
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "checker_revision": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=HERE.parent,
                capture_output=True, text=True, check=False).stdout.strip(),
            "checker_git_dirty": bool(subprocess.run(
                ["git", "status", "--porcelain"], cwd=HERE.parent,
                capture_output=True, text=True, check=False).stdout.strip()),
            "checker_python": __import__("platform").python_version(),
            "checker_numpy": np.__version__, "checker_scipy": scipy.__version__,
            "elapsed_seconds": time.monotonic()-started}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--partial", action="store_true",
                        help="audit an interrupted .partial.jsonl journal")
    args = parser.parse_args()
    report = check_partial(args.result) if args.partial else check(args.result)
    if args.output:
        if args.output.exists() or args.output.is_symlink():
            parser.error("output already exists")
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return 0 if report["status"] in ("passed", "partial") else 1


if __name__ == "__main__":
    raise SystemExit(main())
