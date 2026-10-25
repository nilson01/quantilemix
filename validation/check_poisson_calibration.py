# SPDX-License-Identifier: GPL-2.0-or-later
"""Recompute saved Poisson-log cases and summaries independently."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
SPEC = importlib.util.spec_from_file_location("poisson_calibration", HERE / "poisson_calibration.py")
calibration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(calibration)


def independent_fit(x: np.ndarray, y: np.ndarray) -> dict:
    design = np.column_stack((np.ones(len(y)), x.astype(float)))

    def equations(beta: np.ndarray) -> np.ndarray:
        return design.T @ (y - np.exp(design @ beta))

    def derivative(beta: np.ndarray) -> np.ndarray:
        mu = np.exp(design @ beta)
        return -(design.T @ (mu[:, None] * design))

    observed = float(y.mean())
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


def check(path: Path) -> dict:
    started = time.monotonic()
    payload = json.loads(path.read_text())
    plan = payload.get("plan")
    if not isinstance(plan, dict) or plan != calibration.design(plan.get("mode")):
        # JSON changes tuples to lists; normalize the fixed design once.
        expected = calibration.design(plan.get("mode")) if isinstance(plan, dict) else None
        if expected is None or json.dumps(plan, sort_keys=True) != json.dumps(expected, sort_keys=True):
            raise ValueError("plan differs from fixed design")
    source = payload.get("source_identity")
    if not isinstance(source, dict):
        raise ValueError("source identity absent")
    for label, source_path in (("runner_sha256", HERE / "poisson_calibration.py"),
                               ("implementation_sha256", Path(implementation.__file__))):
        if source.get(label) != hashlib.sha256(source_path.read_bytes()).hexdigest():
            raise ValueError(f"{label} differs")
    if plan["mode"] == "study":
        commit = source.get("git_commit")
        if (not isinstance(commit, str) or len(commit) != 40
                or source.get("git_dirty") is not False):
            raise ValueError("study lacks clean committed provenance")
        root = HERE.parent
        for source_path, rel in ((HERE / "poisson_calibration.py", Path("validation/poisson_calibration.py")),
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
                for spec in calibration.CELLS}
    independent_errors = {str(spec["cell"]): [] for spec in calibration.CELLS}
    for row in records:
        cell, repetition = row.get("cell"), row.get("repetition")
        if (type(cell) is not int or type(repetition) is not int
                or not 0 <= cell < len(calibration.CELLS)
                or not 0 <= repetition < plan["repetitions_per_cell"]
                or (cell, repetition) in seen):
            raise ValueError("case identity invalid or repeated")
        seen.add((cell, repetition))
        x, y = calibration.generate(plan["mode"], cell, repetition)
        if row.get("input_sha256") != calibration.input_hash(x, y):
            raise ValueError("case input differs")
        if row.get("count_total") != int(y.sum()) or row.get("zero_count") != int(np.count_nonzero(y == 0)):
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
                fit_poisson(y, x, q=None)
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
            other = independent_fit(x, y)
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
        truth = calibration.CELLS[cell]["truth"]
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
    expected_summary = calibration.summarize(records, plan)
    if json.dumps(payload.get("summaries"), sort_keys=True) != json.dumps(expected_summary, sort_keys=True):
        raise ValueError("summary differs from records")
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
    args = parser.parse_args()
    report = check(args.result)
    if args.output:
        if args.output.exists() or args.output.is_symlink():
            parser.error("output already exists")
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
