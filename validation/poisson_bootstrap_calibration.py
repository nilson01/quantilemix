# SPDX-License-Identifier: GPL-2.0-or-later
"""Known-score marginal Poisson row-bootstrap study; default prints its plan."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
import warnings

import numpy as np
import scipy
from scipy.stats import norm

from quantilemix import fit_poisson, fit_poisson_bootstrap
import quantilemix._poisson as poisson_implementation
import quantilemix._poisson_bootstrap as bootstrap_implementation

PROTOCOL = "poisson-bootstrap-known-scores-v1"
MASTER_SEED = 20261023
CELLS = tuple({"cell": i * 2 + j, "center_mean": m, "truth": b}
              for i, m in enumerate((0.20, 2.00))
              for j, b in enumerate((0.0, 0.40)))
B = 200
N = 500


def design(mode: str) -> dict:
    if mode not in ("pilot", "study"):
        raise ValueError("mode must be pilot or study")
    repetitions = 5 if mode == "pilot" else 1000
    return {"protocol": PROTOCOL, "mode": mode, "n": N, "B": B,
            "cells": CELLS, "repetitions_per_cell": repetitions,
            "total_datasets": len(CELLS)*repetitions,
            "total_bootstrap_fits": len(CELLS)*repetitions*B,
            "master_seed": MASTER_SEED}


def generate(mode: str, cell: int, repetition: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    plan = design(mode)
    if type(cell) is not int or not 0 <= cell < len(CELLS):
        raise ValueError("cell outside plan")
    if type(repetition) is not int or not 0 <= repetition < plan["repetitions_per_cell"]:
        raise ValueError("repetition outside plan")
    spec = CELLS[cell]
    rng = np.random.default_rng(np.random.SeedSequence(
        [MASTER_SEED, 91 if mode == "pilot" else 92, cell, repetition]))
    x = rng.integers(0, 4, size=(N, 4), dtype=np.int64)
    z = rng.integers(0, 2, size=(N, 1), dtype=np.int64)
    eta = math.log(spec["center_mean"]) + spec["truth"] * (x.sum(axis=1)-6)/4 + 0.4*z[:, 0]
    y = rng.poisson(np.exp(eta)).astype(np.int64)
    return x, z, y


def bootstrap_seed(mode: str, cell: int, repetition: int) -> int:
    design(mode)
    return int(np.random.SeedSequence(
        [MASTER_SEED, 93, 1 if mode == "pilot" else 2, cell, repetition]
    ).generate_state(1, dtype=np.uint64)[0])


def input_hash(x: np.ndarray, z: np.ndarray, y: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x, dtype="<i8").tobytes() +
                          np.asarray(z, dtype="<i8").tobytes() +
                          np.asarray(y, dtype="<i8").tobytes()).hexdigest()


def provenance(mode: str) -> dict:
    root = Path(__file__).resolve().parents[1]
    commit_result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                   capture_output=True, text=True, check=False)
    status_result = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                                   capture_output=True, text=True, check=False)
    commit = commit_result.stdout.strip() if commit_result.returncode == 0 else None
    dirty = bool(status_result.stdout.strip()) if status_result.returncode == 0 else None
    paths = {"runner": Path(__file__),
             "checker": Path(__file__).with_name("check_poisson_bootstrap_calibration.py"),
             "poisson_estimator": Path(poisson_implementation.__file__),
             "bootstrap_estimator": Path(bootstrap_implementation.__file__)}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
              for name, path in paths.items()}
    if mode == "study":
        if not commit or len(commit) != 40 or dirty is not False:
            raise ValueError("study requires a clean committed checkout")
        rels = {"runner": "validation/poisson_bootstrap_calibration.py",
                "checker": "validation/check_poisson_bootstrap_calibration.py",
                "poisson_estimator": "src/quantilemix/_poisson.py",
                "bootstrap_estimator": "src/quantilemix/_poisson_bootstrap.py"}
        for name, rel in rels.items():
            committed = subprocess.run(["git", "show", f"{commit}:{rel}"],
                                       cwd=root, capture_output=True, check=False)
            if committed.returncode or committed.stdout != paths[name].read_bytes():
                raise ValueError(f"study source differs from commit: {name}")
    return {"git_commit": commit, "git_dirty": dirty, "source_sha256": hashes,
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "scipy": scipy.__version__}


def wilson(yes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = float(norm.ppf(.975))
    p = yes/total
    den = 1+z*z/total
    mid = (p+z*z/(2*total))/den
    half = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [mid-half, mid+half]


def summarize(records: list[dict], plan: dict) -> dict:
    result = {}
    for spec in CELLS:
        rows = [r for r in records if r["cell"] == spec["cell"]]
        cell = {"planned": plan["repetitions_per_cell"], "attempted": len(rows),
                "missing": plan["repetitions_per_cell"]-len(rows),
                "observed_mean": sum(r["count_total"] for r in rows)/(N*len(rows)) if rows else None,
                "zero_fraction": sum(r["zero_count"] for r in rows)/(N*len(rows)) if rows else None}
        for name in ("bootstrap", "model_based"):
            good = [r[name] for r in rows if r[name]["status"] == "ok"]
            m = len(good)
            covered = sum(v["ci"][0] <= spec["truth"] <= v["ci"][1] for v in good)
            rejected = sum(v["p_value"] < .05 for v in good)
            part = {"successful": m, "failed": len(rows)-m,
                    "warnings": sum(bool(r[name]["warnings"]) for r in rows),
                    "covered": covered, "rejected": rejected,
                    "coverage_all_planned_bounds": [covered/plan["repetitions_per_cell"],
                        (covered+plan["repetitions_per_cell"]-m)/plan["repetitions_per_cell"]]}
            if m:
                errors = np.array([v["estimate"]-spec["truth"] for v in good])
                variances = np.array([v["se"]**2 for v in good])
                part.update({"bias": float(errors.mean()),
                             "bias_mcse": float(errors.std(ddof=1)/math.sqrt(m)) if m > 1 else None,
                             "empirical_sd": float(errors.std(ddof=1)) if m > 1 else None,
                             "root_mean_variance": float(np.sqrt(variances.mean())),
                             "coverage": covered/m, "coverage_wilson_95": wilson(covered,m),
                             "rejection_rate": rejected/m,
                             "rejection_wilson_95": wilson(rejected,m)})
            cell[name] = part
        result[str(spec["cell"])] = cell
    return result


def _capture(operation, *, bootstrap: bool) -> dict:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            fit = operation()
            result = {"status": "ok", "estimate": float(fit.psi),
                      "se": float(fit.se), "ci": list(map(float, fit.ci)),
                      "p_value": float(fit.p_value), "failure_class": None,
                      "failure_detail": None}
            if bootstrap:
                result["bootstrap_effects"] = list(map(float, fit.bootstrap_effects))
                result["intervention_means"] = list(map(float, fit.intervention_means))
        except Exception as exc:
            result = {"status": "failed", "estimate": None, "se": None,
                      "ci": None, "p_value": None, "failure_class": type(exc).__name__,
                      "failure_detail": str(exc)[:200]}
            if bootstrap:
                result["bootstrap_effects"] = None
                result["intervention_means"] = None
        result["warnings"] = [{"category": type(w.message).__name__,
                               "message": str(w.message)[:300]} for w in caught]
    return result


def run(mode: str, output: Path) -> dict:
    plan = design(mode)
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ValueError("output must be a new path in an existing directory")
    journal_path = output.with_name(output.name + ".partial.jsonl")
    if journal_path.exists() or journal_path.is_symlink():
        raise ValueError("partial journal already exists")
    identity = provenance(mode)
    records = []
    started = time.monotonic()
    with journal_path.open("x") as journal:
        journal.write(json.dumps({"plan": plan, "source_identity": identity}) + "\n")
        journal.flush()
        os.fsync(journal.fileno())
        for spec in CELLS:
            for repetition in range(plan["repetitions_per_cell"]):
                x, z, y = generate(mode, spec["cell"], repetition)
                seed = bootstrap_seed(mode, spec["cell"], repetition)
                row = {"cell": spec["cell"], "repetition": repetition,
                       "input_sha256": input_hash(x,z,y), "bootstrap_seed": seed,
                       "count_total": int(y.sum()), "zero_count": int(np.count_nonzero(y == 0))}
                row["bootstrap"] = _capture(
                    lambda: fit_poisson_bootstrap(y,x,z,q=None,B=B,seed=seed), bootstrap=True)
                row["model_based"] = _capture(
                    lambda: fit_poisson(y,x,z,q=None), bootstrap=False)
                records.append(row)
                journal.write(json.dumps(row, allow_nan=False) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
    result = {"plan": plan, "source_identity": identity, "records": records,
              "summaries": summarize(records,plan),
              "elapsed_seconds": time.monotonic()-started}
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    journal_path.unlink()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot","study"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode is None and args.output is None:
        print(json.dumps(design("study"),indent=2))
        return 0
    if args.mode is None or args.output is None:
        parser.error("--mode and --output are required together")
    result = run(args.mode,args.output)
    print(f"{len(result['records'])} attempted; {result['elapsed_seconds']:.3f} seconds")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
