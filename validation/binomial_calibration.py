# SPDX-License-Identifier: GPL-2.0-or-later
"""Prospective known-score Bernoulli-logit validation; default prints the plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time
import warnings

import numpy as np
import scipy
from scipy import special
from scipy.stats import norm

from quantilemix import fit_binomial
import quantilemix._binomial as implementation


PROTOCOL = "binomial-known-scores-v1"
MASTER_SEED = 20261023
CELLS = tuple({"cell": i * 2 + j, "center_event_probability": p,
               "truth": b} for i, p in enumerate((0.10, 0.50))
              for j, b in enumerate((0.0, 0.40)))


def design(mode: str) -> dict:
    if mode not in ("pilot", "study"):
        raise ValueError("mode must be pilot or study")
    repetitions = 5 if mode == "pilot" else 1000
    return {"protocol": PROTOCOL, "mode": mode, "n": 500,
            "cells": CELLS, "repetitions_per_cell": repetitions,
            "total_fits": len(CELLS) * repetitions, "seed": MASTER_SEED}


def generate(mode: str, cell: int, repetition: int) -> tuple[np.ndarray, np.ndarray]:
    plan = design(mode)
    if type(cell) is not int or not 0 <= cell < len(CELLS):
        raise ValueError("cell outside plan")
    if type(repetition) is not int or not 0 <= repetition < plan["repetitions_per_cell"]:
        raise ValueError("repetition outside plan")
    spec = CELLS[cell]
    rng = np.random.default_rng(np.random.SeedSequence(
        [MASTER_SEED, 71 if mode == "pilot" else 72, cell, repetition]))
    x = rng.integers(0, 4, size=(plan["n"], 4), dtype=np.int64)
    eta = special.logit(spec["center_event_probability"]) + (
        spec["truth"] / 4.0) * (x.sum(axis=1) - 6)
    y = (rng.random(plan["n"]) < special.expit(eta)).astype(np.int64)
    return x, y


def input_hash(x: np.ndarray, y: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x, dtype="<i8").tobytes() +
                          np.asarray(y, dtype="<i8").tobytes()).hexdigest()


def provenance(mode: str) -> dict:
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                              capture_output=True, text=True, check=False)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                            capture_output=True, text=True, check=False)
    commit = revision.stdout.strip() if revision.returncode == 0 else None
    dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    if mode == "study" and (not commit or len(commit) != 40 or dirty is not False):
        raise ValueError("study requires a clean committed checkout")
    if mode == "study":
        for path in (Path(__file__).resolve(), Path(implementation.__file__).resolve()):
            rel = path.relative_to(root)
            committed = subprocess.run(["git", "show", f"{commit}:{rel.as_posix()}"],
                                       cwd=root, capture_output=True, check=False)
            if committed.returncode or committed.stdout != path.read_bytes():
                raise ValueError(f"study source differs from commit: {rel}")
    return {"git_commit": commit, "git_dirty": dirty,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "implementation_sha256": hashlib.sha256(Path(implementation.__file__).read_bytes()).hexdigest(),
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "scipy": scipy.__version__}


def wilson(yes: int, total: int) -> list[float]:
    z = float(norm.ppf(.975))
    p = yes / total
    den = 1 + z*z/total
    center = (p + z*z/(2*total))/den
    half = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/den
    return [center-half, center+half]


def summarize(records: list[dict], plan: dict) -> dict:
    result = {}
    for spec in CELLS:
        rows = [r for r in records if r["cell"] == spec["cell"]]
        good = [r for r in rows if r["failure"] is None]
        m = len(good)
        covered = sum(r["ci"][0] <= spec["truth"] <= r["ci"][1] for r in good)
        rejected = sum(r["p_value"] < .05 for r in good)
        summary = {"planned": plan["repetitions_per_cell"], "attempted": len(rows),
                   "successful": m, "failed": len(rows)-m,
                   "missing": plan["repetitions_per_cell"]-len(rows),
                   "event_fraction": sum(r["events"] for r in rows)/(plan["n"]*len(rows)),
                   "warnings": sum(bool(r["warnings"]) for r in rows),
                   "covered": covered, "rejected": rejected,
                   "coverage_all_planned_bounds": [covered/plan["repetitions_per_cell"],
                       (covered+plan["repetitions_per_cell"]-m)/plan["repetitions_per_cell"]]}
        if m:
            errors = np.array([r["estimate"]-spec["truth"] for r in good])
            variances = np.array([r["variance"] for r in good])
            summary.update({"bias": float(errors.mean()),
                            "bias_mcse": float(errors.std(ddof=1)/math.sqrt(m)) if m > 1 else None,
                            "empirical_sd": float(errors.std(ddof=1)) if m > 1 else None,
                            "root_mean_variance": float(np.sqrt(variances.mean())),
                            "coverage": covered/m, "coverage_wilson_95": wilson(covered, m),
                            "rejection_rate": rejected/m,
                            "rejection_wilson_95": wilson(rejected, m)})
        result[str(spec["cell"])] = summary
    return result


def run(mode: str, output: Path) -> dict:
    plan = design(mode)
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ValueError("output must be a new file in an existing directory")
    identity = provenance(mode)
    records = []
    started = time.monotonic()
    for spec in CELLS:
        for repetition in range(plan["repetitions_per_cell"]):
            x, y = generate(mode, spec["cell"], repetition)
            row = {"cell": spec["cell"], "repetition": repetition,
                   "input_sha256": input_hash(x, y), "events": int(y.sum()),
                   "warnings": []}
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    fit = fit_binomial(y, x, q=None)
                    row.update({"estimate": fit.psi, "variance": fit.var_psi,
                                "ci": list(fit.ci), "p_value": fit.p_value,
                                "failure": None, "failure_detail": None})
                except (ValueError, np.linalg.LinAlgError) as exc:
                    row.update({"estimate": None, "variance": None,
                                "ci": None, "p_value": None,
                                "failure": type(exc).__name__,
                                "failure_detail": str(exc)[:200]})
                row["warnings"] = [type(w.message).__name__ for w in caught]
            records.append(row)
    result = {"plan": plan, "source_identity": identity, "records": records,
              "summaries": summarize(records, plan),
              "elapsed_seconds": time.monotonic()-started}
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "study"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode is None and args.output is None:
        print(json.dumps(design("study"), indent=2))
        return 0
    if args.mode is None or args.output is None:
        parser.error("--mode and --output are required together")
    result = run(args.mode, args.output)
    print(f"{len(result['records'])} attempted; {result['elapsed_seconds']:.3f} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
