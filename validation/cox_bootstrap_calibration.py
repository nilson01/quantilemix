# SPDX-License-Identifier: GPL-2.0-or-later
"""Batched known-truth marginal Cox bootstrap study; prints its plan by default."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import warnings
from importlib import metadata

import numpy as np
import scipy

from quantilemix import fit_cox_bootstrap
import quantilemix._cox as cox_implementation
import quantilemix._cox_bootstrap as bootstrap_implementation
import quantilemix._core as core_implementation
import quantilemix as package_implementation


PROTOCOL = "cox-bootstrap-known-truth-v2"
RUNS_ADMITTED = {"pilot": True, "study": True}
MASTER_SEED = 20261024
N = 300
B = 200
MCSIZE = 10000
BREAKS = ((-.5, .5, 1.5, 2.5, 3.5),) * 4
BASELINE_RATE = 0.08
ADMIN_END = 12.0
CELLS = tuple({"cell": 2*i+j, "censor_rate": c, "truth": b}
              for i, c in enumerate((0.10, 0.30)) for j, b in enumerate((0.0, 0.4)))


def design(mode: str) -> dict:
    if mode not in ("pilot", "study"):
        raise ValueError("mode must be pilot or study")
    repetitions = 5 if mode == "pilot" else 2000
    return {"protocol": PROTOCOL, "mode": mode, "n": N, "B": B,
            "MCsize": MCSIZE, "degree": 1, "q": 4, "breaks": BREAKS, "alpha": 0.05,
            "baseline_event_rate": BASELINE_RATE, "administrative_end": ADMIN_END,
            "cells": CELLS, "repetitions_per_cell": repetitions,
            "total_datasets": len(CELLS)*repetitions,
            "total_bootstrap_fits": len(CELLS)*repetitions*B,
            "master_seed": MASTER_SEED,
            "run_admitted": RUNS_ADMITTED[mode]}


def _check_case(mode: str, cell: int, repetition: int) -> dict:
    plan = design(mode)
    if type(cell) is not int or not 0 <= cell < len(CELLS):
        raise ValueError("cell outside plan")
    if type(repetition) is not int or not 0 <= repetition < plan["repetitions_per_cell"]:
        raise ValueError("repetition outside plan")
    return CELLS[cell]


def generate(mode: str, cell: int, repetition: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = _check_case(mode, cell, repetition)
    stream = 301 if mode == "pilot" else 302
    rng = np.random.default_rng(np.random.SeedSequence(
        [MASTER_SEED, stream, cell, repetition]))
    x = rng.integers(0, 4, size=(N, 4), dtype=np.int64)
    hazard = BASELINE_RATE*np.exp(spec["truth"]*(x.sum(axis=1)-6)/4)
    latent = rng.exponential(scale=1/hazard)
    censor = rng.exponential(scale=1/spec["censor_rate"], size=N)
    duration = np.minimum(np.minimum(latent, censor), ADMIN_END)
    event = ((latent <= censor) & (latent <= ADMIN_END)).astype(float)
    return x, duration, event


def fit_seed(mode: str, cell: int, repetition: int) -> int:
    _check_case(mode, cell, repetition)
    return int(np.random.SeedSequence(
        [MASTER_SEED, 303, 1 if mode == "pilot" else 2, cell, repetition]
    ).generate_state(1, dtype=np.uint64)[0])


def input_hash(x: np.ndarray, duration: np.ndarray, event: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x, dtype="<i8").tobytes()
                          + np.asarray(duration, dtype="<f8").tobytes()
                          + np.asarray(event, dtype="<f8").tobytes()).hexdigest()


def _check_committed_file(root: Path, commit: str, rel: str, path: Path) -> None:
    saved = subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=root,
                           capture_output=True, check=False)
    if saved.returncode or saved.stdout != path.read_bytes():
        raise ValueError(f"installed wheel or source differs from commit: {rel}")


def provenance(mode: str, *, enforce_clean: bool = True) -> dict:
    design(mode)
    root = Path(__file__).resolve().parents[1]
    distribution = metadata.distribution("quantilemix")
    record = distribution.read_text("RECORD")
    direct_url = distribution.read_text("direct_url.json")
    paths = {"protocol": Path(__file__).with_name("COX_BOOTSTRAP_PROTOCOL.md"),
             "runner": Path(__file__),
             "checker": Path(__file__).with_name("check_cox_bootstrap_calibration.py"),
             "pilot_auditor": Path(__file__).with_name("server_cox_bootstrap") / "audit_pilot.py",
             "cox_estimator": Path(cox_implementation.__file__),
             "bootstrap_estimator": Path(bootstrap_implementation.__file__),
             "core": Path(core_implementation.__file__),
             "package": Path(package_implementation.__file__)}
    rels = {"protocol": "validation/COX_BOOTSTRAP_PROTOCOL.md",
            "runner": "validation/cox_bootstrap_calibration.py",
            "checker": "validation/check_cox_bootstrap_calibration.py",
            "pilot_auditor": "validation/server_cox_bootstrap/audit_pilot.py",
            "cox_estimator": "src/quantilemix/_cox.py",
            "bootstrap_estimator": "src/quantilemix/_cox_bootstrap.py",
            "core": "src/quantilemix/_core.py",
            "package": "src/quantilemix/__init__.py"}
    if mode == "study":
        paths["study_auditor"] = Path(__file__).with_name("server_cox_bootstrap") / "audit_study.py"
        paths["study_job"] = Path(__file__).with_name("server_cox_bootstrap") / "study.sbatch"
        rels["study_auditor"] = "validation/server_cox_bootstrap/audit_study.py"
        rels["study_job"] = "validation/server_cox_bootstrap/study.sbatch"
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                            capture_output=True, text=True, check=False)
    state = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                           capture_output=True, text=True, check=False)
    identity = {"git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
                "git_dirty": bool(state.stdout.strip()) if state.returncode == 0 else None,
                "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                  for name, path in paths.items()},
                "python": platform.python_version(), "platform": platform.platform(),
                "python_executable": sys.executable,
                "qgcomp_distribution_version": metadata.version("quantilemix"),
                "distribution_record_sha256": (
                    hashlib.sha256(record.encode()).hexdigest() if record else None),
                "distribution_direct_url": direct_url,
                "imported_paths": {name: str(path.resolve()) for name, path in paths.items()
                                   if name in ("cox_estimator", "bootstrap_estimator",
                                               "core", "package")},
                "thread_settings": {name: os.environ.get(name) for name in
                                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                     "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
                "numpy": np.__version__, "scipy": scipy.__version__,
                "numpy_build": np.__config__.show(mode="dicts")}
    if enforce_clean and mode in ("pilot", "study"):
        if identity["git_dirty"] is not False or not identity["git_commit"]:
            raise ValueError(f"{mode} requires a clean committed checkout")
        if not record or (direct_url and
                          json.loads(direct_url).get("dir_info", {}).get("editable")):
            raise ValueError(f"{mode} requires a clean installed wheel")
        installed_root = Path(distribution.locate_file("quantilemix")).resolve()
        for name, rel in rels.items():
            if name in ("cox_estimator", "bootstrap_estimator", "core", "package"):
                if not paths[name].resolve().is_relative_to(installed_root):
                    raise ValueError(f"{mode} imported module outside installed wheel: {name}")
            _check_committed_file(root, identity["git_commit"], rel, paths[name])
    return identity


def _fit_one(x: np.ndarray, duration: np.ndarray, event: np.ndarray, seed: int) -> dict:
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            fit = fit_cox_bootstrap(duration, event, x, q=4, breaks=BREAKS, degree=1, B=B,
                                    MCsize=MCSIZE, alpha=.05, seed=seed)
            result = {"status": "ok", "estimate": float(fit.psi[0]),
                      "se": float(fit.se[0]), "ci": fit.ci[0].tolist(),
                      "p_value": float(fit.p_value[0]),
                      "intervention_levels": fit.intervention_levels.tolist(),
                      "bootstrap_effects": fit.bootstrap_effects[:, 0].tolist(),
                      "failure_class": None, "failure_detail": None,
                      "original_estimate": None, "failed_draw": None,
                      "completed_draws": B}
        except Exception as exc:
            partial = isinstance(exc, bootstrap_implementation.CoxBootstrapFitError)
            result = {"status": "failed", "estimate": None, "se": None,
                      "ci": None, "p_value": None,
                      "bootstrap_effects": [row[0] for row in exc.effects] if partial else None,
                      "intervention_levels": list(exc.levels) if partial else None,
                      "original_estimate": exc.estimate[0] if partial else None,
                      "failed_draw": exc.draw if partial else None,
                      "completed_draws": len(exc.effects) if partial else None,
                      "failure_class": type(exc).__name__,
                      "failure_detail": str(exc)[:300]}
        result["warnings"] = [{"category": type(w.message).__name__,
                               "message": str(w.message)[:300]} for w in caught]
    result["elapsed_seconds"] = time.perf_counter()-started
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as exc:
        result = {"status": "failed", "estimate": None, "se": None, "ci": None,
                  "p_value": None, "bootstrap_effects": None,
                  "intervention_levels": None,
                  "original_estimate": None, "failed_draw": None,
                  "completed_draws": None,
                  "failure_class": "NonfiniteOrUnserializableFit",
                  "failure_detail": str(exc)[:300], "warnings": result["warnings"],
                  "elapsed_seconds": time.perf_counter()-started}
    return result


def run_batch(mode: str, cell: int, start: int, count: int, output: Path) -> dict:
    plan = design(mode)
    if not plan["run_admitted"]:
        raise ValueError(f"{mode} run requires protocol and compute admission")
    if (type(cell) is not int or not 0 <= cell < len(CELLS)
            or type(start) is not int or type(count) is not int or count < 1
            or start < 0 or start+count > plan["repetitions_per_cell"]):
        raise ValueError("batch must be a nonempty in-range cell and repetition interval")
    partial = output.with_name(output.name+".partial.jsonl")
    if (output.exists() or output.is_symlink() or partial.exists() or partial.is_symlink()
            or not output.parent.is_dir()):
        raise ValueError("output and partial journal must be new paths in an existing directory")
    identity = provenance(mode)
    header = {"plan": plan, "source_identity": identity,
              "batch": {"cell": cell, "start": start, "count": count},
              "invocation": list(sys.argv)}
    records = []
    started = time.perf_counter()
    with partial.open("x") as journal:
        def append(item: dict) -> None:
            journal.write(json.dumps(item, allow_nan=False)+"\n")
            journal.flush()
            os.fsync(journal.fileno())

        append({"kind": "header", "header": header})
        try:
            for repetition in range(start, start+count):
                append({"kind": "start", "cell": cell, "repetition": repetition})
                x, duration, event = generate(mode, cell, repetition)
                seed = fit_seed(mode, cell, repetition)
                row = {"cell": cell, "repetition": repetition,
                       "truth": CELLS[cell]["truth"], "fit_seed": seed,
                       "input_sha256": input_hash(x, duration, event),
                       "observed_events": int(event.sum()),
                       "administratively_censored": int(np.sum(
                           (duration == ADMIN_END) & (event == 0))),
                       "fit": _fit_one(x, duration, event, seed)}
                append({"kind": "record", "row": row})
                records.append(row)
        except BaseException as exc:
            append({"kind": "terminal", "status": "interrupted",
                    "failure_class": type(exc).__name__,
                    "failure_detail": str(exc)[:300]})
            raise
    report = {**header, "records": records, "elapsed_seconds": time.perf_counter()-started}
    with output.open("x") as target:
        target.write(json.dumps(report, indent=2, allow_nan=False)+"\n")
        target.flush()
        os.fsync(target.fileno())
    partial.unlink()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "study"))
    parser.add_argument("--cell", type=int)
    parser.add_argument("--start", type=int)
    parser.add_argument("--count", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if all(value is None for value in vars(args).values()):
        print(json.dumps({name: design(name) for name in ("pilot", "study")}, indent=2))
        return 0
    if any(value is None for value in vars(args).values()):
        parser.error("--mode, --cell, --start, --count and --output are required together")
    result = run_batch(args.mode, args.cell, args.start, args.count, args.output)
    print(f"saved {len(result['records'])} case records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
