# SPDX-License-Identifier: GPL-2.0-or-later
"""Bounded Gaussian scenario-5 translation; the default command only prints a plan."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
from importlib.metadata import version
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
from quantilemix import fit
import quantilemix._core as core


PROTOCOL = "keil-2020-scenario5-qgcomp-v1"
MASTER_SEED = 20261023
VARIANTS = {"released_ncor2": 2, "two_causal_ncor1": 1}
BETA2 = (-0.20, -0.10, -0.05)
RHO = (0.0, 0.4, 0.75)
CODE_ZIP_SHA256 = "ba2a8a858a31f24593d38d94dd204db2a18a214b4aa61645cf7748f988f54ccd"


def cells() -> list[dict]:
    return [{"cell": i * len(RHO) + j, "beta2": beta2, "rho": rho,
             "truth": 0.25 + beta2}
            for i, beta2 in enumerate(BETA2) for j, rho in enumerate(RHO)]


def design(mode: str) -> dict:
    if mode not in {"pilot", "study"}:
        raise ValueError("mode must be pilot or study")
    repetitions = 5 if mode == "pilot" else 1000
    return {"protocol": PROTOCOL, "mode": mode, "n": 500, "exposures": 9,
            "variants": VARIANTS, "cells": cells(),
            "repetitions_per_cell": repetitions,
            "total_fits": repetitions * len(VARIANTS) * len(cells()),
            "seed": MASTER_SEED, "code_zip_sha256": CODE_ZIP_SHA256}


def generate(variant: str, cell: int, repetition: int, *, mode: str = "pilot") -> tuple[np.ndarray, np.ndarray]:
    """Translate the supplement's balanced shared-master score generator."""
    plan = design(mode)
    if variant not in VARIANTS or type(cell) is not int or not 0 <= cell < len(plan["cells"]):
        raise ValueError("unknown variant or cell")
    if type(repetition) is not int or not 0 <= repetition < plan["repetitions_per_cell"]:
        raise ValueError("repetition outside plan")
    variant_number = list(VARIANTS).index(variant)
    rng = np.random.default_rng(np.random.SeedSequence(
        [MASTER_SEED, 51 if mode == "pilot" else 52, variant_number, cell, repetition]))
    specification = plan["cells"][cell]
    master = rng.permutation(np.tile(np.arange(4, dtype=np.int64), plan["n"] // 4))
    x = np.empty((plan["n"], plan["exposures"]), dtype=np.int64)
    for k in range(plan["exposures"]):
        if k <= VARIANTS[variant]:
            mask = rng.binomial(1, math.sqrt(specification["rho"]), plan["n"]).astype(bool)
            column = np.empty(plan["n"], dtype=np.int64)
            column[mask] = master[mask]
            column[~mask] = rng.permutation(master[~mask])
            x[:, k] = column
        else:
            x[:, k] = rng.permutation(master)
    coefficients = np.zeros(plan["exposures"])
    coefficients[:2] = [0.25, specification["beta2"]]
    y = x @ coefficients + rng.standard_normal(plan["n"])
    return x, y


def input_hash(x: np.ndarray, y: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x, dtype="<i8").tobytes() +
                          np.asarray(y, dtype="<f8").tobytes()).hexdigest()


def provenance(mode: str) -> dict:
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                              capture_output=True, text=True, check=False)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                            capture_output=True, text=True, check=False)
    commit = revision.stdout.strip() if revision.returncode == 0 else None
    dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    source_revision = commit or os.environ.get("QGCOMP_SOURCE_REVISION")
    if mode == "study" and (not commit or len(commit) != 40 or dirty is not False):
        raise ValueError("full study requires a clean, committed Git checkout")
    if mode == "study":
        for path in (Path(__file__).resolve(), Path(core.__file__).resolve()):
            try:
                relative = path.relative_to(root)
            except ValueError as exc:
                raise ValueError("full study must import committed checkout sources") from exc
            committed = subprocess.run(["git", "show", f"{commit}:{relative.as_posix()}"],
                                       cwd=root, capture_output=True, check=False)
            if committed.returncode or committed.stdout != path.read_bytes():
                raise ValueError(f"full study source differs from commit: {relative}")
    return {"git_commit": commit, "git_dirty": dirty,
            "source_revision": source_revision,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "core_sha256": hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest()}


def correlations(x: np.ndarray) -> list[float]:
    return [float(np.corrcoef(x[:, a], x[:, b])[0, 1])
            for a, b in ((0, 1), (0, 2), (1, 2))]


def summarize(records: list[dict], plan: dict) -> dict:
    summaries = {}
    for variant in VARIANTS:
        for specification in plan["cells"]:
            cell = specification["cell"]
            rows = [r for r in records if r["variant"] == variant and r["cell"] == cell]
            good = [r for r in rows if r["failure"] is None]
            key = f"{variant}:cell{cell}"
            corr = np.asarray([r["correlations"] for r in rows])
            summary = {"planned": len(rows), "successful": len(good),
                       "failed": len(rows) - len(good),
                       "warnings": sum(bool(r["warnings"]) for r in rows),
                       "mean_correlations": corr.mean(axis=0).tolist(),
                       "sd_correlations": corr.std(axis=0, ddof=1).tolist() if len(rows) > 1 else None}
            if good:
                deviations = np.array([r["estimate"] - r["truth"] for r in good])
                variance = np.array([r["variance"] for r in good])
                covered = sum(r["ci"][0] <= r["truth"] <= r["ci"][1] for r in good)
                rejected = sum(r["p_value"] < .05 for r in good)
                summary.update({"bias": float(deviations.mean()),
                                "bias_mcse": float(deviations.std(ddof=1) / math.sqrt(len(good)))
                                if len(good) > 1 else None,
                                "empirical_sd": float(deviations.std(ddof=1)) if len(good) > 1 else None,
                                "root_mean_variance": float(np.sqrt(variance.mean())),
                                "bias_quartiles": np.quantile(deviations, [.25, .5, .75]).tolist(),
                                "coverage": covered / len(good),
                                "rejection_rate": rejected / len(good),
                                "coverage_all_planned_bounds": [covered / len(rows),
                                    (covered + len(rows) - len(good)) / len(rows)],
                                "rejection_all_planned_bounds": [rejected / len(rows),
                                    (rejected + len(rows) - len(good)) / len(rows)]})
            else:
                summary.update({"coverage_all_planned_bounds": [0., 1.],
                                "rejection_all_planned_bounds": [0., 1.]})
            summaries[key] = summary
    return summaries


def run(mode: str, *, journal: Path | None = None) -> dict:
    plan = design(mode)
    source = provenance(mode)
    records = []
    started = time.monotonic()
    if journal is not None and (journal.exists() or journal.is_symlink()):
        raise ValueError("journal exists; use a fresh output path")
    with (journal.open("x", encoding="utf-8") if journal is not None else nullcontext()) as stream:
        for variant in VARIANTS:
            for specification in plan["cells"]:
                for repetition in range(plan["repetitions_per_cell"]):
                    cell = specification["cell"]
                    x, y = generate(variant, cell, repetition, mode=mode)
                    record = {"variant": variant, "cell": cell, "repetition": repetition,
                              "input_sha256": input_hash(x, y), "truth": specification["truth"],
                              "correlations": correlations(x), "warnings": []}
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        try:
                            fitted = fit(y, x, q=None)
                            if not all(math.isfinite(v) for v in
                                       (fitted.psi, fitted.var_psi, fitted.p_value, *fitted.ci)):
                                raise ValueError("nonfinite fit result")
                            record.update({"estimate": fitted.psi, "variance": fitted.var_psi,
                                           "ci": list(fitted.ci), "p_value": fitted.p_value,
                                           "failure": None})
                        except (ValueError, np.linalg.LinAlgError) as exc:
                            record.update({"estimate": None, "variance": None, "ci": None,
                                           "p_value": None, "failure": f"{type(exc).__name__}: {exc}"})
                    record["warnings"] = [f"{w.category.__name__}: {w.message}" for w in caught]
                    records.append(record)
                    if stream is not None:
                        stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
    return {"plan": plan, "source_identity": source,
            "journal_sha256": hashlib.sha256(journal.read_bytes()).hexdigest() if journal is not None else None,
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "scipy": scipy.__version__, "qgcomp_python": version("quantilemix"),
                            "platform": platform.platform()},
            "elapsed_seconds": time.monotonic() - started,
            "summaries": summarize(records, plan), "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "pilot", "study"), nargs="?", default="plan")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "plan":
        print(json.dumps({"pilot": design("pilot"), "study": design("study")}, indent=2))
        return
    if args.output is None or args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir():
        parser.error("pilot and study require a fresh --output in an existing directory")
    journal = args.output.with_suffix(args.output.suffix + ".jsonl")
    if journal.exists() or journal.is_symlink():
        parser.error("journal exists; choose a fresh output path")
    result = run(args.mode, journal=journal)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "elapsed_seconds": result["elapsed_seconds"],
                      "summaries": result["summaries"]}, indent=2))


if __name__ == "__main__":
    main()
