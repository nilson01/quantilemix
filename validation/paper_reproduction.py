# SPDX-License-Identifier: GPL-2.0-or-later
"""Reproduce the Gaussian qgcomp arm of Keil et al. (2020), scenarios 1–4.

The paper's R supplement defines balanced, prequantized exposure scores. This
runner translates that data-generating process, not R's random-number stream.
No simulation runs on import. The default command only prints the plan.
"""

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

import numpy as np
import scipy
from quantilemix import fit
import quantilemix._core as core


SCENARIOS = {
    "null": (0.0, 0.0, 0.0, 0.0),
    "cancellation": (0.25, -0.25, 0.0, 0.0),
    "single": (0.25, 0.0, 0.0, 0.0),
    "distributed": (0.0625, 0.0625, 0.0625, 0.0625),
}
MASTER_SEED = 20261023


def design(mode: str) -> dict:
    if mode not in {"pilot", "study"}:
        raise ValueError("mode must be pilot or study")
    repetitions = 5 if mode == "pilot" else 1000
    return {"protocol": "keil-2020-gaussian-scenarios-1-4-v2", "mode": mode,
            "n": 500, "exposures": 4, "repetitions_per_scenario": repetitions,
            "scenarios": list(SCENARIOS), "seed": MASTER_SEED,
            "total_fits": repetitions * len(SCENARIOS),
            "source": "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7228100/supplementaryFiles",
            "supplement_zip_sha256": "c7a936552eaaaf06272b2dbb6a9f5eda34d176ccf07964855a6a70a687eb636d",
            "code_zip_sha256": "ba2a8a858a31f24593d38d94dd204db2a18a214b4aa61645cf7748f988f54ccd"}


def generate(scenario: str, repetition: int, *, mode: str = "pilot") -> tuple[np.ndarray, np.ndarray]:
    """Independent columns are permutations of exactly balanced 0–3 scores."""
    plan = design(mode)
    if scenario not in SCENARIOS or not isinstance(repetition, int) or isinstance(repetition, bool):
        raise ValueError("unknown scenario or repetition")
    if not 0 <= repetition < plan["repetitions_per_scenario"]:
        raise ValueError("repetition outside plan")
    seed = np.random.SeedSequence([MASTER_SEED, 1 if mode == "pilot" else 2,
                                   list(SCENARIOS).index(scenario), repetition])
    rng = np.random.default_rng(seed)
    master = np.tile(np.arange(4), plan["n"] // 4)
    x = np.column_stack([rng.permutation(master) for _ in range(4)])
    y = x @ np.asarray(SCENARIOS[scenario]) + rng.standard_normal(plan["n"])
    return x, y


def input_hash(x: np.ndarray, y: np.ndarray) -> str:
    """Hash the exact little-endian score and outcome arrays in row order."""
    return hashlib.sha256(np.asarray(x, dtype="<i8").tobytes() +
                          np.asarray(y, dtype="<f8").tobytes()).hexdigest()


def provenance(mode: str) -> dict:
    """Record exact fitted source and refuse an unversioned full study."""
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                              capture_output=True, text=True, check=False)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                            capture_output=True, text=True, check=False)
    commit = revision.stdout.strip() if revision.returncode == 0 else None
    dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    source_revision = commit or os.environ.get("QGCOMP_SOURCE_REVISION")
    if mode == "study" and (not source_revision or dirty is True):
        raise ValueError("full study requires a source revision and no dirty checkout")
    return {"git_commit": commit, "git_dirty": dirty,
            "source_revision": source_revision,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "core_sha256": hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest()}


def run(mode: str, *, journal: Path | None = None) -> dict:
    plan = design(mode)
    source = provenance(mode)
    records = []
    started = time.monotonic()
    if journal is not None and (journal.exists() or journal.is_symlink()):
        raise ValueError("journal exists; use a fresh output path")
    with (journal.open("x", encoding="utf-8") if journal is not None else nullcontext()) as stream:
        for scenario, beta in SCENARIOS.items():
            truth = sum(beta)
            for repetition in range(plan["repetitions_per_scenario"]):
                x, y = generate(scenario, repetition, mode=mode)
                try:
                    result = fit(y, x, q=None)
                    if not all(math.isfinite(v) for v in (result.psi, result.se, result.p_value, *result.ci)):
                        raise ValueError("nonfinite fit result")
                    record = {"scenario": scenario, "repetition": repetition,
                              "input_sha256": input_hash(x, y),
                              "estimate": result.psi, "variance": result.var_psi,
                              "ci": list(result.ci), "p_value": result.p_value,
                              "truth": truth, "failure": None}
                except (ValueError, np.linalg.LinAlgError) as exc:
                    record = {"scenario": scenario, "repetition": repetition,
                              "input_sha256": input_hash(x, y),
                              "estimate": None, "variance": None, "ci": None,
                              "p_value": None, "truth": truth,
                              "failure": f"{type(exc).__name__}: {exc}"}
                records.append(record)
                if stream is not None:
                    stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
    summaries = {}
    for scenario in SCENARIOS:
        rows = [r for r in records if r["scenario"] == scenario]
        good = [r for r in rows if r["failure"] is None]
        if good:
            estimates = np.array([r["estimate"] for r in good])
            variances = np.array([r["variance"] for r in good])
            covered = sum(r["ci"][0] <= r["truth"] <= r["ci"][1] for r in good)
            rejected = sum(r["p_value"] < .05 for r in good)
            summaries[scenario] = {"planned": len(rows), "successful": len(good),
                                   "failed": len(rows) - len(good),
                                   "bias": float(estimates.mean() - sum(SCENARIOS[scenario])),
                                   "empirical_sd": float(estimates.std(ddof=1)) if len(good) > 1 else None,
                                   "root_mean_variance": float(np.sqrt(variances.mean())),
                                   "coverage": covered / len(good),
                                   "rejection_rate": rejected / len(good),
                                   "coverage_all_planned_bounds": [covered / len(rows),
                                                                   (covered+len(rows)-len(good)) / len(rows)],
                                   "rejection_all_planned_bounds": [rejected / len(rows),
                                                                    (rejected+len(rows)-len(good)) / len(rows)],
                                   "bias_mcse": float(estimates.std(ddof=1) / np.sqrt(len(good))) if len(good) > 1 else None}
        else:
            summaries[scenario] = {"planned": len(rows), "successful": 0,
                                   "failed": len(rows),
                                   "coverage_all_planned_bounds": [0.0, 1.0],
                                   "rejection_all_planned_bounds": [0.0, 1.0]}
    return {"plan": plan, "source_identity": source,
            "journal_sha256": hashlib.sha256(journal.read_bytes()).hexdigest() if journal is not None else None,
            "environment": {"python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__,
            "qgcomp_python": version("quantilemix"), "platform": platform.platform()},
            "elapsed_seconds": time.monotonic() - started, "summaries": summaries,
            "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "pilot", "study"), nargs="?", default="plan")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "plan":
        print(json.dumps({"pilot": design("pilot"), "study": design("study")}, indent=2))
        return
    if args.output is None:
        parser.error("pilot and study require --output")
    if args.output.exists() or args.output.is_symlink():
        parser.error("output exists; choose a fresh path")
    if not args.output.parent.is_dir():
        parser.error("output parent does not exist")
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
