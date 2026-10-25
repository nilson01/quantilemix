# SPDX-License-Identifier: GPL-2.0-or-later
"""One fixed known-truth Cox bootstrap smoke; not a coverage study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import warnings

import numpy as np
import scipy

from quantilemix import fit_cox_bootstrap
import quantilemix._cox as cox_implementation
import quantilemix._cox_bootstrap as bootstrap_implementation


PROTOCOL = "cox-bootstrap-calibration-preflight-v1"
N = 300
B = 2
MC_SIZE = 300
MASTER_SEED = 20261024
TRUE_LOG_HR = 0.4
BASELINE_RATE = 0.08
CENSOR_RATE = 0.10
ADMIN_END = 12.0


def conditional_hazard(scores: np.ndarray) -> np.ndarray:
    """Exponential event hazard at four prequantized exposure scores."""
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape[1] != 4:
        raise ValueError("scores must have four columns")
    return BASELINE_RATE*np.exp(TRUE_LOG_HR*(scores.sum(axis=1)-6)/4)


def generate() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, 201]))
    scores = rng.integers(0, 4, size=(N, 4), dtype=np.int64)
    latent_event = rng.exponential(scale=1/conditional_hazard(scores))
    independent_censor = rng.exponential(scale=1/CENSOR_RATE, size=N)
    duration = np.minimum(np.minimum(latent_event, independent_censor), ADMIN_END)
    event = ((latent_event <= independent_censor) & (latent_event <= ADMIN_END)).astype(float)
    return scores, duration, event


def fit_seed() -> int:
    return int(np.random.SeedSequence([MASTER_SEED, 202]).generate_state(
        1, dtype=np.uint64)[0])


def digest(scores: np.ndarray, duration: np.ndarray, event: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(scores, dtype="<i8").tobytes()
                          + np.asarray(duration, dtype="<f8").tobytes()
                          + np.asarray(event, dtype="<f8").tobytes()).hexdigest()


def source_identity() -> dict:
    root = Path(__file__).resolve().parents[1]
    paths = {"preflight": Path(__file__), "cox": Path(cox_implementation.__file__),
             "cox_bootstrap": Path(bootstrap_implementation.__file__)}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                            capture_output=True, text=True, check=False)
    state = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                           capture_output=True, text=True, check=False)
    return {"git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "git_dirty": bool(state.stdout.strip()) if state.returncode == 0 else None,
            "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for name, path in paths.items()},
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "scipy": scipy.__version__}


def design() -> dict:
    return {"protocol": PROTOCOL, "purpose": "bounded software and timing preflight",
            "n": N, "B": B, "MCsize": MC_SIZE, "degree": 1, "q": None,
            "master_seed": MASTER_SEED, "true_log_hazard_ratio": TRUE_LOG_HR,
            "baseline_event_rate": BASELINE_RATE, "independent_censor_rate": CENSOR_RATE,
            "administrative_end": ADMIN_END,
            "coverage_result": False, "full_study_authorized_by_this_file": False}


def run(output: Path) -> dict:
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ValueError("output must be a new path in an existing directory")
    scores, duration, event = generate()
    report = {"design": design(), "source_identity": source_identity(),
              "input_sha256": digest(scores, duration, event),
              "fit_seed": fit_seed(), "observed_events": int(np.sum(event)),
              "administratively_censored": int(np.sum((duration == ADMIN_END) & (event == 0))),
              "status": "running"}
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            fit = fit_cox_bootstrap(duration, event, scores, q=None, degree=1,
                                    B=B, MCsize=MC_SIZE, seed=fit_seed())
            report.update(status="passed", point_estimate=float(fit.psi[0]),
                          bootstrap_se=float(fit.se[0]),
                          bootstrap_effects=fit.bootstrap_effects[:, 0].tolist(),
                          intervention_levels=fit.intervention_levels.tolist(),
                          simulated_events=int(np.sum(fit.simulated_event)),
                          error_class=None, error_message=None)
        except Exception as exc:
            report.update(status="failed", point_estimate=None, bootstrap_se=None,
                          bootstrap_effects=None, intervention_levels=None,
                          simulated_events=None, error_class=type(exc).__name__,
                          error_message=str(exc)[:200])
        report["warnings"] = [{"category": type(w.message).__name__,
                               "message": str(w.message)[:200]} for w in caught]
    report["elapsed_seconds"] = time.perf_counter()-started
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        print(json.dumps(design(), indent=2))
        return 0
    result = run(args.output)
    print(f"{result['status']}: {result['elapsed_seconds']:.3f} seconds")
    return 0 if result["status"] == "passed" and not result["warnings"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
