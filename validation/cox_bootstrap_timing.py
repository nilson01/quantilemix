# SPDX-License-Identifier: GPL-2.0-or-later
"""Fixed marginal Cox bootstrap timing profiles; no coverage inference."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np

from quantilemix import fit_cox_bootstrap


PREFLIGHT_PATH = Path(__file__).with_name("cox_bootstrap_preflight.py")
SPEC = importlib.util.spec_from_file_location("cox_bootstrap_preflight", PREFLIGHT_PATH)
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)
PROFILES = {"medium": {"B": 20, "MCsize": 1000, "seed_namespace": 1},
            "default_mc": {"B": 20, "MCsize": 10000, "seed_namespace": 2}}


def plan(profile: str) -> dict:
    if profile not in PROFILES:
        raise ValueError("profile must be medium or default_mc")
    settings = PROFILES[profile]
    return {"protocol": "cox-bootstrap-timing-v1", "profile": profile,
            "n": preflight.N, "degree": 1, "q": None,
            "B": settings["B"], "MCsize": settings["MCsize"],
            "true_log_hazard_ratio": preflight.TRUE_LOG_HR,
            "coverage_result": False, "full_study_authorized_by_this_file": False}


def seed(profile: str) -> int:
    settings = PROFILES[profile]
    return int(np.random.SeedSequence([
        preflight.MASTER_SEED, 203, settings["seed_namespace"]
    ]).generate_state(1, dtype=np.uint64)[0])


def peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw*1024


def run(profile: str, output: Path) -> dict:
    design = plan(profile)
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ValueError("output must be a new path in an existing directory")
    x, duration, event = preflight.generate()
    identity = preflight.source_identity()
    identity["source_sha256"]["timing"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = {"design": design, "source_identity": identity,
              "input_sha256": preflight.digest(x, duration, event),
              "fit_seed": seed(profile), "observed_events": int(np.sum(event)),
              "status": "running"}
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            fit = fit_cox_bootstrap(duration, event, x, q=None, degree=1,
                                    B=design["B"], MCsize=design["MCsize"],
                                    seed=seed(profile))
            report.update(status="passed", point_estimate=float(fit.psi[0]),
                          bootstrap_se=float(fit.se[0]),
                          bootstrap_effects=fit.bootstrap_effects[:, 0].tolist(),
                          simulated_events=int(np.sum(fit.simulated_event)),
                          error_class=None, error_message=None)
        except Exception as exc:
            report.update(status="failed", point_estimate=None, bootstrap_se=None,
                          bootstrap_effects=None, simulated_events=None,
                          error_class=type(exc).__name__, error_message=str(exc)[:200])
        report["warnings"] = [{"category": type(w.message).__name__,
                               "message": str(w.message)[:200]} for w in caught]
    report["elapsed_seconds"] = time.perf_counter()-started
    report["peak_process_rss_bytes"] = peak_rss_bytes()
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.profile is None and args.output is None:
        print(json.dumps({key: plan(key) for key in PROFILES}, indent=2))
        return 0
    if args.profile is None or args.output is None:
        parser.error("--profile and --output are required together")
    result = run(args.profile, args.output)
    print(f"{result['status']}: {result['elapsed_seconds']:.3f} seconds")
    return 0 if result["status"] == "passed" and not result["warnings"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
