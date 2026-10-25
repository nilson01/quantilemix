# SPDX-License-Identifier: GPL-2.0-or-later
"""Four-level binary odds-ratio row-bootstrap study; default prints its plan."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import time
import warnings

import numpy as np
import scipy
from scipy import optimize, special
from scipy.stats import norm

import quantilemix
from quantilemix import fit_binomial_odds_ratio
import quantilemix._odds_ratio as odds_ratio_implementation

PROTOCOL = "binary-odds-ratio-four-level-known-scores-v1"
MASTER_SEED = 2026102402
B = 200
N = 500


def _intercept(r0: float) -> float:
    return float(optimize.brentq(
        lambda a: (special.expit(a) + special.expit(a + .5))/2 - r0,
        -30, 30, xtol=1e-14))


def _truth(r0: float, b: float) -> float:
    """Population logistic MSM slope over all four joint interventions."""
    if b == 0:
        return 0.0
    a = _intercept(r0)
    levels = np.arange(4, dtype=float)
    risks = (special.expit(a+b*levels/3) +
             special.expit(a+b*levels/3+.5))/2
    design = np.column_stack((np.ones(4),levels))
    def equations(beta: np.ndarray) -> np.ndarray:
        mu = special.expit(design@beta)
        return design.T@(risks-mu)
    answer = optimize.root(equations,[special.logit(r0),b/3],tol=1e-12)
    beta = answer.x
    if (not np.all(np.isfinite(beta)) or
            np.max(np.abs(equations(beta))) > 1e-10):
        raise ValueError("population marginal target could not be solved")
    return float(beta[1])


CELLS = tuple({"cell": 2*i+j, "baseline_risk": r0, "joint_log_odds": b,
               "true_log_odds_ratio": _truth(r0,b)}
              for i,r0 in enumerate((.10,.40)) for j,b in enumerate((0.,.80)))


def design(mode: str) -> dict:
    if mode not in ("pilot", "study"):
        raise ValueError("mode must be pilot or study")
    repetitions = 5 if mode == "pilot" else 1000
    return {"protocol": PROTOCOL, "mode": mode, "n": N, "B": B,
            "cells": CELLS, "repetitions_per_cell": repetitions,
            "total_datasets": len(CELLS)*repetitions,
            "total_bootstrap_fits": len(CELLS)*repetitions*B,
            "master_seed": MASTER_SEED}


def generate(mode: str, cell: int, repetition: int) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    plan = design(mode)
    if type(cell) is not int or not 0 <= cell < len(CELLS):
        raise ValueError("cell outside plan")
    if type(repetition) is not int or not 0 <= repetition < plan["repetitions_per_cell"]:
        raise ValueError("repetition outside plan")
    spec = CELLS[cell]
    rng = np.random.default_rng(np.random.SeedSequence(
        [MASTER_SEED, 111 if mode == "pilot" else 112, cell, repetition]))
    x = rng.integers(0,4,size=(N,4),dtype=np.int64)
    z = rng.integers(0,2,size=(N,1),dtype=np.int64)
    eta = (_intercept(spec["baseline_risk"]) +
           spec["joint_log_odds"]*x.sum(axis=1)/12 + .5*z[:,0])
    y = rng.binomial(1,special.expit(eta)).astype(np.int64)
    return x,z,y


def bootstrap_seed(mode: str, cell: int, repetition: int) -> int:
    design(mode)
    return int(np.random.SeedSequence(
        [MASTER_SEED,113,1 if mode == "pilot" else 2,cell,repetition]
    ).generate_state(1,dtype=np.uint64)[0])


def input_hash(x: np.ndarray,z: np.ndarray,y: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x,dtype="<i8").tobytes() +
                          np.asarray(z,dtype="<i8").tobytes() +
                          np.asarray(y,dtype="<i8").tobytes()).hexdigest()


def provenance(mode: str) -> dict:
    root = Path(__file__).resolve().parents[1]
    commit_result = subprocess.run(["git","rev-parse","HEAD"],cwd=root,
                                   capture_output=True,text=True,check=False)
    status_result = subprocess.run(["git","status","--porcelain"],cwd=root,
                                   capture_output=True,text=True,check=False)
    commit = commit_result.stdout.strip() if commit_result.returncode == 0 else None
    dirty = bool(status_result.stdout.strip()) if status_result.returncode == 0 else None
    paths = {"runner":Path(__file__),
             "checker":Path(__file__).with_name("check_odds_ratio_q4_calibration.py"),
             "reference_cases":Path(__file__).with_name("reference_odds_ratio_q4_cases.py"),
             "project_metadata":root/"pyproject.toml"}
    package_root = Path(quantilemix.__file__).resolve().parent
    for path in package_root.rglob("*.py"):
        paths[f"package/{path.relative_to(package_root).as_posix()}"] = path
    hashes = {name:hashlib.sha256(path.read_bytes()).hexdigest()
              for name,path in paths.items()}
    if mode == "study":
        if not commit or len(commit) != 40 or dirty is not False:
            raise ValueError("study requires a clean committed checkout")
        for name,path in paths.items():
            rel = (f"src/quantilemix/{name.removeprefix('package/')}"
                   if name.startswith("package/") else
                   ("pyproject.toml" if name == "project_metadata" else
                    f"validation/{path.name}"))
            committed = subprocess.run(["git","show",f"{commit}:{rel}"],cwd=root,
                                       capture_output=True,check=False)
            if committed.returncode or committed.stdout != path.read_bytes():
                raise ValueError(f"study source differs from commit: {name}")
    return {"git_commit":commit,"git_dirty":dirty,"source_sha256":hashes,
            "package_origin":str(package_root),
            "package_version":importlib.metadata.version("quantilemix"),
            "python":platform.python_version(),"platform":platform.platform(),
            "numpy":np.__version__,"scipy":scipy.__version__}


def wilson(yes: int,total: int) -> list[float] | None:
    if total == 0:
        return None
    z = float(norm.ppf(.975))
    p = yes/total
    den = 1+z*z/total
    mid = (p+z*z/(2*total))/den
    half = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [max(0.0,mid-half),min(1.0,mid+half)]


def summarize(records: list[dict],plan: dict) -> dict:
    result = {}
    for spec in CELLS:
        rows = [r for r in records if r["cell"] == spec["cell"]]
        good = [r for r in rows if r["fit"]["status"] == "ok"]
        m = len(good)
        covered = sum(r["fit"]["ci"][0] <= spec["true_log_odds_ratio"] <= r["fit"]["ci"][1]
                      for r in good)
        rejected = sum(r["fit"]["p_value"] < .05 for r in good)
        part = {"planned":plan["repetitions_per_cell"],"attempted":len(rows),
                "missing":plan["repetitions_per_cell"]-len(rows),"successful":m,
                "failed":len(rows)-m,"warnings":sum(bool(r["fit"]["warnings"]) for r in rows),
                "observed_event_rate":sum(r["events"] for r in rows)/(N*len(rows)) if rows else None,
                "covered":covered,"rejected":rejected,
                "coverage_all_planned_bounds":[covered/plan["repetitions_per_cell"],
                   (covered+plan["repetitions_per_cell"]-m)/plan["repetitions_per_cell"]],
                "rejection_all_planned_bounds":[rejected/plan["repetitions_per_cell"],
                   (rejected+plan["repetitions_per_cell"]-m)/plan["repetitions_per_cell"]]}
        if m:
            errors = np.array([r["fit"]["estimate"]-spec["true_log_odds_ratio"] for r in good])
            ses = np.array([r["fit"]["se"] for r in good])
            part.update({"bias":float(errors.mean()),
                         "bias_mcse":float(errors.std(ddof=1)/math.sqrt(m)) if m>1 else None,
                         "empirical_sd":float(errors.std(ddof=1)) if m>1 else None,
                         "root_mean_variance":float(np.sqrt(np.mean(ses**2))),
                         "coverage":covered/m,"coverage_wilson_95":wilson(covered,m),
                         "rejection_rate":rejected/m,"rejection_wilson_95":wilson(rejected,m)})
        result[str(spec["cell"])] = part
    return result


def _capture(y: np.ndarray,x: np.ndarray,z: np.ndarray,seed: int) -> dict:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            fit = fit_binomial_odds_ratio(y,x,z,q=None,B=B,seed=seed)
            result = {"status":"ok","estimate":float(fit.psi),"se":float(fit.se),
                      "ci":list(map(float,fit.ci)),"p_value":float(fit.p_value),
                      "bootstrap_effects":list(map(float,fit.bootstrap_effects)),
                      "intervention_risks":list(map(float,fit.intervention_risks)),
                      "failure_class":None,"failure_detail":None,
                      "failed_draw":None,"completed_draws":B}
        except Exception as exc:
            detail = str(exc)
            draw = re.search(r"bootstrap replicate (\d+) failed:",detail)
            failed_draw = int(draw.group(1)) if draw is not None else None
            partial = isinstance(exc, odds_ratio_implementation.BootstrapFitError)
            result = {"status":"failed","estimate":None,"se":None,"ci":None,
                      "p_value":None,
                      "bootstrap_effects":list(exc.effects) if partial else None,
                      "intervention_risks":list(exc.risks) if partial else None,
                      "original_estimate":exc.estimate if partial else None,
                      "failure_class":type(exc).__name__,"failure_detail":detail,
                      "failed_draw":failed_draw,
                      "completed_draws":failed_draw-1 if failed_draw is not None else None}
        result["warnings"] = [{"category":type(w.message).__name__,
                               "message":str(w.message)} for w in caught]
    return result


def _paths(output: Path) -> dict[str,Path]:
    return {name:output.with_name(output.name+suffix) for name,suffix in
            (("partial",".partial.jsonl"),("journal",".journal.jsonl"),
             ("receipt",".complete.json"),("lock",".lock"),("temporary",".tmp"))}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sync_line(handle,text: str) -> None:
    handle.write(text+"\n")
    handle.flush()
    os.fsync(handle.fileno())


def _publish_no_replace(source: Path,target: Path) -> None:
    """Publish an already synced file only when the destination is absent."""
    os.link(source,target)


def run(mode: str,output: Path,registry: Path | None = None) -> dict:
    plan = design(mode)
    paths = _paths(output)
    identity = provenance(mode)
    run_id = hashlib.sha256(json.dumps(
        {"plan":plan,"git_commit":identity["git_commit"],
         "source_sha256":identity["source_sha256"]},sort_keys=True).encode()).hexdigest()
    if registry is None:
        if mode == "study":
            raise ValueError("study requires an authoritative identity registry")
        registry = output.parent
    if not registry.is_dir():
        raise ValueError("identity registry must be an existing directory")
    global_lock = registry/f"{PROTOCOL}-{mode}-{run_id}.json"
    if not output.parent.is_dir() or any(
            path.exists() or path.is_symlink() for path in
            (output,*paths.values(),global_lock)):
        raise ValueError("output and run identity must be new in an existing directory")
    with global_lock.open("x") as lock:
        _sync_line(lock,json.dumps({"run_id":run_id,"pid":os.getpid(),
                                    "mode":mode,"source_identity":identity}))
    with paths["lock"].open("x") as lock:
        _sync_line(lock,json.dumps({"run_id":run_id,"pid":os.getpid(),
                                    "output":output.name,"source_identity":identity}))
    records = []
    started = time.monotonic()
    with paths["partial"].open("x") as journal:
        _sync_line(journal,json.dumps({"event":"header","run_id":run_id,
                                       "plan":plan,"source_identity":identity}))
        for spec in CELLS:
            for repetition in range(plan["repetitions_per_cell"]):
                case = {"cell":spec["cell"],"repetition":repetition}
                _sync_line(journal,json.dumps({"event":"case_start",**case}))
                try:
                    x,z,y = generate(mode,spec["cell"],repetition)
                    seed = bootstrap_seed(mode,spec["cell"],repetition)
                    row = {**case,"input_sha256":input_hash(x,z,y),
                           "bootstrap_seed":seed,"events":int(y.sum()),
                           "fit":_capture(y,x,z,seed)}
                    line = json.dumps({"event":"case_complete","record":row},allow_nan=False)
                except Exception as exc:
                    _sync_line(journal,json.dumps({"event":"case_error",**case,
                                                   "stage":"generation_or_recording",
                                                   "error_class":type(exc).__name__,
                                                   "error_detail":str(exc)}))
                    raise
                _sync_line(journal,line)
                records.append(row)
        _sync_line(journal,json.dumps({"event":"end","completed":len(records)}))
    journal_hash = _sha256(paths["partial"])
    result = {"run_id":run_id,"plan":plan,"source_identity":identity,
              "journal_sha256":journal_hash,"records":records,
              "summaries":summarize(records,plan),"elapsed_seconds":time.monotonic()-started}
    with paths["temporary"].open("x") as temp:
        temp.write(json.dumps(result,indent=2,allow_nan=False)+"\n")
        temp.flush()
        os.fsync(temp.fileno())
    _publish_no_replace(paths["temporary"],output)
    paths["temporary"].unlink()
    _publish_no_replace(paths["partial"],paths["journal"])
    paths["partial"].unlink()
    receipt = {"run_id":run_id,"output_sha256":_sha256(output),
               "journal_sha256":_sha256(paths["journal"]),
               "plan_sha256":hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest(),
               "source_sha256":identity["source_sha256"],"record_count":len(records)}
    with paths["receipt"].open("x") as marker:
        marker.write(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
        marker.flush()
        os.fsync(marker.fileno())
    paths["lock"].unlink()
    with global_lock.open("a") as lock:
        _sync_line(lock,json.dumps({"status":"complete","output":str(output.resolve()),
                                    "output_sha256":receipt["output_sha256"],
                                    "receipt":str(paths["receipt"].resolve())}))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("pilot","study"))
    parser.add_argument("--output",type=Path)
    parser.add_argument("--identity-registry",type=Path)
    args = parser.parse_args()
    if args.mode is None and args.output is None:
        print(json.dumps(design("study"),indent=2))
        return 0
    if args.mode is None or args.output is None:
        parser.error("--mode and --output are required together")
    result = run(args.mode,args.output,args.identity_registry)
    print(f"{len(result['records'])} attempted; {result['elapsed_seconds']:.3f} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
