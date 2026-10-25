# SPDX-License-Identifier: GPL-2.0-or-later
"""Independent all-case Poisson bootstrap and saved-summary audit."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import subprocess
import time
import warnings

import numpy as np
import scipy
from scipy import optimize
from scipy.stats import norm
import quantilemix._poisson as poisson_implementation
import quantilemix._poisson_bootstrap as bootstrap_implementation
from quantilemix import fit_poisson, fit_poisson_bootstrap

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "poisson_bootstrap_offset_calibration", HERE / "poisson_bootstrap_offset_calibration.py")
calibration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(calibration)


def independent_fit(x: np.ndarray, z: np.ndarray, exposure_time: np.ndarray, y: np.ndarray,
                    initial: np.ndarray | None = None) -> dict:
    design = np.column_stack((np.ones(len(y)), x.astype(float), z.astype(float)))
    def score(beta):
        return design.T @ (y-exposure_time*np.exp(design@beta))
    def jac(beta):
        mu = exposure_time*np.exp(design@beta)
        return -(design.T @ (mu[:,None]*design))
    if initial is None:
        initial = np.zeros(6)
        mean = float(y.sum()/exposure_time.sum())
        if mean <= 0:
            raise ValueError("independent fit has no observed counts")
        initial[0] = math.log(mean)
    solved = optimize.root(score, initial, jac=jac, tol=1e-10)
    beta = np.asarray(solved.x)
    residual = score(beta)
    if (not np.all(np.isfinite(beta)) or not np.all(np.isfinite(residual))
            or float(np.max(np.abs(residual))) > 1e-7):
        raise ValueError(f"independent score residual unacceptable: {np.max(np.abs(residual)):.6g}")
    information = -jac(beta)
    eig = np.linalg.eigvalsh(information)
    if not np.all(np.isfinite(eig)) or eig[0] <= 0:
        raise ValueError("independent information not positive definite")
    contrast = np.array([0.,1.,1.,1.,1.,0.])
    variance = float(contrast @ np.linalg.solve(information,contrast))
    return {"beta": beta, "estimate": float(contrast@beta),
            "variance": variance, "score_max_abs": float(np.max(np.abs(residual))),
            "information_min_eigenvalue": float(eig[0]),
            "solver_success": bool(solved.success)}


def independent_case(x: np.ndarray, z: np.ndarray, exposure_time: np.ndarray,
                     y: np.ndarray, seed: int, B: int) -> dict:
    original = independent_fit(x,z,exposure_time,y)
    levels = np.arange(4)
    means = np.array([np.mean(exposure_time*np.exp(original["beta"][0] + original["beta"][5]*z[:,0]
                                     + a*original["estimate"])) for a in levels])
    marginal = float(np.log(means[1]/means[0]))
    rng = np.random.default_rng(seed)
    effects = []
    max_score = original["score_max_abs"]
    min_eig = original["information_min_eigenvalue"]
    for draw in range(B):
        rows = rng.integers(0,len(y),size=len(y),dtype=np.intp)
        try:
            replica = independent_fit(x[rows],z[rows],exposure_time[rows],y[rows],original["beta"])
        except Exception as exc:
            raise ValueError(f"independent bootstrap draw {draw}: {exc}") from exc
        effects.append(replica["estimate"])
        max_score = max(max_score,replica["score_max_abs"])
        min_eig = min(min_eig,replica["information_min_eigenvalue"])
    effects = np.asarray(effects)
    se = float(effects.std(ddof=1))
    critical = float(norm.ppf(.975))
    model_se = math.sqrt(original["variance"])
    return {"bootstrap": {"estimate": marginal, "se": se,
                          "ci": [marginal-critical*se,marginal+critical*se],
                          "p_value": float(2*norm.sf(abs(marginal/se))),
                          "bootstrap_effects": effects.tolist(),
                          "intervention_means": means.tolist()},
            "model_based": {"estimate": original["estimate"], "se": model_se,
                            "ci": [original["estimate"]-critical*model_se,
                                   original["estimate"]+critical*model_se],
                            "p_value": float(2*norm.sf(abs(original["estimate"]/model_se)))},
            "max_score_residual": max_score, "min_information_eigenvalue": min_eig}


def close(actual, expected):
    return (type(actual) in (int,float) and math.isfinite(actual)
            and math.isclose(actual,expected,abs_tol=1e-7,rel_tol=1e-6))


def compare_method(saved, other, cell, repetition, name):
    differences = []
    maximum = 0.0
    for field in ("estimate","se","p_value"):
        value = saved.get(field)
        target = other[field]
        if type(value) in (int,float) and math.isfinite(value):
            maximum = max(maximum,abs(value-target))
        if not close(value,target):
            differences.append(field)
    for field in ("ci",) + (("bootstrap_effects","intervention_means") if name == "bootstrap" else ()):
        value = saved.get(field)
        target = other[field]
        if not isinstance(value,list) or len(value) != len(target):
            differences.append(field)
            continue
        bad = []
        for i,(a,b) in enumerate(zip(value,target,strict=True)):
            if type(a) in (int,float) and math.isfinite(a):
                maximum = max(maximum,abs(a-b))
            if not close(a,b):
                bad.append(i)
        if bad:
            differences.append({"field":field,"indices":bad})
    return maximum,differences


def audit_saved_method(saved, x, z, exposure_time, y, seed, B, cell, repetition, name):
    """Validate and replay package failures before the independent solve."""
    if not isinstance(saved,dict) or saved.get("status") not in ("ok","failed"):
        raise ValueError("method record invalid")
    caught = saved.get("warnings")
    if not isinstance(caught,list) or any(
            not isinstance(w,dict) or set(w)!={"category","message"}
            or not isinstance(w["category"],str) or not isinstance(w["message"],str)
            for w in caught):
        raise ValueError("warning record invalid")
    problems = []
    if caught:
        problems.append({"cell":cell,"repetition":repetition,"method":name,
                         "code":"warning","warnings":caught})
    if saved["status"] == "ok":
        if saved.get("failure_class") is not None or saved.get("failure_detail") is not None:
            raise ValueError("successful case has failure reason")
        return problems
    if not isinstance(saved.get("failure_class"),str) or not isinstance(saved.get("failure_detail"),str):
        raise ValueError("failure reason absent")
    if any(saved.get(key) is not None for key in ("estimate","se","ci","p_value")):
        raise ValueError("failed method has estimate")
    try:
        if name == "bootstrap":
            fit_poisson_bootstrap(y,x,z,q=None,B=B,seed=seed,exposure_time=exposure_time)
        else:
            fit_poisson(y,x,z,q=None,exposure_time=exposure_time)
    except Exception as exc:
        if type(exc).__name__ != saved["failure_class"] or str(exc)[:200] != saved["failure_detail"]:
            problems.append({"cell":cell,"repetition":repetition,"method":name,
                             "code":"failure_replay_differs"})
    else:
        problems.append({"cell":cell,"repetition":repetition,"method":name,
                         "code":"fabricated_failure"})
    problems.append({"cell":cell,"repetition":repetition,"method":name,
                     "code":"package_fit_failed","reason":saved["failure_detail"]})
    return problems


def check(path: Path) -> dict:
    started = time.monotonic()
    payload = json.loads(path.read_text())
    plan = payload.get("plan")
    if not isinstance(plan,dict) or json.dumps(plan,sort_keys=True) != json.dumps(
            calibration.design(plan.get("mode")),sort_keys=True):
        raise ValueError("plan differs from fixed design")
    source = payload.get("source_identity")
    if not isinstance(source,dict):
        raise ValueError("source identity absent")
    paths = {"runner": HERE/"poisson_bootstrap_offset_calibration.py",
             "checker": Path(__file__),
             "poisson_estimator":Path(poisson_implementation.__file__),
             "bootstrap_estimator":Path(bootstrap_implementation.__file__)}
    for name,source_path in paths.items():
        if source.get("source_sha256",{}).get(name) != hashlib.sha256(source_path.read_bytes()).hexdigest():
            raise ValueError(f"{name} source differs")
    if source.get("numpy") != np.__version__:
        raise ValueError("NumPy version differs from result; exact random-stream replay is unsupported")
    if plan["mode"] == "study":
        commit = source.get("git_commit")
        if not isinstance(commit,str) or len(commit) != 40 or source.get("git_dirty") is not False:
            raise ValueError("study lacks clean committed provenance")
        rels = {"runner":"validation/poisson_bootstrap_offset_calibration.py",
                "poisson_estimator":"src/quantilemix/_poisson.py",
                "bootstrap_estimator":"src/quantilemix/_poisson_bootstrap.py",
                "checker":"validation/check_poisson_bootstrap_offset_calibration.py"}
        for name,rel in rels.items():
            committed = subprocess.run(["git","show",f"{commit}:{rel}"],cwd=HERE.parent,
                                       capture_output=True,check=False)
            source_path = Path(__file__) if name == "checker" else paths[name]
            if committed.returncode or committed.stdout != source_path.read_bytes():
                raise ValueError(f"{name} differs from study commit")
    records = payload.get("records")
    if not isinstance(records,list):
        raise ValueError("records are not a list")
    seen = set()
    issues = []
    audit = []
    maximum = 0.0
    paired = 0
    independent_failures = 0
    if len(records) != plan["total_datasets"]:
        issues.append({"code":"record_count_differs","observed":len(records),
                       "planned":plan["total_datasets"]})
    for row_index,row in enumerate(records):
        try:
            case = audit_row(row, plan, seen)
        except (ValueError,TypeError,KeyError,AttributeError) as exc:
            issue={"row_index":row_index,"code":"case_integrity_error",
                   "class":type(exc).__name__,"reason":str(exc)[:200]}
            issues.append(issue)
            audit.append({"row_index":row_index,"status":"integrity_failed","reason":issue["reason"]})
            continue
        cell,repetition,x,z,exposure_time,y,seed = case
        case_audit = {"cell":cell,"repetition":repetition,"row_index":row_index}
        for name in ("bootstrap","model_based"):
            try:
                issues.extend(audit_saved_method(row.get(name),x,z,exposure_time,y,seed,
                                                 plan["B"],cell,repetition,name))
            except (ValueError,TypeError,KeyError,AttributeError) as exc:
                issues.append({"cell":cell,"repetition":repetition,"method":name,
                               "code":"method_integrity_error","reason":str(exc)[:200]})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                independent = independent_case(x,z,exposure_time,y,seed,plan["B"])
            except Exception as exc:
                independent_failures += 1
                case_audit.update({"status":"independent_failed","reason":str(exc)[:200]})
                issues.append({"cell":cell,"repetition":repetition,"code":"independent_fit_failed",
                               "class":type(exc).__name__,"reason":str(exc)[:200]})
                independent = None
        if caught:
            details=[{"category":type(w.message).__name__,"message":str(w.message)[:300]} for w in caught]
            issues.append({"cell":cell,"repetition":repetition,
                           "code":"independent_warning","warnings":details})
        if independent is None:
            audit.append(case_audit)
            continue
        case_audit.update({"status":"independent_ok",
                           "max_score_residual":independent["max_score_residual"],
                           "min_information_eigenvalue":independent["min_information_eigenvalue"]})
        for name in ("bootstrap","model_based"):
            saved = row[name]
            if not isinstance(saved,dict) or saved.get("status") != "ok":
                continue
            try:
                difference,mismatched = compare_method(saved,independent[name],cell,repetition,name)
            except (ValueError,TypeError,KeyError,AttributeError) as exc:
                issues.append({"cell":cell,"repetition":repetition,"method":name,
                               "code":"comparison_error","reason":str(exc)[:200]})
                continue
            maximum=max(maximum,difference)
            if mismatched:
                issues.append({"cell":cell,"repetition":repetition,"method":name,
                               "code":"numerical_mismatch","fields":mismatched,"maximum_absolute_difference":difference})
            else:
                paired += 1
        audit.append(case_audit)
    missing={(cell,repetition) for cell in range(len(calibration.CELLS))
             for repetition in range(plan["repetitions_per_cell"])}-seen
    if missing:
        issues.append({"code":"planned_identities_missing","count":len(missing),
                       "first":list(sorted(missing)[0])})
    try:
        expected_summary = calibration.summarize(records,plan)
        if json.dumps(payload.get("summaries"),sort_keys=True) != json.dumps(expected_summary,sort_keys=True):
            issues.append({"code":"summary_differs_from_records"})
    except (ValueError,TypeError,KeyError,AttributeError) as exc:
        issues.append({"code":"summary_cannot_be_recomputed","reason":str(exc)[:200]})
    return {"status":"passed" if not issues else "needs_review", "datasets":len(records),
            "method_pairs":paired,"independent_failures":independent_failures,
            "issues":issues,"case_audit":audit,"maximum_absolute_difference":maximum,
            "result_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "checker_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "checker_revision":subprocess.run(["git","rev-parse","HEAD"],cwd=HERE.parent,
                              capture_output=True,text=True,check=False).stdout.strip(),
            "python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,
            "elapsed_seconds":time.monotonic()-started}


def audit_row(row, plan, seen):
    if not isinstance(row,dict):
        raise ValueError("case is not an object")
    cell,repetition = row.get("cell"),row.get("repetition")
    if (type(cell) is not int or type(repetition) is not int or not 0 <= cell < len(calibration.CELLS)
            or not 0 <= repetition < plan["repetitions_per_cell"] or (cell,repetition) in seen):
        raise ValueError("case identity invalid or repeated")
    x,z,exposure_time,y = calibration.generate(plan["mode"],cell,repetition)
    seed = calibration.bootstrap_seed(plan["mode"],cell,repetition)
    if (row.get("input_sha256") != calibration.input_hash(x,z,exposure_time,y)
            or row.get("bootstrap_seed") != seed):
        raise ValueError("input or bootstrap seed differs")
    if row.get("count_total") != int(y.sum()) or row.get("zero_count") != int(np.count_nonzero(y==0)):
        raise ValueError("count totals differ")
    if row.get("time_total") != float(exposure_time.sum()):
        raise ValueError("observation-time total differs")
    seen.add((cell,repetition))
    return cell,repetition,x,z,exposure_time,y,seed


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    report=check(args.result)
    if args.output:
        if args.output.exists() or args.output.is_symlink():
            parser.error("output already exists")
        pending = args.output.with_name(args.output.name + ".partial")
        if pending.exists() or pending.is_symlink():
            parser.error("partial audit already exists")
        with pending.open("x") as stream:
            stream.write(json.dumps(report,indent=2)+"\n")
            stream.flush()
            import os
            os.fsync(stream.fileno())
        pending.replace(args.output)
    print(json.dumps({k:report[k] for k in ("status","datasets","method_pairs","independent_failures",
                                              "maximum_absolute_difference","elapsed_seconds")}))
    return 0 if report["status"]=="passed" else 1

if __name__=="__main__":
    raise SystemExit(main())
