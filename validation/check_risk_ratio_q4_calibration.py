# SPDX-License-Identifier: GPL-2.0-or-later
"""Independently solve saved four-level risk-ratio bootstrap fits."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import time

import numpy as np
import scipy
from scipy import optimize, special
from scipy.stats import norm

from quantilemix import fit_binomial_risk_ratio
import quantilemix._binomial as binomial_implementation
import quantilemix._risk_ratio as risk_ratio_implementation

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "risk_ratio_q4_calibration",HERE/"risk_ratio_q4_calibration.py")
calibration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(calibration)


def independent_fit(x: np.ndarray,z: np.ndarray,y: np.ndarray,
                    initial: np.ndarray | None = None) -> dict:
    design = np.column_stack((np.ones(len(y)),x.astype(float),z.astype(float)))
    if initial is None:
        event_rate = float(np.mean(y))
        if not 0 < event_rate < 1:
            raise ValueError("independent fit needs both outcome classes")
        initial = np.zeros(6)
        initial[0] = float(special.logit(event_rate))

    def score(beta):
        return design.T@(y-special.expit(design@beta))

    def jac(beta):
        p = special.expit(design@beta)
        return -(design.T@((p*(1-p))[:,None]*design))

    solution = optimize.root(score,initial,jac=jac,tol=1e-10)
    beta = np.asarray(solution.x)
    residual = score(beta)
    if (not np.all(np.isfinite(beta)) or not np.all(np.isfinite(residual))
            or float(np.max(np.abs(residual))) > 1e-7):
        raise ValueError("independent logistic score residual unacceptable")
    probability = special.expit(design@beta)
    if (float(np.max(np.abs(beta))) > 30
            or float(np.min(probability*(1-probability))) < 1e-12):
        raise ValueError("logistic separation or boundary")
    information = -jac(beta)
    eigenvalues = np.linalg.eigvalsh(information)
    if not np.all(np.isfinite(eigenvalues)) or eigenvalues[0] <= 0:
        raise ValueError("independent logistic information is not positive definite")
    slope = float(np.sum(beta[1:5]))
    levels = np.arange(4,dtype=float)
    risks = [float(np.mean(special.expit(beta[0]+beta[5]*z[:,0]+a*slope)))
             for a in levels]
    if not all(0 < risk < 1 for risk in risks):
        raise ValueError("independent standardized risks outside (0,1)")
    marginal = np.asarray(risks)
    msm_design = np.column_stack((np.ones(4),levels))
    def msm_score(value):
        fitted = np.exp(msm_design@value)
        return msm_design.T@((marginal-fitted)/(1-fitted))
    starting = np.array([math.log(marginal[0]),
                         math.log(marginal[-1]/marginal[0])/3])
    msm = optimize.root(msm_score,starting,tol=1e-11)
    if (not np.all(np.isfinite(msm.x)) or np.any(msm_design@msm.x >= 0)
            or np.max(np.abs(msm_score(msm.x))) > 1e-9):
        raise ValueError("independent marginal score residual unacceptable")
    return {"beta":beta,"estimate":float(msm.x[1]),
            "intervention_risks":risks,
            "score_max_abs":float(np.max(np.abs(residual))),
            "information_min_eigenvalue":float(eigenvalues[0]),
            "solver_success":bool(solution.success)}


def independent_case(x: np.ndarray,z: np.ndarray,y: np.ndarray,seed: int,B: int) -> dict:
    original = independent_fit(x,z,y)
    rng = np.random.default_rng(seed)
    effects = []
    max_score = original["score_max_abs"]
    min_eigenvalue = original["information_min_eigenvalue"]
    for draw in range(B):
        rows = rng.integers(0,len(y),size=len(y),dtype=np.intp)
        try:
            replica = independent_fit(x[rows],z[rows],y[rows],original["beta"])
        except Exception as exc:
            raise ValueError(f"independent bootstrap draw {draw+1}: {exc}") from exc
        effects.append(replica["estimate"])
        max_score = max(max_score,replica["score_max_abs"])
        min_eigenvalue = min(min_eigenvalue,replica["information_min_eigenvalue"])
    values = np.asarray(effects)
    se = float(np.std(values,ddof=1))
    if not np.isfinite(se) or se <= 0:
        raise ValueError("independent bootstrap standard error invalid")
    estimate = original["estimate"]
    critical = float(norm.ppf(.975))
    return {"estimate":estimate,"se":se,
            "ci":[estimate-critical*se,estimate+critical*se],
            "p_value":float(2*norm.sf(abs(estimate/se))),
            "bootstrap_effects":values.tolist(),
            "intervention_risks":original["intervention_risks"],
            "max_score_residual":max_score,
            "min_information_eigenvalue":min_eigenvalue}


def _close(a: object,b: float) -> bool:
    return (type(a) in (int,float) and math.isfinite(a)
            and math.isclose(a,b,abs_tol=1e-7,rel_tol=1e-6))


def compare_fit(saved: dict,other: dict) -> tuple[float,list]:
    maximum = 0.
    differences = []
    for field in ("estimate","se","p_value","ci","bootstrap_effects","intervention_risks"):
        actual,target = saved.get(field),other[field]
        if isinstance(target,list):
            if not isinstance(actual,list) or len(actual)!=len(target):
                differences.append({"field":field,"reason":"length_or_type"})
                continue
            bad = []
            for i,(a,b) in enumerate(zip(actual,target,strict=True)):
                if type(a) in (int,float) and math.isfinite(a):
                    maximum = max(maximum,abs(a-b))
                if not _close(a,b):
                    bad.append(i)
            if bad:
                differences.append({"field":field,"indices":bad})
        else:
            if type(actual) in (int,float) and math.isfinite(actual):
                maximum = max(maximum,abs(actual-target))
            if not _close(actual,target):
                differences.append(field)
    return maximum,differences


def _source_check(source: dict,mode: str) -> None:
    paths = {"runner":HERE/"risk_ratio_q4_calibration.py",
             "checker":Path(__file__),
             "binomial_estimator":Path(binomial_implementation.__file__),
             "risk_ratio_estimator":Path(risk_ratio_implementation.__file__)}
    hashes = source.get("source_sha256")
    if not isinstance(hashes,dict):
        raise ValueError("source hashes absent")
    for name,path in paths.items():
        if hashes.get(name)!=hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f"{name} hash differs")
    if source.get("numpy")!=np.__version__ or source.get("scipy")!=scipy.__version__:
        raise ValueError("numerical dependency versions differ")
    if mode=="study":
        commit = source.get("git_commit")
        if not isinstance(commit,str) or len(commit)!=40 or source.get("git_dirty") is not False:
            raise ValueError("study source is not clean committed source")
        root = HERE.parent
        for path in paths.values():
            rel = path.relative_to(root).as_posix()
            result = subprocess.run(["git","show",f"{commit}:{rel}"],cwd=root,
                                    capture_output=True,check=False)
            if result.returncode or result.stdout!=path.read_bytes():
                raise ValueError(f"study source differs from commit: {rel}")


def _audit_saved_failure(saved: dict,y: np.ndarray,x: np.ndarray,z: np.ndarray,
                         seed: int,cell: int,repetition: int) -> list[dict]:
    issues = []
    if not isinstance(saved.get("failure_class"),str) or not isinstance(saved.get("failure_detail"),str):
        issues.append({"cell":cell,"repetition":repetition,"code":"failure_reason_invalid"})
    draw = saved.get("failed_draw")
    count = saved.get("completed_draws")
    if draw is None and count is not None:
        issues.append({"cell":cell,"repetition":repetition,"code":"completed_draw_count_without_failure"})
    if draw is not None and (type(draw) is not int or not 1<=draw<=calibration.B
                             or count!=draw-1):
        issues.append({"cell":cell,"repetition":repetition,"code":"failed_draw_invalid"})
    if any(saved.get(k) is not None for k in
           ("estimate","se","ci","p_value","bootstrap_effects","intervention_risks")):
        issues.append({"cell":cell,"repetition":repetition,"code":"failed_fit_has_values"})
    try:
        fit_binomial_risk_ratio(y,x,z,q=None,B=calibration.B,seed=seed)
    except Exception as exc:
        if (type(exc).__name__!=saved.get("failure_class") or
                str(exc)!=saved.get("failure_detail")):
            issues.append({"cell":cell,"repetition":repetition,"code":"failure_replay_differs"})
    else:
        issues.append({"cell":cell,"repetition":repetition,"code":"fabricated_failure"})
    return issues


def _matching_separation(saved: dict,independent_error: str) -> bool:
    """Recognize the same failed draw without treating a boundary root as a fit."""
    match = re.fullmatch(r"independent bootstrap draw (\d+): logistic separation or boundary",
                         independent_error)
    return (match is not None and saved.get("failure_class")=="ValueError"
            and saved.get("failed_draw")==int(match.group(1))
            and "check separation" in saved.get("failure_detail",""))


def audit_journal(path: Path) -> dict:
    """Read a completed or interrupted journal without fitting new models."""
    issues = []
    entries = []
    for line_number,line in enumerate(path.read_text().splitlines(),1):
        try:
            item = json.loads(line)
        except (ValueError,TypeError):
            issues.append({"code":"malformed_line","line":line_number})
            continue
        if not isinstance(item,dict):
            issues.append({"code":"nonobject_line","line":line_number})
            continue
        entries.append(item)
    if not entries or entries[0].get("event")!="header":
        raise ValueError("journal header absent")
    header = entries[0]
    plan = header.get("plan")
    if not isinstance(plan,dict) or json.dumps(plan,sort_keys=True)!=json.dumps(
            calibration.design(plan.get("mode")),sort_keys=True):
        raise ValueError("journal plan differs from fixed design")
    expected = {(c,r) for c in range(len(calibration.CELLS))
                for r in range(plan["repetitions_per_cell"])}
    started = set()
    terminal = {}
    ended = False
    for index,item in enumerate(entries[1:],2):
        event = item.get("event")
        if event=="end":
            if ended or index!=len(entries):
                issues.append({"code":"end_position_invalid","line":index})
            ended = True
            if item.get("completed")!=len(terminal):
                issues.append({"code":"end_count_differs","line":index})
            continue
        if event=="case_complete":
            row = item.get("record")
            key = (row.get("cell"),row.get("repetition")) if isinstance(row,dict) else None
        elif event in ("case_start","case_error"):
            key = (item.get("cell"),item.get("repetition"))
        else:
            issues.append({"code":"unknown_event","line":index})
            continue
        if key not in expected:
            issues.append({"code":"invalid_case_identity","line":index})
            continue
        if event=="case_start":
            if key in started or key in terminal:
                issues.append({"code":"duplicate_start","line":index})
            started.add(key)
        else:
            if key not in started or key in terminal:
                issues.append({"code":"missing_start_or_duplicate_terminal","line":index})
            terminal[key] = item
            if event=="case_error":
                issues.append({"code":"case_error","cell":key[0],"repetition":key[1],
                               "stage":item.get("stage"),"reason":item.get("error_detail")})
    missing = sorted(expected-started)
    incomplete = sorted(started-set(terminal))
    if ended and (missing or incomplete):
        issues.append({"code":"ended_with_incomplete_cases"})
    status = ("complete" if ended and not issues and not missing and not incomplete
              and len(terminal)==len(expected) else "failed" if ended or issues
              else "incomplete")
    return {"status":status,"plan":plan,"run_id":header.get("run_id"),
            "source_identity":header.get("source_identity"),
            "started":len(started),"terminal":len(terminal),
            "unattempted":missing,"started_without_terminal":incomplete,
            "end_seen":ended,"issues":issues,
            "records":[item["record"] for item in entries[1:]
                       if item.get("event")=="case_complete" and isinstance(item.get("record"),dict)]}


def independent_targets(plan: dict) -> list[dict]:
    """Check all population risks and the four-level marginal score equations."""
    issues = []
    c = math.exp(.5)
    for spec in plan["cells"]:
        r0 = spec["baseline_risk"]
        b = spec["joint_log_odds"]
        # t=exp(intercept): 2*r0 = t/(1+t) + c*t/(1+c*t).
        roots = np.roots([2*c*(r0-1),(1+c)*(2*r0-1),2*r0])
        positive = [float(v.real) for v in roots if abs(v.imag)<1e-13 and v.real>0]
        if len(positive)!=1:
            issues.append({"cell":spec["cell"],"code":"target_root_invalid"})
            continue
        t = positive[0]
        def risk(shift):
            u = t*math.exp(shift)
            return (u/(1+u)+c*u/(1+c*u))/2
        levels = np.arange(4,dtype=float)
        risks = np.array([risk(b*level/3) for level in levels])
        design = np.column_stack((np.ones(4),levels))
        def score(value):
            mu = np.exp(design@value)
            return design.T@((risks-mu)/(1-mu))
        solved = optimize.root(score,[math.log(r0),b/3],tol=1e-12)
        if (not np.all(np.isfinite(solved.x)) or np.any(design@solved.x >= 0)
                or np.max(np.abs(score(solved.x))) > 1e-10):
            issues.append({"cell":spec["cell"],"code":"analytic_msm_invalid"})
            continue
        target = float(solved.x[1])
        if not math.isclose(target,spec["true_log_risk_ratio"],abs_tol=1e-13,rel_tol=1e-12):
            issues.append({"cell":spec["cell"],"code":"analytic_target_differs",
                           "independent_target":target})
    return issues


def check(path: Path) -> dict:
    started = time.monotonic()
    payload = json.loads(path.read_text())
    plan = payload.get("plan")
    if not isinstance(plan,dict) or json.dumps(plan,sort_keys=True)!=json.dumps(
            calibration.design(plan.get("mode")),sort_keys=True):
        raise ValueError("plan differs from fixed design")
    source = payload.get("source_identity")
    if not isinstance(source,dict):
        raise ValueError("source identity absent")
    _source_check(source,plan["mode"])
    expected_run_id = hashlib.sha256(json.dumps(
        {"plan":plan,"git_commit":source.get("git_commit"),
         "source_sha256":source.get("source_sha256")},sort_keys=True).encode()).hexdigest()
    if payload.get("run_id") != expected_run_id:
        raise ValueError("run identity differs from plan and source")
    paths = calibration._paths(path)
    if not paths["receipt"].is_file() or not paths["journal"].is_file():
        raise ValueError("completion receipt or retained journal absent")
    receipt = json.loads(paths["receipt"].read_text())
    journal = audit_journal(paths["journal"])
    if journal["status"]!="complete":
        raise ValueError("retained journal is not complete")
    if (receipt.get("run_id")!=payload.get("run_id")
            or journal["run_id"]!=payload.get("run_id")
            or receipt.get("output_sha256")!=hashlib.sha256(path.read_bytes()).hexdigest()
            or receipt.get("journal_sha256")!=hashlib.sha256(paths["journal"].read_bytes()).hexdigest()
            or payload.get("journal_sha256")!=receipt["journal_sha256"]
            or receipt.get("source_sha256")!=source.get("source_sha256")
            or receipt.get("plan_sha256")!=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
            or journal["source_identity"]!=source):
        raise ValueError("completion receipt, source or journal digest differs")
    records = payload.get("records")
    if not isinstance(records,list):
        raise ValueError("records absent")
    if receipt.get("record_count")!=len(records) or journal["records"]!=records:
        raise ValueError("retained journal records differ from final output")
    seen = set()
    issues = independent_targets(plan)
    case_audit = []
    maximum = 0.
    independent_failures = 0
    matching_failures = 0
    package_failures = 0
    paired = 0
    for row in records:
        if not isinstance(row,dict):
            issues.append({"code":"record_not_object"})
            continue
        cell,repetition = row.get("cell"),row.get("repetition")
        if (type(cell) is not int or type(repetition) is not int
                or not 0<=cell<len(calibration.CELLS)
                or not 0<=repetition<plan["repetitions_per_cell"]
                or (cell,repetition) in seen):
            issues.append({"cell":cell,"repetition":repetition,"code":"invalid_or_duplicate_identity"})
            continue
        seen.add((cell,repetition))
        audit = {"cell":cell,"repetition":repetition}
        try:
            x,z,y = calibration.generate(plan["mode"],cell,repetition)
            seed = calibration.bootstrap_seed(plan["mode"],cell,repetition)
            if (row.get("input_sha256")!=calibration.input_hash(x,z,y)
                    or row.get("bootstrap_seed")!=seed or row.get("events")!=int(y.sum())):
                raise ValueError("input, seed or event count differs")
            saved = row.get("fit")
            if not isinstance(saved,dict) or saved.get("status") not in ("ok","failed"):
                raise ValueError("fit record invalid")
            warnings_list = saved.get("warnings")
            if not isinstance(warnings_list,list) or any(
                    not isinstance(w,dict) or set(w)!={"category","message"}
                    or not isinstance(w["category"],str) or not isinstance(w["message"],str)
                    for w in warnings_list):
                raise ValueError("warning record invalid")
            if warnings_list:
                issues.append({"cell":cell,"repetition":repetition,"code":"package_warning",
                               "warnings":warnings_list})
            audit["package_status"] = saved["status"]
            if saved["status"]=="failed":
                package_failures += 1
                issues.extend(_audit_saved_failure(saved,y,x,z,seed,cell,repetition))
            elif saved.get("failure_class") is not None or saved.get("failure_detail") is not None:
                issues.append({"cell":cell,"repetition":repetition,"code":"successful_fit_has_failure_reason"})
            elif saved.get("completed_draws")!=plan["B"] or saved.get("failed_draw") is not None:
                issues.append({"cell":cell,"repetition":repetition,"code":"successful_draw_count_invalid"})
            try:
                other = independent_case(x,z,y,seed,plan["B"])
            except Exception as exc:
                independent_failures += 1
                audit.update({"independent_status":"failed",
                              "independent_failure":str(exc)[:200]})
                if saved["status"]=="failed" and _matching_separation(saved,str(exc)):
                    matching_failures += 1
                    audit["failure_agreement"] = "separation_at_same_draw"
                else:
                    issues.append({"cell":cell,"repetition":repetition,
                                   "code":"independent_fit_failed","reason":str(exc)[:200]})
                    if saved["status"]=="failed":
                        issues.append({"cell":cell,"repetition":repetition,
                                       "code":"package_fit_failed",
                                       "reason":saved.get("failure_detail")})
            else:
                audit.update({"independent_status":"ok",
                              "score_max_abs":other["max_score_residual"],
                              "information_min_eigenvalue":other["min_information_eigenvalue"]})
                if saved["status"]=="ok":
                    paired += 1
                    difference,bad = compare_fit(saved,other)
                    maximum = max(maximum,difference)
                    if bad:
                        issues.append({"cell":cell,"repetition":repetition,
                                       "code":"numerical_mismatch","fields":bad})
                else:
                    issues.append({"cell":cell,"repetition":repetition,
                                   "code":"package_fit_failed",
                                   "reason":saved.get("failure_detail")})
                    issues.append({"cell":cell,"repetition":repetition,
                                   "code":"independent_success_package_failure"})
        except Exception as exc:
            audit.update({"audit_status":"failed","audit_error":str(exc)[:200]})
            issues.append({"cell":cell,"repetition":repetition,
                           "code":"case_audit_failed","reason":str(exc)[:200]})
        case_audit.append(audit)
    expected = {(c,r) for c in range(len(calibration.CELLS))
                for r in range(plan["repetitions_per_cell"])}
    missing = sorted(expected-seen)
    if missing:
        issues.append({"code":"missing_cases","identities":missing})
    if len(records)!=plan["total_datasets"]:
        issues.append({"code":"record_count_differs","observed":len(records)})
    try:
        summary = calibration.summarize([r for r in records if isinstance(r,dict)
                                         and type(r.get("cell")) is int
                                         and 0<=r["cell"]<len(calibration.CELLS)
                                         and isinstance(r.get("fit"),dict)
                                         and r["fit"].get("status") in ("ok","failed")
                                         and type(r.get("events")) is int],plan)
    except Exception as exc:
        issues.append({"code":"summary_recalculation_failed","reason":str(exc)[:200]})
    else:
        if json.dumps(summary,sort_keys=True)!=json.dumps(payload.get("summaries"),sort_keys=True):
            issues.append({"code":"summary_differs"})
    return {"status":"passed" if not issues else "failed","plan":plan,
            "source_identity":source,"input_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "datasets":len(seen),"paired":paired,"independent_failures":independent_failures,
            "package_failures":package_failures,"matching_failures":matching_failures,
            "maximum_absolute_difference":maximum,"case_audit":case_audit,
            "issue_count":len(issues),"issues":issues,"elapsed_seconds":time.monotonic()-started}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input",type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--partial-journal",action="store_true",
                        help="audit a journal without treating an interrupted run as complete")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir():
        parser.error("output must be a new path in an existing directory")
    result = audit_journal(args.input) if args.partial_journal else check(args.input)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    if args.partial_journal:
        print(f"{result['started']} started; {result['terminal']} terminal; "
              f"{len(result['issues'])} journal issues")
        return 0
    print(f"{result['datasets']} audited; {result['issue_count']} issues")
    return 0 if result["status"]=="passed" else 1


if __name__=="__main__":
    raise SystemExit(main())
