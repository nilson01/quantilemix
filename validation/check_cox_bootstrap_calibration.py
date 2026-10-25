# SPDX-License-Identifier: GPL-2.0-or-later
"""Audit Cox calibration batches and summarize all attempted cases."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from statistics import NormalDist

import numpy as np
import scipy


RUNNER_PATH = Path(__file__).with_name("cox_bootstrap_calibration.py")
SPEC = importlib.util.spec_from_file_location("cox_bootstrap_calibration", RUNNER_PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
NORMAL = NormalDist()


def _same(actual: float, expected: float, name: str) -> None:
    if not (math.isfinite(actual) and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10)):
        raise ValueError(f"{name} differs from independent arithmetic")


def wilson(yes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = NORMAL.inv_cdf(.975)
    p = yes/total
    den = 1+z*z/total
    center = (p+z*z/(2*total))/den
    half = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [center-half, center+half]


def _audit_fit(fit: dict, label: str) -> None:
    if not isinstance(fit.get("warnings"), list) or not math.isfinite(fit.get("elapsed_seconds", float("nan"))):
        raise ValueError(f"{label}: warning or timing record invalid")
    if fit["elapsed_seconds"] < 0:
        raise ValueError(f"{label}: negative elapsed time")
    if fit.get("status") == "failed":
        if (not isinstance(fit.get("failure_class"), str)
                or any(fit.get(key) is not None for key in
                       ("estimate", "se", "ci", "p_value"))):
            raise ValueError(f"{label}: failed fit fields invalid")
        if fit["failure_class"] == "CoxBootstrapFitError":
            draw = fit.get("failed_draw")
            completed = fit.get("completed_draws")
            effects = np.asarray(fit.get("bootstrap_effects"), dtype=float)
            levels = np.asarray(fit.get("intervention_levels"), dtype=float)
            original = fit.get("original_estimate")
            detail = fit.get("failure_detail")
            if (type(draw) is not int or not 1 <= draw <= runner.B
                    or type(completed) is not int or completed != draw-1
                    or effects.shape != (completed,) or not np.all(np.isfinite(effects))
                    or levels.shape != (4,) or not np.all(np.isfinite(levels))
                    or not np.all(np.diff(levels) > 0)
                    or isinstance(original, bool)
                    or not isinstance(original, (int, float)) or not math.isfinite(original)
                    or not isinstance(detail, str)
                    or f"bootstrap replicate {draw} failed:" not in detail):
                raise ValueError(f"{label}: partial bootstrap record invalid")
        elif any(fit.get(key) is not None for key in
                 ("bootstrap_effects", "intervention_levels", "original_estimate",
                  "failed_draw", "completed_draws")):
            raise ValueError(f"{label}: failed fit fields invalid")
        return
    if fit.get("status") != "ok" or fit.get("failure_class") is not None:
        raise ValueError(f"{label}: unknown fit status")
    if (fit.get("failed_draw") is not None or fit.get("original_estimate") is not None
            or fit.get("completed_draws", runner.B) != runner.B):
        raise ValueError(f"{label}: successful fit draw record invalid")
    effects = np.asarray(fit.get("bootstrap_effects"), dtype=float)
    if effects.shape != (runner.B,) or not np.all(np.isfinite(effects)):
        raise ValueError(f"{label}: missing or nonfinite bootstrap effects")
    estimate = float(fit["estimate"])
    se = float(fit["se"])
    ci = np.asarray(fit["ci"], dtype=float)
    p_value = float(fit["p_value"])
    levels = np.asarray(fit.get("intervention_levels"), dtype=float)
    if (not math.isfinite(estimate) or not math.isfinite(se) or se <= 0
            or ci.shape != (2,) or not np.all(np.isfinite(ci))
            or levels.shape != (4,) or not np.all(np.isfinite(levels))
            or not np.all(np.diff(levels) > 0)
            or not math.isfinite(p_value) or not 0 <= p_value <= 1):
        raise ValueError(f"{label}: estimate or interval invalid")
    _same(se, float(effects.std(ddof=1)), f"{label}.se")
    critical = NORMAL.inv_cdf(.975)
    for position, expected in enumerate((estimate-critical*se, estimate+critical*se)):
        _same(float(ci[position]), expected, f"{label}.ci[{position}]")
    _same(p_value, math.erfc(abs(estimate/se)/math.sqrt(2)), f"{label}.p_value")


def audit_batch(report: dict) -> list[dict]:
    batch = report.get("batch")
    plan = report.get("plan")
    if not isinstance(batch, dict) or not isinstance(plan, dict):
        raise ValueError("batch header missing")
    mode = plan.get("mode")
    if plan != json.loads(json.dumps(runner.design(mode))):
        raise ValueError("batch plan differs from fixed design")
    cell, start, count = (batch.get(key) for key in ("cell", "start", "count"))
    if (type(cell) is not int or type(start) is not int or type(count) is not int
            or not 0 <= cell < len(runner.CELLS) or count < 1 or start < 0
            or start+count > plan["repetitions_per_cell"]):
        raise ValueError("invalid batch interval")
    rows = report.get("records")
    if not isinstance(rows, list) or (len(rows) > count if report.get("_partial")
                                       else len(rows) != count):
        raise ValueError("batch records missing or count differs")
    if not report.get("_partial") and not math.isfinite(
            report.get("elapsed_seconds", float("nan"))):
        raise ValueError("batch elapsed time invalid")
    for offset, row in enumerate(rows):
        repetition = start+offset
        label = f"cell {cell}, repetition {repetition}"
        x, duration, event = runner.generate(mode, cell, repetition)
        if (row.get("cell") != cell or row.get("repetition") != repetition
                or row.get("truth") != runner.CELLS[cell]["truth"]
                or row.get("fit_seed") != runner.fit_seed(mode, cell, repetition)
                or row.get("input_sha256") != runner.input_hash(x, duration, event)
                or row.get("observed_events") != int(event.sum())
                or row.get("administratively_censored") !=
                int(np.sum((duration == runner.ADMIN_END) & (event == 0)))):
            raise ValueError(f"{label}: data, seed or identity mismatch")
        _audit_fit(row.get("fit", {}), label)
        if row["fit"]["status"] == "ok":
            selected = np.arange(4, dtype=float)
            if not np.allclose(row["fit"]["intervention_levels"], selected,
                               rtol=0, atol=1e-12):
                raise ValueError(f"{label}: intervention levels differ")
    return rows


def _source_matches(identity: dict) -> None:
    paths = {"protocol": RUNNER_PATH.with_name("COX_BOOTSTRAP_PROTOCOL.md"),
             "runner": RUNNER_PATH, "checker": Path(__file__),
             "pilot_auditor": RUNNER_PATH.with_name("server_cox_bootstrap") / "audit_pilot.py",
             "cox_estimator": Path(runner.cox_implementation.__file__),
             "bootstrap_estimator": Path(runner.bootstrap_implementation.__file__),
             "core": Path(runner.core_implementation.__file__),
             "package": Path(runner.package_implementation.__file__)}
    expected = identity.get("source_sha256") if isinstance(identity, dict) else None
    if isinstance(expected, dict) and "study_auditor" in expected:
        paths["study_auditor"] = RUNNER_PATH.with_name("server_cox_bootstrap") / "audit_study.py"
        paths["study_job"] = RUNNER_PATH.with_name("server_cox_bootstrap") / "study.sbatch"
    if (not isinstance(expected, dict) or set(expected) != set(paths)
            or any(expected[name] != hashlib.sha256(path.read_bytes()).hexdigest()
                   for name, path in paths.items())):
        raise ValueError("batch source hashes differ from this checkout")
    record = metadata.distribution("quantilemix").read_text("RECORD")
    installed_record = hashlib.sha256(record.encode()).hexdigest() if record else None
    if (identity.get("qgcomp_distribution_version") != metadata.version("quantilemix")
            or identity.get("distribution_record_sha256") != installed_record
            or identity.get("python") != platform.python_version()
            or identity.get("numpy") != np.__version__
            or identity.get("scipy") != scipy.__version__
            or identity.get("numpy_build") != np.__config__.show(mode="dicts")):
        raise ValueError("batch installed environment differs from audit environment")
    if identity.get("git_dirty") is False:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=RUNNER_PATH.parents[1],
                                capture_output=True, text=True, check=False)
        state = subprocess.run(["git", "status", "--porcelain"],
                               cwd=RUNNER_PATH.parents[1],
                               capture_output=True, text=True, check=False)
        if (commit.returncode or state.returncode or state.stdout.strip()
                or commit.stdout.strip() != identity.get("git_commit")):
            raise ValueError("batch commit differs from this checkout")


def summarize(rows: list[dict], mode: str, interrupted: set[tuple[int, int]]) -> dict:
    plan = runner.design(mode)
    result = {}
    for spec in runner.CELLS:
        cell_rows = [row for row in rows if row["cell"] == spec["cell"]]
        good = [row["fit"] for row in cell_rows if row["fit"]["status"] == "ok"]
        count = len(good)
        covered = sum(fit["ci"][0] <= spec["truth"] <= fit["ci"][1] for fit in good)
        rejected = sum(fit["p_value"] < .05 for fit in good)
        planned = plan["repetitions_per_cell"]
        interrupted_count = sum(key[0] == spec["cell"] for key in interrupted)
        observed_events = sum(row["observed_events"] for row in cell_rows)
        administrative_censoring = sum(row["administratively_censored"]
                                        for row in cell_rows)
        observed_people = runner.N*len(cell_rows)
        part = {"planned": planned, "attempted": len(cell_rows)+interrupted_count,
                "interrupted": interrupted_count,
                "never_started": planned-len(cell_rows)-interrupted_count,
                "missing": planned-len(cell_rows),
                "successful": count, "failed": len(cell_rows)-count,
                "warnings": sum(bool(row["fit"]["warnings"]) for row in cell_rows),
                "observed_event_fraction": (observed_events/observed_people
                                            if cell_rows else None),
                "censored_fraction": (1-observed_events/observed_people
                                       if cell_rows else None),
                "administrative_censoring_fraction": (
                    administrative_censoring/observed_people if cell_rows else None),
                "independent_censoring_fraction": (
                    (observed_people-observed_events-administrative_censoring)
                    /observed_people if cell_rows else None),
                "covered": covered, "rejected": rejected,
                "coverage_all_planned_bounds": [covered/planned, (covered+planned-count)/planned],
                "rejection_all_planned_bounds": [rejected/planned,
                                                 (rejected+planned-count)/planned]}
        if good:
            estimates = np.asarray([fit["estimate"] for fit in good])
            ses = np.asarray([fit["se"] for fit in good])
            errors = estimates-spec["truth"]
            part.update(bias=float(errors.mean()),
                        bias_mcse=float(errors.std(ddof=1)/math.sqrt(count)) if count > 1 else None,
                        empirical_sd=float(estimates.std(ddof=1)) if count > 1 else None,
                        root_mean_variance=float(np.sqrt(np.mean(ses**2))),
                        mean_interval_width=float(np.mean([fit["ci"][1]-fit["ci"][0] for fit in good])),
                        coverage=covered/count, coverage_wilson_95=wilson(covered, count),
                        rejection_among_successful=rejected/count,
                        rejection_wilson_95=wilson(rejected, count))
        result[str(spec["cell"])] = part
    return result


def _read_partial(path: Path) -> tuple[dict, set[tuple[int, int]], str]:
    content = path.read_text()
    lines = content.splitlines()
    if not lines:
        raise ValueError("empty partial journal")
    entries = []
    truncated = False
    for index, line in enumerate(lines):
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines)-1 and not content.endswith("\n"):
                truncated = True
            else:
                raise ValueError("partial journal has corrupt complete line") from None
    if entries[0].get("kind") != "header":
        raise ValueError("partial journal header missing")
    report = dict(entries[0]["header"])
    report["_partial"] = True
    report["records"] = []
    cell = report["batch"]["cell"]
    start = report["batch"]["start"]
    count = report["batch"]["count"]
    starts: list[tuple[int, int]] = []
    status = "truncated_unconfirmed_exit" if truncated else "unconfirmed_exit"
    for index, entry in enumerate(entries[1:], start=1):
        kind = entry.get("kind")
        if kind == "start":
            key = entry.get("cell"), entry.get("repetition")
            if (key != (cell, start+len(starts)) or len(starts) >= count
                    or len(starts) != len(report["records"])):
                raise ValueError("partial journal starts are out of order")
            starts.append(key)
        elif kind == "record":
            row = entry.get("row")
            if (not starts or len(report["records"]) != len(starts)-1
                    or (row.get("cell"), row.get("repetition")) != starts[-1]):
                raise ValueError("partial journal record lacks matching start")
            report["records"].append(row)
        elif kind == "terminal":
            if index != len(entries)-1 or entry.get("status") != "interrupted":
                raise ValueError("partial journal terminal invalid")
            status = "interrupted"
        else:
            raise ValueError("partial journal entry invalid")
    return report, set(starts), status


def audit_files(paths: list[Path], *, complete: bool) -> dict:
    if not paths:
        raise ValueError("at least one batch file is required")
    parsed = [(_read_partial(path) if path.name.endswith(".partial.jsonl")
               else (json.loads(path.read_text()), set(), "complete")) for path in paths]
    reports = [item[0] for item in parsed]
    first = reports[0]
    identity = first.get("source_identity")
    _source_matches(identity)
    mode = first["plan"]["mode"]
    seen = set()
    claimed = set()
    started = set()
    rows = []
    exits = []
    for report, partial_starts, exit_status in parsed:
        if report.get("source_identity") != identity or report.get("plan") != first["plan"]:
            raise ValueError("batch plans or source identities differ")
        exits.append(exit_status)
        batch_rows = audit_batch(report)
        batch_keys = {(row["cell"], row["repetition"]) for row in batch_rows}
        if not batch_keys.issubset(partial_starts) and report.get("_partial"):
            raise ValueError("partial records lack starts")
        this_claim = partial_starts | batch_keys
        if claimed.intersection(this_claim):
            raise ValueError("duplicate cell and repetition")
        claimed.update(this_claim)
        started.update(partial_starts)
        for row in batch_rows:
            key = row["cell"], row["repetition"]
            if key in seen:
                raise ValueError("duplicate cell and repetition")
            seen.add(key)
            rows.append(row)
    interrupted = started-seen
    if any(report.get("_partial") for report in reports) and complete:
        raise ValueError("complete audit cannot include partial journals")
    if complete and len(rows) != runner.design(mode)["total_datasets"]:
        raise ValueError("complete audit has missing datasets")
    summaries = summarize(rows, mode, interrupted)
    return {"status": "partial" if interrupted or not complete else "passed",
            "mode": mode, "source_identity": identity,
            "audited_batch_files": [str(path) for path in paths],
            "audited_records": len(rows), "complete": complete,
            "interrupted_case_ids": sorted([list(key) for key in interrupted]),
            "journal_exit_statuses": exits,
            "summaries": summaries}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--complete", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir():
        raise ValueError("output must be a new path in an existing directory")
    result = audit_files(args.input, complete=args.complete)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"audited {result['audited_records']} Cox calibration records")


if __name__ == "__main__":
    main()
