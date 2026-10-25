#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Independently reconcile the fixed completed study without fitting any model.

Usage: python validation/check_calibration_results.py RUN_DIRECTORY NEW_REPORT.json

This optional tool checks gaussian-known-scores-v1 at the recorded study source
revision 8cb4619934e1cf02ed8ade3b09f09b25115f9381. It does not fit models, draw
data or start R. A later study source revision needs its own reviewed identity.

Only the standard library is used. At most one batch of retained data is loaded
at a time. Scalar accumulators replace the production aggregator. The report is
created exclusively outside the run directory; the run is never modified.
This checker verifies stored evidence, not statistical validity or calibration.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import NormalDist
import sys


METHODS = ("python", "r", "qr_t")
ATOL, RTOL = 1e-10, 1e-8
SUMMARY_ATOL, SUMMARY_RTOL = 1e-12, 1e-10
SOURCE_HASHES = {
    "calibration.py": "c4aa619dba8a6bf7116ef4f119716ff5fb840ea378beca4140fb86e513d3fb0e",
    "qgcomp/__init__.py": "59f333a2061f7420b09093e4fb8e2de66212823a0b591bf62a75c57cbf8f119a",
    "qgcomp/_core.py": "efad59c9d420696fa6619795fdf5cd40c9f34829a32625a215ce0b7200f2f447",
    "calibration_reference.py": "8a22c59e49b492cbddac32d802f3c7f86fe0b883e8608383c2c93c6f551293d1",
    "calibration_r.R": "a9d2db991cccf994974e34ce59d1943217a7a9e2155e0c2581ce8dcc6c984dca",
}
EXPECTED_CONFIG = {
    "schema_version": 1, "protocol_id": "gaussian-known-scores-v1",
    "reference_version": "2.19.6", "target": "known_scores", "alpha": 0.05,
    "design": {
        "sample_sizes": [100, 500], "rho": [0, 0.75],
        "beta": {"null": [0, 0, 0, 0], "cancellation": [0.25, -0.25, 0, 0],
                 "positive": [0.1, 0.1, 0.05, 0]},
        "intercept": 1, "covariate_coefficients": [0.5, -0.25],
        "exposure_z1_loading": 0.5, "error_sd": 1, "quartile_variance": 1.25,
    },
    "study": {"replicates_per_cell": 2000, "seed_namespace": 2},
    "pilot": {"replicates": 20, "sample_size": 500, "rho": 0.75,
              "beta_name": "positive", "seed_namespace": 1},
    "execution": {"batch_size": 20, "master_seed": 20261022,
                  "retain_inputs": True, "storage_budget_bytes": 3000000000},
}


class EvidenceError(ValueError):
    """The requested completed-study evidence is absent or inconsistent."""


def require(condition, label):
    if not condition:
        raise EvidenceError(label)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def finite_tree(value):
    if isinstance(value, float):
        require(math.isfinite(value), "nonfinite JSON number")
    elif isinstance(value, dict):
        for part in value.values():
            finite_tree(part)
    elif isinstance(value, list):
        for part in value:
            finite_tree(part)


def read_json(path, *, with_identity=False):
    def invalid_constant(value):
        raise EvidenceError(f"nonfinite JSON constant: {value}")
    require(path.is_file() and not path.is_symlink(), f"missing or unsafe file: {path.name}")
    raw = path.read_bytes()
    result = json.loads(raw, object_pairs_hook=no_duplicate_keys,
                        parse_constant=invalid_constant)
    finite_tree(result)
    if with_identity:
        return result, {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    return result


def keys(value, expected, label):
    require(isinstance(value, dict) and set(value) == set(expected),
            f"{label}: unexpected schema")


def number(value, label):
    require(type(value) in (int, float) and math.isfinite(value),
            f"{label}: not a finite number")


def same(actual, expected, label):
    """Compare derived summaries with a stricter tolerance than fit agreement."""
    if isinstance(expected, dict):
        keys(actual, expected, label)
        for key in expected:
            same(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected),
                f"{label}: list length differs")
        for index, value in enumerate(expected):
            same(actual[index], value, f"{label}[{index}]")
    elif type(expected) is float:
        number(actual, label)
        require(math.isclose(actual, expected, rel_tol=SUMMARY_RTOL, abs_tol=SUMMARY_ATOL),
                f"{label}: saved {actual!r} differs from independently calculated {expected!r}")
    else:
        require(type(actual) is type(expected) and actual == expected,
                f"{label}: saved {actual!r} differs from independently calculated {expected!r}")


def expected_cells():
    output = []
    for n in (100, 500):
        for rho, rho_name in ((0, "0"), (0.75, "0p75")):
            for beta_name, beta in EXPECTED_CONFIG["design"]["beta"].items():
                output.append({"id": f"n{n}-rho{rho_name}-{beta_name}",
                               "index": len(output), "n": n, "rho": rho,
                               "beta_name": beta_name, "beta": beta,
                               "true_psi": sum(beta), "planned": 2000})
    return output


def check_manifest(manifest):
    keys(manifest, ("schema_version", "identity", "identity_sha256", "plan"), "manifest")
    require(manifest["schema_version"] == 1, "manifest schema version differs")
    identity = manifest["identity"]
    keys(identity, ("config_sha256", "config", "mode", "sources_sha256", "environment", "reference"), "identity")
    require(identity["mode"] == "study", "pilot evidence cannot establish study completion")
    require(canonical(identity["config"]) == canonical(EXPECTED_CONFIG), "configuration differs from the exact approved study")
    require(identity["config_sha256"] == digest(EXPECTED_CONFIG), "configuration hash differs")
    require(identity["sources_sha256"] == SOURCE_HASHES, "source hashes differ from delivery revision 8cb4619")
    require(manifest["identity_sha256"] == digest(identity), "manifest identity hash differs")
    reference = identity["reference"]
    require(isinstance(reference, dict), "missing R reference identity")
    for key, value in {"package": "qgcomp", "version": "2.19.6", "mode": "installed package",
                       "library_verified": True, "unexpected_loaded_namespaces": []}.items():
        require(type(reference.get(key)) is type(value) and reference[key] == value,
                f"R reference identity differs: {key}")
    for key in ("R_version", "platform"):
        require(isinstance(reference.get(key), str) and bool(reference[key]), f"missing R identity: {key}")
    for key in ("package_versions", "installed_packages", "package_library_origins"):
        require(isinstance(reference.get(key), dict) and bool(reference[key]), f"missing R dependency identity: {key}")
    versions, installed, origins = (reference[key] for key in (
        "package_versions", "installed_packages", "package_library_origins"))
    require(set(versions) == set(origins), "R loaded version/origin inventories differ")
    require(versions.get("qgcomp") == "2.19.6" and origins.get("qgcomp") == "isolated",
            "qgcomp namespace is not the isolated pinned package")
    for name, version in versions.items():
        require(isinstance(name, str) and bool(name) and isinstance(version, str) and bool(version),
                "invalid loaded R package identity")
        require(origins[name] in ("isolated", "base-recommended"), "unadmitted R package origin")
        if origins[name] == "isolated":
            require(isinstance(installed.get(name), dict) and installed[name].get("version") == version,
                    f"R isolated installation and loaded version differ: {name}")
    for name, detail in installed.items():
        require(isinstance(name, str) and bool(name) and isinstance(detail, dict), "invalid installed R package metadata")
        require(isinstance(detail.get("version"), str) and bool(detail["version"]), "missing installed R version")
        require("Built" in detail and (detail["Built"] is None or isinstance(detail["Built"], str)),
                "missing installed R build metadata")
    for key in ("installed_package_md5", "r_runtime_md5"):
        hashes = reference.get(key)
        require(isinstance(hashes, dict) and bool(hashes), f"missing R fingerprints: {key}")
        if key == "installed_package_md5":
            keys(hashes, ("DESCRIPTION", "NAMESPACE", "R/qgcomp.rdb", "R/qgcomp.rdx"), "qgcomp fingerprints")
        for name, value in hashes.items():
            require(isinstance(name, str) and bool(re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", name))
                    and all(part not in (".", "..") for part in name.split("/"))
                    and isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{32}", value)),
                    "invalid R file fingerprint")
    environment = identity["environment"]
    keys(environment, ("python", "numpy", "scipy", "platform", "machine", "threads", "numerical_libraries"), "Python environment")
    for key in ("python", "numpy", "scipy", "platform", "machine"):
        require(isinstance(environment[key], str) and bool(environment[key]), f"missing Python environment: {key}")
    require(isinstance(environment["numerical_libraries"], dict) and bool(environment["numerical_libraries"]),
            "missing numerical library identity")
    require(identity["environment"]["threads"] == {key: "1" for key in (
        "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}, "thread identity differs")
    plan = manifest["plan"]
    keys(plan, ("schema_version", "protocol_id", "mode", "cells", "planned", "batch_size",
                "input_storage_estimate_upper_bytes", "storage_budget_bytes", "storage_estimate_within_budget",
                "execution_performed", "calibration_validated", "tolerances", "peak_memory_bytes"), "plan")
    cells = expected_cells()
    require(isinstance(plan["cells"], list) and len(plan["cells"]) == 12, "expected twelve cells")
    for actual, expected in zip(plan["cells"], cells):
        keys(actual, set(expected) | {"analytic"}, "planned cell")
        for key, value in expected.items():
            same(actual[key], value, f"plan.{expected['id']}.{key}")
        analytic = actual["analytic"]
        keys(analytic, ("condition", "df_residual", "df_reference_pvalue", "reference_coverage",
                        "reference_null_rejection", "qr_t_coverage", "qr_t_null_rejection"), "analytic targets")
        require(analytic["condition"] == "correct normal homoscedastic model and full-rank design",
                "analytic assumptions changed")
        for key, value in {"df_residual": expected["n"]-7, "df_reference_pvalue": expected["n"]-5,
                           "qr_t_coverage": 0.95, "qr_t_null_rejection": 0.05}.items():
            same(analytic[key], value, f"analytic.{key}")
        rounded = (0.94700655, 0.05006228) if expected["n"] == 100 else (0.94943710, 0.05000227)
        for key, value in zip(("reference_coverage", "reference_null_rejection"), rounded):
            number(analytic[key], key)
            require(abs(analytic[key]-value) <= 5.01e-9, f"{key}: differs from protocol benchmark")
        expected["analytic"] = analytic
    rest = {key: value for key, value in plan.items() if key != "cells"}
    same(rest, {
        "schema_version": 1, "protocol_id": "gaussian-known-scores-v1", "mode": "study",
        "planned": 24000, "batch_size": 20,
        "input_storage_estimate_upper_bytes": sum(2000*(256*c["n"]+16384) for c in cells),
        "storage_budget_bytes": 3000000000, "storage_estimate_within_budget": True,
        "execution_performed": False, "calibration_validated": False,
        "tolerances": {"atol": ATOL, "rtol": RTOL, "p_value_atol": ATOL},
        "peak_memory_bytes": None,
    }, "plan")
    return cells


def check_inputs(inputs, n):
    keys(inputs, ("exposures", "covariates", "y", "q", "alpha"), "retained inputs")
    require(inputs["q"] is None and type(inputs["alpha"]) is float and inputs["alpha"] == 0.05,
            "inputs do not use fixed scores and alpha=0.05")
    for key, width in (("exposures", 4), ("covariates", 2), ("y", None)):
        values = inputs[key]
        require(isinstance(values, list) and len(values) == n, f"{key}: sample size differs")
        for row in values:
            if width is None:
                number(row, key)
            else:
                require(isinstance(row, list) and len(row) == width, f"{key}: width differs")
                for value in row:
                    number(value, key)
                    if key == "exposures":
                        require(0 <= value <= 3 and int(value) == value, "score outside integers 0..3")


def check_result(result, n, method):
    keys(result, ("psi", "variance", "se", "ci", "p_value", "df_pvalue", "df_residual"), "result")
    for key in ("psi", "variance", "se", "p_value"):
        number(result[key], key)
    require(result["variance"] > 0 and result["se"] > 0 and 0 <= result["p_value"] <= 1,
            "invalid variance, standard error or p-value")
    require(isinstance(result["ci"], list) and len(result["ci"]) == 2, "interval must have two endpoints")
    for endpoint in result["ci"]:
        number(endpoint, "interval endpoint")
    require(result["ci"][0] <= result["psi"] <= result["ci"][1], "interval does not enclose estimate")
    require(math.isclose(result["se"]**2, result["variance"], abs_tol=ATOL, rel_tol=RTOL),
            "variance and squared standard error disagree")
    for key, expected in (("df_residual", n-7), ("df_pvalue", n-7 if method == "qr_t" else n-5)):
        require(type(result[key]) is int and result[key] > 0, "degrees of freedom must be positive integers")
        if method != "r":
            require(result[key] == expected, "Python/QR degrees of freedom differ from protocol")


def wilson(count, denominator):
    if not denominator:
        return None
    z = NormalDist().inv_cdf(0.975)
    z2 = z*z
    # Count-based form, algebraically equivalent to the probability form.
    center = (count+z2/2)/(denominator+z2)
    radius = z*math.sqrt(count*(denominator-count)/denominator+z2/4)/(denominator+z2)
    return [max(0.0, center-radius), min(1.0, center+radius)]


class MethodTotals:
    """Welford moments and scalar sums; retain no individual fitted results."""

    def __init__(self, truth):
        self.truth = truth
        self.attempted = self.successful = self.covered = self.rejected = 0
        self.mean_error = self.m2 = self.sum_variance = self.sum_se = self.sum_width = self.sum_sq_error = 0.0
        self.failure_classes, self.failure_codes, self.warnings = Counter(), Counter(), Counter()

    def add(self, entry, method):
        self.attempted += 1
        warnings = entry.get("warnings")
        require(isinstance(warnings, list) and all(isinstance(item, str) for item in warnings),
                "method warnings must be an array of strings")
        self.warnings.update(warnings)
        if entry.get("status") == "error":
            require(isinstance(entry.get("error"), str) and bool(entry["error"]), "method failure lacks explanation")
            self.failure_classes[entry.get("error_class", "UnclassifiedError")] += 1
            self.failure_codes[entry.get("error_code", "r_case_failure" if method == "r" else "unclassified_failure")] += 1
            return
        require(entry.get("status") == "ok", "unknown terminal method status")
        result = entry["result"]
        self.successful += 1
        error = result["psi"]-self.truth
        delta = error-self.mean_error
        self.mean_error += delta/self.successful
        self.m2 += delta*(error-self.mean_error)
        self.sum_sq_error += error*error
        self.sum_variance += result["variance"]
        self.sum_se += result["se"]
        self.sum_width += result["ci"][1]-result["ci"][0]
        self.covered += int(result["ci"][0] <= self.truth <= result["ci"][1])
        self.rejected += int(result["p_value"] < 0.05)

    def summary(self, planned):
        m = self.successful
        failed = sum(self.failure_classes.values())
        missing = planned-m-failed
        require(0 <= missing <= planned and self.attempted <= planned, "invalid method denominators")
        sd = math.sqrt(max(0.0, self.m2)/(m-1)) if m >= 2 else None
        bias = self.mean_error if m else None
        mcse = sd/math.sqrt(m) if sd is not None else None
        interval = wilson(self.covered, m)
        return {
            "planned": planned, "attempted": self.attempted, "successful": m, "failed": failed,
            "missing": missing, "unattempted": planned-self.attempted,
            "failure_classes": dict(self.failure_classes), "failure_codes": dict(self.failure_codes),
            "failed_fraction_planned": failed/planned, "missing_fraction_planned": missing/planned,
            "bias": bias, "bias_mcse": mcse, "empirical_sd": sd,
            "rms_se": math.sqrt(self.sum_variance/m) if m else None,
            "mean_interval_width": self.sum_width/m if m else None,
            "covered": self.covered, "coverage_successful": self.covered/m if m else None,
            "coverage_wilson": interval, "covered_fraction_planned": self.covered/planned,
            "coverage_all_planned_bounds": [self.covered/planned, (self.covered+failed+missing)/planned],
            "rejected": self.rejected,
            "rejection_kind": "false_positive_rate" if self.truth == 0 else "power",
            "rejection_successful": self.rejected/m if m else None,
            "rejection_wilson": wilson(self.rejected, m), "rejected_fraction_planned": self.rejected/planned,
            "diagnostics_only": {
                "bias_exceeds_3_mcse": abs(bias) > 3*mcse if mcse is not None else None,
                "coverage_wilson_outside_093_097": (interval[1] < 0.93 or interval[0] > 0.97) if interval else None,
            },
        }

    def additional(self):
        m = self.successful
        return {
            "mean_se": self.sum_se/m if m else None,
            "rmse": math.sqrt(self.sum_sq_error/m) if m else None,
            "coverage_mcse": math.sqrt((self.covered/m)*(1-self.covered/m)/m) if m else None,
            "rejection_mcse": math.sqrt((self.rejected/m)*(1-self.rejected/m)/m) if m else None,
            "warning_counts": dict(self.warnings),
        }


def compare_methods(methods, maxima, case_name):
    expected = []
    for other in ("r", "qr_t"):
        name = f"python_vs_{other}"
        row = {"comparison": name, "status": "unavailable", "fields": [], "differences": {}}
        if methods["python"]["status"] == methods[other]["status"] == "ok":
            left, right = methods["python"]["result"], methods[other]["result"]
            fields = ["psi", "variance", "se"]
            if other == "r":
                fields += ["ci", "p_value", "df_pvalue", "df_residual"]
            for field in fields:
                pairs = list(zip(left[field], right[field])) if field == "ci" else [(left[field], right[field])]
                equal = True
                for index, (actual, reference) in enumerate(pairs):
                    difference = abs(actual-reference)
                    threshold = (0 if field.startswith("df_") else
                                 ATOL+(0 if field == "p_value" else RTOL*abs(reference)))
                    equal = equal and difference <= threshold
                    key = f"{name}:{field}" + (f"_{('lower', 'upper')[index]}" if field == "ci" else "")
                    maximum = maxima.setdefault(key, {"absolute_difference": -1.0, "case": None,
                                                       "maximum_tolerance_ratio": 0.0})
                    if difference > maximum["absolute_difference"]:
                        maximum.update(absolute_difference=difference, case=case_name)
                    if threshold:
                        maximum["maximum_tolerance_ratio"] = max(maximum["maximum_tolerance_ratio"], difference/threshold)
                if not equal:
                    row["fields"].append(field)
                    row["differences"][field] = {"python": left[field], "other": right[field]}
            row["status"] = "mismatch" if row["fields"] else "agree"
        expected.append(row)
    return expected


def check_case(record, cell, repetition, totals, paired, conventions, maxima):
    keys(record, ("name", "cell", "repetition", "seed", "inputs", "input_sha256", "methods", "comparisons"), "case")
    name = f"study-{cell['id']}-r{repetition:05d}"
    require(record["name"] == name and record["cell"] == cell["id"], "case name or cell differs from its exact batch position")
    require(type(record["repetition"]) is int and record["repetition"] == repetition, f"{name}: repetition differs")
    require(canonical(record["seed"]) == canonical([20261022, 2, cell["index"], repetition]), f"{name}: seed identity differs")
    require(record["input_sha256"] == digest(record["inputs"]), f"{name}: input hash differs")
    check_inputs(record["inputs"], cell["n"])
    methods = record["methods"]
    keys(methods, METHODS, f"{name}: methods")
    for method in METHODS:
        entry = methods[method]
        require(isinstance(entry, dict), f"{name}: malformed method entry")
        if entry.get("status") == "ok":
            check_result(entry.get("result"), cell["n"], method)
        totals[method].add(entry, method)
    r_entry = methods["r"]
    if r_entry["status"] == "ok":
        conventions["checked"] += 1
        result = r_entry["result"]
        differences = {key: {"expected": value, "observed": result[key]} for key, value in (
            ("df_pvalue", cell["n"]-5), ("df_residual", cell["n"]-7)) if result[key] != value}
        require(r_entry.get("convention_differences", {}) == differences, f"{name}: saved R convention check differs")
        if differences:
            conventions["mismatch"] += 1
            conventions["mismatching_cases"].append(name)
    comparisons = compare_methods(methods, maxima, name)
    require(record["comparisons"] == comparisons, f"{name}: saved numerical comparison differs")
    for comparison in comparisons:
        counts = paired[comparison["comparison"]]
        counts[comparison["status"]] += 1
        if comparison["status"] == "mismatch":
            counts["mismatching_cases"].append(name)


def check_infrastructure(data):
    """An infrastructure failure must remain the same terminal R failure."""
    error = data["infrastructure_error"]
    if error is None:
        return
    keys(error, ("status", "error_class", "error_code", "error", "warnings"), "batch infrastructure failure")
    require(error["status"] == "error", "invalid batch infrastructure status")
    for key in ("error_class", "error_code", "error"):
        require(isinstance(error[key], str) and bool(error[key]), "missing infrastructure failure diagnostics")
    require(isinstance(error["warnings"], list) and all(isinstance(value, str) for value in error["warnings"]),
            "invalid infrastructure warning categories")
    for record in data["records"]:
        require(isinstance(record, dict) and isinstance(record.get("methods"), dict)
                and record["methods"].get("r") == error,
                "batch infrastructure failure conflicts with retained R result")


def verify(run):
    require(run.is_dir() and not run.is_symlink(), "run directory is absent or a symlink")
    writer = run.with_name(run.name + ".writer.lock")
    require(not writer.exists() and not writer.is_symlink(), "writer lock is present; inspect completion before verification")
    paths = list(run.rglob("*"))
    require(not any(path.is_symlink() for path in paths), "symlink inside run evidence")
    total_bytes = sum(path.stat().st_size for path in paths if path.is_file())
    manifest, manifest_file = read_json(run / "manifest.json", with_identity=True)
    cells = check_manifest(manifest)
    summary, summary_file = read_json(run / "summary.json", with_identity=True)
    batch_directory = run / "batches"
    require(batch_directory.is_dir(), "batch directory is absent")
    batch_names = sorted(path.name for path in batch_directory.iterdir())
    require(batch_names == [f"batch-{index:06d}.json" for index in range(1200)],
            "expected exactly 1200 completed batch files, with no pending, temporary or unknown files")
    stats = [{method: MethodTotals(cell["true_psi"]) for method in METHODS} for cell in cells]
    paired = [{name: {"agree": 0, "mismatch": 0, "unavailable": 0, "missing": 0,
                      "planned": 2000, "mismatching_cases": []}
               for name in ("python_vs_r", "python_vs_qr_t")} for _ in cells]
    conventions = [{"checked": 0, "mismatch": 0, "unavailable": 0, "mismatching_cases": []} for _ in cells]
    maxima, inventory = {}, []
    cases = 0
    elapsed_batches = 0.0
    infrastructure_batches = []
    for index, filename in enumerate(batch_names):
        path = batch_directory / filename
        data, batch_file = read_json(path, with_identity=True)
        keys(data, ("schema_version", "identity_sha256", "batch_index", "status", "records",
                    "elapsed_seconds", "infrastructure_error"), f"batch {index}")
        require(type(data["batch_index"]) is int and data["batch_index"] == index,
                f"batch {index}: index differs")
        require(data["schema_version"] == 1 and data["identity_sha256"] == manifest["identity_sha256"]
                and data["status"] == "completed", f"batch {index}: identity or status differs")
        number(data["elapsed_seconds"], "batch elapsed time")
        require(data["elapsed_seconds"] >= 0, "negative batch elapsed time")
        elapsed_batches += data["elapsed_seconds"]
        require(isinstance(data["records"], list) and len(data["records"]) == 20,
                f"batch {index}: expected twenty records")
        check_infrastructure(data)
        if data["infrastructure_error"] is not None:
            infrastructure_batches.append(index)
        for within, record in enumerate(data["records"]):
            position = 20*index+within
            cell_index, repetition = divmod(position, 2000)
            check_case(record, cells[cell_index], repetition, stats[cell_index], paired[cell_index],
                       conventions[cell_index], maxima)
            cases += 1
        inventory.append({"name": filename, **batch_file})
        del data
    cell_summaries, additional = [], []
    failures = mismatches = 0
    for index, cell in enumerate(cells):
        methods = {method: stats[index][method].summary(2000) for method in METHODS}
        failures += sum(method["failed"] for method in methods.values())
        mismatches += sum(row["mismatch"] for row in paired[index].values())+conventions[index]["mismatch"]
        conventions[index]["unavailable"] = 2000-conventions[index]["checked"]
        cell_summaries.append({"cell": cell["id"], "true_psi": cell["true_psi"], "analytic": cell["analytic"],
                               "methods": methods, "paired": paired[index], "reference_conventions": conventions[index]})
        additional.append({"cell": cell["id"], "methods": {method: stats[index][method].additional() for method in METHODS}})
    expected_summary = {
        "schema_version": 1, "mode": "study", "identity_sha256": manifest["identity_sha256"],
        "status": "completed_with_failures_or_mismatches" if failures or mismatches else "complete_numerical_agreement",
        "execution_complete": True, "calibration_validated": False, "planned": 24000,
        "attempted": cases, "terminal": cases, "missing": 0, "completed_batches": 1200,
        "cells": cell_summaries, "peak_memory_bytes": None,
        "interpretation": "Diagnostics are descriptive, not a calibration pass/fail or release decision.",
    }
    keys(summary, set(expected_summary) | {"output_bytes"}, "summary")
    require(type(summary["output_bytes"]) is int and 0 < summary["output_bytes"] <= total_bytes,
            "saved output byte count is impossible")
    same({key: value for key, value in summary.items() if key != "output_bytes"}, expected_summary, "summary")
    return {
        "schema_version": 1, "status": "verified_complete_study_records",
        "calibration_validated": False, "study_status": expected_summary["status"],
        "checker_sha256": file_hash(Path(__file__)), "manifest_sha256": manifest_file["sha256"],
        "summary_sha256": summary_file["sha256"], "identity_sha256": manifest["identity_sha256"],
        "source_delivery_revision": "8cb4619934e1cf02ed8ade3b09f09b25115f9381",
        "planned_cases": 24000, "verified_unique_expected_cases": cases,
        "completed_batches": 1200, "seed_namespace": 2, "master_seed": 20261022,
        "retained_input_hashes_checked": cases, "method_failures": failures,
        "paired_and_convention_mismatches": mismatches,
        "batches_with_infrastructure_errors": infrastructure_batches,
        "maximum_absolute_differences": maxima,
        "summary_reconciliation": {"status": "all stored inferential and count fields agree",
                                   "absolute_tolerance": SUMMARY_ATOL, "relative_tolerance": SUMMARY_RTOL},
        "cells": cell_summaries, "additional_descriptive_metrics": additional,
        "batch_elapsed_seconds_sum": elapsed_batches, "observed_directory_bytes": total_bytes,
        "batch_inventory_sha256": digest(inventory), "batch_inventory": inventory,
        "limits": [
            "No model was refitted and no data were regenerated. Input hashes check retained bytes, not generator correctness or PRNG independence.",
            "This checks retained Python/R/QR outputs. It does not independently prove R parsed the identical arrays or restore the server environment.",
            "Analytic targets are checked against protocol values rounded to eight decimal places, not recomputed from a distribution library.",
            "The saved output_bytes value precedes summary/timing writes. Its range is checked; its historical exact value and external memory/runtime logs are not reconciled.",
            "Mean SE, RMSE and binomial MCSE are extra descriptive calculations; they are not stored production summary fields or additional release criteria.",
            "Successful arithmetic and numerical agreement do not establish interval calibration, causal validity, generalizability or release readiness.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("new_report", type=Path)
    args = parser.parse_args(argv)
    run = args.run_directory.absolute()
    report = args.new_report.absolute()
    require(not report.exists() and not report.is_symlink(), "report must be a new file")
    require(run.resolve() not in report.resolve().parents, "report must be outside the immutable run directory")
    require(report.parent.is_dir(), "report parent directory must already exist")
    try:
        result = verify(run)
        code = 0
    except (EvidenceError, OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        result = {"schema_version": 1, "status": "verification_failed", "calibration_validated": False,
                  "checker_sha256": file_hash(Path(__file__)), "error_class": type(exc).__name__, "error": str(exc)}
        code = 2
    with report.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "report": str(report),
                      **({"error": result["error"]} if code else {})}))
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EvidenceError, OSError) as error:
        print(json.dumps({"status": "verification_not_started", "error": str(error)}), file=sys.stderr)
        raise SystemExit(2)
