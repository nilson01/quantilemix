# SPDX-License-Identifier: GPL-2.0-or-later
"""Prospective Gaussian known-score calibration; importing never runs a study.

The adjacent frozen protocol/configuration define the only supported design.
``plan`` is read-only and the default. Pilot/study require explicit execution,
an installed R reference, and a fresh directory (or explicit identity-checked
resume). Software tests are not calibration evidence. Tolerances are frozen:
absolute 1e-10 / relative 1e-8 for estimates and uncertainty; absolute 1e-10
for ordinary Python/R p-values; exact degrees of freedom. The QR comparator
uses full-model t inference, so its interval and p-value intentionally differ.

Atomic pending files retain all inputs before fitting. A completed checkpoint
is never refitted, including failures. Pending batches may be replayed with
the same inputs. Attempted means admitted to a persisted batch; missing means
no terminal method result, including interruption after admission. Execution
completion and numerical agreement are reported separately from calibration.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
import warnings

import numpy as np
import scipy
from scipy import linalg, special, stats

import quantilemix as package
import quantilemix._core as core
from quantilemix import fit


ATOL, RTOL = 1e-10, 1e-8
METHODS = ("python", "r", "qr_t")
BETA_NAMES = ("null", "cancellation", "positive")
HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "calibration-config.json"
RESULT_FIELDS = {"psi", "variance", "se", "ci", "p_value", "df_pvalue", "df_residual"}
THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


class CalibrationError(ValueError):
    """Invalid design, evidence, or execution prerequisite."""


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise CalibrationError("duplicate JSON key")
        result[key] = value
    return result


def _finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise CalibrationError("nonfinite JSON number")
    if isinstance(value, dict):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            _finite_tree(item)


def load_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(
                               CalibrationError("nonfinite JSON number")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CalibrationError("cannot read valid JSON evidence") from exc
    _finite_tree(value)
    return value


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def digest(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def _keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise CalibrationError(f"{label}: fields differ from the frozen schema")


def _integer(value, label, minimum=1):
    if type(value) is not int or value < minimum:
        raise CalibrationError(f"{label}: expected an integer >= {minimum}")


def _number(value, label):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise CalibrationError(f"{label}: expected a finite real number")
    return float(value)


def validate_config(config):
    """Reject schema drift and unreviewed designs; seeds/batch budget are explicit."""
    _keys(config, ("schema_version", "protocol_id", "reference_version", "target",
                   "alpha", "design", "study", "pilot", "execution"), "config")
    fixed = {"schema_version": 1, "protocol_id": "gaussian-known-scores-v1",
             "reference_version": "2.19.6", "target": "known_scores", "alpha": .05}
    for key, value in fixed.items():
        if type(config[key]) is not type(value) or config[key] != value:
            raise CalibrationError(f"config.{key}: unsupported protocol")
    design = {"sample_sizes": [100, 500], "rho": [0, .75],
              "beta": {"null": [0, 0, 0, 0], "cancellation": [.25, -.25, 0, 0],
                       "positive": [.1, .1, .05, 0]}, "intercept": 1,
              "covariate_coefficients": [.5, -.25], "exposure_z1_loading": .5,
              "error_sd": 1, "quartile_variance": 1.25}
    # Canonical serialization distinguishes bools from numbers, unlike ==.
    if _bytes(config["design"]) != _bytes(design):
        raise CalibrationError("design differs from the reviewed fixed design")
    _keys(config["study"], ("replicates_per_cell", "seed_namespace"), "study")
    _keys(config["pilot"], ("replicates", "sample_size", "rho", "beta_name", "seed_namespace"), "pilot")
    if type(config["study"]["replicates_per_cell"]) is not int or config["study"]["replicates_per_cell"] != 2000:
        raise CalibrationError("study requires 2000 replicates per cell")
    for key, value in {"replicates": 20, "sample_size": 500, "rho": .75,
                       "beta_name": "positive"}.items():
        if type(config["pilot"][key]) is not type(value) or config["pilot"][key] != value:
            raise CalibrationError("pilot differs from the reviewed fixed pilot")
    for mode in ("pilot", "study"):
        _integer(config[mode]["seed_namespace"], "seed namespace")
    if config["pilot"]["seed_namespace"] == config["study"]["seed_namespace"]:
        raise CalibrationError("pilot and study seed namespaces must be disjoint")
    execution = config["execution"]
    _keys(execution, ("batch_size", "master_seed", "retain_inputs", "storage_budget_bytes"), "execution")
    for key in ("batch_size", "master_seed", "storage_budget_bytes"):
        _integer(execution[key], key)
    if execution["batch_size"] > 20 or execution["retain_inputs"] is not True:
        raise CalibrationError("batches must be <=20 and inputs must be retained")
    if execution["storage_budget_bytes"] > 3_000_000_000:
        raise CalibrationError("the reviewed output ceiling is 3 GB")
    return config


def analytic_targets(n, alpha=.05):
    residual_df, reference_df = n - 7, n - 5
    return {"condition": "correct normal homoscedastic model and full-rank design",
            "df_residual": residual_df, "df_reference_pvalue": reference_df,
            "reference_coverage": float(2 * stats.t.cdf(stats.norm.ppf(1-alpha/2), residual_df)-1),
            "reference_null_rejection": float(2 * stats.t.sf(stats.t.ppf(1-alpha/2, reference_df), residual_df)),
            "qr_t_coverage": 1-alpha, "qr_t_null_rejection": alpha}


def make_plan(config, mode="study"):
    validate_config(config)
    if mode not in ("study", "pilot"):
        raise CalibrationError("plan mode must be pilot or study")
    cells = []
    for n in config["design"]["sample_sizes"]:
        for rho in config["design"]["rho"]:
            for beta_name in BETA_NAMES:
                index = len(cells)
                beta = config["design"]["beta"][beta_name]
                cells.append({"id": f"n{n}-rho{str(rho).replace('.', 'p')}-{beta_name}",
                              "index": index, "n": n, "rho": rho, "beta_name": beta_name,
                              "beta": beta, "true_psi": sum(beta),
                              "planned": config["study"]["replicates_per_cell"],
                              "analytic": analytic_targets(n, config["alpha"])})
    if mode == "pilot":
        cells = [dict(cells[-1], planned=config["pilot"]["replicates"])]
    planned = sum(cell["planned"] for cell in cells)
    # Conservative planning envelope, not a guarantee or a disk reservation.
    # 256 bytes/row for seven JSON values and delimiters; 16 KiB/result metadata.
    upper = sum(cell["planned"] * (256 * cell["n"] + 16384) for cell in cells)
    return {"schema_version": 1, "protocol_id": config["protocol_id"], "mode": mode,
            "cells": cells, "planned": planned, "batch_size": config["execution"]["batch_size"],
            "input_storage_estimate_upper_bytes": upper,
            "storage_budget_bytes": config["execution"]["storage_budget_bytes"],
            "storage_estimate_within_budget": upper <= config["execution"]["storage_budget_bytes"],
            "execution_performed": False, "calibration_validated": False,
            "tolerances": {"atol": ATOL, "rtol": RTOL, "p_value_atol": ATOL},
            "peak_memory_bytes": None}


def _case_name(mode, cell, repetition):
    return f"{mode}-{cell['id']}-r{repetition:05d}"


def generate_case(config, mode, cell, repetition):
    """Generate one case by stable ID; independent of completion/batch order."""
    validate_config(config)
    if mode not in ("pilot", "study") or not any(cell == c for c in make_plan(config, mode)["cells"]):
        raise CalibrationError("unknown design cell")
    _integer(repetition, "repetition", 0)
    if repetition >= cell["planned"]:
        raise CalibrationError("repetition is outside the frozen plan")
    seed = [config["execution"]["master_seed"], config[mode]["seed_namespace"], cell["index"], repetition]
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    normal = rng.standard_normal((cell["n"], 8))
    z = normal[:, :2]
    latent = .5*z[:, 0, None] + math.sqrt(cell["rho"])*normal[:, 2, None] + math.sqrt(1-cell["rho"])*normal[:, 3:7]
    cuts = math.sqrt(1.25) * special.ndtri([.25, .5, .75])
    scores = np.searchsorted(cuts, latent, side="left")
    y = 1 + scores @ np.asarray(cell["beta"]) + z @ np.array([.5, -.25]) + normal[:, 7]
    inputs = {"exposures": scores.tolist(), "covariates": z.tolist(), "y": y.tolist(), "q": None, "alpha": .05}
    return {"name": _case_name(mode, cell, repetition), "cell": cell["id"],
            "repetition": repetition, "seed": seed, "inputs": inputs, "input_sha256": digest(inputs)}


def _arrays(inputs):
    _keys(inputs, ("exposures", "covariates", "y", "q", "alpha"), "inputs")
    if inputs["q"] is not None or type(inputs["alpha"]) is not float or inputs["alpha"] != .05:
        raise CalibrationError("only prequantized alpha=.05 inputs are supported")
    def convert(value):
        # Reject bool/null and strings before float conversion.
        def check(part):
            if isinstance(part, list):
                for element in part:
                    check(element)
            else:
                _number(part, "input array")
        check(value)
        try:
            return np.asarray(value, dtype=float)
        except (ValueError, TypeError) as exc:
            raise CalibrationError("nonrectangular input array") from exc
    x, z, y = [convert(inputs[key]) for key in ("exposures", "covariates", "y")]
    if y.ndim != 1 or x.shape != (len(y), 4) or z.shape != (len(y), 2) or len(y) <= 7:
        raise CalibrationError("expected n>7, four scores and two covariates")
    if np.any((x < 0) | (x > 3) | (x != np.floor(x))):
        raise CalibrationError("exposure scores must be integers 0..3")
    return x, z, y


def validate_result(result, n, method):
    _keys(result, RESULT_FIELDS, "method result")
    for key in ("psi", "variance", "se", "p_value"):
        _number(result[key], key)
    if result["variance"] <= 0 or result["se"] <= 0 or not 0 <= result["p_value"] <= 1:
        raise CalibrationError("degenerate or invalid uncertainty")
    if not isinstance(result["ci"], list) or len(result["ci"]) != 2:
        raise CalibrationError("confidence interval must have two endpoints")
    for value in result["ci"]:
        _number(value, "interval endpoint")
    if not result["ci"][0] <= result["psi"] <= result["ci"][1]:
        raise CalibrationError("confidence interval does not enclose its estimate")
    for key, expected in (("df_residual", n-7), ("df_pvalue", n-7 if method == "qr_t" else n-5)):
        _integer(result[key], key)
        if method != "r" and result[key] != expected:
            raise CalibrationError("incorrect degrees of freedom")
    if not math.isclose(result["se"]**2, result["variance"], abs_tol=ATOL, rel_tol=RTOL):
        raise CalibrationError("SE and variance disagree")
    return result


def native_result(inputs):
    x, z, y = _arrays(inputs)
    fitted = fit(y, x, z, q=None, alpha=inputs["alpha"])
    return {"psi": fitted.psi, "variance": fitted.var_psi, "se": fitted.se,
            "ci": list(fitted.ci), "p_value": fitted.p_value,
            "df_pvalue": fitted.p_value_df, "df_residual": fitted.residual_df}


def qr_result(inputs):
    """Independent QR contrast and full-model Student t inference."""
    x, z, y = _arrays(inputs)
    design = np.column_stack((np.ones(len(y)), x, z))
    q_matrix, r_matrix = np.linalg.qr(design, mode="reduced")
    if np.linalg.matrix_rank(r_matrix) != design.shape[1]:
        raise CalibrationError("QR design is rank deficient")
    coefficients = linalg.solve_triangular(r_matrix, q_matrix.T @ y)
    residuals = y-design @ coefficients
    df = len(y)-design.shape[1]
    contrast = np.array([0., 1., 1., 1., 1., 0., 0.])
    factor = linalg.solve_triangular(r_matrix.T, contrast, lower=True)
    variance = float((residuals @ residuals)/df * (factor @ factor))
    psi, se = float(contrast @ coefficients), math.sqrt(variance)
    if se <= 0:
        raise CalibrationError("QR uncertainty is degenerate")
    critical = float(stats.t.ppf(1-inputs["alpha"]/2, df))
    return {"psi": psi, "variance": variance, "se": se,
            "ci": [psi-critical*se, psi+critical*se],
            "p_value": float(2*stats.t.sf(abs(psi/se), df)), "df_pvalue": df, "df_residual": df}


ERROR_MESSAGES = {
    "r_invalid_configuration": "The R executable or isolated library configuration is invalid; check the supplied paths.",
    "r_missing_executable": "Rscript is unavailable; check the configured executable.",
    "r_invalid_library": "The isolated R library is unavailable; check the configured library.",
    "r_missing_driver": "The R calibration driver is missing from the development files.",
    "r_startup_failure": "R startup or dependency loading failed; verify the isolated installation.",
    "r_timeout": "The R process exceeded its time limit; inspect the retained batch and resource limits.",
    "r_io_error": "The R process or temporary file operation failed; check executable and filesystem access.",
    "r_invalid_response": "The R response failed validation; inspect reference version, schema and finite outputs.",
    "dependency_unavailable": "A required executable or file is unavailable; check the configured dependencies.",
    "rank_deficient": "The design is rank deficient; inspect the retained design matrix.",
    "invalid_numerical_result": "Numerical uncertainty is invalid or degenerate; inspect the retained inputs and scale.",
    "linear_algebra_failure": "The linear algebra operation failed; inspect the retained design and scale.",
    "invalid_evidence": "Input, result or checkpoint validation failed; inspect the retained evidence.",
    "unclassified_failure": "An unclassified method or infrastructure failure occurred; inspect retained inputs.",
}


def _error(exc, caught=(), *, reference=False):
    # Never persist arbitrary exception text, paths or input values. Adapter
    # codes and known numerical failures map to bounded actionable messages.
    code = getattr(exc, "code", None)
    if code not in ERROR_MESSAGES:
        message = str(exc).lower()
        if isinstance(exc, FileNotFoundError):
            code = "dependency_unavailable"
        elif reference:
            code = "r_invalid_response"
        elif isinstance(exc, np.linalg.LinAlgError):
            code = "linear_algebra_failure"
        elif "rank deficient" in message or "rank-deficient" in message:
            code = "rank_deficient"
        elif any(term in message for term in ("degenerate", "variance", "nonfinite", "double-precision")):
            code = "invalid_numerical_result"
        elif isinstance(exc, CalibrationError):
            code = "invalid_evidence"
        else:
            code = "unclassified_failure"
    return {"status": "error", "error_class": type(exc).__name__, "error_code": code,
            "error": ERROR_MESSAGES[code], "warnings": [w.category.__name__ for w in caught]}


def validate_reference(reference):
    if not isinstance(reference, dict):
        raise CalibrationError("missing installed R identity")
    for key, value in {"package": "qgcomp", "version": "2.19.6", "mode": "installed package", "library_verified": True}.items():
        if type(reference.get(key)) is not type(value) or reference[key] != value:
            raise CalibrationError("unverified installed R identity")
    for key in ("R_version", "platform"):
        if not isinstance(reference.get(key), str) or not reference[key]:
            raise CalibrationError("incomplete R runtime identity")
    for key in ("package_versions", "installed_packages", "package_library_origins"):
        if not isinstance(reference.get(key), dict) or not reference[key]:
            raise CalibrationError("missing R dependency identity")
    if reference.get("unexpected_loaded_namespaces") != []:
        raise CalibrationError("unexpected R namespaces")
    for key in ("installed_package_md5", "r_runtime_md5"):
        hashes = reference.get(key)
        if not isinstance(hashes, dict) or not hashes:
            raise CalibrationError("missing R code fingerprints")
        if key == "installed_package_md5" and set(hashes) != {"DESCRIPTION", "NAMESPACE", "R/qgcomp.rdb", "R/qgcomp.rdx"}:
            raise CalibrationError("incomplete R package fingerprints")
        for name, value in hashes.items():
            if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", name)
                    or any(part in (".", "..") for part in name.split("/"))
                    or not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{32}", value)):
                raise CalibrationError("invalid R code fingerprint")
    _finite_tree(reference)
    return reference


def validate_r_batch(payload, cases, reference=None):
    _keys(payload, ("reference", "cases"), "R response")
    validate_reference(payload["reference"])
    if reference is not None and payload["reference"] != reference:
        raise CalibrationError("R identity changed during execution")
    rows = payload["cases"]
    if not isinstance(rows, list) or len(rows) != len(cases):
        raise CalibrationError("R response has missing or extra cases")
    output = {}
    expected = {case["name"]: case for case in cases}
    for row in rows:
        if not isinstance(row, dict) or row.get("name") not in expected or row["name"] in output:
            raise CalibrationError("R response has duplicate or unknown case identity")
        if not isinstance(row.get("warnings"), list) or not all(isinstance(w, str) for w in row["warnings"]):
            raise CalibrationError("R response warnings must be an array of strings")
        if row.get("status") == "ok":
            _keys(row, ("name", "status", "result", "warnings"), "R successful case")
            validate_result(row.get("result"), len(expected[row["name"]]["inputs"]["y"]), "r")
        elif row.get("status") == "error":
            _keys(row, ("name", "status", "error", "error_class", "warnings"), "R failed case")
            if (not isinstance(row.get("error"), str) or not row["error"]
                    or not isinstance(row.get("error_class"), str) or not row["error_class"]):
                raise CalibrationError("R failure lacks diagnostics")
        else:
            raise CalibrationError("unknown R case status")
        output[row["name"]] = {key: value for key, value in row.items() if key != "name"}
        if row["status"] == "ok":
            output[row["name"]]["convention_differences"] = reference_convention_differences(
                row["result"], len(expected[row["name"]]["inputs"]["y"]))
    return output


def reference_convention_differences(result, n):
    """Check R against its specified convention even when another method fails."""
    expected = {"df_pvalue": n-5, "df_residual": n-7}
    return {key: {"expected": value, "observed": result[key]}
            for key, value in expected.items() if result[key] != value}


def comparisons(methods):
    output = []
    for other in ("r", "qr_t"):
        row = {"comparison": f"python_vs_{other}", "status": "unavailable", "fields": [], "differences": {}}
        if methods.get("python", {}).get("status") == methods.get(other, {}).get("status") == "ok":
            left, right = methods["python"]["result"], methods[other]["result"]
            fields = ["psi", "variance", "se"] + (["ci", "p_value", "df_pvalue", "df_residual"] if other == "r" else [])
            for key in fields:
                equal = (left[key] == right[key] if key.startswith("df_") else
                         np.allclose(left[key], right[key], atol=ATOL, rtol=0 if key == "p_value" else RTOL))
                if not equal:
                    row["fields"].append(key)
                    row["differences"][key] = {"python": left[key], "other": right[key]}
            row["status"] = "mismatch" if row["fields"] else "agree"
        output.append(row)
    return output


def evaluate_batch(cases, adapter, reference, *, rscript, library, timeout=180):
    """Evaluate literal cases independently; malformed R output is infrastructure failure.

    Returns all valid Python/QR results even on R infrastructure failure. This
    is a software-test seam, not permission to shorten the frozen study.
    """
    if not cases or len(cases) > 20 or len({c["name"] for c in cases}) != len(cases):
        raise CalibrationError("batch must contain 1..20 unique cases")
    records = []
    for case in cases:
        if digest(case["inputs"]) != case["input_sha256"]:
            raise CalibrationError("paired input hash mismatch")
        _arrays(case["inputs"])
        methods = {}
        for method, operation in (("python", native_result), ("qr_t", qr_result)):
            caught = []
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    result = operation(case["inputs"])
                validate_result(result, len(case["inputs"]["y"]), method)
                methods[method] = {"status": "ok", "result": result,
                                   "warnings": [w.category.__name__ for w in caught]}
            except Exception as exc:
                methods[method] = _error(exc, caught)
        records.append(dict(case, methods=methods))
    infrastructure_error = None
    try:
        request = [{"name": c["name"], "inputs": c["inputs"]} for c in cases]
        payload = adapter(request, rscript, library, timeout=timeout)
        r_results = validate_r_batch(payload, cases, reference)
    except Exception as exc:
        infrastructure_error = _error(exc, reference=True)
        r_results = {case["name"]: infrastructure_error for case in cases}
    for record in records:
        record["methods"]["r"] = r_results[record["name"]]
        record["comparisons"] = comparisons(record["methods"])
    return {"records": records, "infrastructure_error": infrastructure_error}


def _load_adapter():
    spec = importlib.util.spec_from_file_location("qgcomp_calibration_reference", HERE / "calibration_reference.py")
    if spec is None or spec.loader is None:
        raise CalibrationError("R adapter is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate_r_batch


def build_identity(config, mode, reference):
    validate_config(config)
    validate_reference(reference)
    sources = {"calibration.py": Path(__file__), "qgcomp/__init__.py": Path(package.__file__),
               "qgcomp/_core.py": Path(core.__file__),
               "calibration_reference.py": HERE / "calibration_reference.py", "calibration_r.R": HERE / "calibration_r.R"}
    try:
        hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in sources.items()}
    except OSError as exc:
        raise CalibrationError("required source identity is unavailable") from exc
    uname = os.uname() if hasattr(os, "uname") else None
    environment = {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                   "platform": sys.platform, "machine": uname.machine if uname else os.environ.get("PROCESSOR_ARCHITECTURE", "unknown"),
                   "threads": {key: os.environ.get(key) for key in THREAD_VARIABLES}}
    # Build names/versions are useful without recording build-machine paths.
    config_dict = np.__config__.show(mode="dicts")
    environment["numerical_libraries"] = {
        key: {field: value for field, value in detail.items() if field in ("name", "version", "openblas configuration")}
        for key, detail in config_dict.get("Build Dependencies", {}).items() if isinstance(detail, dict)}
    return {"config_sha256": digest(config), "config": config, "mode": mode,
            "sources_sha256": hashes, "environment": environment, "reference": reference}


def _directory_bytes(directory):
    total = 0
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise CalibrationError("symlinks are not valid run evidence")
        if path.is_file():
            total += path.stat().st_size
    return total


def atomic_json(path, payload, budget):
    """Atomic replacement; only derived/pending files may be replaced by callers."""
    path = Path(path)
    data = _bytes(payload) + b"\n"
    root = path.parent.parent if path.parent.name == "batches" else path.parent
    if path.is_symlink() or _directory_bytes(root) + len(data) > budget:
        raise CalibrationError("unsafe output path or storage ceiling reached")
    temporary = path.with_name(path.name + ".tmp")
    if temporary.is_symlink():
        raise CalibrationError("unsafe temporary output path")
    # A stale .tmp is ignored evidence; replace it only with the same named operation.
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def prepare_directory(directory, identity, plan, resume=False):
    directory = Path(directory)
    if directory.is_symlink():
        raise CalibrationError("output directory cannot be a symlink")
    manifest = {"schema_version": 1, "identity": identity, "identity_sha256": digest(identity), "plan": plan}
    if resume:
        old = load_json(directory / "manifest.json")
        if old != manifest:
            raise CalibrationError("resume identity or frozen plan mismatch")
        _directory_bytes(directory)
    else:
        try:
            directory.mkdir(parents=False, exist_ok=False)
        except OSError as exc:
            raise CalibrationError("execution requires a fresh output directory") from exc
        (directory / "batches").mkdir()
        atomic_json(directory / "manifest.json", manifest, plan["storage_budget_bytes"])
    return manifest


def _expected_cases(manifest):
    output = {}
    config, plan = manifest["identity"]["config"], manifest["plan"]
    for cell in plan["cells"]:
        for repetition in range(cell["planned"]):
            name = _case_name(plan["mode"], cell, repetition)
            output[name] = (cell, repetition, [config["execution"]["master_seed"],
                            config[plan["mode"]]["seed_namespace"], cell["index"], repetition])
    return output


def read_checkpoints(directory, manifest):
    """Validate identity, names, inputs and terminal results before aggregation/resume."""
    directory = Path(directory)
    _directory_bytes(directory)
    expected = _expected_cases(manifest)
    ordered_names = list(expected)
    size = manifest["plan"]["batch_size"]
    records, completed = {}, set()
    for path in sorted((directory / "batches").glob("*.json")):
        match = re.fullmatch(r"batch-(\d{6})(\.pending)?\.json", path.name)
        if not match:
            raise CalibrationError("unknown checkpoint filename")
        index, pending = int(match[1]), bool(match[2])
        payload = load_json(path)
        _keys(payload, ("schema_version", "identity_sha256", "batch_index", "status", "records", "elapsed_seconds", "infrastructure_error"), "checkpoint")
        if (payload["schema_version"] != 1 or payload["identity_sha256"] != manifest["identity_sha256"]
                or payload["batch_index"] != index or payload["status"] != ("pending" if pending else "completed")):
            raise CalibrationError("stale or invalid checkpoint identity")
        expected_names = ordered_names[index*size:(index+1)*size]
        if not expected_names or not isinstance(payload["records"], list) or [r.get("name") for r in payload["records"] if isinstance(r, dict)] != expected_names:
            raise CalibrationError("checkpoint case inventory differs from plan")
        if not pending:
            _number(payload["elapsed_seconds"], "elapsed seconds")
            completed.add(index)
        for record in payload["records"]:
            fields = {"name", "cell", "repetition", "seed", "inputs", "input_sha256"}
            _keys(record, fields if pending else fields | {"methods", "comparisons"}, "checkpoint case")
            cell, repetition, seed = expected[record["name"]]
            if (record.get("cell") != cell["id"] or record.get("repetition") != repetition
                    or record.get("seed") != seed or record.get("input_sha256") != digest(record.get("inputs"))):
                raise CalibrationError("case identity or paired data hash mismatch")
            _arrays(record["inputs"])
            if len(record["inputs"]["y"]) != cell["n"]:
                raise CalibrationError("case sample size differs from plan")
            if not pending:
                _keys(record.get("methods"), METHODS, "checkpoint methods")
                for method, entry in record["methods"].items():
                    if not isinstance(entry, dict):
                        raise CalibrationError("invalid stored method status")
                    if entry.get("status") == "ok":
                        validate_result(entry.get("result"), cell["n"], method)
                        if method == "r" and entry.get("convention_differences", {}) != reference_convention_differences(entry["result"], cell["n"]):
                            raise CalibrationError("stored R convention check differs from returned values")
                    elif entry.get("status") != "error" or not isinstance(entry.get("error"), str) or not entry["error"]:
                        raise CalibrationError("invalid stored method failure")
                if record.get("comparisons") != comparisons(record["methods"]):
                    raise CalibrationError("stored comparison differs from method results")
            previous = records.get(record["name"])
            if previous is not None:
                # Crash after commit but before removing its pending file is safe.
                if previous["input_sha256"] != record["input_sha256"] or ("methods" in previous) == ("methods" in record):
                    raise CalibrationError("duplicate or conflicting case evidence")
                if "methods" in previous:
                    continue
            records[record["name"]] = record
    return records, completed


def wilson(count, denominator):
    if denominator == 0:
        return None
    z = float(stats.norm.ppf(.975))
    p = count/denominator
    center = (p+z*z/(2*denominator))/(1+z*z/denominator)
    half = z*math.sqrt(p*(1-p)/denominator+z*z/(4*denominator**2))/(1+z*z/denominator)
    return [max(0., center-half), min(1., center+half)]


def summarize_method(records, method, planned, truth, alpha=.05):
    successes, failures, failure_codes = [], {}, {}
    for record in records:
        entry = record.get("methods", {}).get(method, {})
        if entry.get("status") == "ok":
            successes.append(entry["result"])
        elif entry.get("status") == "error":
            label = entry.get("error_class", "UnclassifiedError")
            failures[label] = failures.get(label, 0)+1
            code = entry.get("error_code", "r_case_failure" if method == "r" else "unclassified_failure")
            failure_codes[code] = failure_codes.get(code, 0)+1
    m, failed = len(successes), sum(failures.values())
    missing = planned-m-failed
    covered = sum(row["ci"][0] <= truth <= row["ci"][1] for row in successes)
    rejected = sum(row["p_value"] < alpha for row in successes)
    psi = np.array([row["psi"] for row in successes])
    sd = float(np.std(psi, ddof=1)) if m >= 2 else None
    bias = float(np.mean(psi-truth)) if m else None
    mcse = sd/math.sqrt(m) if sd is not None else None
    coverage_interval = wilson(covered, m)
    return {"planned": planned, "attempted": len(records), "successful": m, "failed": failed,
            "missing": missing, "unattempted": planned-len(records), "failure_classes": failures,
            "failure_codes": failure_codes,
            "failed_fraction_planned": failed/planned, "missing_fraction_planned": missing/planned,
            "bias": bias, "bias_mcse": mcse, "empirical_sd": sd,
            "rms_se": math.sqrt(sum(row["variance"] for row in successes)/m) if m else None,
            "mean_interval_width": sum(row["ci"][1]-row["ci"][0] for row in successes)/m if m else None,
            "covered": covered, "coverage_successful": covered/m if m else None,
            "coverage_wilson": coverage_interval, "covered_fraction_planned": covered/planned,
            "coverage_all_planned_bounds": [covered/planned, (covered+failed+missing)/planned],
            "rejected": rejected, "rejection_kind": "false_positive_rate" if truth == 0 else "power",
            "rejection_successful": rejected/m if m else None, "rejection_wilson": wilson(rejected, m),
            "rejected_fraction_planned": rejected/planned,
            "diagnostics_only": {"bias_exceeds_3_mcse": abs(bias) > 3*mcse if mcse is not None else None,
                                  "coverage_wilson_outside_093_097": (coverage_interval[1] < .93 or coverage_interval[0] > .97) if coverage_interval else None}}


def aggregate(directory):
    directory = Path(directory)
    manifest = load_json(directory / "manifest.json")
    _keys(manifest, ("schema_version", "identity", "identity_sha256", "plan"), "manifest")
    identity = manifest["identity"]
    validate_config(identity["config"])
    validate_reference(identity["reference"])
    if manifest["identity_sha256"] != digest(identity) or manifest["plan"] != make_plan(identity["config"], identity["mode"]):
        raise CalibrationError("manifest hash or plan mismatch")
    records, completed = read_checkpoints(directory, manifest)
    cells, total_failures, total_mismatches = [], 0, 0
    for cell in manifest["plan"]["cells"]:
        selected = [r for r in records.values() if r["cell"] == cell["id"]]
        methods = {method: summarize_method(selected, method, cell["planned"], cell["true_psi"]) for method in METHODS}
        paired = {}
        for name in ("python_vs_r", "python_vs_qr_t"):
            rows = [row for r in selected for row in r.get("comparisons", []) if row["comparison"] == name]
            counts = {status: sum(row["status"] == status for row in rows) for status in ("agree", "mismatch", "unavailable")}
            counts["missing"] = cell["planned"]-len(rows)
            counts["planned"] = cell["planned"]
            counts["mismatching_cases"] = [r["name"] for r in selected if any(row["comparison"] == name and row["status"] == "mismatch" for row in r.get("comparisons", []))]
            paired[name] = counts
            total_mismatches += counts["mismatch"]
        total_failures += sum(m["failed"] for m in methods.values())
        r_valid = [r for r in selected if r.get("methods", {}).get("r", {}).get("status") == "ok"]
        r_differences = [r["name"] for r in r_valid if r["methods"]["r"].get("convention_differences")]
        total_mismatches += len(r_differences)
        conventions = {"checked": len(r_valid), "mismatch": len(r_differences),
                       "unavailable": cell["planned"]-len(r_valid), "mismatching_cases": r_differences}
        cells.append({"cell": cell["id"], "true_psi": cell["true_psi"], "analytic": cell["analytic"], "methods": methods, "paired": paired,
                      "reference_conventions": conventions})
    terminal = sum("methods" in record for record in records.values())
    planned = manifest["plan"]["planned"]
    complete = terminal == planned
    status = ("completed_with_failures_or_mismatches" if total_failures or total_mismatches else "complete_numerical_agreement") if complete else "incomplete"
    return {"schema_version": 1, "mode": identity["mode"], "identity_sha256": manifest["identity_sha256"],
            "status": status, "execution_complete": complete, "calibration_validated": False,
            "planned": planned, "attempted": len(records), "terminal": terminal, "missing": planned-terminal,
            "completed_batches": len(completed), "cells": cells, "output_bytes": _directory_bytes(directory),
            "peak_memory_bytes": None, "interpretation": "Diagnostics are descriptive, not a calibration pass/fail or release decision."}


@contextmanager
def writer_lock(directory):
    """One foreground writer, including resumes; never automatically clear a lock.

    The sibling ``<output-name>.writer.lock`` is created exclusively. A stale
    lock requires manual recovery only after checking that its recorded PID
    and any other writer are no longer active. PID reuse means existence alone
    cannot establish ownership. Normal errors and KeyboardInterrupt release it;
    process termination may intentionally leave fail-closed evidence behind.
    """
    directory = Path(directory)
    lock = directory.with_name(directory.name + ".writer.lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except OSError as exc:
        raise CalibrationError("writer lock unavailable; inspect active processes before manual stale-lock recovery") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            json.dump({"pid": os.getpid(), "recovery": "manual only after checking no active writer"}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        yield
    finally:
        lock.unlink()


def execute(config, mode, directory, *, rscript, library, timeout=180, resume=False, adapter=None):
    """Explicit execution boundary. Never called by plan or import."""
    start = time.perf_counter()
    with writer_lock(directory):
        return _execute_locked(config, mode, directory, rscript=rscript, library=library,
                               timeout=timeout, resume=resume, adapter=adapter, start=start)


def _execute_locked(config, mode, directory, *, rscript, library, timeout, resume, adapter, start):
    plan = make_plan(config, mode)
    if not plan["storage_estimate_within_budget"]:
        raise CalibrationError("static storage estimate exceeds the configured ceiling")
    if any(os.environ.get(key) != "1" for key in THREAD_VARIABLES):
        raise CalibrationError("execution requires all documented numerical thread limits set to 1 before Python starts")
    parent = Path(directory).parent
    if shutil.disk_usage(parent).free < plan["storage_budget_bytes"]:
        raise CalibrationError("insufficient free space for the configured output ceiling")
    adapter = adapter or _load_adapter()
    preflight_start = time.perf_counter()
    preflight = adapter([], rscript, library, timeout=timeout)
    validate_r_batch(preflight, [])
    preflight_seconds = time.perf_counter()-preflight_start
    identity = build_identity(config, mode, preflight["reference"])
    manifest = prepare_directory(directory, identity, plan, resume=resume)
    existing, completed = read_checkpoints(directory, manifest)
    specs = list(_expected_cases(manifest).items())
    batch_size = plan["batch_size"]
    batches_start = time.perf_counter()
    setup_seconds = batches_start-start-preflight_seconds
    for index, offset in enumerate(range(0, len(specs), batch_size)):
        if index in completed:
            continue
        cases = []
        for name, (cell, repetition, _) in specs[offset:offset+batch_size]:
            cases.append(existing.get(name) or generate_case(config, mode, cell, repetition))
        pending_path = Path(directory) / "batches" / f"batch-{index:06d}.pending.json"
        final_path = Path(directory) / "batches" / f"batch-{index:06d}.json"
        checkpoint = {"schema_version": 1, "identity_sha256": manifest["identity_sha256"],
                      "batch_index": index, "status": "pending", "records": cases,
                      "elapsed_seconds": None, "infrastructure_error": None}
        atomic_json(pending_path, checkpoint, plan["storage_budget_bytes"])
        batch_start = time.perf_counter()
        evaluated = evaluate_batch(cases, adapter, identity["reference"], rscript=rscript, library=library, timeout=timeout)
        checkpoint.update(status="completed", records=evaluated["records"],
                          elapsed_seconds=time.perf_counter()-batch_start,
                          infrastructure_error=evaluated["infrastructure_error"])
        if final_path.exists():
            raise CalibrationError("completed checkpoints cannot be overwritten")
        atomic_json(final_path, checkpoint, plan["storage_budget_bytes"])
        pending_path.unlink()
        if evaluated["infrastructure_error"] is not None:
            raise CalibrationError("R infrastructure failure; retained checkpoints are incomplete evidence")
        if any(row["methods"]["r"].get("convention_differences") for row in evaluated["records"]):
            raise CalibrationError("R degrees-of-freedom mismatch; returned values and valid sibling results are retained")
    batches_seconds = time.perf_counter()-batches_start
    aggregate_start = time.perf_counter()
    report = aggregate(directory)
    aggregation_seconds = time.perf_counter()-aggregate_start
    summary_start = time.perf_counter()
    atomic_json(Path(directory) / "summary.json", report, plan["storage_budget_bytes"])
    timing = {"preflight_seconds": preflight_seconds, "setup_seconds": setup_seconds,
              "batch_seconds_including_generation_and_checkpoints": batches_seconds,
              "aggregation_seconds": aggregation_seconds,
              "summary_write_seconds": time.perf_counter()-summary_start,
              "total_invocation_seconds": time.perf_counter()-start,
              "scope": "this invocation through summary write, including R preflight; excludes timing record write and lock release",
              "resumed": resume, "peak_memory_bytes": None}
    # Timing is distinct from inferential evidence. On resume it describes only
    # this invocation; it must not be presented as a fresh full-study runtime.
    atomic_json(Path(directory) / "timing.json", timing, plan["storage_budget_bytes"])
    report["timing"] = timing
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "pilot", "study", "aggregate"), default="plan", nargs="?")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--r-library", type=Path)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args(argv)
    try:
        if args.mode == "aggregate":
            if args.output is None or args.execute or args.resume:
                raise CalibrationError("aggregate requires --output and does not execute or resume")
            report = aggregate(args.output)
        else:
            config = validate_config(load_json(args.config))
            if args.mode == "plan":
                if args.execute or args.resume or args.output is not None:
                    raise CalibrationError("plan is read-only; execution/output flags are unsupported")
                report = {"status": "planned_not_executed", "study": make_plan(config), "pilot": make_plan(config, "pilot")}
            else:
                if not args.execute or args.output is None or args.r_library is None:
                    raise CalibrationError("pilot/study require --execute, --output and --r-library")
                if not math.isfinite(args.timeout) or args.timeout <= 0:
                    raise CalibrationError("timeout must be finite and positive")
                report = execute(config, args.mode, args.output, rscript=args.rscript, library=args.r_library,
                                 timeout=args.timeout, resume=args.resume)
        print(json.dumps(report, indent=2, allow_nan=False))
        return 2 if args.mode in ("pilot", "study") and report.get("status") != "complete_numerical_agreement" else 0
    except KeyboardInterrupt:
        print(json.dumps({"status": "interrupted", "calibration_validated": False,
                          "error": "pending inputs and completed checkpoints remain resumable"}), file=sys.stderr)
        return 130
    except Exception as exc:
        # Do not expose private executable/library/output paths in public reports.
        diagnostic = _error(exc)
        message = str(exc) if isinstance(exc, CalibrationError) else diagnostic["error"]
        print(json.dumps({"status": "failed", "calibration_validated": False,
                          "error_class": type(exc).__name__, "error_code": diagnostic["error_code"],
                          "error": message}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
