#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Reproducible known-score Gaussian bootstrap study; import and plan never fit.

The fixed config and BOOTSTRAP_PROTOCOL.md define the study. Use ``pilot`` for
an admitted small timing run. ``study`` is a separate server-class decision.
Completed case records, including failures, are never rerun by resume.
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
from quantilemix import fit_bootstrap

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "bootstrap-calibration-config.json"
BASE_CONFIG = HERE / "calibration-config.json"
GENERATOR = HERE / "calibration.py"
REFERENCE_REPORT = HERE / "bootstrap-reference-results.json"
REFERENCE_FIXTURE = HERE.parent / "tests/fixtures/r_bootstrap_2_19_6.json"
THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")

spec = importlib.util.spec_from_file_location("qgcomp_known_score_generator", GENERATOR)
if spec is None or spec.loader is None:
    raise RuntimeError("known-score generator is unavailable")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


class BootstrapStudyError(ValueError):
    """The frozen design or run evidence is invalid."""


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generator_design_hash():
    """Check the frozen generator with only its package imports normalized.

    Run identity still records the actual source bytes separately.
    """
    source = GENERATOR.read_bytes()
    for current, original in (
        (b"import quantilemix as package", b"import qgcomp as package"),
        (b"import quantilemix._core as core", b"import qgcomp._core as core"),
        (b"from quantilemix import fit", b"from qgcomp import fit"),
    ):
        source = source.replace(current, original)
    return hashlib.sha256(source).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("ascii")).hexdigest()


def load_config(path=CONFIG):
    data = base.load_json(path)
    expected = {
        "schema_version": 1, "protocol_id": "gaussian-bootstrap-known-scores-v1",
        "data_protocol_id": "gaussian-known-scores-v1",
        "data_config_sha256": "25a8f6d98dcfcf6a9eb0afbe8c8eaf6c9d9702b8db420af9a10df28f13841de1",
        "data_generator_sha256": "c4aa619dba8a6bf7116ef4f119716ff5fb840ea378beca4140fb86e513d3fb0e",
        "alpha": .05, "bootstrap_resamples": 200,
        "study_replicates_per_cell": 2000, "pilot_replicates": 20,
        "master_seed": 20261023, "pilot_seed_namespace": 31,
        "study_seed_namespace": 32, "bootstrap_seed_namespace": 33,
        "batch_size": 20, "pilot_storage_budget_bytes": 100_000_000,
        "study_storage_budget_bytes": 3_000_000_000,
    }
    if digest(data) != digest(expected):
        raise BootstrapStudyError("bootstrap design differs from reviewed configuration")
    if file_hash(BASE_CONFIG) != data["data_config_sha256"] or generator_design_hash() != data["data_generator_sha256"]:
        raise BootstrapStudyError("known-score design or generator identity changed")
    return data


def data_config(config):
    result = base.load_json(BASE_CONFIG)
    result["execution"]["master_seed"] = config["master_seed"]
    result["pilot"]["seed_namespace"] = config["pilot_seed_namespace"]
    result["study"]["seed_namespace"] = config["study_seed_namespace"]
    base.validate_config(result)
    return result


def plan(config, mode):
    if mode not in ("pilot", "study"):
        raise BootstrapStudyError("mode must be pilot or study")
    source = base.make_plan(data_config(config), mode)
    return {"protocol_id": config["protocol_id"], "mode": mode,
            "cells": source["cells"], "planned": source["planned"],
            "batch_size": config["batch_size"], "B": config["bootstrap_resamples"],
            "storage_budget_bytes": config[f"{mode}_storage_budget_bytes"],
            "fits_expected": source["planned"] * config["bootstrap_resamples"],
            "execution_performed": False, "calibration_validated": False}


def bootstrap_seed(config, mode, cell, repetition):
    source = [config["master_seed"], config["bootstrap_seed_namespace"],
              1 if mode == "pilot" else 2, cell["index"], repetition]
    state = np.random.SeedSequence(source).generate_state(2, dtype=np.uint32)
    return int(state[0]) | (int(state[1]) << 32)


def make_case(config, mode, cell, repetition):
    case = base.generate_case(data_config(config), mode, cell, repetition)
    case["bootstrap_seed"] = bootstrap_seed(config, mode, cell, repetition)
    return case


def _result(fitted):
    return {"psi": float(fitted.psi), "variance": float(fitted.var_psi),
            "se": float(fitted.se), "ci": list(map(float, fitted.ci)),
            "p_value": float(fitted.p_value), "df_pvalue": int(fitted.p_value_df),
            "df_residual": int(fitted.point_fit.residual_df) if hasattr(fitted, "point_fit") else int(fitted.residual_df)}


def validate_method_result(result, n, name, B):
    common = {key: value for key, value in result.items() if key != "bootstrap_effects"}
    base.validate_result(common, n, "r" if name == "bootstrap" else
                         "qr_t" if name == "qr_t" else "python")
    if name == "bootstrap":
        if result["df_pvalue"] != n - 8:
            raise BootstrapStudyError("bootstrap reference degrees of freedom differ")
        draws = result.get("bootstrap_effects")
        if not isinstance(draws, list) or len(draws) != B or not all(
                type(value) in (int, float) and math.isfinite(value) for value in draws):
            raise BootstrapStudyError("bootstrap draws are missing or nonfinite")
        if not math.isclose(float(np.std(draws, ddof=1)), result["se"],
                            rel_tol=1e-12, abs_tol=1e-12):
            raise BootstrapStudyError("bootstrap draw variance differs")
    elif "bootstrap_effects" in result:
        raise BootstrapStudyError("non-bootstrap method has resampling draws")
    return result


def fit_case(case, B=200):
    if digest(case["inputs"]) != case["input_sha256"]:
        raise BootstrapStudyError("case input hash differs")
    x, z, y = base._arrays(case["inputs"])
    methods = {}
    for name, operation in (
        ("bootstrap", lambda: fit_bootstrap(y, x, z, q=None, alpha=.05,
                                            B=B, seed=case["bootstrap_seed"])),
        ("model_based", lambda: base.fit(y, x, z, q=None, alpha=.05)),
        ("qr_t", lambda: base.qr_result(case["inputs"])),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                fitted = operation()
                result = fitted if name == "qr_t" else _result(fitted)
                if name == "bootstrap":
                    draws = list(map(float, fitted.bootstrap_effects))
                    result["bootstrap_effects"] = draws
                validate_method_result(result, len(y), name, B)
                methods[name] = {"status": "ok", "result": result,
                                 "warnings": [w.category.__name__ for w in caught]}
            except Exception as exc:
                # Error codes and classes are retained; arbitrary exception
                # strings may contain paths or data and are not persisted.
                item = base._error(exc, caught)
                methods[name] = item
    return dict(case, methods=methods)


def identity(config, mode):
    import quantilemix
    import quantilemix._bootstrap as bootstrap
    import quantilemix._core as core
    report = base.load_json(REFERENCE_REPORT)
    if (report.get("status") != "passed" or report.get("case_count") != 2
            or report.get("reference", {}).get("version") != "2.19.6"):
        raise BootstrapStudyError("fixed installed-R bootstrap evidence is unavailable")
    paths = {"bootstrap_calibration.py": Path(__file__),
             "bootstrap-calibration-config.json": CONFIG,
             "calibration.py": GENERATOR, "calibration-config.json": BASE_CONFIG,
             "qgcomp/__init__.py": Path(quantilemix.__file__),
             "qgcomp/_bootstrap.py": Path(bootstrap.__file__),
             "bootstrap-reference-results.json": REFERENCE_REPORT,
             "r_bootstrap_2_19_6.json": REFERENCE_FIXTURE,
             "qgcomp/_core.py": Path(core.__file__)}
    return {"protocol_id": config["protocol_id"], "config_sha256": digest(config),
            "mode": mode, "sources_sha256": {key: file_hash(path) for key, path in paths.items()},
            "environment": {"python": sys.version, "numpy": np.__version__,
                            "scipy": scipy.__version__, "platform": sys.platform,
                            "threads": {key: os.environ.get(key) for key in THREADS}},
            "installed_r_fixed_cases": 2}


def _expected(plan_data):
    return [(cell, repetition) for cell in plan_data["cells"]
            for repetition in range(cell["planned"])]


def _check_cases(records, specs, config, mode, completed):
    if not isinstance(records, list) or len(records) != len(specs):
        raise BootstrapStudyError("checkpoint case count differs")
    for record, (cell, repetition) in zip(records, specs):
        expected = make_case(config, mode, cell, repetition)
        for key in ("name", "cell", "repetition", "seed", "input_sha256", "bootstrap_seed"):
            if record.get(key) != expected[key]:
                raise BootstrapStudyError("checkpoint identity differs")
        if digest(record.get("inputs")) != expected["input_sha256"]:
            raise BootstrapStudyError("checkpoint inputs differ")
        if completed:
            if not isinstance(record.get("methods"), dict) or set(record["methods"]) != {"bootstrap", "model_based", "qr_t"}:
                raise BootstrapStudyError("checkpoint methods differ")
            for name, entry in record["methods"].items():
                if entry.get("status") == "ok":
                    result = entry.get("result")
                    validate_method_result(result, cell["n"], name,
                                           config["bootstrap_resamples"])
                elif entry.get("status") != "error" or not isinstance(entry.get("error_code"), str):
                    raise BootstrapStudyError("invalid saved failure")


def read_batches(directory, manifest, config):
    folder = Path(directory) / "batches"
    base._directory_bytes(Path(directory))
    specs = _expected(manifest["plan"])
    size = manifest["plan"]["batch_size"]
    records, completed = [], set()
    for index, offset in enumerate(range(0, len(specs), size)):
        final = folder / f"batch-{index:06d}.json"
        pending = folder / f"batch-{index:06d}.pending.json"
        if final.exists() and pending.exists():
            raise BootstrapStudyError("completed and pending batch both exist")
        path = final if final.exists() else pending
        if not path.exists():
            continue
        payload = base.load_json(path)
        if (payload.get("schema_version") != 1 or payload.get("identity_sha256") != manifest["identity_sha256"]
                or payload.get("batch_index") != index or payload.get("status") != ("completed" if path == final else "pending")):
            raise BootstrapStudyError("checkpoint header differs")
        _check_cases(payload.get("records"), specs[offset:offset+size], config,
                     manifest["plan"]["mode"], path == final)
        if path == final:
            completed.add(index)
            records.extend(payload["records"])
    expected_files = {f"batch-{i:06d}{suffix}.json"
                      for i in range((len(specs) + size - 1) // size)
                      for suffix in ("", ".pending")}
    if any(p.name not in expected_files or not re.fullmatch(
            r"batch-\d{6}(?:\.pending)?\.json", p.name) for p in folder.iterdir()):
        raise BootstrapStudyError("unknown checkpoint file")
    return records, completed


@contextmanager
def writer_lock(directory):
    lock = Path(directory).with_name(Path(directory).name + ".writer.lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump({"pid": os.getpid(), "recovery": "check live writer before removal"}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        yield
    finally:
        lock.unlink()


def summarize(records, plan_data):
    cells = []
    for cell in plan_data["cells"]:
        selected = [r for r in records if r["cell"] == cell["id"]]
        methods = {}
        for name in ("bootstrap", "model_based", "qr_t"):
            slim = [{"methods": {name: {**entry, "result": {k: v for k, v in entry.get("result", {}).items()
                                                    if k != "bootstrap_effects"}}}}
                    for r in selected if (entry := r["methods"][name])]
            methods[name] = base.summarize_method(slim, name, cell["planned"], cell["true_psi"])
            success = [r["methods"][name]["result"] for r in selected if r["methods"][name]["status"] == "ok"]
            methods[name]["interval_excludes_zero"] = sum(not (v["ci"][0] <= 0 <= v["ci"][1]) for v in success)
        paired = [r for r in selected if r["methods"]["bootstrap"]["status"] == r["methods"]["model_based"]["status"] == "ok"]
        width = [((r["methods"]["bootstrap"]["result"]["ci"][1] - r["methods"]["bootstrap"]["result"]["ci"][0])
                  - (r["methods"]["model_based"]["result"]["ci"][1] - r["methods"]["model_based"]["result"]["ci"][0])) for r in paired]
        bootstrap_only = 0
        model_only = 0
        for record in paired:
            boot_ci = record["methods"]["bootstrap"]["result"]["ci"]
            model_ci = record["methods"]["model_based"]["result"]["ci"]
            boot_covers = boot_ci[0] <= cell["true_psi"] <= boot_ci[1]
            model_covers = model_ci[0] <= cell["true_psi"] <= model_ci[1]
            bootstrap_only += boot_covers and not model_covers
            model_only += model_covers and not boot_covers
        cells.append({"cell": cell["id"], "true_psi": cell["true_psi"], "methods": methods,
                      "paired": {"successful": len(paired), "mean_width_difference": float(np.mean(width)) if width else None,
                                 "coverage_discordance": bootstrap_only + model_only,
                                 "bootstrap_only_covered": bootstrap_only,
                                 "model_based_only_covered": model_only,
                                 "mean_coverage_difference": (bootstrap_only - model_only) / len(paired) if paired else None}})
    terminal = len(records)
    return {"protocol_id": plan_data["protocol_id"], "mode": plan_data["mode"],
            "status": "complete" if terminal == plan_data["planned"] else "incomplete",
            "execution_complete": terminal == plan_data["planned"], "calibration_validated": False,
            "planned": plan_data["planned"], "terminal": terminal,
            "cells": cells, "interpretation": "Pilot is timing evidence only; study needs separate statistical interpretation."}


def run(config, mode, output, *, resume=False):
    plan_data = plan(config, mode)
    if any(os.environ.get(key) != "1" for key in THREADS):
        raise BootstrapStudyError("all numerical thread variables must be 1 before Python starts")
    output = Path(output)
    if output.is_symlink() or output.with_name(output.name + ".writer.lock").exists():
        raise BootstrapStudyError("unsafe output path or writer lock exists")
    if shutil.disk_usage(output.parent).free < plan_data["storage_budget_bytes"]:
        raise BootstrapStudyError("insufficient free space for output ceiling")
    start = time.perf_counter()
    with writer_lock(output):
        current = identity(config, mode)
        manifest = {"schema_version": 1, "identity": current,
                    "identity_sha256": digest(current), "plan": plan_data}
        if resume:
            if base.load_json(output / "manifest.json") != manifest:
                raise BootstrapStudyError("resume identity or plan differs")
        else:
            output.mkdir(exist_ok=False)
            (output / "batches").mkdir()
            base.atomic_json(output / "manifest.json", manifest, plan_data["storage_budget_bytes"])
        _, completed = read_batches(output, manifest, config)
        specs = _expected(plan_data)
        size = plan_data["batch_size"]
        for index, offset in enumerate(range(0, len(specs), size)):
            if index in completed:
                continue
            pending = output / "batches" / f"batch-{index:06d}.pending.json"
            final = output / "batches" / f"batch-{index:06d}.json"
            if pending.exists():
                cases = base.load_json(pending)["records"]
            else:
                cases = [make_case(config, mode, cell, repetition)
                         for cell, repetition in specs[offset:offset+size]]
                base.atomic_json(pending, {"schema_version": 1,
                    "identity_sha256": manifest["identity_sha256"], "batch_index": index,
                    "status": "pending", "records": cases}, plan_data["storage_budget_bytes"])
            batch_start = time.perf_counter()
            records = [fit_case(case, plan_data["B"]) for case in cases]
            base.atomic_json(final, {"schema_version": 1,
                "identity_sha256": manifest["identity_sha256"], "batch_index": index,
                "status": "completed", "records": records,
                "elapsed_seconds": time.perf_counter() - batch_start}, plan_data["storage_budget_bytes"])
            pending.unlink()
        records, _ = read_batches(output, manifest, config)
        result = summarize(records, plan_data)
        result.update(identity_sha256=manifest["identity_sha256"],
                      elapsed_seconds=time.perf_counter()-start,
                      output_bytes=base._directory_bytes(output))
        base.atomic_json(output / "summary.json", result, plan_data["storage_budget_bytes"])
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "pilot", "study"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = load_config()
    if args.mode == "plan":
        if args.output or args.resume:
            parser.error("plan mode is read-only")
        print(json.dumps({mode: plan(config, mode) for mode in ("pilot", "study")}, indent=2))
        return 0
    if args.output is None:
        parser.error("execution requires --output")
    result = run(config, args.mode, args.output, resume=args.resume)
    print(json.dumps({key: result[key] for key in ("status", "mode", "planned", "terminal", "elapsed_seconds", "output_bytes")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
