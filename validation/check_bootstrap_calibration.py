#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Read-only reaggregation of a saved Gaussian bootstrap pilot or study.

This checker does not fit a model, draw data, import quantilemix or modify a run.
It verifies identities and counts, recomputes key statistics from terminal
records, and writes a new report outside the run directory.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

import numpy as np
from scipy import special, stats

CONFIG = Path(__file__).with_name("bootstrap-calibration-config.json")
BASE_CONFIG = Path(__file__).with_name("calibration-config.json")
GENERATOR = Path(__file__).with_name("calibration.py")
FROZEN_CONFIG_DIGEST = "039e88c1b27029997a9778b8bed0387a5f9b7b258e1eaa2ef8ef7d4941dcccda"
REQUIRED_METHODS = ("bootstrap", "model_based", "qr_t")


class EvidenceError(ValueError):
    """Saved study evidence differs from its fixed design or summary."""


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("ascii")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def no_duplicates(pairs):
    out = {}
    for key, value in pairs:
        require(key not in out, "duplicate JSON key")
        out[key] = value
    return out


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "missing or unsafe JSON evidence")
    return json.loads(path.read_text(), object_pairs_hook=no_duplicates,
                      parse_constant=lambda _: (_ for _ in ()).throw(EvidenceError("nonfinite JSON number")))


def close(actual, expected, label, *, atol=1e-12, rtol=1e-10):
    require(type(actual) in (int, float) and math.isfinite(actual), f"{label}: invalid number")
    require(math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol), f"{label}: differs")


def same(actual, expected, label):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f"{label}: fields differ")
        for key, value in expected.items():
            same(actual[key], value, f"{label}.{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"{label}: length differs")
        for index, value in enumerate(expected):
            same(actual[index], value, f"{label}[{index}]")
    elif type(expected) is float:
        close(actual, expected, label)
    else:
        require(actual == expected and type(actual) is type(expected), f"{label}: differs")


def expected_cells(config, base_config, mode):
    design = base_config["design"]
    cells = []
    for n in design["sample_sizes"]:
        for rho in design["rho"]:
            for beta_name in ("null", "cancellation", "positive"):
                beta = design["beta"][beta_name]
                index = len(cells)
                name = f"n{n}-rho{str(rho).replace('.', 'p')}-{beta_name}"
                cells.append({"id": name, "index": index, "n": n,
                              "rho": rho, "beta_name": beta_name, "beta": beta,
                              "true_psi": sum(beta)})
    if mode == "pilot":
        cells = [cells[-1]]
    return [dict(cell, planned=config["pilot_replicates"] if mode == "pilot" else
                 config["study_replicates_per_cell"]) for cell in cells]


def expected_bootstrap_seed(config, mode, cell, rep):
    parts = [config["master_seed"], config["bootstrap_seed_namespace"],
             1 if mode == "pilot" else 2, cell["index"], rep]
    words = np.random.SeedSequence(parts).generate_state(2, dtype=np.uint32)
    return int(words[0]) + (int(words[1]) << 32)


def expected_inputs(config, mode, cell, rep):
    seed = [config["master_seed"], config[f"{mode}_seed_namespace"], cell["index"], rep]
    generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    normals = generator.standard_normal((cell["n"], 8))
    covariates = normals[:, :2]
    shared = math.sqrt(cell["rho"]) * normals[:, 2, None]
    individual = math.sqrt(1 - cell["rho"]) * normals[:, 3:7]
    latent = .5 * covariates[:, 0, None] + shared + individual
    cuts = math.sqrt(1.25) * special.ndtri([.25, .5, .75])
    scores = np.searchsorted(cuts, latent, side="left")
    outcome = 1 + scores @ np.asarray(cell["beta"]) + covariates @ np.asarray([.5, -.25]) + normals[:, 7]
    return {"exposures": scores.tolist(), "covariates": covariates.tolist(),
            "y": outcome.tolist(), "q": None, "alpha": .05}


def generated_inputs_agree(saved, expected):
    """Allow only observed cross-platform float rounding in generated rows.

    The saved input hash is still checked exactly before this comparison.
    Integer score categories and scalar analysis settings must agree exactly.
    """
    if not isinstance(saved, dict) or set(saved) != set(expected):
        return False
    if any(saved[key] != expected[key] for key in ("exposures", "q", "alpha")):
        return False
    for key in ("covariates", "y"):
        actual = np.asarray(saved[key])
        target = np.asarray(expected[key])
        if (actual.shape != target.shape or actual.dtype.kind != "f" or
                not np.all(np.isfinite(actual)) or
                not np.all(np.abs(actual-target) <= 2e-15)):
            return False
    return True


def wilson(count, total):
    if total == 0:
        return None
    critical = float(stats.norm.ppf(.975))
    rate = count / total
    denom = 1 + critical**2 / total
    mid = (rate + critical**2 / (2 * total)) / denom
    half = critical * math.sqrt(rate * (1-rate) / total + critical**2 / (4 * total**2)) / denom
    return [max(0.0, mid-half), min(1.0, mid+half)]


def method_summary(entries, planned, truth, alpha):
    good = [entry["result"] for entry in entries if entry["status"] == "ok"]
    bad = [entry for entry in entries if entry["status"] == "error"]
    failures, codes = {}, {}
    for entry in bad:
        label = entry.get("error_class", "UnclassifiedError")
        code = entry["error_code"]
        failures[label] = failures.get(label, 0) + 1
        codes[code] = codes.get(code, 0) + 1
    m = len(good)
    errors = [row["psi"] - truth for row in good]
    empirical_sd = statistics.stdev(errors) if m >= 2 else None
    bias = statistics.mean(errors) if m else None
    covered = sum(row["ci"][0] <= truth <= row["ci"][1] for row in good)
    rejected = sum(row["p_value"] < alpha for row in good)
    failed = len(bad)
    missing = planned - len(entries)
    coverage_interval = wilson(covered, m)
    return {"planned": planned, "attempted": len(entries), "successful": m,
            "failed": failed, "missing": missing, "unattempted": missing,
            "failure_classes": failures, "failure_codes": codes,
            "failed_fraction_planned": failed/planned,
            "missing_fraction_planned": missing/planned,
            "bias": bias, "bias_mcse": empirical_sd/math.sqrt(m) if empirical_sd is not None else None,
            "empirical_sd": empirical_sd,
            "rms_se": math.sqrt(statistics.mean(row["variance"] for row in good)) if m else None,
            "mean_interval_width": statistics.mean(row["ci"][1]-row["ci"][0] for row in good) if m else None,
            "covered": covered, "coverage_successful": covered/m if m else None,
            "coverage_wilson": coverage_interval,
            "covered_fraction_planned": covered/planned,
            "coverage_all_planned_bounds": [covered/planned, (covered+failed+missing)/planned],
            "rejected": rejected,
            "rejection_kind": "false_positive_rate" if truth == 0 else "power",
            "rejection_successful": rejected/m if m else None,
            "rejection_wilson": wilson(rejected, m),
            "rejected_fraction_planned": rejected/planned,
            "interval_excludes_zero": sum(not (row["ci"][0] <= 0 <= row["ci"][1])
                                          for row in good),
            "diagnostics_only": {
                "bias_exceeds_3_mcse": abs(bias) > 3*empirical_sd/math.sqrt(m) if empirical_sd is not None else None,
                "coverage_wilson_outside_093_097":
                    coverage_interval[1] < .93 or coverage_interval[0] > .97 if coverage_interval else None,
            }}


def paired_summary(records, truth):
    pair = [(r["methods"]["bootstrap"]["result"], r["methods"]["model_based"]["result"])
            for r in records if r["methods"]["bootstrap"]["status"] == "ok" and
            r["methods"]["model_based"]["status"] == "ok"]
    widths, bootstrap_only, model_only = [], 0, 0
    for boot, model in pair:
        widths.append((boot["ci"][1]-boot["ci"][0]) - (model["ci"][1]-model["ci"][0]))
        bc = boot["ci"][0] <= truth <= boot["ci"][1]
        mc = model["ci"][0] <= truth <= model["ci"][1]
        bootstrap_only += bool(bc and not mc)
        model_only += bool(mc and not bc)
    return {"successful": len(pair),
            "mean_width_difference": statistics.mean(widths) if widths else None,
            "coverage_discordance": bootstrap_only+model_only,
            "bootstrap_only_covered": bootstrap_only,
            "model_based_only_covered": model_only,
            "mean_coverage_difference": (bootstrap_only-model_only)/len(pair) if pair else None}


def audit(directory):
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink(), "run directory is unavailable or unsafe")
    config = read(CONFIG)
    require(digest(config) == FROZEN_CONFIG_DIGEST, "checker design configuration differs")
    # Preserve the original design fingerprint across the import-namespace rename.
    generator_source = GENERATOR.read_bytes()
    for current, original in (
        (b"import quantilemix as package", b"import qgcomp as package"),
        (b"import quantilemix._core as core", b"import qgcomp._core as core"),
        (b"from quantilemix import fit", b"from qgcomp import fit"),
    ):
        generator_source = generator_source.replace(current, original)
    require(hashlib.sha256(BASE_CONFIG.read_bytes()).hexdigest() == config["data_config_sha256"] and
            hashlib.sha256(generator_source).hexdigest() == config["data_generator_sha256"],
            "fixed data generator identity differs")
    base_config = read(BASE_CONFIG)
    manifest = read(directory / "manifest.json")
    require(manifest.get("schema_version") == 1, "manifest schema differs")
    identity = manifest.get("identity")
    require(isinstance(identity, dict) and manifest.get("identity_sha256") == digest(identity),
            "manifest identity hash differs")
    require(identity.get("config_sha256") == digest(config), "run config differs from saved design")
    mode = identity.get("mode")
    require(mode in ("pilot", "study"), "invalid run mode")
    plan = manifest.get("plan")
    require(isinstance(plan, dict) and plan.get("mode") == mode and
            plan.get("protocol_id") == config["protocol_id"], "run plan differs")
    require(plan.get("B") == config["bootstrap_resamples"] and
            plan.get("planned") == (config["pilot_replicates"] if mode == "pilot" else
                                    config["study_replicates_per_cell"]*12), "resample or case count differs")
    cells = plan.get("cells")
    expected_design = expected_cells(config, base_config, mode)
    require(isinstance(cells, list) and len(cells) == len(expected_design), "planned cells differ")
    require(plan.get("batch_size") == config["batch_size"] and
            plan.get("storage_budget_bytes") == config[f"{mode}_storage_budget_bytes"] and
            plan.get("fits_expected") == plan["planned"] * config["bootstrap_resamples"],
            "run plan resources differ")
    expected_names = []
    cell_by_name = {}
    for cell, expected_cell in zip(cells, expected_design):
        for key, value in expected_cell.items():
            same(cell.get(key), value, f"cell design {key}")
        for rep in range(cell["planned"]):
            name = f"{mode}-{cell['id']}-r{rep:05d}"
            expected_names.append(name)
            cell_by_name[name] = (cell, rep)
    require(len(expected_names) == plan["planned"], "cell totals differ")
    folder = directory / "batches"
    require(folder.is_dir() and not folder.is_symlink(), "batch directory is unavailable")
    files = sorted(folder.iterdir())
    require(all(p.is_file() and not p.is_symlink() and re.fullmatch(r"batch-\d{6}(?:\.pending)?\.json", p.name)
                for p in files), "unknown or unsafe checkpoint")
    size = plan["batch_size"]
    terminal = 0
    rows = defaultdict(lambda: defaultdict(list))
    records_by_cell = defaultdict(list)
    batch_hashes = {}
    seen = set()
    for path in files:
        match = re.fullmatch(r"batch-(\d{6})(\.pending)?\.json", path.name)
        index, pending = int(match[1]), bool(match[2])
        require(index*size < len(expected_names), "extra checkpoint")
        require(index not in seen, "duplicate completed/pending batch")
        seen.add(index)
        payload = read(path)
        require(payload.get("identity_sha256") == manifest["identity_sha256"] and
                payload.get("batch_index") == index and
                payload.get("status") == ("pending" if pending else "completed"),
                "checkpoint identity differs")
        expected = expected_names[index*size:(index+1)*size]
        records = payload.get("records")
        require(isinstance(records, list) and [r.get("name") for r in records] == expected,
                "checkpoint case order differs")
        batch_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for record in records:
            cell, rep = cell_by_name[record["name"]]
            require(record.get("cell") == cell["id"] and record.get("repetition") == rep and
                    record.get("seed") == [config["master_seed"], config[f"{mode}_seed_namespace"],
                                            cell["index"], rep], "case seed or identity differs")
            inputs = record.get("inputs")
            require(isinstance(inputs, dict) and record.get("input_sha256") == digest(inputs),
                    "case input hash differs")
            require(generated_inputs_agree(inputs, expected_inputs(config, mode, cell, rep)),
                    "case inputs differ from independently generated design")
            require(len(inputs.get("y", [])) == cell["n"] and
                    len(inputs.get("exposures", [])) == cell["n"] and
                    len(inputs.get("covariates", [])) == cell["n"], "case shape differs")
            require(record.get("bootstrap_seed") == expected_bootstrap_seed(config, mode, cell, rep),
                    "bootstrap seed differs")
            if pending:
                require("methods" not in record, "pending case has terminal results")
                continue
            terminal += 1
            records_by_cell[cell["id"]].append(record)
            methods = record.get("methods")
            require(isinstance(methods, dict) and set(methods) == set(REQUIRED_METHODS),
                    "terminal methods differ")
            for method, entry in methods.items():
                require(entry.get("status") in ("ok", "error"), "method status differs")
                if entry["status"] == "error":
                    require(isinstance(entry.get("error_code"), str), "failure lacks code")
                    rows[cell["id"]][method].append(None)
                    continue
                result = entry.get("result")
                require(isinstance(result, dict), "missing method result")
                for field in ("psi", "variance", "se", "p_value"):
                    require(type(result.get(field)) in (int, float) and math.isfinite(result[field]),
                            "nonfinite method result")
                require(isinstance(result.get("ci"), list) and len(result["ci"]) == 2 and
                        all(type(x) in (int, float) and math.isfinite(x) for x in result["ci"]),
                        "invalid interval")
                close(result["variance"], result["se"]**2, "reported variance")
                require(result["df_residual"] == cell["n"]-7 and
                        result["df_pvalue"] == cell["n"]-(8 if method == "bootstrap" else
                                                     7 if method == "qr_t" else 5),
                        "degrees of freedom differ")
                require(result["se"] > 0 and 0 <= result["p_value"] <= 1,
                        "invalid uncertainty or p-value")
                if method == "bootstrap":
                    draws = result.get("bootstrap_effects")
                    require(isinstance(draws, list) and len(draws) == config["bootstrap_resamples"] and
                            all(type(x) in (int, float) and math.isfinite(x) for x in draws),
                            "bootstrap draws differ")
                    close(result["se"], statistics.stdev(draws), "bootstrap sample SD")
                critical = (stats.t.ppf(.975, result["df_pvalue"]) if method == "qr_t"
                            else stats.norm.ppf(.975))
                close(result["ci"][0], result["psi"]-critical*result["se"],
                      f"{method} interval lower")
                close(result["ci"][1], result["psi"]+critical*result["se"],
                      f"{method} interval upper")
                p_value = 2*stats.t.sf(abs(result["psi"]/result["se"]), result["df_pvalue"])
                close(result["p_value"], p_value, f"{method} p-value", atol=1e-14)
                rows[cell["id"]][method].append(result)
    saved = read(directory / "summary.json")
    require(saved.get("identity_sha256") == manifest["identity_sha256"] and
            saved.get("terminal") == terminal and
            saved.get("planned") == plan["planned"], "summary totals differ")
    require(saved.get("execution_complete") == (terminal == plan["planned"]),
            "summary completion differs")
    require(saved.get("status") == ("complete" if terminal == plan["planned"] else "incomplete") and
            saved.get("calibration_validated") is False and saved.get("mode") == mode and
            saved.get("protocol_id") == config["protocol_id"], "summary status differs")
    by_id = {cell["cell"]: cell for cell in saved.get("cells", [])}
    require(len(by_id) == len(cells) and set(by_id) == {cell["id"] for cell in cells},
            "summary cell inventory differs")
    for cell in cells:
        for method in REQUIRED_METHODS:
            expected = method_summary([r["methods"][method] for r in records_by_cell[cell["id"]]],
                                      cell["planned"], cell["true_psi"], config["alpha"])
            same(by_id[cell["id"]]["methods"][method], expected, f"summary {cell['id']} {method}")
            require(len(rows[cell["id"]][method]) == len(records_by_cell[cell["id"]]),
                    "method record count differs")
        same(by_id[cell["id"]]["paired"], paired_summary(records_by_cell[cell["id"]],
                                                         cell["true_psi"]), "paired summary")
        close(by_id[cell["id"]]["true_psi"], cell["true_psi"], "summary truth")
    return {"status": "passed", "protocol_id": config["protocol_id"], "mode": mode,
            "planned": plan["planned"], "terminal": terminal,
            "execution_complete": terminal == plan["planned"],
            "source_summary_sha256": hashlib.sha256((directory / "summary.json").read_bytes()).hexdigest(),
            "batch_sha256": batch_hashes,
            "scope": "Independent design, input, inference and summary audit; no model fitting or interval calibration decision."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args(argv)
    if args.report.exists() or args.report.is_symlink() or args.run_directory == args.report or args.run_directory in args.report.parents:
        parser.error("report must be new and outside the run directory")
    result = audit(args.run_directory)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{result['mode']}: {result['terminal']}/{result['planned']} verified terminal cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
