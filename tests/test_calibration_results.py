# SPDX-License-Identifier: GPL-2.0-or-later
"""Tiny literal software checks. No generated study data or fitted models."""

from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "validation" / "check_calibration_results.py"
spec = importlib.util.spec_from_file_location("independent_check_study", SCRIPT)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def literal_entry(psi=0.2, se=0.1, interval=None, p=0.04, method="python"):
    result = {"psi": psi, "se": se, "variance": se*se,
              "ci": interval if interval is not None else [0.0, 0.4],
              "p_value": p, "df_pvalue": 93 if method == "qr_t" else 95,
              "df_residual": 93}
    return {"status": "ok", "result": result, "warnings": []}


def literal_case():
    cell = checker.expected_cells()[0]
    inputs = {"exposures": [[0, 1, 2, 3] for _ in range(100)],
              "covariates": [[0.0, 0.0] for _ in range(100)],
              "y": [1.0 for _ in range(100)], "q": None, "alpha": 0.05}
    methods = {method: literal_entry(method=method) for method in checker.METHODS}
    methods["r"]["convention_differences"] = {}
    return {
        "name": "study-n100-rho0-null-r00000", "cell": cell["id"], "repetition": 0,
        "seed": [20261022, 2, 0, 0], "inputs": inputs, "input_sha256": checker.digest(inputs),
        "methods": methods,
        "comparisons": [{"comparison": f"python_vs_{method}", "status": "agree", "fields": [], "differences": {}}
                        for method in ("r", "qr_t")],
    }


def literal_manifest():
    cells = checker.expected_cells()
    for cell in cells:
        coverage, rejection = ((0.94700655, 0.05006228) if cell["n"] == 100 else (0.94943710, 0.05000227))
        cell["analytic"] = {"condition": "correct normal homoscedastic model and full-rank design",
                            "df_residual": cell["n"]-7, "df_reference_pvalue": cell["n"]-5,
                            "reference_coverage": coverage, "reference_null_rejection": rejection,
                            "qr_t_coverage": 0.95, "qr_t_null_rejection": 0.05}
    identity = {
        "config": deepcopy(checker.EXPECTED_CONFIG), "config_sha256": checker.digest(checker.EXPECTED_CONFIG),
        "mode": "study", "sources_sha256": dict(checker.SOURCE_HASHES),
        "environment": {"python": "literal-test", "numpy": "literal-test", "scipy": "literal-test",
                        "platform": "literal-test", "machine": "literal-test",
                        "numerical_libraries": {"blas": {"name": "literal-test"}},
                        "threads": {key: "1" for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                                    "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}},
        "reference": {"package": "qgcomp", "version": "2.19.6", "mode": "installed package",
                      "library_verified": True, "unexpected_loaded_namespaces": [],
                      "R_version": "literal-test", "platform": "literal-test",
                      "package_versions": {"qgcomp": "2.19.6"},
                      "installed_packages": {"qgcomp": {"version": "2.19.6", "Built": None}},
                      "package_library_origins": {"qgcomp": "isolated"},
                      "installed_package_md5": {key: "0"*32 for key in ("DESCRIPTION", "NAMESPACE", "R/qgcomp.rdb", "R/qgcomp.rdx")},
                      "r_runtime_md5": {"bin/exec/R": "0"*32}},
    }
    plan = {
        "schema_version": 1, "protocol_id": "gaussian-known-scores-v1", "mode": "study",
        "cells": cells, "planned": 24000, "batch_size": 20,
        "input_storage_estimate_upper_bytes": 2236416000,
        "storage_budget_bytes": 3000000000, "storage_estimate_within_budget": True,
        "execution_performed": False, "calibration_validated": False,
        "tolerances": {"atol": 1e-10, "rtol": 1e-8, "p_value_atol": 1e-10},
        "peak_memory_bytes": None,
    }
    return {"schema_version": 1, "identity": identity, "identity_sha256": checker.digest(identity), "plan": plan}


def fresh_accumulators():
    return ({method: checker.MethodTotals(0) for method in checker.METHODS},
            {name: {"agree": 0, "mismatch": 0, "unavailable": 0, "mismatching_cases": []}
             for name in ("python_vs_r", "python_vs_qr_t")},
            {"checked": 0, "mismatch": 0, "unavailable": 0, "mismatching_cases": []}, {})


class LiteralChecks(unittest.TestCase):
    def test_hand_calculated_failed_and_missing_denominators(self):
        totals = checker.MethodTotals(0.25)
        totals.add(literal_entry(), "python")
        totals.add(literal_entry(0.6, 0.2, [0.3, 0.9], 0.05), "python")
        totals.add({"status": "error", "error": "literal failure", "error_class": "LiteralFailure",
                    "error_code": "literal_failure", "warnings": ["LiteralWarning"]}, "python")
        summary = totals.summary(4)
        self.assertEqual([summary[key] for key in ("planned", "attempted", "successful", "failed", "missing", "unattempted")],
                         [4, 3, 2, 1, 1, 1])
        self.assertEqual(summary["covered"], 1)
        self.assertEqual(summary["rejected"], 1)  # p exactly 0.05 is not rejection.
        self.assertEqual(summary["coverage_all_planned_bounds"], [0.25, 0.75])
        self.assertAlmostEqual(summary["bias"], 0.15)
        self.assertAlmostEqual(summary["empirical_sd"], math.sqrt(0.08))
        self.assertAlmostEqual(summary["bias_mcse"], 0.2)
        self.assertAlmostEqual(summary["rms_se"], math.sqrt(0.025))
        self.assertAlmostEqual(summary["mean_interval_width"], 0.5)
        self.assertAlmostEqual(summary["coverage_wilson"][0], 0.09453120573423074)
        self.assertAlmostEqual(summary["coverage_wilson"][1], 0.9054687942657693)
        additional = totals.additional()
        self.assertAlmostEqual(additional["mean_se"], 0.15)
        self.assertAlmostEqual(additional["rmse"], 0.25)
        self.assertAlmostEqual(additional["coverage_mcse"], math.sqrt(0.125))
        self.assertEqual(additional["warning_counts"], {"LiteralWarning": 1})
        altered = deepcopy(summary)
        altered["failed"] = 0
        with self.assertRaises(checker.EvidenceError):
            checker.same(altered, summary, "summary")

    def test_empty_and_one_success_keep_uncertainty_unavailable(self):
        totals = checker.MethodTotals(0)
        empty = totals.summary(4)
        self.assertIsNone(empty["bias"])
        self.assertIsNone(empty["coverage_wilson"])
        self.assertEqual(empty["coverage_all_planned_bounds"], [0.0, 1.0])
        totals.add(literal_entry(), "python")
        one = totals.summary(4)
        self.assertIsNone(one["empirical_sd"])
        self.assertIsNone(one["bias_mcse"])

    def test_valid_literal_case_and_wrong_seed_and_duplicate_position(self):
        record = literal_case()
        accumulators = fresh_accumulators()
        checker.check_case(record, checker.expected_cells()[0], 0, *accumulators)
        self.assertEqual(accumulators[1]["python_vs_r"]["agree"], 1)
        self.assertEqual(accumulators[2]["checked"], 1)
        altered = deepcopy(record)
        altered["seed"][1] = 1
        with self.assertRaisesRegex(checker.EvidenceError, "seed identity"):
            checker.check_case(altered, checker.expected_cells()[0], 0, *fresh_accumulators())
        with self.assertRaisesRegex(checker.EvidenceError, "exact batch position"):
            checker.check_case(record, checker.expected_cells()[0], 1, *fresh_accumulators())

    def test_retained_hash_and_saved_agreement_are_checked(self):
        record = literal_case()
        record["inputs"]["y"][0] = 2.0
        with self.assertRaisesRegex(checker.EvidenceError, "input hash"):
            checker.check_case(record, checker.expected_cells()[0], 0, *fresh_accumulators())
        record = literal_case()
        record["methods"]["r"]["result"]["p_value"] += 2e-10
        with self.assertRaisesRegex(checker.EvidenceError, "saved numerical comparison"):
            checker.check_case(record, checker.expected_cells()[0], 0, *fresh_accumulators())

    def test_failure_is_unavailable_pair_but_sibling_remains_successful(self):
        record = literal_case()
        record["methods"]["r"] = {"status": "error", "error": "literal failure", "error_class": "LiteralFailure", "warnings": []}
        record["comparisons"][0]["status"] = "unavailable"
        totals, pairs, conventions, maxima = fresh_accumulators()
        checker.check_case(record, checker.expected_cells()[0], 0, totals, pairs, conventions, maxima)
        self.assertEqual(totals["python"].summary(2000)["successful"], 1)
        self.assertEqual(totals["r"].summary(2000)["failed"], 1)
        self.assertEqual(pairs["python_vs_r"]["unavailable"], 1)

    def test_wrong_r_df_is_retained_mismatch_not_invalid_result(self):
        record = literal_case()
        record["methods"]["r"]["result"]["df_pvalue"] = 94
        record["methods"]["r"]["convention_differences"] = {"df_pvalue": {"expected": 95, "observed": 94}}
        record["comparisons"][0] = {"comparison": "python_vs_r", "status": "mismatch", "fields": ["df_pvalue"],
                                     "differences": {"df_pvalue": {"python": 95, "other": 94}}}
        totals, pairs, conventions, maxima = fresh_accumulators()
        checker.check_case(record, checker.expected_cells()[0], 0, totals, pairs, conventions, maxima)
        self.assertEqual(conventions["mismatch"], 1)
        self.assertEqual(pairs["python_vs_r"]["mismatch"], 1)
        self.assertEqual(maxima["python_vs_r:df_pvalue"]["absolute_difference"], 1)

    def test_comparison_tolerances_and_qr_interval_exclusion(self):
        methods = literal_case()["methods"]
        methods["qr_t"]["result"]["ci"] = [-1.0, 2.0]
        methods["qr_t"]["result"]["p_value"] = 0.5
        methods["r"]["result"]["p_value"] += 5e-11
        self.assertTrue(all(row["status"] == "agree" for row in checker.compare_methods(methods, {}, "literal")))
        methods["r"]["result"]["p_value"] += 2e-10
        rows = checker.compare_methods(methods, {}, "literal")
        self.assertEqual(rows[0]["fields"], ["p_value"])
        self.assertEqual(rows[1]["status"], "agree")

    def test_manifest_rejects_changed_source_and_incomplete_identity(self):
        manifest = literal_manifest()
        self.assertEqual(len(checker.check_manifest(manifest)), 12)
        changed = deepcopy(manifest)
        changed["identity"]["sources_sha256"]["calibration.py"] = "0"*64
        changed["identity_sha256"] = checker.digest(changed["identity"])
        with self.assertRaisesRegex(checker.EvidenceError, "source hashes"):
            checker.check_manifest(changed)
        for field in ("R_version", "installed_package_md5", "r_runtime_md5"):
            changed = deepcopy(manifest)
            del changed["identity"]["reference"][field]
            changed["identity_sha256"] = checker.digest(changed["identity"])
            with self.assertRaises(checker.EvidenceError):
                checker.check_manifest(changed)
        changed = deepcopy(manifest)
        changed["identity"]["reference"]["installed_packages"]["qgcomp"] = []
        changed["identity_sha256"] = checker.digest(changed["identity"])
        with self.assertRaises(checker.EvidenceError):
            checker.check_manifest(changed)
        for reference in (None, []):
            changed = deepcopy(manifest)
            changed["identity"]["reference"] = reference
            changed["identity_sha256"] = checker.digest(changed["identity"])
            with self.assertRaises(checker.EvidenceError):
                checker.check_manifest(changed)

    def test_infrastructure_failure_must_match_all_r_failures(self):
        error = {"status": "error", "error_class": "LiteralTimeout", "error_code": "r_timeout",
                 "error": "literal infrastructure failure", "warnings": []}
        batch = {"infrastructure_error": error,
                 "records": [{"methods": {"r": deepcopy(error)}} for _ in range(20)]}
        checker.check_infrastructure(batch)
        batch["records"][0]["methods"]["r"] = literal_entry(method="r")
        with self.assertRaisesRegex(checker.EvidenceError, "conflicts with retained R"):
            checker.check_infrastructure(batch)

    def test_json_duplicate_and_nonfinite_rejected_and_raw_hash_bound(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "literal.json"
            for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}'):
                path.write_text(text)
                with self.assertRaises(checker.EvidenceError):
                    checker.read_json(path)
            path.write_text('{ "a": 1 }\n')
            data, identity = checker.read_json(path, with_identity=True)
            self.assertEqual(data, {"a": 1})
            self.assertEqual(identity["sha256"], checker.file_hash(path))

    def test_cli_rejects_partial_study_and_preserves_inputs_and_reports(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            run = root / "literal-run"
            run.mkdir()
            (run / "batches").mkdir()
            (run / "manifest.json").write_text(json.dumps(literal_manifest()))
            (run / "summary.json").write_text("{}")
            before = {p.name: checker.file_hash(p) for p in run.glob("*.json")}
            report = root / "check.json"
            command = [sys.executable, str(SCRIPT), str(run), str(report)]
            finished = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(finished.returncode, 2, finished.stderr)
            result = json.loads(report.read_text())
            self.assertEqual(result["status"], "verification_failed")
            self.assertIn("1200 completed batch", result["error"])
            previous = report.read_bytes()
            second = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(second.returncode, 2)
            self.assertEqual(report.read_bytes(), previous)
            inside = subprocess.run(command[:-1]+[str(run / "forbidden-report.json")], capture_output=True, text=True, check=False)
            self.assertEqual(inside.returncode, 2)
            self.assertFalse((run / "forbidden-report.json").exists())
            self.assertEqual(before, {p.name: checker.file_hash(p) for p in run.glob("*.json")})

    def test_writer_lock_prevents_concurrent_read(self):
        with tempfile.TemporaryDirectory() as name:
            run = Path(name) / "literal-run"
            run.mkdir()
            run.with_name("literal-run.writer.lock").write_text("literal lock")
            with self.assertRaisesRegex(checker.EvidenceError, "writer lock"):
                checker.verify(run)


if __name__ == "__main__":
    unittest.main(verbosity=2)
