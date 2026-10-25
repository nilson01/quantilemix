"""Exercise checkpoint orchestration with literal data and fake method results.

The approved 20-case pilot inventory provides names and shapes only. Its random
generator, numerical fitters, and R adapter are replaced before execution. These
checks are software evidence, not a pilot, study, or timing measurement.
"""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "validation" / "calibration.py"
SPEC = importlib.util.spec_from_file_location("calibration_execution_under_test", SCRIPT)
cal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cal)


def literal_inputs(repetition):
    """Fixed rows with a case marker, deliberately unrelated to the data model."""
    exposure_rows = [[0, 0, 0, 0], [1, 2, 3, 0], [2, 1, 0, 3], [3, 3, 1, 2]]
    covariate_rows = [[0., 1.], [1., 0.], [-1., 0.], [0., -1.]]
    return {"exposures": [row[:] for _ in range(125) for row in exposure_rows],
            "covariates": [row[:] for _ in range(125) for row in covariate_rows],
            "y": [float(repetition)] + [1.] * 499, "q": None, "alpha": .05}


def literal_result(method):
    """A complete response stub; these numbers are not computed from the rows."""
    return {"psi": .25, "variance": .0625, "se": .25, "ci": [-.25, .75],
            "p_value": .2, "df_pvalue": 493 if method == "qr_t" else 495,
            "df_residual": 493}


class LiteralExecution:
    def __init__(self, tmp_path, monkeypatch):
        self.directory = tmp_path / "literal-run"
        self.config_path = tmp_path / "literal-config.json"
        self.config = cal.load_json(SCRIPT.with_name("calibration-config.json"))
        self.config["execution"]["batch_size"] = 2
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        self.reference = {
            "package": "qgcomp", "version": "2.19.6", "mode": "installed package",
            "library_verified": True, "R_version": "synthetic-R", "platform": "synthetic-platform",
            "package_versions": {"qgcomp": "2.19.6"},
            "installed_packages": {"qgcomp": {"version": "2.19.6", "Built": "synthetic"}},
            "package_library_origins": {"qgcomp": "isolated"},
            "unexpected_loaded_namespaces": [],
            "installed_package_md5": {key: "a" * 32 for key in
                                      ("DESCRIPTION", "NAMESPACE", "R/qgcomp.rdb", "R/qgcomp.rdx")},
            "r_runtime_md5": {"bin/exec/R": "b" * 32}}
        self.generated = []
        self.native_calls = []
        self.qr_calls = []
        self.r_calls = []
        self.preflights = 0
        self.native_failures = set()
        self.interrupt_native_at = None
        self.infrastructure_failure_at = None
        self.infrastructure_failure_kind = "exception"
        self.preflight_failure = False

        for key in cal.THREAD_VARIABLES:
            monkeypatch.setenv(key, "1")
        # Only replace this module's disk-capacity dependency. Checkpoint I/O,
        # byte accounting, atomic writes, and lock operations stay real.
        monkeypatch.setattr(cal, "shutil", SimpleNamespace(
            disk_usage=lambda _: SimpleNamespace(free=4_000_000_000)))
        monkeypatch.setattr(cal, "generate_case", self.generate)
        monkeypatch.setattr(cal, "native_result", self.native)
        monkeypatch.setattr(cal, "qr_result", self.qr)
        monkeypatch.setattr(cal, "_load_adapter", lambda: self.adapter)
        monkeypatch.setattr(cal, "fit", self.forbidden_fit)

    @staticmethod
    def forbidden_fit(*args, **kwargs):
        pytest.fail("lifecycle tests must not fit a statistical model")

    @property
    def lock(self):
        return self.directory.with_name(self.directory.name + ".writer.lock")

    def generate(self, config, mode, cell, repetition):
        assert config == self.config and mode == "pilot"
        assert cell["n"] == 500 and cell["planned"] == 20
        self.generated.append(repetition)
        inputs = literal_inputs(repetition)
        return {"name": f"pilot-{cell['id']}-r{repetition:05d}", "cell": cell["id"],
                "repetition": repetition,
                "seed": [config["execution"]["master_seed"], config["pilot"]["seed_namespace"],
                         cell["index"], repetition],
                "inputs": inputs, "input_sha256": cal.digest(inputs)}

    def native(self, inputs):
        repetition = int(inputs["y"][0])
        self.native_calls.append(deepcopy(inputs))
        if repetition == self.interrupt_native_at:
            self.interrupt_native_at = None
            # The public runner must have admitted these inputs before fitting.
            pending = self.checkpoint(repetition // 2, pending=True)
            assert pending.exists()
            assert cal.load_json(pending)["records"][0]["inputs"] == inputs
            raise KeyboardInterrupt
        if repetition in self.native_failures:
            raise ValueError("rank deficient")
        return literal_result("python")

    def qr(self, inputs):
        self.qr_calls.append(deepcopy(inputs))
        return literal_result("qr_t")

    def adapter(self, cases, rscript, library, timeout=180):
        assert rscript == "synthetic-Rscript" and str(library) == "synthetic-library"
        assert timeout == 180
        if not cases:
            self.preflights += 1
            if self.preflight_failure:
                raise FileNotFoundError("synthetic missing executable")
        else:
            self.r_calls.append(deepcopy(cases))
            if int(cases[0]["inputs"]["y"][0]) == self.infrastructure_failure_at:
                self.infrastructure_failure_at = None
                if self.infrastructure_failure_kind == "exception":
                    raise TimeoutError("synthetic reference timeout")
                return {"reference": deepcopy(self.reference), "cases": []}
        return {"reference": deepcopy(self.reference), "cases": [
            {"name": case["name"], "status": "ok", "result": literal_result("r"), "warnings": []}
            for case in cases]}

    def checkpoint(self, index, *, pending=False):
        suffix = ".pending" if pending else ""
        return self.directory / "batches" / f"batch-{index:06d}{suffix}.json"

    def main(self, *, resume=False):
        args = ["pilot", "--execute", "--config", str(self.config_path),
                "--output", str(self.directory), "--rscript", "synthetic-Rscript",
                "--r-library", "synthetic-library"]
        return cal.main(args + (["--resume"] if resume else []))

    def records(self):
        manifest = cal.load_json(self.directory / "manifest.json")
        return cal.read_checkpoints(self.directory, manifest)


@pytest.fixture
def lifecycle(tmp_path, monkeypatch):
    return LiteralExecution(tmp_path, monkeypatch)


def markers(inputs):
    return [int(item["y"][0]) for item in inputs]


def r_markers(lifecycle):
    return markers([case["inputs"] for batch in lifecycle.r_calls for case in batch])


def assert_counts(report, *, attempted, terminal, failures=None):
    failures = failures or {}
    assert report["planned"] == 20
    assert report["attempted"] == attempted
    assert report["terminal"] == terminal
    assert report["missing"] == 20 - terminal
    assert report["completed_batches"] == terminal // 2
    assert report["execution_complete"] is (terminal == 20)
    assert report["calibration_validated"] is False
    assert len(report["cells"]) == 1
    for method in ("python", "r", "qr_t"):
        row = report["cells"][0]["methods"][method]
        failed = failures.get(method, 0)
        assert (row["planned"], row["attempted"], row["successful"], row["failed"],
                row["missing"], row["unattempted"]) == (
                    20, attempted, terminal - failed, failed, 20 - terminal, 20 - attempted)


def assert_completed_inventory(lifecycle):
    records, completed = lifecycle.records()
    assert len(records) == 20 and completed == set(range(10))
    assert sorted(record["repetition"] for record in records.values()) == list(range(20))
    assert all("methods" in record for record in records.values())
    assert lifecycle.generated == list(range(20))
    assert not lifecycle.lock.exists()
    assert not list(lifecycle.directory.rglob("*.tmp"))
    return records


@pytest.mark.parametrize("entrypoint", ["main", "execute"])
def test_literal_run_commits_all_batches_and_reports_completion(lifecycle, capsys, entrypoint):
    if entrypoint == "main":
        assert lifecycle.main() == 0
        captured = capsys.readouterr()
        assert captured.err == ""
        report = json.loads(captured.out)
    else:
        report = cal.execute(lifecycle.config, "pilot", lifecycle.directory,
                             rscript="synthetic-Rscript", library="synthetic-library",
                             adapter=lifecycle.adapter)
    assert report["status"] == "complete_numerical_agreement"
    assert_counts(report, attempted=20, terminal=20)
    records = assert_completed_inventory(lifecycle)
    assert markers(lifecycle.native_calls) == markers(lifecycle.qr_calls) == r_markers(lifecycle) == list(range(20))
    assert lifecycle.preflights == 1
    assert not list((lifecycle.directory / "batches").glob("*.pending.json"))
    assert cal.load_json(lifecycle.directory / "summary.json") == {
        key: value for key, value in report.items() if key != "timing"}
    assert cal.load_json(lifecycle.directory / "timing.json") == report["timing"]
    assert report["timing"]["resumed"] is False
    for record in records.values():
        inputs = literal_inputs(record["repetition"])
        assert record["inputs"] == inputs and record["input_sha256"] == cal.digest(inputs)
    for comparison in report["cells"][0]["paired"].values():
        assert (comparison["agree"], comparison["mismatch"], comparison["unavailable"], comparison["missing"]) == (20, 0, 0, 0)


def test_interrupt_after_pending_resumes_same_inputs_without_refitting_completed_failure(lifecycle, capsys):
    lifecycle.native_failures = {1}
    lifecycle.interrupt_native_at = 2
    assert lifecycle.main() == 130
    captured = capsys.readouterr()
    assert captured.out == "" and json.loads(captured.err)["status"] == "interrupted"
    assert not lifecycle.lock.exists()
    assert not (lifecycle.directory / "summary.json").exists()
    completed_bytes = lifecycle.checkpoint(0).read_bytes()
    manifest_bytes = (lifecycle.directory / "manifest.json").read_bytes()
    pending = cal.load_json(lifecycle.checkpoint(1, pending=True))
    assert not lifecycle.checkpoint(1).exists()
    assert lifecycle.generated == [0, 1, 2, 3]
    assert markers(lifecycle.native_calls) == [0, 1, 2]
    assert markers(lifecycle.qr_calls) == r_markers(lifecycle) == [0, 1]
    partial = cal.aggregate(lifecycle.directory)
    assert partial["status"] == "incomplete"
    assert_counts(partial, attempted=4, terminal=2, failures={"python": 1})

    assert lifecycle.main(resume=True) == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    report = json.loads(captured.out)
    assert report["status"] == "completed_with_failures_or_mismatches"
    assert_counts(report, attempted=20, terminal=20, failures={"python": 1})
    records = assert_completed_inventory(lifecycle)
    assert lifecycle.checkpoint(0).read_bytes() == completed_bytes
    assert (lifecycle.directory / "manifest.json").read_bytes() == manifest_bytes
    assert not lifecycle.checkpoint(1, pending=True).exists()
    assert markers(lifecycle.native_calls) == [0, 1, 2] + list(range(2, 20))
    assert markers(lifecycle.qr_calls) == r_markers(lifecycle) == list(range(20))
    assert lifecycle.preflights == 2 and report["timing"]["resumed"] is True
    for admitted in pending["records"]:
        resumed = records[admitted["name"]]
        assert resumed["inputs"] == admitted["inputs"]
        assert resumed["input_sha256"] == admitted["input_sha256"]
    assert lifecycle.native_calls[2] == lifecycle.native_calls[3] == pending["records"][0]["inputs"]
    for comparison in report["cells"][0]["paired"].values():
        assert (comparison["agree"], comparison["unavailable"], comparison["missing"]) == (19, 1, 0)


def test_interrupt_after_atomic_commit_preserves_success_and_failure_once(lifecycle, capsys, monkeypatch):
    lifecycle.native_failures = {1}
    original_os = cal.os

    class InterruptAfterCommit:
        """Delegate real OS operations without changing the shared os module."""
        interrupted = False

        def __getattr__(self, name):
            return getattr(original_os, name)

        def replace(self, source, destination):
            original_os.replace(source, destination)
            if Path(destination) == lifecycle.checkpoint(0) and not self.interrupted:
                self.interrupted = True
                raise KeyboardInterrupt

    monkeypatch.setattr(cal, "os", InterruptAfterCommit())
    assert lifecycle.main() == 130
    captured = capsys.readouterr()
    assert captured.out == "" and json.loads(captured.err)["status"] == "interrupted"
    assert not lifecycle.lock.exists()
    completed_bytes = lifecycle.checkpoint(0).read_bytes()
    pending = cal.load_json(lifecycle.checkpoint(0, pending=True))
    records, completed = lifecycle.records()
    assert len(records) == 2 and completed == {0}
    assert all("methods" in record for record in records.values())
    partial = cal.aggregate(lifecycle.directory)
    assert_counts(partial, attempted=2, terminal=2, failures={"python": 1})
    assert partial["status"] == "incomplete"

    assert lifecycle.main(resume=True) == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    report = json.loads(captured.out)
    assert report["status"] == "completed_with_failures_or_mismatches"
    assert_counts(report, attempted=20, terminal=20, failures={"python": 1})
    records = assert_completed_inventory(lifecycle)
    assert lifecycle.checkpoint(0).read_bytes() == completed_bytes
    assert markers(lifecycle.native_calls) == markers(lifecycle.qr_calls) == r_markers(lifecycle) == list(range(20))
    assert lifecycle.preflights == 2
    for admitted in pending["records"]:
        record = records[admitted["name"]]
        assert record["inputs"] == admitted["inputs"]
        assert record["input_sha256"] == admitted["input_sha256"]
    first = cal.load_json(lifecycle.checkpoint(0))["records"]
    assert first[0]["methods"]["python"]["status"] == "ok"
    assert first[1]["methods"]["python"]["status"] == "error"


@pytest.mark.parametrize("failure_kind", ["exception", "malformed"])
def test_r_infrastructure_failure_is_terminal_evidence_preserved_on_resume(lifecycle, capsys, failure_kind):
    lifecycle.infrastructure_failure_at = 2
    lifecycle.infrastructure_failure_kind = failure_kind
    assert lifecycle.main() == 2
    captured = capsys.readouterr()
    assert captured.out == "" and json.loads(captured.err)["status"] == "failed"
    assert not lifecycle.lock.exists()
    assert not (lifecycle.directory / "summary.json").exists()
    assert not list((lifecycle.directory / "batches").glob("*.pending.json"))
    completed_bytes = {index: lifecycle.checkpoint(index).read_bytes() for index in (0, 1)}
    failed_batch = cal.load_json(lifecycle.checkpoint(1))
    assert failed_batch["infrastructure_error"]["status"] == "error"
    assert all(record["methods"]["r"]["status"] == "error" for record in failed_batch["records"])
    assert lifecycle.generated == markers(lifecycle.native_calls) == markers(lifecycle.qr_calls) == r_markers(lifecycle) == [0, 1, 2, 3]
    partial = cal.aggregate(lifecycle.directory)
    assert partial["status"] == "incomplete"
    assert_counts(partial, attempted=4, terminal=4, failures={"r": 2})

    assert lifecycle.main(resume=True) == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    report = json.loads(captured.out)
    assert report["status"] == "completed_with_failures_or_mismatches"
    assert_counts(report, attempted=20, terminal=20, failures={"r": 2})
    assert_completed_inventory(lifecycle)
    assert markers(lifecycle.native_calls) == markers(lifecycle.qr_calls) == r_markers(lifecycle) == list(range(20))
    assert lifecycle.preflights == 2
    for index, saved in completed_bytes.items():
        assert lifecycle.checkpoint(index).read_bytes() == saved
    paired = report["cells"][0]["paired"]
    assert (paired["python_vs_r"]["agree"], paired["python_vs_r"]["unavailable"], paired["python_vs_r"]["missing"]) == (18, 2, 0)
    assert paired["python_vs_qr_t"]["agree"] == 20


def test_missing_reference_preflight_does_not_admit_cases_and_releases_lock(lifecycle, capsys):
    lifecycle.preflight_failure = True
    assert lifecycle.main() == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    error = json.loads(captured.err)
    assert error["status"] == "failed" and error["error_class"] == "FileNotFoundError"
    assert not lifecycle.directory.exists() and not lifecycle.lock.exists()
    assert lifecycle.preflights == 1
    assert lifecycle.generated == lifecycle.native_calls == lifecycle.qr_calls == lifecycle.r_calls == []


@pytest.mark.parametrize("python_fails", [False, True])
def test_unexpected_r_degrees_of_freedom_commits_both_cases_before_stopping(lifecycle, capsys, monkeypatch, python_fails):
    if python_fails:
        lifecycle.native_failures = {1}
    def unexpected_df(cases, *args, **kwargs):
        payload = lifecycle.adapter(cases, *args, **kwargs)
        if cases:
            payload["cases"][1]["result"].update(df_pvalue=494, df_residual=492)
        return payload

    monkeypatch.setattr(cal, "_load_adapter", lambda: unexpected_df)
    assert lifecycle.main() == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    error = json.loads(captured.err)
    assert error["status"] == "failed" and "degrees-of-freedom mismatch" in error["error"]
    assert not lifecycle.lock.exists()
    assert not lifecycle.checkpoint(0, pending=True).exists()
    assert not lifecycle.checkpoint(1).exists()
    assert not (lifecycle.directory / "summary.json").exists()
    assert lifecycle.generated == markers(lifecycle.native_calls) == markers(lifecycle.qr_calls) == r_markers(lifecycle) == [0, 1]

    checkpoint = cal.load_json(lifecycle.checkpoint(0))
    assert checkpoint["status"] == "completed" and checkpoint["infrastructure_error"] is None
    good, mismatched = checkpoint["records"]
    assert all(record["methods"]["r"]["status"] == "ok" for record in (good, mismatched))
    assert good["methods"]["r"]["result"] == literal_result("r")
    assert good["comparisons"][0]["status"] == "agree"
    assert mismatched["methods"]["r"]["result"] == {
        **literal_result("r"), "df_pvalue": 494, "df_residual": 492}
    expected_comparison = {
        "comparison": "python_vs_r", "status": "mismatch",
        "fields": ["df_pvalue", "df_residual"],
        "differences": {"df_pvalue": {"python": 495, "other": 494},
                        "df_residual": {"python": 493, "other": 492}}}
    if python_fails:
        expected_comparison.update(status="unavailable", fields=[], differences={})
    assert mismatched["comparisons"][0] == expected_comparison
    assert mismatched["methods"]["r"]["convention_differences"] == {
        "df_pvalue": {"expected": 495, "observed": 494},
        "df_residual": {"expected": 493, "observed": 492}}
    records, completed = lifecycle.records()
    assert len(records) == 2 and completed == {0}
    partial = cal.aggregate(lifecycle.directory)
    assert partial["status"] == "incomplete"
    assert_counts(partial, attempted=2, terminal=2, failures={"python": 1} if python_fails else None)
    paired = partial["cells"][0]["paired"]["python_vs_r"]
    assert (paired["agree"], paired["mismatch"], paired["unavailable"], paired["missing"]) == ((1, 0, 1, 18) if python_fails else (1, 1, 0, 18))
    assert paired["mismatching_cases"] == ([] if python_fails else [mismatched["name"]])
    assert partial["cells"][0]["reference_conventions"] == {
        "checked": 2, "mismatch": 1, "unavailable": 18, "mismatching_cases": [mismatched["name"]]}
