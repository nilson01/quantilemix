"""Cox coverage runner guards, deterministic inputs, and audit fault detection."""

import importlib.util
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    path = ROOT / "validation" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("cox_bootstrap_calibration")
checker = load("check_cox_bootstrap_calibration")


def valid_fake_fit(x, *args):
    effects = np.linspace(.1, .5, runner.B)
    estimate = .3
    se = float(effects.std(ddof=1))
    critical = checker.NORMAL.inv_cdf(.975)
    return {"status": "ok", "estimate": estimate, "se": se,
            "ci": [estimate-critical*se, estimate+critical*se],
            "p_value": math.erfc(abs(estimate/se)/math.sqrt(2)),
            "intervention_levels": [0., 1., 2., 3.],
            "bootstrap_effects": effects.tolist(), "failure_class": None,
            "failure_detail": None, "warnings": [], "elapsed_seconds": .01}


def test_failed_draw_record_preserves_partial_diagnostics_and_audit(monkeypatch):
    error = runner.bootstrap_implementation.CoxBootstrapFitError(
        "bootstrap replicate 3 failed: Cox score did not converge",
        draw=3, effects=np.array([[.1], [.2]]), estimate=np.array([.3]),
        levels=np.array([0., 1., 2., 3.]),
    )

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(runner, "fit_cox_bootstrap", fail)
    record = runner._fit_one(np.zeros((4, 4)), np.ones(4), np.ones(4), 7)
    assert record["status"] == "failed"
    assert record["bootstrap_effects"] == [.1, .2]
    assert record["original_estimate"] == .3
    assert record["failed_draw"] == 3
    assert record["completed_draws"] == 2
    checker._audit_fit(record, "injected failure")
    record["completed_draws"] = 1
    with pytest.raises(ValueError, match="partial bootstrap record invalid"):
        checker._audit_fit(record, "tampered failure")


def admit_pilot(monkeypatch):
    monkeypatch.setitem(runner.RUNS_ADMITTED, "pilot", True)
    monkeypatch.setitem(checker.runner.RUNS_ADMITTED, "pilot", True)
    original = runner.provenance
    monkeypatch.setattr(runner, "provenance",
                        lambda mode: original(mode, enforce_clean=False))


def test_committed_source_guard_rejects_a_different_file(tmp_path):
    original = tmp_path / "original.py"
    original.write_text("# committed source\n")
    runner.subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    runner.subprocess.run(["git", "add", "original.py"], cwd=tmp_path, check=True)
    runner.subprocess.run(["git", "-c", "user.name=Package Test", "-c",
                           "user.email=package-test@example.invalid", "commit", "-qm", "fixture"],
                          cwd=tmp_path, check=True)
    changed = tmp_path / "different.py"
    changed.write_text("# changed source\n")
    commit = runner.subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path,
                                   capture_output=True, text=True, check=True).stdout.strip()
    with pytest.raises(ValueError, match="differs from commit"):
        runner._check_committed_file(tmp_path, commit, "original.py", changed)


def test_terminal_pilot_auditor_rejects_nonzero_task_exit(tmp_path, monkeypatch):
    path = ROOT / "validation" / "server_cox_bootstrap" / "audit_pilot.py"
    spec = importlib.util.spec_from_file_location("cox_pilot_auditor", path)
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    (tmp_path / "code").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "SOURCE_REVISION.txt").write_text("abc\n")
    for filename, digest in (("source.bundle", "CODE_BUNDLE_SHA256.txt"),
                             (auditor.WHEEL, "WHEEL_SHA256.txt")):
        (tmp_path / filename).write_bytes(b"frozen")
        (tmp_path / digest).write_text(auditor._hash(tmp_path / filename)+"\n")
    for index in range(20):
        directory = tmp_path / f"pilot-case-{index}"
        directory.mkdir()
        (directory / "result.json").write_text("{}\n")
        (directory / "exit-code.txt").write_text("0\n")
        (directory / "output-sha256.txt").write_text(
            f"{auditor._hash(directory / 'result.json')}  result.json\n")
    def fake_run(args, **kwargs):
        if args[:2] == ["git", "rev-parse"]:
            return SimpleNamespace(stdout="abc\n")
        if args[:2] == ["git", "status"]:
            return SimpleNamespace(stdout="")
        assert args[0] == "sacct"
        assert "--format=JobID,State,ExitCode,ElapsedRaw,MaxRSS" in args
        lines = [f"42_{index}|COMPLETED|0:0|600|\n"
                 f"42_{index}.batch|COMPLETED|0:0|600|64M\n"
                 for index in range(20)]
        return SimpleNamespace(stdout="".join(lines))
    monkeypatch.setattr(auditor.subprocess, "run", fake_run)
    monkeypatch.setattr(auditor.checker, "audit_files", lambda *args, **kwargs: {
        "status": "passed", "audited_records": 20,
        "summaries": {str(i): {"successful": 5, "failed": 0, "warnings": 0}
                      for i in range(4)}})
    assert auditor.audit(tmp_path, "42")["accepted"]
    (tmp_path / "pilot-case-7" / "exit-code.txt").write_text("1\n")
    result = auditor.audit(tmp_path, "42")
    assert not result["accepted"]
    assert "task 7: nonzero wrapper exit" in result["issues"]


def test_study_auditor_checks_array_mapping_and_wrapper_exit(tmp_path, monkeypatch):
    path = ROOT / "validation" / "server_cox_bootstrap" / "audit_study.py"
    spec = importlib.util.spec_from_file_location("cox_study_auditor", path)
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    monkeypatch.setattr(auditor, "TASKS", 201)
    (tmp_path / "code").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "SOURCE_REVISION.txt").write_text("abc\n")
    for filename, digest in (("source.bundle", "CODE_BUNDLE_SHA256.txt"),
                             (auditor.WHEEL, "WHEEL_SHA256.txt")):
        (tmp_path / filename).write_bytes(b"frozen")
        (tmp_path / digest).write_text(auditor._hash(tmp_path / filename)+"\n")
    for index in range(auditor.TASKS):
        directory = tmp_path / f"study-batch-{index}"
        directory.mkdir()
        batch = {"cell": index // 200, "start": (index % 200)*10, "count": 10}
        (directory / "result.json").write_text(json.dumps({
            "batch": batch, "plan": {"mode": "study"}}))
        for name in auditor.MANIFEST_FILES - {"result.json"}:
            (directory / name).write_text(f"{name}\n")
        (directory / "exit-code.txt").write_text("0\n")
        (directory / "output-sha256.txt").write_text(
            "".join(f"{auditor._hash(directory / name)}  {name}\n"
                    for name in sorted(auditor.MANIFEST_FILES)))

    missing_rss = set()
    excessive_rss = set()
    malformed_rows = set()

    def fake_run(args, **kwargs):
        if args[:2] == ["git", "rev-parse"]:
            return SimpleNamespace(stdout="abc\n")
        if args[:2] == ["git", "status"]:
            return SimpleNamespace(stdout="")
        assert args[0] == "sacct"
        return SimpleNamespace(stdout="".join(
            (f"42_{index}|COMPLETED|0:0\n" if index in malformed_rows
             else f"42_{index}|COMPLETED|0:0|7200|\n")
            +
            f"42_{index}.batch|COMPLETED|0:0|7200|"
            f"{'' if index in missing_rss else ('3G' if index in excessive_rss else '64M')}\n"
            for index in range(auditor.TASKS)))

    monkeypatch.setattr(auditor.subprocess, "run", fake_run)
    monkeypatch.setattr(auditor.checker, "audit_files", lambda *args, **kwargs: {
        "status": "passed", "mode": "study", "audited_records": 2010})
    assert auditor.audit(tmp_path, "42")["accepted"]
    missing_rss.add(1)
    result = auditor.audit(tmp_path, "42")
    assert "task 1: MaxRSS missing or above 2 GiB" in result["issues"]
    assert "task 10: MaxRSS missing or above 2 GiB" not in result["issues"]
    missing_rss.clear()
    excessive_rss.add(100)
    result = auditor.audit(tmp_path, "42")
    assert "task 100: MaxRSS missing or above 2 GiB" in result["issues"]
    assert "task 10: MaxRSS missing or above 2 GiB" not in result["issues"]
    excessive_rss.clear()
    malformed_rows.add(100)
    assert "Slurm accounting has malformed rows" in auditor.audit(tmp_path, "42")["issues"]
    malformed_rows.clear()
    directory = tmp_path / "study-batch-200"
    manifest = directory / "output-sha256.txt"
    original_manifest = manifest.read_text()
    for damaged in ("", original_manifest.splitlines()[0]+"\n",
                    "not a digest line\n"):
        manifest.write_text(damaged)
        assert "task 200: manifest lacks required files" in auditor.audit(tmp_path, "42")["issues"]
    manifest.write_text(original_manifest+original_manifest.splitlines()[0]+"\n")
    assert any("duplicate manifest entry" in issue for issue in auditor.audit(tmp_path, "42")["issues"])
    manifest.write_text(original_manifest)
    (directory / "result.json").write_text(json.dumps({
        "batch": {"cell": 0, "start": 0, "count": 10},
        "plan": {"mode": "study"}}))
    result = auditor.audit(tmp_path, "42")
    assert not result["accepted"]
    assert "task 200: batch mapping or mode differs" in result["issues"]
    (directory / "exit-code.txt").write_text("1\n")
    result = auditor.audit(tmp_path, "42")
    assert "task 200: nonzero wrapper exit" in result["issues"]


@pytest.mark.skipif(os.name == "nt", reason="Slurm wrapper requires a POSIX shell")
@pytest.mark.parametrize("failure", ["bad_receipt", "missing_python"])
def test_study_wrapper_cancels_array_on_early_preflight_failure(tmp_path, failure):
    root = tmp_path / "qgcomp-cox-bootstrap-study-001"
    root.mkdir()
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    (stubs / "module").write_text("#!/bin/sh\nexit 0\n")
    (stubs / "scancel").write_text(
        '#!/bin/sh\nprintf "%s\\n" "$1" >> "$CANCEL_LOG"\n')
    for name in ("module", "scancel"):
        (stubs / name).chmod(0o755)
    (root / "source.bundle").write_bytes(b"source")
    (root / "CODE_BUNDLE_SHA256.txt").write_text(
        ("0"*64 if failure == "bad_receipt" else hashlib.sha256(b"source").hexdigest())+"\n")
    if failure == "bad_receipt":
        executable = root / "python/bin/python"
        executable.parent.mkdir(parents=True)
        executable.write_text("#!/bin/sh\nexit 0\n")
        executable.chmod(0o755)
    cancel_log = tmp_path / "cancel.log"
    env = dict(os.environ, SLURM_SUBMIT_DIR=str(root), SLURM_CPUS_PER_TASK="1",
               SLURM_ARRAY_TASK_ID="0", SLURM_ARRAY_JOB_ID="42",
               CANCEL_LOG=str(cancel_log), PATH=str(stubs)+os.pathsep+os.environ["PATH"])
    completed = subprocess.run(["bash", str(ROOT / "validation/server_cox_bootstrap/study.sbatch")],
                               cwd=root, env=env, capture_output=True, text=True, check=False)
    assert completed.returncode != 0
    assert cancel_log.read_text().strip() == "42"
    output = root / "study-batch-0"
    assert (output / "exit-code.txt").read_text().strip() != "0"
    assert (output / "stop-reason.txt").is_file()


def test_study_auditor_cli_preserves_rejected_missing_dependency_receipt(
        tmp_path, monkeypatch):
    path = ROOT / "validation/server_cox_bootstrap/audit_study.py"
    spec = importlib.util.spec_from_file_location("cox_study_auditor_cli", path)
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    def missing_accounting(*args):
        raise FileNotFoundError("sacct unavailable")
    monkeypatch.setattr(auditor, "audit", missing_accounting)
    output = tmp_path / "study-acceptance.json"
    monkeypatch.setattr(sys, "argv", ["audit_study.py", "--root", str(tmp_path),
                                   "--job", "42", "--output", str(output)])
    assert auditor.main() == 1
    receipt = json.loads(output.read_text())
    assert not receipt["accepted"]
    assert "FileNotFoundError" in receipt["issues"][0]


def test_fixed_design_and_distinct_deterministic_streams():
    pilot, study = runner.design("pilot"), runner.design("study")
    assert pilot["total_datasets"] == 20
    assert study["total_datasets"] == 8000
    assert pilot["B"] == study["B"] == 200
    assert pilot["run_admitted"] and study["run_admitted"]
    assert pilot["q"] == 4 and len(pilot["breaks"]) == 4
    assert all(list(column) == [-.5, .5, 1.5, 2.5, 3.5]
               for column in pilot["breaks"])
    assert runner.fit_seed("pilot", 0, 0) != runner.fit_seed("study", 0, 0)
    x, duration, event = runner.generate("pilot", 0, 0)
    assert runner.input_hash(x, duration, event) == (
        "4a4d3b2108723168a14459cdf798581c930f5a519b00c136b4186bdc90b6d50b")
    assert int(event.sum()) == 129
    again = runner.generate("pilot", 0, 0)
    assert all(np.array_equal(a, b) for a, b in zip((x, duration, event), again))
    assert len(np.unique(x)) == 4
    assert np.all((duration > 0) & (duration <= runner.ADMIN_END))
    assert set(np.unique(event)).issubset({0., 1.})
    assert runner.input_hash(x, duration, event) != runner.input_hash(
        *runner.generate("study", 0, 0))
    with pytest.raises(ValueError, match="outside plan"):
        runner.generate("pilot", 4, 0)


def test_fixed_breaks_preserve_scores_when_prequantized_levels_repeat():
    from quantilemix import quantize

    x, _, _ = runner.generate("pilot", 1, 1)
    assert np.quantile(x.ravel(), [.25, .5, .75]).tolist() == [1., 2., 2.]
    fitted = quantize(x, q=4, breaks=runner.BREAKS)
    assert np.array_equal(fitted.scores, x)
    assert fitted.category_counts == (4, 4, 4, 4)


def test_admitted_study_still_rejects_out_of_range_batches(tmp_path):
    output = tmp_path / "invalid.json"
    with pytest.raises(ValueError, match="nonempty in-range"):
        runner.run_batch("study", 0, 1999, 10, output)
    with pytest.raises(ValueError, match="nonempty in-range"):
        runner.run_batch("study", 4, 0, 10, output)
    assert not output.exists()


def test_study_batch_audits_ten_distinct_fresh_cases(tmp_path, monkeypatch):
    monkeypatch.setitem(runner.RUNS_ADMITTED, "study", True)
    monkeypatch.setitem(checker.runner.RUNS_ADMITTED, "study", True)
    original = runner.provenance
    monkeypatch.setattr(runner, "provenance",
                        lambda mode: original(mode, enforce_clean=False))
    monkeypatch.setattr(runner, "_fit_one", valid_fake_fit)
    output = tmp_path / "study-batch.json"
    report = runner.run_batch("study", 1, 1990, 10, output)
    assert [row["repetition"] for row in report["records"]] == list(range(1990, 2000))
    assert {row["cell"] for row in report["records"]} == {1}
    assert {row["input_sha256"] for row in report["records"]}.isdisjoint({
        runner.input_hash(*runner.generate("pilot", 1, repetition))
        for repetition in range(5)})
    audited = checker.audit_files([output], complete=False)
    assert audited["audited_records"] == 10
    assert audited["summaries"]["1"]["successful"] == 10
    cell = audited["summaries"]["1"]
    expected_admin = sum(row["administratively_censored"] for row in report["records"])
    expected_events = sum(row["observed_events"] for row in report["records"])
    assert cell["administrative_censoring_fraction"] == pytest.approx(
        expected_admin/(10*runner.N))
    assert cell["independent_censoring_fraction"] == pytest.approx(
        1-(expected_admin+expected_events)/(10*runner.N))
    assert cell["censored_fraction"] == pytest.approx(
        cell["administrative_censoring_fraction"]
        + cell["independent_censoring_fraction"])


def test_batch_audit_recomputes_interval_and_detects_corruption(tmp_path, monkeypatch):
    admit_pilot(monkeypatch)
    monkeypatch.setattr(runner, "_fit_one", valid_fake_fit)
    output = tmp_path / "batch.json"
    report = runner.run_batch("pilot", 0, 0, 2, output)
    assert len(report["records"]) == 2
    assert not (tmp_path / "batch.json.partial.jsonl").exists()
    audited = checker.audit_files([output], complete=False)
    assert audited["audited_records"] == 2
    assert audited["summaries"]["0"]["successful"] == 2
    with pytest.raises(ValueError, match="complete audit"):
        checker.audit_files([output], complete=True)
    with pytest.raises(ValueError, match="duplicate"):
        checker.audit_files([output, output], complete=False)
    with pytest.raises(ValueError, match="new paths"):
        runner.run_batch("pilot", 0, 0, 2, output)
    damaged = json.loads(output.read_text())
    damaged["records"][0]["fit"]["se"] += .01
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="independent arithmetic"):
        checker.audit_files([bad], complete=False)
    damaged = json.loads(output.read_text())
    damaged["source_identity"]["distribution_record_sha256"] = "0"*64
    bad_environment = tmp_path / "bad-environment.json"
    bad_environment.write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="installed environment"):
        checker.audit_files([bad_environment], complete=False)


def test_failed_fit_is_kept_and_counted(tmp_path, monkeypatch):
    admit_pilot(monkeypatch)
    def failed(*args):
        return {"status": "failed", "estimate": None, "se": None, "ci": None,
                "p_value": None, "bootstrap_effects": None,
                "intervention_levels": None,
                "failure_class": "ValueError", "failure_detail": "test failure",
                "warnings": [], "elapsed_seconds": .01}
    monkeypatch.setattr(runner, "_fit_one", failed)
    output = tmp_path / "failed.json"
    runner.run_batch("pilot", 0, 0, 1, output)
    audited = checker.audit_files([output], complete=False)
    summary = audited["summaries"]["0"]
    assert summary["attempted"] == summary["failed"] == 1
    assert summary["successful"] == 0
    assert summary["coverage_all_planned_bounds"] == [0, 1]
    assert summary["rejection_all_planned_bounds"] == [0, 1]


def test_nonfinite_fit_is_recorded_as_failure(tmp_path, monkeypatch):
    admit_pilot(monkeypatch)
    def nonfinite(*args, **kwargs):
        return SimpleNamespace(
            psi=np.array([float("nan")]), se=np.array([.1]),
            ci=np.array([[0., 1.]]), p_value=np.array([.5]),
            bootstrap_effects=np.zeros((runner.B, 1)),
            intervention_levels=np.array([0., 1., 2.]))
    monkeypatch.setattr(runner, "fit_cox_bootstrap", nonfinite)
    output = tmp_path / "nonfinite.json"
    runner.run_batch("pilot", 0, 0, 1, output)
    audited = checker.audit_files([output], complete=False)
    assert audited["summaries"]["0"]["failed"] == 1
    assert json.loads(output.read_text())["records"][0]["fit"]["failure_class"] == (
        "NonfiniteOrUnserializableFit")


def test_interrupted_journal_accounts_for_started_case_and_rejects_duplicate(
        tmp_path, monkeypatch):
    admit_pilot(monkeypatch)
    calls = 0
    def interrupted(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt()
        return valid_fake_fit(*args)
    monkeypatch.setattr(runner, "_fit_one", interrupted)
    output = tmp_path / "interrupted.json"
    with pytest.raises(KeyboardInterrupt):
        runner.run_batch("pilot", 0, 0, 3, output)
    partial = tmp_path / "interrupted.json.partial.jsonl"
    assert partial.exists() and not output.exists()
    audited = checker.audit_files([partial], complete=False)
    assert audited["interrupted_case_ids"] == [[0, 1]]
    assert audited["journal_exit_statuses"] == ["interrupted"]
    summary = audited["summaries"]["0"]
    assert (summary["attempted"], summary["interrupted"],
            summary["never_started"], summary["successful"]) == (2, 1, 3, 1)
    with pytest.raises(ValueError, match="partial journals"):
        checker.audit_files([partial], complete=True)
    torn = tmp_path / "torn.partial.jsonl"
    lines = partial.read_text().splitlines()
    torn.write_text("\n".join(lines[:-1])+"\n"+'{"kind":"record",')
    torn_audit = checker.audit_files([torn], complete=False)
    assert torn_audit["journal_exit_statuses"] == ["truncated_unconfirmed_exit"]
    assert torn_audit["interrupted_case_ids"] == [[0, 1]]
    monkeypatch.setattr(runner, "_fit_one", valid_fake_fit)
    duplicate = tmp_path / "duplicate.json"
    runner.run_batch("pilot", 0, 0, 1, duplicate)
    with pytest.raises(ValueError, match="duplicate"):
        checker.audit_files([partial, duplicate], complete=False)


@pytest.mark.parametrize(
    ("cell", "failed_replicate", "reference_coefficient"),
    [(2, 72, -0.132083323393032), (3, 43, 0.0772201193120002)],
)
def test_pilot_roundoff_failures_converge_to_independent_score_root(
        cell, failed_replicate, reference_coefficient):
    """Replay failed pilot draws against installed R survival 3.8.3 coefficients."""
    from quantilemix import fit_cox
    from quantilemix._cox import _partial_likelihood, _solve_cox
    from quantilemix._cox_bootstrap import _levels, _simulate_survival

    x, duration, event = runner.generate("pilot", cell, 0)
    point = fit_cox(duration, event, x, q=None)
    levels = _levels(point)
    rng = np.random.default_rng(runner.fit_seed("pilot", cell, 0))
    rng.integers(0, len(x), size=runner.MCSIZE, dtype=np.intp)
    rng.exponential(size=(len(levels), runner.MCSIZE))
    for replicate in range(1, failed_replicate+1):
        sampled = rng.integers(0, len(x), size=len(x), dtype=np.intp)
        if replicate < failed_replicate:
            rng.integers(0, len(x), size=runner.MCSIZE, dtype=np.intp)
            rng.exponential(size=(len(levels), runner.MCSIZE))
    conditional = fit_cox(duration[sampled], event[sampled], x[sampled], q=None)
    rng.integers(0, len(x), size=runner.MCSIZE, dtype=np.intp)
    thresholds = rng.exponential(size=(len(levels), runner.MCSIZE))
    simulated_time, simulated_event, simulated_level = _simulate_survival(
        conditional, levels, np.empty((runner.MCSIZE, 0)), thresholds)
    order = np.argsort(-simulated_time, kind="stable")
    design = simulated_level[order, None]
    beta, _, _, _ = _solve_cox(design, simulated_event[order], simulated_time[order])
    _, score, _ = _partial_likelihood(
        design, simulated_event[order], beta, simulated_time[order])
    assert beta[0] == pytest.approx(reference_coefficient, abs=1e-8)
    assert abs(score[0]) < 1e-7
