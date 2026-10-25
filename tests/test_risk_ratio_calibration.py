"""Execution and fault checks for the optional risk-ratio validation study."""
from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pytest
from quantilemix import fit_binomial

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str,filename: str):
    spec = importlib.util.spec_from_file_location(name,ROOT/"validation"/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load("risk_ratio_calibration", "risk_ratio_calibration.py")
checker = _load("check_risk_ratio_calibration", "check_risk_ratio_calibration.py")


@pytest.fixture(scope="module")
def pilot_source(tmp_path_factory):
    path = tmp_path_factory.mktemp("risk_ratio_calibration")/"pilot.json"
    runner.run("pilot",path)
    return path


@pytest.fixture
def pilot(pilot_source):
    return json.loads(pilot_source.read_text())


def _checked(tmp_path,payload):
    path = tmp_path/"payload.json"
    paths = runner._paths(path)
    journal = [{"event":"header","run_id":payload["run_id"],"plan":payload["plan"],
                "source_identity":payload["source_identity"]}]
    for row in payload["records"]:
        journal.append({"event":"case_start","cell":row["cell"],
                        "repetition":row["repetition"]})
        journal.append({"event":"case_complete","record":row})
    journal.append({"event":"end","completed":len(payload["records"])})
    paths["journal"].write_text("".join(json.dumps(item)+"\n" for item in journal))
    payload["journal_sha256"] = hashlib.sha256(paths["journal"].read_bytes()).hexdigest()
    path.write_text(json.dumps(payload))
    receipt = {"run_id":payload["run_id"],
               "output_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
               "journal_sha256":payload["journal_sha256"],
               "plan_sha256":hashlib.sha256(json.dumps(payload["plan"],sort_keys=True).encode()).hexdigest(),
               "source_sha256":payload["source_identity"]["source_sha256"],
               "record_count":len(payload["records"])}
    paths["receipt"].write_text(json.dumps(receipt))
    return checker.check(path)


def test_pilot_checks_every_case_and_bootstrap_draw(tmp_path,pilot):
    audit = _checked(tmp_path,pilot)
    assert audit["status"] == "passed"
    assert audit["datasets"] == 20
    assert audit["paired"] == 20
    assert audit["independent_failures"] == 0
    assert len(audit["case_audit"]) == 20
    assert all(len(row["fit"]["bootstrap_effects"]) == 200 for row in pilot["records"])
    assert runner.CELLS[0]["true_log_risk_ratio"] == 0
    assert runner.CELLS[2]["true_log_risk_ratio"] == 0
    assert pilot["journal_sha256"]


def test_last_input_corruption_is_reported(tmp_path,pilot):
    pilot["records"][-1]["input_sha256"] = "0"*64
    audit = _checked(tmp_path,pilot)
    assert audit["status"] == "failed"
    assert any(x["cell"] == 3 and x["repetition"] == 4 and
               x["code"] == "case_audit_failed" for x in audit["issues"])
    assert audit["datasets"] == 20


def test_last_bootstrap_effect_corruption_is_reported(tmp_path,pilot):
    pilot["records"][-1]["fit"]["bootstrap_effects"][-1] += .1
    audit = _checked(tmp_path,pilot)
    assert any(x["code"] == "numerical_mismatch" and x["cell"] == 3 and
               x["repetition"] == 4 for x in audit["issues"])
    assert audit["paired"] == 20


def test_missing_last_case_is_reported(tmp_path,pilot):
    pilot["records"].pop()
    with pytest.raises(ValueError,match="retained journal is not complete"):
        _checked(tmp_path,pilot)
    journal = checker.audit_journal(runner._paths(tmp_path/"payload.json")["journal"])
    assert journal["unattempted"] == [[3,4]] or journal["unattempted"] == [(3,4)]


def test_fabricated_failure_is_replayed_and_reported(tmp_path,pilot):
    fit = pilot["records"][-1]["fit"]
    fit.update({"status":"failed","estimate":None,"se":None,"ci":None,
                "p_value":None,"bootstrap_effects":None,"intervention_risks":None,
                "failure_class":"ValueError","failure_detail":"fabricated",
                "failed_draw":3,"completed_draws":2})
    audit = _checked(tmp_path,pilot)
    codes = {x["code"] for x in audit["issues"]}
    assert {"fabricated_failure","package_fit_failed",
            "independent_success_package_failure"} <= codes
    assert audit["case_audit"][-1]["independent_status"] == "ok"


def test_sparse_study_draw_is_separation_not_a_valid_independent_fit():
    x,z,y = runner.generate("study",0,190)
    rng = np.random.default_rng(runner.bootstrap_seed("study",0,190))
    for _ in range(83):
        sampled = rng.integers(0,len(y),size=len(y),dtype=np.intp)
    with pytest.raises(ValueError,match="check separation") as package_error:
        fit_binomial(y[sampled],x[sampled],z[sampled],q=None)
    with pytest.raises(ValueError,match="logistic separation or boundary"):
        checker.independent_fit(x[sampled],z[sampled],y[sampled])
    with pytest.raises(ValueError,match="independent bootstrap draw 83: logistic separation"):
        checker.independent_case(x,z,y,runner.bootstrap_seed("study",0,190),200)
    saved = {"failure_class":"ValueError","failed_draw":83,
             "failure_detail":f"bootstrap replicate 83 failed: {package_error.value}"}
    assert checker._matching_separation(saved,
        "independent bootstrap draw 83: logistic separation or boundary")
    assert not checker._matching_separation(saved,
        "independent bootstrap draw 82: logistic separation or boundary")


def test_summary_tamper_and_source_tamper_are_detected(tmp_path,pilot):
    pilot["summaries"]["3"]["covered"] += 1
    assert any(x["code"] == "summary_differs" for x in _checked(tmp_path,pilot)["issues"])
    pilot["source_identity"]["source_sha256"]["runner"] = "0"*64
    with pytest.raises(ValueError,match="runner hash differs"):
        _checked(tmp_path,pilot)


def test_independent_two_state_target_algebra_rejects_wrong_truth():
    plan = json.loads(json.dumps(runner.design("pilot")))
    assert checker.independent_targets(plan) == []
    plan["cells"][3]["true_log_risk_ratio"] += .01
    assert checker.independent_targets(plan)[0]["code"] == "analytic_target_differs"


def test_runner_will_not_replace_output_or_partial_journal(tmp_path):
    output = tmp_path/"study.json"
    output.write_text("existing")
    with pytest.raises(ValueError,match="must be new"):
        runner.run("pilot",output)
    output.unlink()
    output.with_name(output.name+".partial.jsonl").write_text("partial")
    with pytest.raises(ValueError,match="must be new"):
        runner.run("pilot",output)


def test_actual_producer_failure_is_saved_with_draw_number(tmp_path,monkeypatch):
    original = runner.fit_binomial_risk_ratio
    calls = 0

    def failing_once(*args,**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("bootstrap replicate 3 failed: injected failure")
        return original(*args,**kwargs)

    monkeypatch.setattr(runner,"fit_binomial_risk_ratio",failing_once)
    path = tmp_path/"failed-fit.json"
    result = runner.run("pilot",path)
    fit = result["records"][0]["fit"]
    assert fit["status"] == "failed"
    assert fit["failure_detail"] == "bootstrap replicate 3 failed: injected failure"
    assert fit["failed_draw"] == 3 and fit["completed_draws"] == 2
    assert runner._paths(path)["journal"].is_file()
    assert any(issue["code"] == "package_fit_failed" for issue in checker.check(path)["issues"])


def test_actual_warning_is_saved(tmp_path,monkeypatch):
    original = runner.fit_binomial_risk_ratio
    calls = 0

    def warning_once(*args,**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            warnings.warn("injected warning",RuntimeWarning)
        return original(*args,**kwargs)

    monkeypatch.setattr(runner,"fit_binomial_risk_ratio",warning_once)
    path = tmp_path/"warning.json"
    result = runner.run("pilot",path)
    assert result["records"][0]["fit"]["warnings"][0]["message"] == "injected warning"
    assert any(issue["code"] == "package_warning" for issue in checker.check(path)["issues"])


def test_interrupted_generation_remains_auditable(tmp_path,monkeypatch):
    original = runner.generate

    def interrupted(mode,cell,repetition):
        if cell == 3 and repetition == 4:
            raise RuntimeError("injected generation stop")
        return original(mode,cell,repetition)

    monkeypatch.setattr(runner,"generate",interrupted)
    path = tmp_path/"interrupted.json"
    with pytest.raises(RuntimeError,match="generation stop"):
        runner.run("pilot",path)
    paths = runner._paths(path)
    assert not path.exists() and paths["lock"].exists()
    journal = checker.audit_journal(paths["partial"])
    assert journal["started"] == 20 and journal["terminal"] == 20
    assert any(issue["code"] == "case_error" for issue in journal["issues"])


def test_nonfinite_record_is_visible_in_partial_journal(tmp_path,monkeypatch):
    original = runner._capture
    calls = 0

    def nonfinite_once(*args,**kwargs):
        nonlocal calls
        calls += 1
        fit = original(*args,**kwargs)
        if calls == 1:
            fit["estimate"] = float("nan")
        return fit

    monkeypatch.setattr(runner,"_capture",nonfinite_once)
    path = tmp_path/"nonfinite.json"
    with pytest.raises(ValueError,match="Out of range float"):
        runner.run("pilot",path)
    journal = checker.audit_journal(runner._paths(path)["partial"])
    assert journal["started"] == 1 and journal["terminal"] == 1
    assert any(issue["code"] == "case_error" for issue in journal["issues"])


def test_interrupted_journal_write_names_started_case(tmp_path,monkeypatch):
    original = runner._sync_line

    def broken_write(handle,text):
        if '"event": "case_complete"' in text and '"cell": 3, "repetition": 4' in text:
            raise OSError("injected journal write failure")
        return original(handle,text)

    monkeypatch.setattr(runner,"_sync_line",broken_write)
    path = tmp_path/"journal-error.json"
    with pytest.raises(OSError,match="journal write failure"):
        runner.run("pilot",path)
    journal = checker.audit_journal(runner._paths(path)["partial"])
    assert journal["started"] == 20 and journal["terminal"] == 19
    assert journal["started_without_terminal"] == [(3,4)]
    assert not journal["end_seen"]


def test_late_competing_output_is_never_replaced(tmp_path,monkeypatch):
    original = runner._publish_no_replace
    path = tmp_path/"competing.json"

    def competitor(source,target):
        if target == path:
            path.write_text("competing writer")
        return original(source,target)

    monkeypatch.setattr(runner,"_publish_no_replace",competitor)
    with pytest.raises(FileExistsError):
        runner.run("pilot",path)
    assert path.read_text() == "competing writer"
    paths = runner._paths(path)
    assert paths["partial"].exists() and paths["lock"].exists()
    assert checker.audit_journal(paths["partial"])["status"] == "complete"


def test_final_publication_failure_keeps_journal_and_lock(tmp_path,monkeypatch):
    def failed_publication(source,target):
        raise OSError("injected publication failure")

    monkeypatch.setattr(runner,"_publish_no_replace",failed_publication)
    path = tmp_path/"failed-publication.json"
    with pytest.raises(OSError,match="publication failure"):
        runner.run("pilot",path)
    paths = runner._paths(path)
    assert not path.exists()
    assert paths["partial"].is_file() and paths["lock"].is_file()
    assert checker.audit_journal(paths["partial"])["status"] == "complete"
    with pytest.raises(ValueError,match="run identity must be new"):
        runner.run("pilot",tmp_path/"same-run-second-output.json")
