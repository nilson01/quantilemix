"""Four-level study execution and independent-audit fault checks."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "validation" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load("risk_ratio_q4_calibration", "risk_ratio_q4_calibration.py")
checker = _load("check_risk_ratio_q4_calibration", "check_risk_ratio_q4_calibration.py")
aggregator = _load("aggregate_risk_ratio_q4", "aggregate_risk_ratio_q4.py")


def _checked(tmp_path, payload):
    path = tmp_path / "payload.json"
    paths = runner._paths(path)
    journal = [{"event": "header", "run_id": payload["run_id"],
                "plan": payload["plan"], "source_identity": payload["source_identity"]}]
    for row in payload["records"]:
        journal.append({"event": "case_start", "cell": row["cell"],
                        "repetition": row["repetition"]})
        journal.append({"event": "case_complete", "record": row})
    journal.append({"event": "end", "completed": len(payload["records"])})
    paths["journal"].write_text("".join(json.dumps(item) + "\n" for item in journal))
    payload["journal_sha256"] = hashlib.sha256(paths["journal"].read_bytes()).hexdigest()
    path.write_text(json.dumps(payload))
    paths["receipt"].write_text(json.dumps({
        "run_id": payload["run_id"],
        "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "journal_sha256": payload["journal_sha256"],
        "plan_sha256": hashlib.sha256(json.dumps(payload["plan"], sort_keys=True).encode()).hexdigest(),
        "source_sha256": payload["source_identity"]["source_sha256"],
        "record_count": len(payload["records"])}))
    return checker.check(path)


@pytest.fixture(scope="module")
def pilot_source(tmp_path_factory):
    path = tmp_path_factory.mktemp("risk_ratio_q4") / "pilot.json"
    runner.run("pilot", path)
    return path


@pytest.fixture
def pilot(pilot_source):
    return json.loads(pilot_source.read_text())


def test_four_level_truth_and_input():
    assert len(runner.CELLS) == 4
    assert runner.CELLS[0]["true_log_risk_ratio"] == 0
    assert runner.CELLS[2]["true_log_risk_ratio"] == 0
    assert checker.independent_targets(runner.design("pilot")) == []
    x, z, y = runner.generate("pilot", 1, 0)
    assert x.shape == (500, 4) and z.shape == (500, 1) and y.shape == (500,)
    assert x.min() == 0 and x.max() == 3


def test_pilot_all_draws_have_independent_match(tmp_path, pilot):
    audit = _checked(tmp_path, pilot)
    assert audit["status"] == "passed"
    assert audit["datasets"] == 20 and audit["paired"] == 20
    assert audit["issue_count"] == 0
    assert all(len(row["fit"]["bootstrap_effects"]) == 200 for row in pilot["records"])


def test_last_draw_corruption_is_detected(tmp_path, pilot):
    pilot["records"][-1]["fit"]["bootstrap_effects"][-1] += .1
    audit = _checked(tmp_path, pilot)
    assert audit["status"] == "failed"
    assert any(issue["code"] == "numerical_mismatch" and issue["cell"] == 3
               and issue["repetition"] == 4 for issue in audit["issues"])


def test_last_case_omission_is_detected(tmp_path, pilot):
    pilot["records"].pop()
    with pytest.raises(ValueError, match="retained journal is not complete"):
        _checked(tmp_path, pilot)


def test_wrong_truth_is_detected():
    plan = json.loads(json.dumps(runner.design("pilot")))
    plan["cells"][3]["true_log_risk_ratio"] += .01
    assert checker.independent_targets(plan)[0]["code"] == "analytic_target_differs"


def test_run_identity_rebinding_is_detected(tmp_path, pilot):
    pilot["run_id"] = "0" * 64
    with pytest.raises(ValueError, match="run identity differs"):
        _checked(tmp_path, pilot)


def test_malformed_partial_journal_reports_problem(tmp_path):
    path = tmp_path / "partial.jsonl"
    header = {"event": "header", "run_id": "test", "plan": runner.design("pilot"),
              "source_identity": {}}
    path.write_text(json.dumps(header) + "\nnot-json\n" +
                    json.dumps({"event": "case_start", "cell": 0, "repetition": 0}) + "\n")
    audit = checker.audit_journal(path)
    assert audit["status"] == "failed"
    assert audit["started_without_terminal"] == [(0, 0)]
    assert any(issue["code"] == "malformed_line" for issue in audit["issues"])


def test_missing_receipt_fails_closed(tmp_path, pilot):
    _checked(tmp_path, pilot)
    path = tmp_path / "payload.json"
    runner._paths(path)["receipt"].unlink()
    with pytest.raises(ValueError, match="completion receipt"):
        checker.check(path)


def test_standard_library_aggregation_and_tamper(pilot):
    fresh = aggregator.aggregate(pilot)
    assert aggregator.compare(pilot["summaries"], fresh) == []
    pilot["summaries"]["3"]["covered"] += 1
    assert "cell 3: covered differs" in aggregator.compare(pilot["summaries"], fresh)


def test_runner_refuses_duplicate_or_partial_output(tmp_path):
    path = tmp_path / "result.json"
    path.write_text("existing")
    with pytest.raises(ValueError, match="must be new"):
        runner.run("pilot", path)
    path.unlink()
    runner._paths(path)["partial"].write_text("interrupted")
    with pytest.raises(ValueError, match="must be new"):
        runner.run("pilot", path)
