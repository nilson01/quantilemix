"""Software checks for the prospective bootstrap runner; no study is run."""

import copy
from itertools import product
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "validation/bootstrap_calibration.py"
SPEC = importlib.util.spec_from_file_location("bootstrap_calibration_under_test", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
CHECK_SPEC = importlib.util.spec_from_file_location(
    "bootstrap_calibration_checker", SCRIPT.with_name("check_bootstrap_calibration.py"))
checker = importlib.util.module_from_spec(CHECK_SPEC)
CHECK_SPEC.loader.exec_module(checker)


def literal_case(repetition=0, config=None):
    rows = np.array(list(product((0., 1.), repeat=4)))
    x = np.tile(rows, (32, 1))[:500]
    z = np.column_stack((x[:, 0]*x[:, 1], x[:, 2]*x[:, 3]))
    y = 1 + x @ np.array([.1, .1, .05, 0.]) + z @ np.array([.5, -.25])
    y += .2*np.tile(np.arange(5)-2., 100)
    inputs = {"exposures": x.tolist(), "covariates": z.tolist(),
              "y": y.tolist(), "q": None, "alpha": .05}
    config = config or runner.load_config()
    return {"name": f"pilot-n500-rho0p75-positive-r{repetition:05d}",
            "cell": "n500-rho0p75-positive",
            "repetition": repetition,
            "seed": [config["master_seed"], config["pilot_seed_namespace"], 11, repetition],
            "inputs": inputs, "input_sha256": runner.digest(inputs),
            "bootstrap_seed": 17+repetition}


def test_frozen_config_and_fresh_seed_namespaces():
    config = runner.load_config()
    pilot = runner.plan(config, "pilot")
    study = runner.plan(config, "study")
    assert pilot["planned"] == 20 and pilot["fits_expected"] == 4000
    assert study["planned"] == 24000 and study["fits_expected"] == 4_800_000
    first = runner.make_case(config, "pilot", pilot["cells"][0], 0)
    again = runner.make_case(config, "pilot", pilot["cells"][0], 0)
    assert first == again
    assert first["seed"] != [20261022, 1, 11, 0]
    assert first["bootstrap_seed"] != runner.bootstrap_seed(config, "study", study["cells"][0], 0)
    changed = dict(config, bootstrap_resamples=201)
    path = SCRIPT.with_name("bootstrap-calibration-config.json")
    assert runner.load_config(path) == config
    with pytest.raises(runner.BootstrapStudyError):
        runner.plan(changed, "unknown")


@pytest.mark.parametrize("change", ["original_imports", "other_edit"])
def test_generator_identity_limits_namespace_compatibility(tmp_path, monkeypatch, change):
    source = runner.GENERATOR.read_bytes()
    if change == "original_imports":
        source = source.replace(b"import quantilemix", b"import qgcomp")
        source = source.replace(b"from quantilemix import fit", b"from qgcomp import fit")
    else:
        source += b"\n# Source changed beyond the import namespace.\n"
    generator = tmp_path / "calibration.py"
    generator.write_bytes(source)
    monkeypatch.setattr(runner, "GENERATOR", generator)
    if change == "original_imports":
        assert runner.load_config()["data_generator_sha256"] == runner.generator_design_hash()
    else:
        with pytest.raises(runner.BootstrapStudyError, match="generator identity"):
            runner.load_config()
        monkeypatch.setattr(checker, "GENERATOR", generator)
        with pytest.raises(checker.EvidenceError, match="generator identity"):
            checker.audit(tmp_path)


def test_independent_input_check_allows_only_tiny_float_rounding():
    config = runner.load_config()
    cell = runner.plan(config, "pilot")["cells"][0]
    expected = checker.expected_inputs(config, "pilot", cell, 0)
    rounded = copy.deepcopy(expected)
    rounded["covariates"][0][0] = float(np.nextafter(rounded["covariates"][0][0], np.inf))
    assert checker.generated_inputs_agree(rounded, expected)
    rounded["y"][0] += 1e-12
    assert not checker.generated_inputs_agree(rounded, expected)
    rounded = copy.deepcopy(expected)
    rounded["exposures"][0][0] = (rounded["exposures"][0][0] + 1) % 4
    assert not checker.generated_inputs_agree(rounded, expected)


def test_literal_case_and_saved_draw_validation():
    case = literal_case()
    result = runner.fit_case(case, 8)
    assert {name: item["status"] for name, item in result["methods"].items()} == {
        "bootstrap": "ok", "model_based": "ok", "qr_t": "ok"}
    boot = result["methods"]["bootstrap"]["result"]
    model = result["methods"]["model_based"]["result"]
    assert abs(boot["psi"]-model["psi"]) < 1e-12
    assert len(boot["bootstrap_effects"]) == 8
    assert boot["df_pvalue"] == len(case["inputs"]["y"]) - 8
    runner.validate_method_result(boot, len(case["inputs"]["y"]), "bootstrap", 8)
    broken = dict(boot, bootstrap_effects=[*boot["bootstrap_effects"]])
    broken["bootstrap_effects"][0] += 10
    with pytest.raises(runner.BootstrapStudyError, match="variance"):
        runner.validate_method_result(broken, len(case["inputs"]["y"]), "bootstrap", 8)


def test_checkpoint_resume_preserves_completed_results_and_rejects_corruption(tmp_path, monkeypatch):
    config = dict(runner.load_config(), bootstrap_resamples=8)
    for key in runner.THREADS:
        monkeypatch.setenv(key, "1")
    output = tmp_path / "run"
    first = runner.run(config, "pilot", output)
    assert first["status"] == "complete" and first["terminal"] == 20
    paired = first["cells"][0]["paired"]
    assert paired["mean_coverage_difference"] == (
        paired["bootstrap_only_covered"] - paired["model_based_only_covered"]) / 20
    test_config = tmp_path / "config.json"
    test_config.write_text(json.dumps(config))
    monkeypatch.setattr(checker, "CONFIG", test_config)
    monkeypatch.setattr(checker, "FROZEN_CONFIG_DIGEST", checker.digest(config))
    verified = checker.audit(output)
    assert verified["status"] == "passed" and verified["terminal"] == 20
    assert verified["execution_complete"] is True
    summary_path = output / "summary.json"
    original_summary = summary_path.read_bytes()
    altered_summary = json.loads(original_summary)
    altered_summary["cells"][0]["methods"]["bootstrap"]["covered"] += 1
    summary_path.write_text(json.dumps(altered_summary))
    with pytest.raises(checker.EvidenceError, match="summary"):
        checker.audit(output)
    summary_path.write_bytes(original_summary)
    altered_summary = json.loads(original_summary)
    altered_summary["cells"][0]["methods"]["bootstrap"]["bias_mcse"] += .1
    summary_path.write_text(json.dumps(altered_summary))
    with pytest.raises(checker.EvidenceError, match="bias_mcse"):
        checker.audit(output)
    summary_path.write_bytes(original_summary)
    saved = (output / "batches/batch-000000.json").read_bytes()
    payload = json.loads(saved)
    payload["records"][0]["methods"]["bootstrap"]["result"]["ci"][0] += .1
    batch_path = output / "batches/batch-000000.json"
    batch_path.write_text(json.dumps(payload))
    with pytest.raises(checker.EvidenceError, match="interval lower"):
        checker.audit(output)
    batch_path.write_bytes(saved)
    payload = json.loads(saved)
    payload["records"][0]["inputs"]["y"][0] += 1
    payload["records"][0]["input_sha256"] = checker.digest(payload["records"][0]["inputs"])
    batch_path.write_text(json.dumps(payload))
    with pytest.raises(checker.EvidenceError, match="independently generated design"):
        checker.audit(output)
    batch_path.write_bytes(saved)
    payload = json.loads(saved)
    payload["records"][0]["methods"]["bootstrap"]["result"]["p_value"] = .99
    batch_path.write_text(json.dumps(payload))
    with pytest.raises(checker.EvidenceError, match="p-value"):
        checker.audit(output)
    batch_path.write_bytes(saved)
    manifest_path = output / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    altered_manifest = json.loads(manifest_bytes)
    altered_manifest["plan"]["cells"][0]["true_psi"] = 0
    manifest_path.write_text(json.dumps(altered_manifest))
    with pytest.raises(checker.EvidenceError, match="cell design true_psi"):
        checker.audit(output)
    manifest_path.write_bytes(manifest_bytes)
    payload = json.loads(saved)
    payload["records"][0]["bootstrap_seed"] += 1
    batch_path.write_text(json.dumps(payload))
    with pytest.raises(checker.EvidenceError, match="bootstrap seed"):
        checker.audit(output)
    batch_path.write_bytes(saved)
    payload = json.loads(saved)
    payload["records"][0]["methods"]["bootstrap"] = {
        "status": "error", "error_class": "LinAlgError",
        "error_code": "singular_design", "warnings": []}
    batch_path.write_text(json.dumps(payload))
    failed_summary = runner.summarize(payload["records"], runner.plan(config, "pilot"))
    failed_summary.update({key: json.loads(original_summary)[key] for key in
                           ("identity_sha256", "elapsed_seconds", "output_bytes")})
    summary_path.write_text(json.dumps(failed_summary))
    failure_check = checker.audit(output)
    assert failure_check["execution_complete"] is True
    assert failed_summary["cells"][0]["methods"]["bootstrap"]["failed"] == 1
    assert failed_summary["cells"][0]["paired"]["successful"] == 19
    batch_path.write_bytes(saved)
    summary_path.write_bytes(original_summary)
    with monkeypatch.context() as patch:
        patch.setattr(runner, "fit_case", lambda *_args: pytest.fail("completed case was refit"))
        second = runner.run(config, "pilot", output, resume=True)
    assert second["terminal"] == 20
    assert (output / "batches/batch-000000.json").read_bytes() == saved
    payload = json.loads(saved)
    payload["records"][0]["methods"]["bootstrap"]["result"]["bootstrap_effects"][0] += 20
    (output / "batches/batch-000000.json").write_text(json.dumps(payload))
    with pytest.raises(checker.EvidenceError, match="sample SD"):
        checker.audit(output)
    with pytest.raises(runner.BootstrapStudyError, match="variance"):
        runner.run(config, "pilot", output, resume=True)
    assert not output.with_name(output.name + ".writer.lock").exists()
