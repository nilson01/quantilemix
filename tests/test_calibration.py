"""Software checks using literal data/injected R; no calibration study is run."""

from copy import deepcopy
import hashlib
import importlib.util
from itertools import product
import json
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import pytest
from scipy import stats


SCRIPT = Path(__file__).resolve().parents[1] / "validation" / "calibration.py"
SPEC = importlib.util.spec_from_file_location("calibration_under_test", SCRIPT)
cal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cal)


@pytest.fixture
def config():
    return cal.load_json(SCRIPT.with_name("calibration-config.json"))


@pytest.fixture
def reference():
    return {"package": "qgcomp", "version": "2.19.6", "mode": "installed package",
            "library_verified": True, "R_version": "synthetic-R-4.5.0", "platform": "synthetic-platform",
            "package_versions": {"qgcomp": "2.19.6"},
            "installed_packages": {"qgcomp": {"version": "2.19.6", "Built": "synthetic"}},
            "package_library_origins": {"qgcomp": "isolated"}, "unexpected_loaded_namespaces": [],
            "installed_package_md5": {key: "a"*32 for key in ("DESCRIPTION", "NAMESPACE", "R/qgcomp.rdb", "R/qgcomp.rdx")},
            "r_runtime_md5": {"bin/exec/R": "b"*32}}


def literal_case():
    # Orthogonal 16-row +/-1 factorial, two distinct product covariates,
    # and a third product as residual. SSE=4, df=9, variance(sum beta)=1/9.
    rows = np.array(list(product((-1., 1.), repeat=4)))
    scores = rows+1
    z = np.column_stack((rows[:, 0]*rows[:, 1], rows[:, 2]*rows[:, 3]))
    y = 1+scores @ np.array([.1, .1, .05, 0]) + z @ np.array([.5, -.25]) + .5*rows[:, 0]*rows[:, 2]
    inputs = {"exposures": scores.tolist(), "covariates": z.tolist(), "y": y.tolist(), "q": None, "alpha": .05}
    return {"name": "literal", "cell": "literal", "repetition": 0, "seed": [1, 1, 1, 0],
            "inputs": inputs, "input_sha256": cal.digest(inputs)}


def analytic_result(n=16, psi=.25, variance=1/9, method="r"):
    df = n-7 if method == "qr_t" else n-5
    se = np.sqrt(variance)
    critical = stats.t.ppf(.975, df) if method == "qr_t" else stats.norm.ppf(.975)
    return {"psi": psi, "variance": variance, "se": float(se),
            "ci": [float(psi-critical*se), float(psi+critical*se)],
            "p_value": float(2*stats.t.sf(abs(psi/se), df)), "df_pvalue": df, "df_residual": n-7}


def adapter_for(reference, result=None):
    def adapter(cases, rscript, library, timeout=180):
        assert rscript == "synthetic-Rscript"
        assert library == "synthetic-library"
        return {"reference": deepcopy(reference), "cases": [
            {"name": case["name"], "status": "ok", "result": deepcopy(result or analytic_result()), "warnings": []}
            for case in cases]}
    return adapter


def identity(config, reference):
    return {"config": config, "config_sha256": cal.digest(config), "mode": "study",
            "sources_sha256": {"fixture": "c"*64}, "environment": {"python": "synthetic"},
            "reference": reference}


def prepare(tmp_path, config, reference):
    config = deepcopy(config)
    config["execution"]["batch_size"] = 1
    plan = cal.make_plan(config)
    directory = tmp_path / "evidence"
    manifest = cal.prepare_directory(directory, identity(config, reference), plan)
    return directory, manifest


def planned_record(manifest, *, methods=True):
    # Literal repeated rows, not generated study data or empirical evidence.
    cell = manifest["plan"]["cells"][0]
    inputs = literal_case()["inputs"]
    inputs = {key: ([value[i % len(value)] for i in range(cell["n"])]) if isinstance(value, list) else value
              for key, value in inputs.items()}
    config = manifest["identity"]["config"]
    record = {"name": cal._case_name("study", cell, 0), "cell": cell["id"], "repetition": 0,
              "seed": [config["execution"]["master_seed"], config["study"]["seed_namespace"], cell["index"], 0],
              "inputs": inputs, "input_sha256": cal.digest(inputs)}
    if methods:
        record["methods"] = {method: {"status": "ok", "result": analytic_result(n=100, method=method), "warnings": []}
                             for method in cal.METHODS}
        record["comparisons"] = cal.comparisons(record["methods"])
    return record


def save_checkpoint(directory, manifest, record, *, pending=False):
    payload = {"schema_version": 1, "identity_sha256": manifest["identity_sha256"], "batch_index": 0,
               "status": "pending" if pending else "completed", "records": [record],
               "elapsed_seconds": None if pending else .001, "infrastructure_error": None}
    name = "batch-000000.pending.json" if pending else "batch-000000.json"
    cal.atomic_json(directory / "batches" / name, payload, manifest["plan"]["storage_budget_bytes"])
    return payload


def test_plan_is_explicit_and_analytic(config):
    study, pilot = cal.make_plan(config), cal.make_plan(config, "pilot")
    assert study["planned"] == 24000 and len(study["cells"]) == 12
    assert pilot["planned"] == 20 and pilot["cells"][0]["n"] == 500
    assert not study["execution_performed"] and not study["calibration_validated"]
    assert study["input_storage_estimate_upper_bytes"] < 3_000_000_000
    assert study["cells"][0]["analytic"]["reference_coverage"] == pytest.approx(.94700655, abs=1e-8)
    assert study["cells"][0]["analytic"]["reference_null_rejection"] == pytest.approx(.05006228, abs=1e-8)
    assert study["cells"][1]["true_psi"] == 0  # cancellation is also a joint null


@pytest.mark.parametrize("mutate", [
    lambda c: c.update(extra=1), lambda c: c.update(alpha=True),
    lambda c: c["study"].update(replicates_per_cell=2),
    lambda c: c["design"].update(sample_sizes=[10, 50]),
    lambda c: c["design"]["beta"]["null"].__setitem__(0, False),
    lambda c: c["execution"].update(batch_size=21),
    lambda c: c["execution"].update(retain_inputs=False),
    lambda c: c["execution"].update(storage_budget_bytes=3_000_000_001),
    lambda c: c["pilot"].update(seed_namespace=c["study"]["seed_namespace"]),
])
def test_config_rejects_drift(config, mutate):
    mutate(config)
    with pytest.raises(cal.CalibrationError):
        cal.validate_config(config)


@pytest.mark.parametrize("text", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}', '{'])
def test_strict_json(tmp_path, text):
    path = tmp_path / "invalid.json"
    path.write_text(text)
    with pytest.raises(cal.CalibrationError):
        cal.load_json(path)


def test_generator_is_stable_by_id_and_pilot_namespace(config):
    # Generator checks only; no fitter, adapter, pilot or study loop is called.
    cell = cal.make_plan(config)["cells"][-1]
    first = cal.generate_case(config, "study", cell, 0)
    assert first == cal.generate_case(config, "study", cell, 0)
    other = cal.generate_case(config, "study", cell, 1)
    pilot_cell = cal.make_plan(config, "pilot")["cells"][0]
    pilot = cal.generate_case(config, "pilot", pilot_cell, 0)
    assert len({first["input_sha256"], other["input_sha256"], pilot["input_sha256"]}) == 3
    assert first["seed"] == [20261022, 2, 11, 0]
    assert pilot["seed"] == [20261022, 1, 11, 0]
    assert np.asarray(first["inputs"]["exposures"]).shape == (500, 4)
    assert set(np.ravel(first["inputs"]["exposures"])) <= {0, 1, 2, 3}


def test_independent_qr_literal_contrast_and_inference():
    result = cal.qr_result(literal_case()["inputs"])
    expected = analytic_result(method="qr_t")
    for key in cal.RESULT_FIELDS:
        assert result[key] == pytest.approx(expected[key], rel=1e-12, abs=1e-12)
    native = cal.native_result(literal_case()["inputs"])
    assert native["variance"] == pytest.approx(1/9)
    assert native["df_pvalue"] == 11 and result["df_pvalue"] == 9
    assert native["ci"] != pytest.approx(result["ci"])


def test_batch_agreement_uses_independent_injected_reference(reference):
    result = cal.evaluate_batch([literal_case()], adapter_for(reference), reference,
                                rscript="synthetic-Rscript", library="synthetic-library")
    assert result["infrastructure_error"] is None
    assert all(row["status"] == "agree" for row in result["records"][0]["comparisons"])


def test_python_failure_does_not_discard_other_methods(reference, monkeypatch):
    def fail(_):
        raise ValueError("/private/input must not be exposed")
    monkeypatch.setattr(cal, "native_result", fail)
    result = cal.evaluate_batch([literal_case()], adapter_for(reference), reference,
                                rscript="synthetic-Rscript", library="synthetic-library")
    methods = result["records"][0]["methods"]
    assert methods["python"]["status"] == "error"
    assert methods["r"]["status"] == methods["qr_t"]["status"] == "ok"
    assert "/private/" not in json.dumps(result)


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(cases=[]),
    lambda p: p["cases"].append(deepcopy(p["cases"][0])),
    lambda p: p["cases"][0].update(name="wrong"),
    lambda p: p["cases"][0]["result"].update(psi=float("nan")),
    lambda p: p["cases"][0]["result"].update(variance=True),
    lambda p: p["cases"][0]["result"].update(ci=[None, 2]),
    lambda p: p["cases"][0]["result"].update(df_pvalue=0),
    lambda p: p["reference"].update(library_verified=False),
    lambda p: p["reference"]["r_runtime_md5"].update({"bin/exec/R": "e"*32}),
])
def test_malformed_r_is_infrastructure_failure_not_silent_fallback(reference, mutate):
    def adapter(cases, *args, **kwargs):
        payload = adapter_for(reference)(cases, "synthetic-Rscript", "synthetic-library")
        mutate(payload)
        return payload
    report = cal.evaluate_batch([literal_case()], adapter, reference, rscript="unused", library="unused")
    assert report["infrastructure_error"] is not None
    methods = report["records"][0]["methods"]
    assert methods["r"]["status"] == "error"
    assert methods["python"]["status"] == methods["qr_t"]["status"] == "ok"


def test_missing_r_and_case_failure_are_distinct(reference):
    def missing(*args, **kwargs):
        raise FileNotFoundError("/private/rscript")
    report = cal.evaluate_batch([literal_case()], missing, reference, rscript="unused", library="unused")
    assert report["infrastructure_error"]["error_class"] == "FileNotFoundError"
    assert "/private" not in json.dumps(report)
    def fit_failure(cases, *args, **kwargs):
        return {"reference": reference, "cases": [{"name": cases[0]["name"], "status": "error",
                                                    "error": "rank deficient", "error_class": "simpleError", "warnings": []}]}
    report = cal.evaluate_batch([literal_case()], fit_failure, reference, rscript="unused", library="unused")
    assert report["infrastructure_error"] is None
    assert report["records"][0]["methods"]["r"]["status"] == "error"


def test_mismatch_retains_input_and_field(reference):
    shifted = analytic_result(psi=.5)
    result = cal.evaluate_batch([literal_case()], adapter_for(reference, shifted), reference,
                                rscript="synthetic-Rscript", library="synthetic-library")
    row = result["records"][0]
    assert row["inputs"] == literal_case()["inputs"]
    assert row["comparisons"][0]["status"] == "mismatch"
    assert "psi" in row["comparisons"][0]["fields"]


def test_unexpected_r_df_preserves_returned_value_and_sibling(reference):
    cases = [literal_case(), dict(literal_case(), name="second")]
    def adapter(cases, *args, **kwargs):
        payload = adapter_for(reference)(cases, "synthetic-Rscript", "synthetic-library")
        payload["cases"][1]["result"]["df_pvalue"] = 9
        return payload
    report = cal.evaluate_batch(cases, adapter, reference, rscript="unused", library="unused")
    assert report["infrastructure_error"] is None
    first, second = report["records"]
    assert first["comparisons"][0]["status"] == "agree"
    assert second["methods"]["r"]["status"] == "ok"
    assert second["methods"]["r"]["result"]["df_pvalue"] == 9
    assert second["comparisons"][0]["differences"]["df_pvalue"] == {"python": 11, "other": 9}
    assert second["comparisons"][0]["status"] == "mismatch"


@pytest.mark.parametrize("message,code", [
    ("the design matrix is rank deficient: /private/data", "rank_deficient"),
    ("degenerate variance: C:\\private\\data", "invalid_numerical_result"),
])
def test_failure_retains_code_and_prior_warning_without_values(reference, monkeypatch, message, code):
    def fail(_):
        warnings.warn("private input 12345", RuntimeWarning)
        raise ValueError(message)
    monkeypatch.setattr(cal, "native_result", fail)
    report = cal.evaluate_batch([literal_case()], adapter_for(reference), reference,
                                rscript="synthetic-Rscript", library="synthetic-library")
    error = report["records"][0]["methods"]["python"]
    assert error["error_code"] == code and error["warnings"] == ["RuntimeWarning"]
    assert "private" not in json.dumps(error) and "12345" not in json.dumps(error)


@pytest.mark.parametrize("code", ["r_missing_executable", "r_invalid_library", "r_startup_failure",
                                  "r_timeout", "r_invalid_response"])
def test_r_failure_codes_survive_checkpoints_and_cli(config, reference, tmp_path, monkeypatch, capsys, code):
    class ProviderFailure(RuntimeError):
        pass
    error = ProviderFailure("private inputs 12345 at /private/path")
    error.code = code
    def fail(*args, **kwargs):
        raise error
    report = cal.evaluate_batch([literal_case()], fail, reference, rscript="unused", library="unused")
    failure = report["records"][0]["methods"]["r"]
    assert failure["error_code"] == code
    directory, manifest = prepare(tmp_path, config, reference)
    record = planned_record(manifest)
    record["methods"]["r"] = failure
    record["comparisons"] = cal.comparisons(record["methods"])
    save_checkpoint(directory, manifest, record)
    saved, _ = cal.read_checkpoints(directory, manifest)
    assert saved[record["name"]]["methods"]["r"] == failure
    assert cal.aggregate(directory)["cells"][0]["methods"]["r"]["failure_codes"] == {code: 1}
    for key in cal.THREAD_VARIABLES:
        monkeypatch.setenv(key, "1")
    monkeypatch.setattr(cal, "_load_adapter", lambda: fail)
    monkeypatch.setattr(cal, "generate_case", lambda *a, **kw: pytest.fail("no generation"))
    assert cal.main(["pilot", "--execute", "--output", str(tmp_path / "preflight"),
                     "--r-library", "unused"]) == 2
    diagnostic = json.loads(capsys.readouterr().err)
    assert diagnostic["error_code"] == code
    assert "private" not in diagnostic["error"] and "12345" not in diagnostic["error"]


def test_aggregate_arithmetic_and_method_denominators():
    def entry(psi, variance, ci, p_value):
        return {"methods": {"python": {"status": "ok", "result": {
            "psi": psi, "variance": variance, "ci": ci, "p_value": p_value}}}}
    records = [entry(1, 1, [0, 2], .01), entry(3, 4, [2, 4], .2),
               {"methods": {"python": {"status": "error", "error_class": "RankError"}}}]
    report = cal.summarize_method(records, "python", planned=5, truth=0)
    assert (report["attempted"], report["successful"], report["failed"], report["missing"]) == (3, 2, 1, 2)
    assert report["bias"] == 2 and report["bias_mcse"] == pytest.approx(1)
    assert report["empirical_sd"] == pytest.approx(np.sqrt(2))
    assert report["rms_se"] == pytest.approx(np.sqrt(2.5))
    assert report["mean_interval_width"] == 2
    assert report["coverage_successful"] == .5 and report["rejection_successful"] == .5
    assert report["coverage_wilson"] == pytest.approx([.0945312057, .9054687943])
    assert report["coverage_all_planned_bounds"] == [.2, .8]
    assert report["rejection_kind"] == "false_positive_rate"
    empty = cal.summarize_method(records, "r", planned=5, truth=.25)
    assert empty["successful"] == 0 and empty["missing"] == 5
    assert empty["bias"] is None and empty["coverage_successful"] is None
    assert empty["coverage_all_planned_bounds"] == [0, 1]
    single = cal.summarize_method(records[:1], "python", planned=5, truth=0)
    assert single["empirical_sd"] is None and single["bias_mcse"] is None


def test_empty_pending_and_complete_batch_remain_incomplete_study(tmp_path, config, reference):
    directory, manifest = prepare(tmp_path, config, reference)
    empty = cal.aggregate(directory)
    assert empty["status"] == "incomplete" and empty["missing"] == 24000
    assert not empty["execution_complete"] and not empty["calibration_validated"]
    save_checkpoint(directory, manifest, planned_record(manifest, methods=False), pending=True)
    pending = cal.aggregate(directory)
    assert pending["attempted"] == 1 and pending["terminal"] == 0
    save_checkpoint(directory, manifest, planned_record(manifest))
    partial = cal.aggregate(directory)
    assert partial["attempted"] == partial["terminal"] == 1 and partial["missing"] == 23999
    assert partial["status"] == "incomplete"
    # Both files may exist after interruption between final commit and cleanup;
    # this is one case, never two.
    assert partial["cells"][0]["methods"]["python"]["successful"] == 1


@pytest.mark.parametrize("mutation", ["hash", "name", "seed", "comparison", "method", "input"])
def test_corrupt_checkpoint_rejected(tmp_path, config, reference, mutation):
    directory, manifest = prepare(tmp_path, config, reference)
    record = planned_record(manifest)
    if mutation == "hash":
        record["input_sha256"] = "f"*64
    elif mutation == "name":
        record["name"] = "unknown"
    elif mutation == "seed":
        record["seed"][0] += 1
    elif mutation == "comparison":
        record["comparisons"] = []
    elif mutation == "method":
        record["methods"]["r"]["result"]["p_value"] = None
    else:
        record["inputs"]["y"][0] += 1
    save_checkpoint(directory, manifest, record)
    with pytest.raises(cal.CalibrationError):
        cal.aggregate(directory)


def test_resume_matches_identity_and_does_not_replace_evidence(tmp_path, config, reference):
    directory, manifest = prepare(tmp_path, config, reference)
    record = planned_record(manifest)
    save_checkpoint(directory, manifest, record)
    before = (directory / "batches" / "batch-000000.json").read_bytes()
    assert cal.prepare_directory(directory, manifest["identity"], manifest["plan"], resume=True) == manifest
    changed = deepcopy(manifest["identity"])
    changed["sources_sha256"]["fixture"] = "f"*64
    with pytest.raises(cal.CalibrationError, match="identity"):
        cal.prepare_directory(directory, changed, manifest["plan"], resume=True)
    assert (directory / "batches" / "batch-000000.json").read_bytes() == before
    with pytest.raises(cal.CalibrationError, match="fresh"):
        cal.prepare_directory(directory, manifest["identity"], manifest["plan"])


def test_resume_rejects_changed_public_package_exports(tmp_path, config, reference, monkeypatch):
    # The public initializer can change which implementation `fit` exposes.
    # Change only a private source copy; never patch the shared package module.
    initializer = tmp_path / "__init__.py"
    original = Path(cal.package.__file__).read_bytes()
    initializer.write_bytes(original)
    monkeypatch.setattr(cal, "package", SimpleNamespace(__file__=str(initializer)))
    before = cal.build_identity(config, "pilot", reference)
    assert before["sources_sha256"]["qgcomp/__init__.py"] == hashlib.sha256(original).hexdigest()
    plan = cal.make_plan(config, "pilot")
    directory = tmp_path / "evidence"
    cal.prepare_directory(directory, before, plan)
    manifest_bytes = (directory / "manifest.json").read_bytes()
    cal.prepare_directory(directory, before, plan, resume=True)

    initializer.write_bytes(original + b"\nfit = another_implementation\n")
    after = cal.build_identity(config, "pilot", reference)
    with pytest.raises(cal.CalibrationError, match="identity"):
        cal.prepare_directory(directory, after, plan, resume=True)
    assert (directory / "manifest.json").read_bytes() == manifest_bytes

    initializer.unlink()
    with pytest.raises(cal.CalibrationError, match="required source identity"):
        cal.build_identity(config, "pilot", reference)


def test_atomic_failure_leaves_previous_checkpoint(tmp_path, config, reference, monkeypatch):
    directory, manifest = prepare(tmp_path, config, reference)
    path = directory / "scratch.json"
    cal.atomic_json(path, {"old": 1}, 3_000_000_000)
    original = path.read_bytes()
    def replace_failure(source, destination):
        raise OSError("synthetic interruption before atomic replacement")
    # cal.os is shared; patching it would be process-wide. Inject a small local
    # facade instead, leaving every other module's os untouched.
    class LocalOS:
        fsync = staticmethod(cal.os.fsync)
        replace = staticmethod(replace_failure)
    monkeypatch.setattr(cal, "os", LocalOS)
    with pytest.raises(OSError):
        cal.atomic_json(path, {"new": 2}, 3_000_000_000)
    assert path.read_bytes() == original
    assert path.with_name("scratch.json.tmp").exists()


def test_storage_ceiling_and_symlink_output(tmp_path, config, reference):
    directory, manifest = prepare(tmp_path, config, reference)
    with pytest.raises(cal.CalibrationError, match="ceiling"):
        cal.atomic_json(directory / "too-large.json", {"large": "x"*100}, 1)
    link = tmp_path / "dangling"
    target = tmp_path / "must-not-exist"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(cal.CalibrationError, match="symlink"):
        cal.prepare_directory(link, manifest["identity"], manifest["plan"])
    assert not target.exists()


def test_default_cli_plan_never_fits_or_loads_r(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("execution was not authorized")
    monkeypatch.setattr(cal, "execute", forbidden)
    monkeypatch.setattr(cal, "_load_adapter", forbidden)
    assert cal.main([]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "planned_not_executed"


@pytest.mark.parametrize("mode", ["pilot", "study"])
def test_cli_requires_explicit_execution(mode, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cal, "execute", lambda *a, **kw: pytest.fail("must not execute"))
    target = tmp_path / "output"
    assert cal.main([mode, "--output", str(target), "--r-library", "unused"]) == 2
    assert not target.exists()
    assert "--execute" in json.loads(capsys.readouterr().err)["error"]


def test_cli_missing_r_fails_before_any_fitting(config, monkeypatch, capsys, tmp_path):
    for key in cal.THREAD_VARIABLES:
        monkeypatch.setenv(key, "1")
    def missing(*args, **kwargs):
        assert args[0] == []  # identity-only preflight; no generated case
        raise FileNotFoundError("/private/rscript")
    monkeypatch.setattr(cal, "_load_adapter", lambda: missing)
    monkeypatch.setattr(cal, "generate_case", lambda *a, **kw: pytest.fail("no simulation"))
    target = tmp_path / "output"
    assert cal.main(["pilot", "--execute", "--output", str(target), "--r-library", "unused"]) == 2
    message = json.loads(capsys.readouterr().err)
    assert message["error_class"] == "FileNotFoundError"
    assert "/private" not in message["error"] and not target.exists()


@pytest.mark.parametrize("exception", [ValueError, KeyboardInterrupt])
def test_writer_lock_releases_on_error_and_rejects_second_writer(tmp_path, exception):
    directory = tmp_path / "run"
    lock = tmp_path / "run.writer.lock"
    with pytest.raises(exception):
        with cal.writer_lock(directory):
            assert lock.exists()
            with pytest.raises(cal.CalibrationError, match="writer lock"):
                with cal.writer_lock(directory):
                    pytest.fail("a second writer must never enter")
            raise exception()
    assert not lock.exists()
    lock.write_text('{"pid":1}')
    with pytest.raises(cal.CalibrationError, match="manual"):
        with cal.writer_lock(directory):
            pytest.fail("stale locks require manual recovery")
    assert lock.read_text() == '{"pid":1}'


def test_invocation_timing_includes_preflight_setup_and_summary(config, reference, tmp_path, monkeypatch):
    # An orchestration unit check with all batches already represented by a
    # stub checkpoint reader. No generator/fitter/experimental loop is entered.
    for key in cal.THREAD_VARIABLES:
        monkeypatch.setenv(key, "1")
    ticks = iter(range(100, 200))
    class Clock:
        perf_counter = staticmethod(lambda: next(ticks))
    monkeypatch.setattr(cal, "time", Clock)
    monkeypatch.setattr(cal, "read_checkpoints", lambda *args: ({}, {0}))
    monkeypatch.setattr(cal, "aggregate", lambda *args: {"status": "complete_numerical_agreement", "test_stub": True})
    monkeypatch.setattr(cal, "generate_case", lambda *args: pytest.fail("no simulation"))
    def preflight_only(cases, *args, **kwargs):
        assert cases == []
        return {"reference": reference, "cases": []}
    directory = tmp_path / "timing"
    report = cal.execute(config, "pilot", directory, rscript="synthetic", library="synthetic", adapter=preflight_only)
    timing = report["timing"]
    assert timing["preflight_seconds"] == 1
    assert timing["setup_seconds"] > 0 and timing["aggregation_seconds"] == 1
    assert timing["summary_write_seconds"] == 1
    assert timing["total_invocation_seconds"] > timing["preflight_seconds"] + timing["setup_seconds"]
    assert cal.load_json(directory / "timing.json") == timing
    assert not (tmp_path / "timing.writer.lock").exists()


@pytest.mark.parametrize("status,code", [("complete_numerical_agreement", 0),
                                        ("completed_with_failures_or_mismatches", 2), ("incomplete", 2)])
def test_execution_cli_status_is_not_calibration_proof(status, code, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cal, "execute", lambda *a, **kw: {"status": status, "calibration_validated": False})
    actual = cal.main(["pilot", "--execute", "--output", str(tmp_path / "unused"), "--r-library", "synthetic"])
    assert actual == code
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == status and report["calibration_validated"] is False
