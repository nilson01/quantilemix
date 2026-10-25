"""Optional poisson checker boundaries without requiring R in normal tests."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "validation/check_poisson_offset_r.py"
SPEC = importlib.util.spec_from_file_location("check_poisson_offset_r", SCRIPT)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def arguments(tmp_path):
    library = tmp_path / "library"
    (library / "qgcomp").mkdir(parents=True)
    (library / "qgcomp/DESCRIPTION").write_text("Package: qgcomp\nVersion: 2.19.6\n")
    return SimpleNamespace(fixture=str(CHECK.FIXTURE), r_library=str(library),
                           rscript="Rscript", output=str(tmp_path / "report.json"),
                           timeout=20)


def fake_export(args, *, capture_output, text, timeout, check):
    assert args[0] == "Rscript" and args[1] == "--vanilla"
    assert capture_output and text and timeout == 20 and not check
    payload = json.loads(CHECK.FIXTURE.read_text())
    payload.update(qgcomp_version="2.19.6", R_version="R 4.5.1",
                   library=args[3], qgcomp_origin=str(Path(args[3]) / "qgcomp"))
    Path(args[4]).write_text(json.dumps(payload))
    return SimpleNamespace(returncode=0)


def test_valid_optional_check_and_output(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    monkeypatch.setattr(CHECK, "run_export", fake_export)
    report = CHECK.check(args)
    assert report["status"] == "passed" and report["case_count"] == 2
    assert json.loads(Path(args.output).read_text())["status"] == "passed"
    with pytest.raises(CHECK.CheckError, match="new path"):
        CHECK.check(args)


def test_missing_isolated_library(tmp_path):
    args = arguments(tmp_path)
    Path(args.r_library, "qgcomp/DESCRIPTION").unlink()
    with pytest.raises(CHECK.CheckError, match="unavailable"):
        CHECK.check(args)


@pytest.mark.parametrize("change", ("input", "numeric", "names"))
def test_changed_export_is_rejected(tmp_path, monkeypatch, change):
    args = arguments(tmp_path)

    def modified(*call_args, **kwargs):
        reply = fake_export(*call_args, **kwargs)
        target = Path(call_args[0][4])
        payload = json.loads(target.read_text())
        case = payload["cases"][0]
        if change == "input":
            case["inputs"]["x1"][0] += 1
        elif change == "numeric":
            case["expected"]["psi"] += 0.1
        else:
            case["expected"]["coefficient_names"][1] = "wrong"
        target.write_text(json.dumps(payload))
        return reply

    monkeypatch.setattr(CHECK, "run_export", modified)
    with pytest.raises(CHECK.CheckError):
        CHECK.check(args)
    assert not Path(args.output).exists()


def test_export_failure_leaves_no_report(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    monkeypatch.setattr(CHECK, "run_export", lambda *a, **k: SimpleNamespace(returncode=2))
    with pytest.raises(CHECK.CheckError, match="exited with 2"):
        CHECK.check(args)
    assert not Path(args.output).exists()
