"""Optional installed-R check of conditional Cox cluster-robust uncertainty."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from quantilemix import fit_cox
from quantilemix._cox import (_partial_likelihood, _weighted_partial_likelihood,
                         _weighted_score_residuals)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/r_cox_cluster_2_19_6.json"
DRIVER = Path(__file__).with_name("export_cox_cluster_r.R")
FIXTURE_SHA = "db4fb5bf2376219e4e47ed761a3656c2f749a322e0ed5fb46a59f1a10e33380d"
DRIVER_SHA = "cfc27e145ae8afa563ba0f638653f4f858f3492610920759f408142338ab69a5"
NAMES = ("cluster_untied", "cluster_tied", "cluster_weighted_offset_tied")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compare(actual, expected, label: str, tolerance: float = 2e-8) -> float:
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError(f"{label}: shape or finiteness differs")
    gap = float(np.max(np.abs(a-b)))
    if gap > tolerance:
        raise ValueError(f"{label}: numerical difference {gap:.3g} exceeds {tolerance:.3g}")
    return gap


def check(saved: dict, live: dict) -> dict:
    for key in ("R_version", "qgcomp_version", "survival_version", "jsonlite_version"):
        if saved.get(key) != live.get(key):
            raise ValueError(f"installed reference {key} differs from pinned fixture")
    if live.get("qgcomp_version") != "2.19.6":
        raise ValueError("installed qgcomp 2.19.6 required")
    old_cases, cases = saved.get("cases"), live.get("cases")
    if (not isinstance(old_cases, list) or not isinstance(cases, list)
            or tuple(c.get("name") for c in old_cases) != NAMES
            or tuple(c.get("name") for c in cases) != NAMES):
        raise ValueError("three pinned Cox cluster cases required")
    details = []
    for old, current in zip(old_cases, cases):
        name = old["name"]
        if old["inputs"] != current["inputs"]:
            raise ValueError(f"{name}: inputs changed")
        expected, reference = old["expected"], current["expected"]
        if set(expected) != set(reference):
            raise ValueError(f"{name}: reference fields changed")
        for key, value in expected.items():
            _compare(reference[key], value, f"{name}.{key}.saved_R", 1e-10)
        inputs = current["inputs"]
        x = np.column_stack((inputs["x1"], inputs["x2"]))
        z = np.asarray(inputs["z"])
        ids = np.asarray(inputs["cluster_ids"])
        weights = inputs["weights"]
        offset = inputs["offset"]
        fitted = fit_cox(inputs["time"], inputs["event"], x, z, q=None,
                         cluster_ids=ids, weights=weights, offset=offset)
        actual = {"coefficients": fitted.coefficients,
                  "covariance": fitted.covariance,
                  "psi": fitted.psi, "variance": fitted.var_psi,
                  "ci": fitted.ci, "p": fitted.p_value,
                  "events": fitted.event_count}
        gaps = {key: _compare(value, reference[key], f"{name}.{key}.Python")
                for key, value in actual.items()}
        order = np.argsort(-np.asarray(inputs["time"]), kind="stable")
        design = np.column_stack((x, z))[order]
        events = np.asarray(inputs["event"])[order]
        times = np.asarray(inputs["time"])[order]
        w = np.ones(len(times)) if weights is None else np.asarray(weights)[order]
        o = None if offset is None else np.asarray(offset)[order]
        if weights is None:
            information = _partial_likelihood(design, events, fitted.coefficients,
                                              times, o)[2]
        else:
            information = _weighted_partial_likelihood(
                design, events, fitted.coefficients, times, w, o)[2]
        bread = np.linalg.solve(information, np.eye(design.shape[1]))
        gaps["naive_covariance"] = _compare(
            bread, reference["naive_covariance"], f"{name}.naive_covariance")
        scores = _weighted_score_residuals(design, events, times,
                                           fitted.coefficients, w, o)
        original_scores = np.empty_like(scores)
        original_scores[order] = scores
        # survival::residuals(type='score') omits the case-weight multiplier.
        gaps["score_residuals"] = _compare(
            original_scores / (np.ones((len(times), 1)) if weights is None
                               else np.asarray(weights)[:, None]),
            reference["score_residuals"], f"{name}.score_residuals")
        r_scores = np.asarray(reference["score_residuals"])*(
            1 if weights is None else np.asarray(weights)[:, None])
        _, inverse = np.unique(ids, return_inverse=True)
        cluster_scores = np.zeros((fitted.cluster_count, design.shape[1]))
        np.add.at(cluster_scores, inverse, r_scores)
        r_bread = np.asarray(reference["naive_covariance"])
        independent_covariance = r_bread @ cluster_scores.T @ cluster_scores @ r_bread
        gaps["R_score_cluster_sandwich"] = _compare(
            independent_covariance, reference["covariance"],
            f"{name}.R_score_cluster_sandwich")
        details.append({"name": name, "cluster_count": fitted.cluster_count,
                        "max_absolute_differences": gaps})
    return {"status": "passed", "checked_utc": datetime.now(timezone.utc).isoformat(),
            "qgcomp_version": "2.19.6", "saved_fixture_sha256": FIXTURE_SHA,
            "r_exporter_sha256": DRIVER_SHA, "cases": details}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise ValueError("output already exists")
    if _sha(FIXTURE) != FIXTURE_SHA or _sha(DRIVER) != DRIVER_SHA:
        raise ValueError("Cox cluster fixture or exporter hash differs")
    with tempfile.TemporaryDirectory() as folder:
        live_path = Path(folder) / "live.json"
        try:
            subprocess.run([args.rscript, str(DRIVER), args.library, str(live_path)], check=True)
        except OSError as exc:
            raise SystemExit(f"Cannot run Rscript at {args.rscript}: {exc}") from exc
        report = check(json.loads(FIXTURE.read_text()), json.loads(live_path.read_text()))
    output.write_text(json.dumps(report, indent=2)+"\n")
    print("Three Cox cluster cases agree with isolated installed R qgcomp 2.19.6")


if __name__ == "__main__":
    main()
