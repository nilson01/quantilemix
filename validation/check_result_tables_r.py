"""Optional fixed-input comparison of public result tables with installed R."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from quantilemix import fit, fit_binomial, fit_cox, fit_poisson, glance, tidy


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
DRIVER = Path(__file__).with_name("export_result_tables_r.R")
SAVED = ROOT / "tests" / "fixtures" / "r_result_tables_2_19_6.json"
SOURCES = {
    "gaussian": "r_gaussian_2_19_6.json",
    "binomial": "r_binomial_2_19_6.json",
    "poisson": "r_poisson_2_19_6.json",
    "cox": "r_cox_2_19_6.json",
}
CASES = (
    ("adjusted_mixed", "gaussian"),
    ("prequantized", "gaussian"),
    ("binary_exact_scores", "binomial"),
    ("binary_quantized_adjusted", "binomial"),
    ("count_exact_scores", "poisson"),
    ("count_quantized_adjusted", "poisson"),
    ("cox_quantized_untied", "cox"),
)
TIDY_FIELDS = {
    "estimate": "estimate", "std_error": "std.error",
    "lower_ci": "Lower CI", "upper_ci": "Upper CI",
    "statistic": "test",
}
GLANCE_FIELDS = {
    "null_deviance": "null.deviance", "df_null": "df.null",
    "log_likelihood": "logLik", "aic": "AIC", "bic": "BIC",
    "deviance": "deviance", "df_residual": "df.residual", "nobs": "nobs",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(actual, expected, label: str, *, atol: float) -> float:
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError(f"{label}: shape or finiteness differs")
    gap = float(np.max(np.abs(a-b)))
    if not np.allclose(a, b, rtol=0, atol=atol):
        raise ValueError(f"{label}: difference {gap:.3g} exceeds {atol:.3g}")
    return gap


def load_inputs() -> dict[str, dict]:
    inputs = {}
    for family, file in SOURCES.items():
        data = json.loads((FIXTURES / file).read_text())
        for case in data["cases"]:
            if (case["name"], family) in CASES:
                inputs[case["name"]] = case["inputs"]
    if set(inputs) != {name for name, _ in CASES}:
        raise ValueError("selected fixed inputs are missing")
    return inputs


def python_fit(family: str, data: dict):
    if family == "gaussian":
        return fit(data["y"], data["exposures"], data["covariates"],
                   q=data["q"], alpha=data["alpha"])
    x = np.column_stack((data["x1"], data["x2"]))
    z = None if data["z"] is None else np.asarray(data["z"])[:, None]
    if family == "binomial":
        return fit_binomial(data["y"], x, z, q=data["q"])
    if family == "poisson":
        return fit_poisson(data["y"], x, z, q=data["q"])
    return fit_cox(data["time"], data["event"], x, z, q=data["q"])


def compare_saved(actual, expected, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{label}: saved-R fields differ")
        for key, value in expected.items():
            compare_saved(actual[key], value, f"{label}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{label}: saved-R rows differ")
        for index, value in enumerate(expected):
            compare_saved(actual[index], value, f"{label}[{index}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        compare(actual, expected, label, atol=1e-5)
    elif actual != expected:
        raise ValueError(f"{label}: saved-R value differs")


def check(live: dict) -> dict:
    if live.get("qgcomp_version") != "2.19.6":
        raise ValueError("installed qgcomp 2.19.6 required")
    cases = live.get("cases")
    if not isinstance(cases, list) or tuple(c.get("name") for c in cases) != tuple(
            name for name, _ in CASES):
        raise ValueError("seven ordered fixed cases required")
    saved = json.loads(SAVED.read_text())
    if saved.get("qgcomp_version") != "2.19.6":
        raise ValueError("saved R reference version differs")
    compare_saved(cases, saved["cases"], "saved_R")
    inputs = load_inputs()
    results = []
    for (name, family), case in zip(CASES, cases):
        result = python_fit(family, inputs[name])
        py_tidy = tidy(result)
        r_tidy = case["tidy"]
        if tuple(py_tidy["term"]) != tuple(row["term"] for row in r_tidy):
            raise ValueError(f"{name}: term order differs")
        gaps = {}
        for py_field, r_field in TIDY_FIELDS.items():
            values = [row[r_field] for row in r_tidy]
            gaps[py_field] = compare(py_tidy[py_field], values,
                                     f"{name}.tidy.{py_field}", atol=1e-5)
        gaps["joint_p_value"] = compare(py_tidy["p_value"][-1],
                                        r_tidy[-1]["Pr(>|z|)"],
                                        f"{name}.tidy.joint_p_value", atol=1e-6)
        quick = case["tidy_quick"]
        if any(set(row) != {"term", "estimate"} for row in quick):
            raise ValueError(f"{name}: R quick columns differ")
        if tuple(row["term"] for row in quick) != tuple(py_tidy["term"]):
            raise ValueError(f"{name}: quick term order differs")
        gaps["quick_estimate"] = compare(
            tidy(result, quick=True)["estimate"], [row["estimate"] for row in quick],
            f"{name}.tidy.quick", atol=1e-7)
        if family != "gaussian":
            py_exp = tidy(result, exponentiate=True)
            r_exp = case["tidy_exponentiated"]
            for field, r_field in (("estimate", "estimate"), ("lower_ci", "Lower CI")):
                gaps[f"exp_{field}"] = compare(
                    py_exp[field], [row[r_field] for row in r_exp],
                    f"{name}.tidy.exp_{field}", atol=1e-5)
            # R 2.19.6 computes exp(exp(lower link bound)) for the upper bound.
            # It first overwrites the local lower bound, then exponentiates it again.
            compare([row["Upper CI"] for row in r_exp],
                    np.exp([row["Lower CI"] for row in r_exp]),
                    f"{name}.R_upper_limit_bug", atol=1e-12)
            if not np.all(py_exp["upper_ci"] > py_exp["lower_ci"]):
                raise ValueError(f"{name}: Python exponentiated interval is invalid")
        if family == "cox":
            if case["glance"] is not None:
                raise ValueError(f"{name}: Cox glance was unexpectedly exported")
        else:
            py_glance = glance(result)
            r_glance = case["glance"]
            if set(r_glance) != set(GLANCE_FIELDS.values()):
                raise ValueError(f"{name}: R glance fields differ")
            for py_field, r_field in GLANCE_FIELDS.items():
                gaps[f"glance_{py_field}"] = compare(
                    py_glance[py_field], r_glance[r_field],
                    f"{name}.glance.{py_field}", atol=1e-7)
        results.append({"name": name, "family": family,
                        "max_absolute_difference": max(gaps.values()),
                        "field_differences": gaps})
    return {
        "status": "passed", "checked_utc": datetime.now(timezone.utc).isoformat(),
        "R_version": live["R_version"], "qgcomp_version": "2.19.6",
        "jsonlite_version": live["jsonlite_version"], "broom_version": live["broom_version"],
        "r_exporter_sha256": sha(DRIVER),
        "saved_reference_sha256": sha(SAVED),
        "fixture_sha256": {family: sha(FIXTURES / file) for family, file in SOURCES.items()},
        "cases": results,
        "scope": ("Seven fixed cases; common tidy/glance fields only. R's intercept p-value "
                  "and exponentiated upper limit are known display errors. "
                  "No Cox glance or repeated-dataset calibration."),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise ValueError("output already exists")
    with tempfile.TemporaryDirectory() as folder:
        live_path = Path(folder) / "live.json"
        subprocess.run([args.rscript, "--vanilla", str(DRIVER), args.library,
                        str(FIXTURES), str(live_path)], check=True, timeout=120)
        report = check(json.loads(live_path.read_text()))
    output.write_text(json.dumps(report, indent=2)+"\n")
    print("Seven result-table cases agree with installed R qgcomp 2.19.6 on shared fields")


if __name__ == "__main__":
    main()
