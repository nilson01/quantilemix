"""Compare unequal-time marginal Poisson fits with installed R qgcomp 2.19.6."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from scipy import optimize

from quantilemix import quantize
from quantilemix._poisson_bootstrap import _from_indices


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/r_poisson_bootstrap_offset_2_19_6.json"
DRIVER = Path(__file__).with_name("export_poisson_bootstrap_offset_r.R")
FIXTURE_SHA = "63c22e14b45ac709e8cf68654c79a51d23d74c117f3e2789197b2c0370988108"
DRIVER_SHA = "89b856960b420de2cc6fee9dc2ea9583fd7c2281bf5b06b487994578e7078a84"
NAMES = ("offset_bootstrap_exact_scores", "offset_bootstrap_quantized_adjusted")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(actual, expected, label: str, *, atol: float) -> float:
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError(f"{label}: shape or finiteness differs")
    gap = float(np.max(np.abs(a-b)))
    if not np.allclose(a, b, atol=atol, rtol=0):
        raise ValueError(f"{label}: numerical difference {gap:.3g} exceeds {atol:.3g}")
    return gap


def independent_effects(data: dict) -> tuple[float, np.ndarray, np.ndarray]:
    """Solve Poisson score equations without the package fitting routine."""
    x = np.column_stack((data["x1"], data["x2"]))
    if data["q"] is not None:
        x = quantize(x, q=data["q"]).scores
    z = np.empty((len(x), 0)) if data["z"] is None else np.asarray(data["z"])[:, None]
    design = np.column_stack((np.ones(len(x)), x, z))
    y, time = np.asarray(data["y"], dtype=float), np.asarray(data["time"], dtype=float)

    def solve(indices) -> np.ndarray:
        a, counts, periods = design[indices], y[indices], time[indices]
        initial = np.zeros(design.shape[1])
        initial[0] = np.log(np.sum(counts) / np.sum(periods))

        def score(beta):
            return a.T @ (counts - periods*np.exp(a @ beta))

        def jacobian(beta):
            means = periods*np.exp(a @ beta)
            return -(a.T*means) @ a

        fitted = optimize.root(score, initial, jac=jacobian, tol=1e-11)
        if not np.all(np.isfinite(fitted.x)) or np.max(np.abs(score(fitted.x))) > 1e-8:
            raise ValueError("independent Poisson score equations did not converge")
        return fitted.x

    beta = solve(np.arange(len(y)))
    effects = np.array([np.sum(solve(row)[1:3]) for row in data["indices"]])
    levels = np.arange(2 if data["q"] is None else data["q"])
    means = np.array([np.mean(time*np.exp(beta[0] + z @ beta[3:] +
                                       level*np.sum(beta[1:3]))) for level in levels])
    return float(np.sum(beta[1:3])), effects, means


def check(saved: dict, live: dict) -> dict:
    if (saved.get("reference", {}).get("version") != "2.19.6"
            or live.get("qgcomp_version") != "2.19.6"):
        raise ValueError("installed R qgcomp 2.19.6 required")
    old_cases, new_cases = saved.get("cases"), live.get("cases")
    if (not isinstance(old_cases, list) or not isinstance(new_cases, list)
            or tuple(c["name"] for c in old_cases) != NAMES
            or tuple(c["name"] for c in new_cases) != NAMES):
        raise ValueError("two fixed offset bootstrap cases required")
    results = []
    for old, new in zip(old_cases, new_cases):
        if old["inputs"] != new["inputs"]:
            raise ValueError(f"{old['name']}: reference inputs changed")
        if set(old["expected"]) != set(new["expected"]):
            raise ValueError(f"{old['name']}: reference fields changed")
        gaps = {}
        for field, value in old["expected"].items():
            if field == "breaks":
                if len(value) != len(new["expected"][field]):
                    raise ValueError(f"{old['name']}: R cutpoint count changed")
                for i, (old_cut, new_cut) in enumerate(zip(value, new["expected"][field])):
                    gaps[f"saved_R_breaks[{i}]"] = compare(new_cut, old_cut,
                        f"{old['name']}.breaks[{i}].saved_R", atol=1e-10)
            else:
                gaps[f"saved_R_{field}"] = compare(new["expected"][field], value,
                    f"{old['name']}.{field}.saved_R", atol=2e-9)
        data, expected = new["inputs"], new["expected"]
        x = np.column_stack((data["x1"], data["x2"]))
        z = None if data["z"] is None else np.asarray(data["z"])[:, None]
        result = _from_indices(data["y"], x, data["indices"], z,
                               q=data["q"], exposure_time=data["time"])
        actual = {"psi": result.psi, "se": result.se, "ci": result.ci,
                  "p": result.p_value, "means": result.intervention_means,
                  "effects": result.bootstrap_effects, "scores": result.point_fit.scores}
        for field, value in actual.items():
            gaps[f"Python_{field}"] = compare(value, expected[field],
                                               f"{old['name']}.{field}.Python", atol=2e-9)
        if data["q"] is None:
            if result.point_fit.breaks is not None:
                raise ValueError("Python prequantized breaks contract changed")
        else:
            for i, (cut, ref) in enumerate(zip(result.point_fit.breaks, expected["breaks"])):
                gaps[f"Python_breaks[{i}]"] = compare(cut, ref,
                    f"{old['name']}.breaks[{i}].Python", atol=1e-10)
        psi, effects, means = independent_effects(data)
        for field, value in (("psi", psi), ("effects", effects), ("means", means)):
            gaps[f"independent_{field}"] = compare(value, expected[field],
                f"{old['name']}.{field}.independent", atol=2e-9)
        results.append({"name": old["name"], "max_absolute_differences": gaps})
    return {"status": "passed", "checked_utc": datetime.now(timezone.utc).isoformat(),
            "qgcomp_version": "2.19.6", "saved_fixture_sha256": FIXTURE_SHA,
            "r_exporter_sha256": DRIVER_SHA, "cases": results,
            "scope": "One prequantized and one quantized unequal-time case; not interval calibration.",
            "prequantized_breaks_difference": "R saves synthetic boundaries; Python retains breaks=None."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise ValueError("output already exists")
    if sha(FIXTURE) != FIXTURE_SHA or sha(DRIVER) != DRIVER_SHA:
        raise ValueError("Poisson offset fixture or exporter hash differs")
    with tempfile.TemporaryDirectory() as folder:
        live_path = Path(folder) / "live.json"
        subprocess.run([args.rscript, str(DRIVER), args.library, str(live_path)], check=True)
        report = check(json.loads(FIXTURE.read_text()), json.loads(live_path.read_text()))
    output.write_text(json.dumps(report, indent=2)+"\n")
    print("Two offset marginal Poisson cases agree with installed R qgcomp 2.19.6")


if __name__ == "__main__":
    main()
