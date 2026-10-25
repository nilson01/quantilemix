# SPDX-License-Identifier: GPL-2.0-or-later
"""Optional installed-R check of the retained separated risk-ratio draw."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from quantilemix import fit_binomial, fit_binomial_risk_ratio
from risk_ratio_calibration import bootstrap_seed, generate, input_hash


DRIVER = Path(__file__).with_suffix(".R")
EXPECTED_INPUT_SHA256 = "c214ef57d695c54e68fb1b7d9f6791d5acff1eda71ad846404873b333aa32f44"
EXPECTED_SEED = 7377651011041772520
EXPECTED_R_PSI = -0.08880640528975485


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_draw() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Regenerate the one retained study draw, without running a new study."""
    x, z, y = generate("study", 0, 190)
    if input_hash(x, z, y) != EXPECTED_INPUT_SHA256:
        raise ValueError("saved study input identity changed")
    seed = bootstrap_seed("study", 0, 190)
    if seed != EXPECTED_SEED:
        raise ValueError("saved bootstrap seed changed")
    rng = np.random.default_rng(seed)
    for _ in range(83):
        rows = rng.integers(0, len(y), size=len(y), dtype=np.intp)
    if (len(rows) != 500 or len(np.unique(rows)) != 313
            or np.sum(z[rows, 0] == 0) != 255
            or np.sum(y[rows]) != 32
            or np.sum(y[rows][z[rows, 0] == 0]) != 0):
        raise ValueError("retained separated draw changed")
    return x, z, y, np.sort(rows)


def write_draw(path: Path, x: np.ndarray, z: np.ndarray, y: np.ndarray,
               rows: np.ndarray) -> None:
    with path.open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("id__", "x1", "x2", "x3", "x4", "z", "y"))
        for row in rows:
            writer.writerow((int(row) + 1, *map(int, x[row]), int(z[row, 0]), int(y[row])))


def validate_r(payload: dict) -> None:
    """Reject a partial or different installed-R result."""
    if (not payload.get("R_version", "").startswith("R version 4.5.1 ")
            or payload.get("qgcomp_version") != "2.19.6"
            or payload.get("draw") != {"rows": 500, "unique_original_ids": 313,
                                        "events": 32, "z0_rows": 255,
                                        "z0_events": 0, "z1_events": 32}):
        raise ValueError("R version, package or draw identity differs")
    inner, public = payload.get("internal", {}), payload.get("public", {})
    if (inner.get("warnings") != [] or public.get("warnings") != []
            or inner.get("conditional_converged") is not True
            or inner.get("msm_converged") is not True
            or public.get("conditional_converged") is not True
            or public.get("B") != 4):
        raise ValueError("installed R convergence, warnings or resamples differ")
    conditional = np.asarray(inner.get("conditional_coefficients"), dtype=float)
    marginal = np.asarray(inner.get("msm_coefficients"), dtype=float)
    public_conditional = np.asarray(public.get("conditional_coefficients"), dtype=float)
    ci = np.asarray(public.get("ci"), dtype=float)
    psi, se = float(public.get("psi")), float(public.get("se"))
    if (conditional.shape != (6,) or marginal.shape != (2,)
            or public_conditional.shape != (6,) or ci.shape != (2,)
            or not all(np.all(np.isfinite(a)) for a in
                       (conditional, marginal, public_conditional, ci))
            or not math.isfinite(psi) or not math.isfinite(se) or se <= 0
            or not np.allclose(conditional, public_conditional, atol=1e-8, rtol=0)
            or not math.isclose(marginal[1], psi, abs_tol=1e-8)
            or not math.isclose(psi, EXPECTED_R_PSI, abs_tol=1e-8)
            or not (conditional[0] < -10 and conditional[-1] > 10)
            or not ci[0] < psi < ci[1]):
        raise ValueError("installed R separated-draw numbers differ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise ValueError("output already exists")
    x, z, y, rows = exact_draw()
    try:
        fit_binomial(y[rows], x[rows], z[rows], q=None)
    except ValueError as exc:
        if "coefficients diverged; check separation" not in str(exc):
            raise ValueError("Python failure reason changed") from exc
        python_failure = str(exc)
    else:
        raise ValueError("Python no longer rejects the separated draw")
    try:
        fit_binomial_risk_ratio(y, x, z, q=None, B=200, seed=EXPECTED_SEED)
    except ValueError as exc:
        if "bootstrap replicate 83 failed: binomial coefficients diverged" not in str(exc):
            raise ValueError("Python public failure draw or reason changed") from exc
        python_public_failure = str(exc)
    else:
        raise ValueError("Python public fit no longer rejects the separated draw")
    with tempfile.TemporaryDirectory() as folder:
        draw_path = Path(folder) / "draw.csv"
        r_path = Path(folder) / "r.json"
        write_draw(draw_path, x, z, y, rows)
        completed = subprocess.run(
            [args.rscript, str(DRIVER), args.library, str(draw_path), str(r_path)],
            check=True, capture_output=True, text=True)
        payload = json.loads(r_path.read_text())
        validate_r(payload)
        report = {"status": "passed", "checked_utc": datetime.now(timezone.utc).isoformat(),
                  "input_sha256": EXPECTED_INPUT_SHA256, "bootstrap_seed": EXPECTED_SEED,
                  "draw_csv_sha256": sha256(draw_path), "r_driver_sha256": sha256(DRIVER),
                  "R": payload, "python_failure": python_failure,
                  "python_public_failure": python_public_failure,
                  "R_stdout": completed.stdout.strip(),
                  "scope": "One retained separated draw; the public R call uses four new resamples, not the original 200-draw Python schedule. This is a failure-policy comparison, not interval calibration."}
    with output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print("Retained separated draw checked against installed R qgcomp 2.19.6")


if __name__ == "__main__":
    main()
