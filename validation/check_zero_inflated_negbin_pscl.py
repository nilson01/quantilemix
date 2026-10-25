"""Optional fixed-case comparison with installed pscl 1.5.9 (R)."""

from pathlib import Path
import json
import os
import subprocess

import numpy as np
from scipy import special, stats

from quantilemix import fit_zero_inflated_negative_binomial


ROOT = Path(__file__).resolve().parents[1]


def case():
    i = np.arange(720)
    x = np.column_stack((i % 4, (i // 4) % 4))
    z = np.sin(i*.17)
    mu = np.exp(.3+.18*x[:, 0]-.11*x[:, 1]+.12*z)
    pi = special.expit(-.8+.15*x[:, 0]-.12*x[:, 1]+.1*z)
    u0 = ((i*37) % 719+.5)/719
    u1 = ((i*113) % 719+.5)/719
    y = np.where(u0 < pi, 0, stats.nbinom.ppf(u1, 1.2, 1.2/(1.2+mu)))
    return y, x, z


def main():
    script = ROOT / "validation/export_zero_inflated_negbin_pscl.R"
    run = subprocess.run([os.environ.get("RSCRIPT", "Rscript"), str(script)],
                         cwd=ROOT, check=True, capture_output=True, text=True)
    parsed = {}
    for line in run.stdout.strip().splitlines():
        name, *values = line.split("\t")
        parsed[name] = np.array([float(v.strip()) for v in values])
    y, x, z = case()
    if int(parsed["n"][0]) != len(y) or int(parsed["sum_y"][0]) != int(sum(y)):
        raise SystemExit("R/Python fixed-case data differ")
    fit = fit_zero_inflated_negative_binomial(y, x, z, q=None)
    covariance = parsed["covariance"].reshape((8, 8), order="F")
    gaps = {"theta": float(abs(fit.theta-parsed["theta"][0])),
            "coefficients": float(np.max(abs(fit.coefficients-parsed["coefficients"]))),
            "covariance": float(np.max(abs(fit.covariance-covariance))),
            "se_logtheta": float(abs(fit.se_logtheta-parsed["se_logtheta"][0]))}
    if any(gap > 2e-6 for gap in gaps.values()):
        raise SystemExit(f"pscl/Python zero-inflated NB comparison failed: {gaps}")
    print(json.dumps({"reference": "pscl 1.5.9", "rows": len(y), "gaps": gaps}, indent=2))


if __name__ == "__main__":
    main()
