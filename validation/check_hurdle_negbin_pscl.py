"""Optional fixed-case comparison with installed pscl 1.5.9 (R)."""

from pathlib import Path
import json
import os
import subprocess

import numpy as np
from scipy import special, stats

from quantilemix import fit_hurdle_negative_binomial


ROOT = Path(__file__).resolve().parents[1]


def case():
    i = np.arange(480)
    x = np.column_stack((i % 4, (i // 4) % 4))
    z = np.sin(i*.17)
    mu = np.exp(.3+.15*x[:, 0]-.1*x[:, 1]+.2*z)
    theta = 1.3
    chance = special.expit(.2+.2*x[:, 0]-.15*x[:, 1]-.1*z)
    u0 = ((i*37) % 479+.5)/479
    u1 = ((i*113) % 479+.5)/479
    p0 = stats.nbinom.pmf(0, theta, theta/(theta+mu))
    y = (u0 < chance)*np.maximum(
        1, stats.nbinom.ppf(np.minimum(p0+u1*(1-p0), np.nextafter(1., 0.)),
                             theta, theta/(theta+mu)))
    return y, x, z


def main():
    script = ROOT / "validation/export_hurdle_negbin_pscl.R"
    run = subprocess.run([os.environ.get("RSCRIPT", "Rscript"), str(script)],
                         cwd=ROOT, check=True, capture_output=True, text=True)
    parsed = {}
    for line in run.stdout.strip().splitlines():
        name, *values = line.split("\t")
        parsed[name] = np.array([float(v.strip()) for v in values])
    y, x, z = case()
    if int(parsed["n"][0]) != len(y) or int(parsed["sum_y"][0]) != int(sum(y)):
        raise SystemExit("R/Python fixed-case data differ")
    fit = fit_hurdle_negative_binomial(y, x, z, q=None)
    coefficients = np.r_[parsed["count"], parsed["zero"]]
    covariance = np.zeros((8, 8))
    covariance[:4, :4] = parsed["count_cov"].reshape((4, 4), order="F")
    covariance[4:, 4:] = parsed["zero_cov"].reshape((4, 4), order="F")
    gaps = {"theta": float(abs(fit.theta-parsed["theta"][0])),
            "coefficients": float(np.max(abs(fit.coefficients-coefficients))),
            "covariance": float(np.max(abs(fit.covariance-covariance)))}
    if gaps["theta"] > 8e-7 or gaps["coefficients"] > 8e-7 or gaps["covariance"] > 2e-6:
        raise SystemExit(f"pscl/Python hurdle NB comparison failed: {gaps}")
    print(json.dumps({"reference": "pscl 1.5.9", "rows": len(y), "gaps": gaps}, indent=2))


if __name__ == "__main__":
    main()
