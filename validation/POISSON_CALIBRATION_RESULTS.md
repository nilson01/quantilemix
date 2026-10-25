# Additive Poisson-log known-score validation

This report tests the existing conditional Poisson-log
coefficient-sum method under four correctly specified, independent-score
conditions. The advance design is in `POISSON_PROTOCOL.md`; the machine-readable
summary is `poisson-calibration-study-summary.json`. This is internal software
validation, not a reproduction of the full Keil et al. paper.

Each of 4,000 independent datasets has 500 rows and four scores drawn from
{0,1,2,3}. The conditional mean is exp(log(m)+b(S-6)/4), where S is the sum
of scores, m is the mean at the hypothetical score vector (1.5,1.5,1.5,1.5),
and b is the true coefficient sum for raising every score by one. The fit uses
the saved scores (`q=None`) and a nominal 95% normal interval. It assumes
equal observation periods and Poisson conditional variance.

| Center mean m | True b | Observed mean | Zero fraction | Bias (MCSE) | Empirical SD / root mean variance | Coverage [95% Wilson] | Rejection [95% Wilson] |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.20 | 0 | 0.1992 | 0.8191 | -0.0034 (0.0060) | 0.1889 / 0.1819 | 0.934 [0.917, 0.948] | 0.066 [0.052, 0.083] |
| 0.20 | 0.40 | 0.2046 | 0.8158 | -0.0060 (0.0057) | 0.1797 / 0.1796 | 0.949 [0.934, 0.961] | 0.599 [0.568, 0.629] |
| 2.00 | 0 | 2.0024 | 0.1343 | +0.0015 (0.0018) | 0.0577 / 0.0568 | 0.945 [0.929, 0.958] | 0.055 [0.042, 0.071] |
| 2.00 | 0.40 | 2.0516 | 0.1414 | -0.0002 (0.0017) | 0.0549 / 0.0564 | 0.955 [0.940, 0.966] | 1.000 [0.996, 1.000] |

All four rows have 1,000 planned, attempted and successful fits, with zero
failed or missing cases and zero captured warnings. Coverage and rejection
denominators are 1,000 per row; the best/worst all-planned coverage bounds
therefore equal the displayed coverage. Bias is mean estimate minus true b;
MCSE is the Monte Carlo standard error of that mean. Rejection is the
two-sided p<0.05 fraction: false-positive rate for b=0 and descriptive power
for b=0.40. For this Wald test, rejection and coverage are complementary at
the null, so they are not independent checks.

The low-count null row merits attention. Coverage is 0.934, and its 95%
Wilson interval [0.917,0.948] sits just below nominal 0.95. Equivalently,
false-positive rejection is 0.066, with Wilson interval [0.052,0.083]. The
prespecified broad screening band [0.93,0.97] was not crossed by the whole
coverage interval, but that screen does not erase this narrower indication
of undercoverage. The other three cell intervals include 0.95. These are
cellwise Monte Carlo intervals; no simultaneous guarantee or equivalence
test was planned. No design or fitting rule was changed after inspecting the
study. In the sparse null cell, empirical estimate SD (0.18893) exceeds root
mean reported variance (0.18189) by about 3.9%. This is a descriptive clue,
not an established cause of the excess rejection. Zero observed failures do
not prove a zero failure probability; the rough one-sided 95% upper bound is
about 0.003 per 1,000-case cell under this design.

## Software agreement and provenance

The original Linux server checker regenerated every case and independently
solved the Poisson score equations. It paired all 4,000 fits, with zero
independent failures or mismatches. The largest absolute difference across
estimates, variances, interval endpoints and p-values was 2.54e-11. A later
Mac recheck found that the checker had assumed its imported package file lived
inside the source checkout. This was a checker path issue, not a fit or study
data issue. After that path was repaired, the Mac recheck again paired all
4,000 with zero mismatches; its checker source and report hashes are recorded
separately in the private evidence directory.

The separate installed-R qgcomp 2.19.6 check passed two **fixed** Poisson
cases. Those cases are different from the 4,000 study datasets. R is an
optional development reference, not a Python runtime dependency.

Server study source commit:
`63d969cf46c4dbc82887b79599a494fc15575d9e`. Python 3.11.5, NumPy
2.0.0 and SciPy 1.14.0 ran on Linux. Slurm pilot 2905431 finished 0:0 in
eight seconds; study 2905432 finished 0:0 in 23 seconds. The study requested
one CPU, 2 GiB RAM, ten minutes and zero GPU. Its saved output used 732 KiB
by `du -sk`. Server-return hashes were verified locally. The complete private
case-level paths are under `../validation-local/poisson-calibration-20261023/`:
`server-study-001/result.json` (SHA-256
`d70626c0f91141e16f516e88f053e8415863896f69d7e45f157a28588d0c68a5`)
and `server-study-001/check.json` (SHA-256
`33a021fc6391b33c50b94f5064d6231ee3b19aca0e303f5b05e2139a2de825b0`).
Exact raw files are private and not available from the package distribution.

This study does not cover estimated quantile cuts, correlated or mixed-sign
exposures, overdispersion, offsets, unequal observation periods, adjusted or
weighted fits, missingness, clusters, marginal count contrasts, or other
outcome families. It does not establish broad paper or R-package coverage.
