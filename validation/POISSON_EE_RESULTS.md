# Clustered Poisson estimating equations

`fit_poisson_ee` estimates the marginal log
expected-count ratio for one joint score increase in an additive Poisson-log
model. It sums conditional and marginal estimating functions within each
cluster before calculating the robust covariance. The result treats estimated
exposure cutpoints as fixed. There is no small-cluster correction, offset or
time-lag model in this subset.

The optional exporter fitted two 96-row, 32-cluster count datasets with an
isolated installation of R qgcomp 2.19.6: one with four quantile levels and
one with four prequantized levels. The [saved fixture](../tests/fixtures/r_poisson_ee_2_19_6.json)
has SHA-256 `d5c4509305e93f8c8814d385f879bd531242bc84a817d9fa89fbda989b5091e3`.
The [machine-readable comparison](poisson-ee-reference-results.json) checks
scores, cuts, conditional and marginal coefficients, full and component
covariances, cluster meat, derivative bread, joint effect, interval, test and
intervention means. The largest difference in the fitted joint effect is
`6.16e-13`; the largest robust-covariance difference is `5.53e-12`.
R stores a Richardson numerical derivative. Python uses the exact derivative,
so the derivative matrices differ by as much as `1.16e-6` in absolute value
without changing the tested estimates materially. R saves synthetic breaks
for prequantized inputs; Python keeps `breaks=None`.

An independent exact-cell check uses count means 2, 4, 6 and 12 for the four
two-exposure cells. It verifies the joint log mean ratio `log(6)` and computes
its cluster sandwich directly from cell means and cluster score totals. This
checks the method separately from the R implementation.

Reproduce the installed-R comparison with a compatible Rscript and isolated
library:

```sh
PYTHONPATH=src python validation/check_poisson_ee_r.py \
  --rscript /path/to/Rscript --library /path/to/isolated-R-library \
  --output /path/to/new-report.json
```

R and `jsonlite` are optional validation tools. The installed Python package
does not require R. Fixed cases and exact-cell arithmetic do not establish
interval coverage across repeated datasets, small-cluster reliability,
overdispersion robustness or broader R options. Those remain open checks.
