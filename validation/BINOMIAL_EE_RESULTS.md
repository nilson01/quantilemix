# Clustered binary estimating equations

`fit_binomial_ee` fits a conditional logistic model
and then a marginal model across joint exposure scores. With `rr=False`, the
marginal link is logit and the reported ratio is an odds ratio. With `rr=True`
(the default), the marginal link is log and the reported ratio is a risk ratio.
Cluster totals of both estimating functions enter the robust covariance.
Estimated exposure cutpoints are treated as fixed; there is no small-cluster
correction, weight, offset or time-lag model in this subset.

The optional exporter fitted four fixed 96-row, 32-cluster datasets in an
isolated installation of R qgcomp 2.19.6: odds and risk links at two and four
joint score levels. The [saved fixture](../tests/fixtures/r_binomial_ee_2_19_6.json)
has SHA-256 `17a7d0d0e379449ca22fafedc22db3b216aa05d731459dff88a24ce6da7a8e98`.
The [machine-readable comparison](binomial-ee-reference-results.json) checks
scores, cuts, conditional and marginal coefficients, full and component
covariances, cluster meat, derivative bread, intervals, tests and standardized
risks. Across all four cases, the largest joint-effect difference is
`9.25e-14`, and the largest full robust-covariance difference is `1.12e-11`.
R saves a Richardson numerical derivative; Python uses exact derivatives.
The largest derivative-matrix difference is `1.40e-8`.

An independent 120-row exact-cell check has standardized risks 1/5 and 2/3
at the two joint levels. It verifies a marginal odds ratio of 8 and a marginal
risk ratio of 10/3. It also computes each cluster sandwich from the logistic
information matrix, cluster score totals and the appropriate effect gradient.
This tests the uncertainty calculation separately from R. At four levels,
`exp(psi)` is the fitted marginal-model slope. In the fixed risk-ratio case it
is 1.3495; the adjacent standardized-risk ratio is 1.4782. They answer
different questions.

Reproduce the installed-R comparison with a compatible Rscript and isolated
library:

```sh
PYTHONPATH=src python validation/check_binomial_ee_r.py \
  --rscript /path/to/Rscript --library /path/to/isolated-R-library \
  --output /path/to/new-report.json
```

R and `jsonlite` are optional validation tools. The installed Python package
does not require R. Fixed cases and exact-cell arithmetic do not establish
interval coverage across repeated datasets, small-cluster reliability,
behavior under separation or the broader weighted, nonlinear and missing-data
options. These remain open checks.
