# Weighted clustered estimating equations

`fit_gaussian_ee`, `fit_poisson_ee`, and
`fit_binomial_ee` now accept strictly positive row-level case weights.
Each weight multiplies that row's conditional score and its score at every
joint intervention level. Scores are summed within cluster before the robust
covariance is calculated. Standardized means or risks use the same weights.
Exposure cutpoints remain unweighted, as in R qgcomp 2.19.6. The result saves
the supplied weights as a read-only array.

An isolated installation of R qgcomp 2.19.6 produced four fixed 96-row,
32-cluster cases: Gaussian, Poisson, binary marginal odds ratio and binary
marginal risk ratio. The [saved fixture](../tests/fixtures/r_weighted_ee_2_19_6.json)
has SHA-256 `61646422bf2d2a8ef99620135bbfe284dc64d420b1f0363e771289420177b450`.
The [comparison report](weighted-ee-reference-results.json) checks the
conditional and marginal coefficients, full and component robust covariance,
cluster meat, derivative bread, intervention scores, cuts, standardized
outcomes, intervals and tests. The largest joint-effect difference is
`9.87e-14`; the largest full-covariance difference is `2.75e-11`.
The largest bread difference is `5.31e-7`: R uses a numerical Richardson
derivative, while Python calculates the derivative exactly.

Independent checks use integer weights from one through four and repeat each
row that many times **within the same cluster**. For all four cases, weighted
and repeated-row fits agree on the conditional and marginal coefficients,
full sandwich covariance, bread, meat and interval. Doubling every weight
leaves the effect and cluster covariance unchanged. Invalid, zero, negative
or nonfinite weights are rejected.

Reproduce the optional installed-R check with a compatible Rscript and an
isolated R library containing qgcomp 2.19.6 and jsonlite:

```sh
PYTHONPATH=src python validation/check_weighted_ee_r.py \
  --rscript /path/to/Rscript --library /path/to/isolated-R-library \
  --output /path/to/new-report.json
```

R is not a Python runtime dependency. These fixed and algebraic checks do
not establish interval coverage across datasets, small-cluster reliability,
or behavior under binary separation or count overdispersion. These cases do not assess weighted nonlinear or time-lag models. A separate
[Poisson EE offset check](POISSON_EE_OFFSET_RESULTS.md) covers unequal observation times.
