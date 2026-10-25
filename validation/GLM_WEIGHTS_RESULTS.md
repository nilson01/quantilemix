# Positive case weights for additive binary and count fits

`fit_binomial`, `fit_binomial_formula`, `fit_poisson` and
`fit_poisson_formula` accept positive case weights. The weights are GLM prior weights: they multiply each
row's log-likelihood and Fisher information. The joint effect remains the sum
of exposure coefficients. Quantile cutpoints use exposure observations before
weighting. The result saves a read-only `case_weights` array and records
`diagnostics["case_weighted"]`. The array calls require complete data; formula
calls can drop rows with missing values or weights when `missing="drop"` is
explicit. Zero, negative, nonfinite and malformed weights fail clearly.

## Independent and R checks

Integer weights from one through three give the same coefficients, covariance
and joint estimate as repeating each row that many times, for both families.
Separate tests check missing-weight omission and input failures. Four fixed
cases were exported from an isolated installation of R qgcomp **2.19.6**:
exact-score and automatically quantized fits in each family. The optional
checker re-exports those same cases from the requested installed package,
verifies its loaded namespace and the fixture hash, then compares inputs and
all saved R results. Run it with:

```sh
python validation/check_glm_weights_r.py --rscript /path/to/Rscript \
  --library /path/to/isolated-r-library
```

The pinned [fixture](../tests/fixtures/r_glm_weights_2_19_6.json) has SHA-256
`cb51d6eb918089c528bf974b9de40954e1f5c92b4c5666320b11c6dabf0727b2`.
The R exporter has SHA-256
`af7b82873978e9c73ea44be74e61162f7c601e1dabab5895a1b8ad9d436c125c`.
Scores and prior weights agree exactly. The largest observed absolute Python/R
difference is approximately `3.1e-7` in a binomial confidence limit;
all coefficients, full covariance entries, joint variance, confidence limits,
p-values, fitted values and predictions agree within `5e-7`. R `glm` stops
at a different tolerance from the Python solver. This tolerance describes
these fixed cases and does not establish repeated-sample interval coverage.

## Limits

These are model-based
uncertainty estimates. They are not survey or cluster-robust variances, and
fractional binomial prior weights can trigger R's noninteger-success warning.
These fixed cases do not assess grouped trials or weighted interval coverage.
