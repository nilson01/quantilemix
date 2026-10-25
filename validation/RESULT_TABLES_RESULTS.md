# Conditional result tables: installed-R comparison

Seven fixed cases compare the public Python `tidy` and `glance` helpers with
the registered methods in **installed R qgcomp 2.19.6**. The cases reuse the
existing Gaussian, binary, count and Cox input fixtures. They include
prequantized and quantized exposures, adjustment variables and a 90% Gaussian
confidence level. The R exporter checks that qgcomp, jsonlite, generics and
broom load from the requested isolated library. The checker compares fresh R
tables with [saved tables](../tests/fixtures/r_result_tables_2_19_6.json) and
then with Python.

All seven cases passed, using R 4.5.1 and an isolated
qgcomp 2.19.6 library. Shared `tidy` terms, estimates, standard errors, link
scale limits, test statistics, joint-effect p-values and quick estimates agree.
For binary, count and Cox fits, exponentiated estimates and lower limits
agree. Shared `glance` values agree for the six Gaussian, binary and count
cases: null and fitted deviance, null and residual degrees of freedom,
log likelihood, AIC, BIC and row count. The largest absolute difference
among compared fields was `5.22e-6`, in a count test statistic near an
exactly fitted input. The comparison tolerance was `1e-5` for `tidy`
statistics and `1e-7` for `glance` measures.

This check found two Gaussian `glance` fields missing in Python:
`null_deviance` and `df_null`. They are now returned. Python also has
additional fit fields, such as Gaussian R-squared and the model degrees of
freedom, which this R table does not return. Python uses its own intercept
p-value. R 2.19.6's exponentiated `tidy` upper limit is
`exp(exp(lower link bound))`, because it exponentiates the overwritten lower
limit. Python uses `exp(upper link bound)`. The checker verifies both values
without treating the R display error as a target.

To repeat the optional installed-R check, install the Python development
dependencies and prepare an isolated R library as described in
[the reference guide](../docs/reference.md#installed-r-gaussian-check). Then run:

```sh
python validation/check_result_tables_r.py \
  --rscript /path/to/matching/Rscript \
  --library /path/to/isolated-r-library \
  --output /path/to/new-result-tables-report.json
```

The checker writes a new JSON report with case-level differences and source
hashes. R, broom and jsonlite are optional check dependencies; Python package
users do not need them. These seven fixed inputs test table agreement. They
do not establish repeated-dataset interval coverage, Cox `glance` support,
or table behavior for other result families.
