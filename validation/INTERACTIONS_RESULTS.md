# Gaussian pairwise exposure interactions

Setting `pairwise_interactions=True` in
`fit_quadratic_bootstrap` adds every distinct pairwise exposure-score product
to the conditional Gaussian model. The joint quadratic coefficient sums both
squared-term and pairwise-product coefficients. The default square-only method
and its inputs remain unchanged.

Two fixed cases from isolated installed R qgcomp 2.19.6 agree with Python
when both use identical bootstrap row schedules. The
[fixture](../tests/fixtures/r_interactions_2_19_6.json) includes full fitted,
marginal and bootstrap outputs. The [saved report](interactions-reference-results.json)
records field-level differences. The largest observed absolute difference is
`3.943512183468556e-13` (one test statistic), below the `1e-10` tolerance.
A separate three-exposure least-squares calculation checks
the conditional design, sum of six quadratic coefficients, marginal means,
bootstrap pairs and full covariance. The optional live-R check is:

```sh
python validation/check_interactions_r.py --rscript /path/to/Rscript \
  --library /path/to/isolated-r-library --output /path/to/new-report.json
```

The method requires complete numeric independent rows, at least two exposure
columns and three consecutive intervention levels. It has no cluster-aware
variance, weights, arbitrary formula, exposure-covariate product or alternate
outcome family. These fixed checks establish implementation agreement in the
stated cases; they do not estimate repeated-sample interval coverage.
