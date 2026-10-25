# Restricted clustered Gaussian estimating equations

`fit_gaussian_ee` estimates an additive joint mean
difference and cluster-robust uncertainty from the stacked conditional and
marginal estimating equations. Rows with the same integer ID contribute one
combined score to the sandwich covariance. The result saves the conditional,
marginal and full covariance blocks, cluster meat, derivative bread and
intervention means. Exposure cutpoints are fixed before interventions.

Two fixed cases agree with isolated installed R qgcomp **2.19.6**: one uses
assigned exposure scores and the other automatic quartiles. Both have 96 rows
in 32 clusters. The [fixture](../tests/fixtures/r_gaussian_ee_2_19_6.json)
saves R inputs and output fields. The
[report](gaussian-ee-reference-results.json) records all field-level
differences. Point, robust covariance, intervention and inference quantities
agree within `2e-10`; the largest gap outside the derivative matrix is below
`1e-10`. Python's exact derivative matrix differs from R's numerically
estimated Richardson derivative by at most `1.4460056263487786e-08`, within
its separate `2e-8` tolerance. An independent conditional cluster-sandwich
calculation reproduces the joint effect and robust variance without using
the stacked-matrix implementation.

Re-run the optional installed-package check with:

```sh
python validation/check_gaussian_ee_r.py --rscript /path/to/Rscript \
  --library /path/to/isolated-r-library --output /path/to/new-report.json
```

The checker pins its R exporter and fixture, rejects changed cases and checks
live installed R values against saved R and Python. R writes synthetic
prequantized cutpoints; the Python API retains `breaks=None` in that mode.
Neither difference changes fitted scores or the effect estimate.

These two cases use complete numeric rows and exact integer cluster IDs.
They do not assess weighted fits or small-cluster reliability. Separate
[weighted comparisons](WEIGHTED_EE_RESULTS.md) cover positive case weights.
Fixed agreement does not establish interval coverage or causal validity.
