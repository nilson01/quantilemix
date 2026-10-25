# Conditional zero-inflated Poisson checks

This is a fixed-case software comparison, not a repeated-dataset test of
interval coverage or causal identification. R qgcomp 2.19.6 was installed in
the isolated `../.r-library-qgcomp-2.19.6` library. The R exporter uses
`qgcomp.zi.noboot` and its installed `pscl::zeroinfl` dependency. The
Python package uses only NumPy and SciPy at runtime.

The two deterministic 240-row cases use complete, prequantized exposure scores,
one numeric covariate, nonnegative counts and the same additive predictors in
the Poisson and structural-zero components. Both include observed zeros and
positive counts. The maximum absolute difference across the two component
effects and standard errors is **1.45 × 10⁻⁶**. The tighter R optimizer
setting is `maxit=1000, reltol=1e-12`; the stated `3 × 10⁻⁶` check tolerance
allows numerical optimizer and Hessian differences. This is agreement within
a tolerance, not exact equality.
The full coefficient vectors differ by at most **3.56 × 10⁻⁶** and full
covariance entries by at most **1.88 × 10⁻⁶**; both are checked with separate
finite tolerances.

The independent central-difference check compares every analytic score and
observed-information entry with derivatives of the full mixture log
likelihood. A separate prediction check verifies that expected observed
counts equal `(1 - structural_zero_probability) * poisson_mean`. The two
reported joint effects are sums of exposure coefficients **within each
component**. The count effect is a log Poisson-mean ratio; the zero effect is
a log odds ratio for membership in the structural-zero component. Neither
alone is the ratio of expected observed counts.

Run the saved comparison without R:

```sh
PYTHONPATH=src python validation/check_zero_inflated_r.py
```

To regenerate from an isolated installed R 2.19.6, set `QGCOMP_R_LIB` to its
library path and, if needed, `RSCRIPT` to the Rscript executable, then use
`--regenerate`. The checker pins the committed fixture SHA-256
`2397821e3989df7ab0b39b098a673b2571c3d8853a9481a31bfc7e9b0b9290ef`
for ordinary runs. It does not install R or download packages.

These Poisson cases do not assess the separate negative-binomial or marginal
bootstrap methods, component-specific predictor sets, weights, clusters,
offsets or formulas. The fixed cases do not measure bias, confidence-interval
coverage, false-positive rate or performance under model misspecification.
