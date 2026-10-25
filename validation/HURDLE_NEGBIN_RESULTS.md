# Conditional negative-binomial hurdle check

The Python fit estimates two separate effects. The positive-count effect is
on the log scale of the underlying negative-binomial mean. The nonzero effect
is on the log odds scale for having any count. Neither effect alone is a
ratio of observed mean counts. The negative-binomial size parameter
`theta` is estimated with the count regression terms. Its covariance is
included when calculating uncertainty for the count effect.

One deterministic 480-row case has two already scored exposures, one numeric
adjustment variable, 209 zeros and 271 positive counts. The optional R
exporter fits `pscl::hurdle(..., dist="negbin", zero.dist="binomial")` from
the isolated `pscl` 1.5.9 installation. This is the fitting engine used by
the R qgcomp hurdle method. The Python and R data generators give identical
counts (sum 610). On this case, the largest absolute gaps are
1.10 × 10⁻⁷ for `theta`, 2.05 × 10⁻⁸ for coefficients, and
7.03 × 10⁻⁸ for their regression covariance. The comparison also checks
an independent finite-difference derivative of the count log likelihood,
expected-count prediction after zero truncation, component placement,
and separate adjustment columns.

Run the optional installed-R check from the package root:

```sh
QGCOMP_R_LIB=../.r-library-qgcomp-2.19.6 \
  PYTHONPATH=src ../.venv/bin/python validation/check_hurdle_negbin_pscl.py
```

This is an installed-`pscl` engine comparison, not an installed-qgcomp
wrapper comparison. Re-exporting needs R and `pscl` 1.5.9; Python fitting
does not need R.

This case checks software agreement, not interval coverage, false-positive
rate, model misspecification, or causal identification. Weights, clusters,
offsets, formula inputs and partial exposure sets are not assessed here.
See the separate [marginal bootstrap checks](HURDLE_NEGBIN_BOOTSTRAP_RESULTS.md).
