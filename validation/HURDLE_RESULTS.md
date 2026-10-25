# Conditional Poisson hurdle checks

This is a fixed-case software comparison, not a repeated-dataset interval
study. R qgcomp 2.19.6 was installed in the isolated
`../.r-library-qgcomp-2.19.6` library. The R exporter uses
`qgcomp.hurdle.noboot` and its installed `pscl::hurdle` dependency. The
Python runtime uses NumPy and SciPy only.

Two deterministic 240-row cases use complete, prequantized exposure scores,
one numeric covariate, zeros and positive counts. Both components have the
same additive predictors. The largest absolute R/Python difference across
the two component effects and standard errors is **1.21 × 10⁻⁷**. Full
coefficient and covariance comparisons also pass at `1 × 10⁻⁷`. The R
optimizer uses `maxit=1000, reltol=1e-12`. These are fixed-case numerical
agreements within stated tolerances, not exact equality.

An independent central-difference check covers every score and
observed-information entry of the positive-truncated Poisson likelihood.
The nonzero component uses logistic regression on the indicator `y > 0`.
The prediction check verifies
`Pr(y > 0) * mu / (1 - exp(-mu))`, where `mu` is the *underlying*
Poisson parameter. The count effect is a log ratio of these underlying
parameters. The hurdle effect is a log odds ratio for having a positive
count. Neither alone is an observed-count ratio. This meaning differs from
the structural-zero effect of a zero-inflated model.

Run the saved comparison without R:

```sh
PYTHONPATH=src python validation/check_hurdle_r.py
```

To regenerate from installed R 2.19.6, set `QGCOMP_R_LIB` to its library
path and, if needed, `RSCRIPT` to the Rscript executable, then use
`--regenerate`. Ordinary runs pin the fixture SHA-256
`cb8a15046fc6c2329ec87e4de589aa8996ed28551dd8a50effe57890c2853ae6`.
Neither the checker nor the Python package installs or downloads R.

Limits: no negative-binomial count component, component-specific predictors,
weights, clusters, offsets, formulas or marginal bootstrap. The two fixed
cases do not measure bias, interval coverage, false-positive rate or model
misspecification.
