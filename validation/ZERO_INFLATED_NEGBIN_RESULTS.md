# Conditional zero-inflated negative-binomial check

The Python fit has an underlying negative-binomial count process and a
separate logistic process for structural zeros. The count effect is a log
ratio of underlying count means. The structural-zero effect is a log odds
ratio. Neither alone is an observed-count ratio. The count size `theta` is
estimated jointly with both components, and its uncertainty enters the
regression covariance.

One deterministic 720-row case has two already scored exposures, one numeric
adjustment variable, 412 observed zeros and a total count of 783. The
optional R exporter fits `pscl::zeroinfl(..., dist="negbin")` with
`maxit=1000`, `reltol=1e-12` and `EM=FALSE` from isolated `pscl` 1.5.9.
This is the fitting engine used by the R qgcomp zero-inflated method.
The R and Python data generators give identical counts. Maximum absolute
gaps are 9.48 × 10⁻⁷ for `theta`, 4.34 × 10⁻⁷ for coefficients,
3.52 × 10⁻⁷ for their regression covariance, and 3.43 × 10⁻⁷ for the
standard error of log `theta`. An independent finite-difference derivative
checks the complete mixture score. Focused tests also cover expected-count
prediction, component placement, separate adjustment, and invalid outcomes.

Run the optional installed-R check from the package root:

```sh
QGCOMP_R_LIB=../.r-library-qgcomp-2.19.6 \
  PYTHONPATH=src ../.venv/bin/python validation/check_zero_inflated_negbin_pscl.py
```

The Python package has no R runtime dependency. The optional check requires
R and installed `pscl` 1.5.9. This is an **installed-pscl engine** comparison, not a
direct installed-qgcomp wrapper comparison. The Poisson zero-inflated
method retains its own direct installed-qgcomp cases.

One fixed case does not establish interval coverage, false-positive rate,
model misspecification behavior, or causal identification. Weights,
clusters, offsets, formulas, partial exposure sets, wider mixture options remain untested by this case.
