# Gaussian exposure-covariate interaction checks

`fit_quadratic_bootstrap(..., covariate_interactions=True)` includes every
product of an exposure score and a numeric adjustment covariate. The result
still reports a degree-two joint-response curve. Its linear coefficient is a
population-average effect: each interaction coefficient is multiplied by the
mean of its covariate. Each bootstrap replicate uses the covariate mean in
that resampled dataset, so the uncertainty includes changes in the sampled
covariate distribution. The flag defaults to `False`, preserving earlier
square-only and pairwise-exposure behavior.

Two fixed cases agree with **installed** R qgcomp 2.19.6: one has prequantized
scores and exposure-covariate products; the other has four score categories,
exposure-covariate products and a pairwise exposure product. Both use the same
12 R-generated row schedules in Python. The optional checker verifies the
joint-curve coefficients, all bootstrap coefficients, covariance, interval,
p-values, standardized means, scores and applicable cutpoints. The largest
absolute Python/R gap is 1.50e-14. An independent three-exposure,
two-covariate least-squares check verifies product ordering and use of the
resampled covariate means.

Run the optional check with an already installed isolated R library:

```sh
PYTHONPATH=src python validation/check_covariate_interactions_r.py \
  --rscript /path/to/Rscript --library /path/to/isolated-R-library \
  --output /path/to/new-report.json
```

The Python runtime needs no R. This subset requires complete numeric rows,
at least three consecutive joint intervention scores, independent row
resampling and no case weights or cluster structure. Fixed-case agreement
does not establish repeated-dataset interval coverage or causal validity.
