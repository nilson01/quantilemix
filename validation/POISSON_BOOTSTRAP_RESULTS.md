# Marginal Poisson bootstrap checks

`fit_poisson_bootstrap` estimates the log ratio of standardized expected
counts when all exposure scores rise by one category. With an additive log
link, this ratio equals the sum of the conditional exposure coefficients.
Its uncertainty comes from resampling independent rows while holding the
original exposure cutpoints fixed. Count outcomes refer to equal observation
periods; this method has no time offset or case weights.

Two fixed cases agree with **installed** R qgcomp 2.19.6: one uses two
already-scored levels without adjustment, and one groups two exposures into
four categories with a numeric adjustment covariate. Both use 24 identical
R-generated bootstrap row schedules in Python and R. The optional checker
verifies scores, saved boundaries where applicable, intervention means, all
replicate effects, the joint log mean ratio, bootstrap standard error, normal
interval and normal p-value. The largest absolute difference is 5.0e-10 for
one replicate. R saves synthetic boundaries in its prequantized case; Python
retains `breaks=None` because no grouping was performed.

Run the optional check with an already installed isolated R library:

```sh
PYTHONPATH=src python validation/check_poisson_bootstrap_r.py \
  --rscript /path/to/Rscript --library /path/to/isolated-R-library \
  --output /path/to/new-report.json
```

The Python runtime needs only NumPy and SciPy. The fixed comparisons establish
software agreement in these two cases. They do not establish repeated-data
interval coverage, an absolute rate interpretation, validity with unequal
observation periods, or broader count-family coverage.
