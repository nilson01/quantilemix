# Two-level marginal binary risk-ratio checks

In these two-level cases, `fit_binomial_risk_ratio` estimates the population-average risk ratio
for setting every exposure score to one versus zero. It standardizes predicted
event probabilities over the observed adjustment covariates. This estimand
differs from the conditional odds ratio returned by `fit_binomial`.
The later [four-level extension](RISK_RATIO_MULTILEVEL_RESULTS.md) fits a
degree-one marginal log-risk model across all four standardized risks.

Two fixed cases were exported from **installed** R qgcomp 2.19.6 in an
isolated library. One has no adjustment covariate; the other includes one.
The saved exporter, inputs and R outputs are under `validation/` and
`tests/fixtures/`. The optional checker reruns the exporter and applies the
same zero-based bootstrap row schedule in Python. It verifies the log risk
ratio, standardized risks, each of 24 bootstrap effects, standard error,
normal interval and normal p-value. The largest absolute Python/R difference
is 1.10e-8 for one replicate; the largest non-replicate difference is
9.54e-10. The different solvers explain these small numerical differences.

Run the optional comparison only where the isolated R qgcomp 2.19.6 library
and `jsonlite` are already installed:

```sh
PYTHONPATH=src python validation/check_risk_ratio_r.py \
  --rscript /path/to/Rscript --library /path/to/isolated-R-library \
  --output /path/to/new-report.json
```

The Python package has no R runtime dependency. These fixed cases use complete numeric rows, two intervention levels and
independent row resampling without case weights. A failed bootstrap replicate is reported.
These fixed comparisons establish software agreement for this subset. A
separate [4,000-dataset known-score study](RISK_RATIO_CALIBRATION_RESULTS.md)
assesses the implemented interval in four specified conditions: coverage was
94.9%–96.7% among 3,999 successful fits, and one sparse-null dataset returned
no interval after a separated bootstrap draw. The fixed comparisons alone do
not establish interval coverage; neither check establishes causal
identification, clustered-data validity, or broader binary R-package coverage.
An [exact-draw installed-R diagnostic](RISK_RATIO_SEPARATION_R_RESULTS.md)
documents R's finite stopping value on that separated input and Python's
deliberate error for a conditional fit with no finite maximum.
