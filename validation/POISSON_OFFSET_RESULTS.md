# Poisson observation-time offset reference check

The conditional Poisson fit now accepts a positive observation time for each
count. Its logarithm enters the model with coefficient fixed at one. The joint
effect compares the fitted count rates at the same observation time.

Two fixed cases passed against **installed R qgcomp 2.19.6** under R 4.5.1.
The exact-rate case has rates 1, 2, 3 and 6, observed for 1 through 5 time
units. Its fitted joint rate ratio is 6. The second case uses automatic
quartile scores and a numeric adjustment. Both compare coefficients, full
covariance, joint inference, training scores, fitted counts and predictions at
new times. The largest absolute covariance difference was
`6.01e-9`; the largest prediction difference was `2.08e-12`.

The [machine-readable result](poisson-offset-reference-results.json) records
field-level differences, versions, fixture/exporter hashes and tolerances.
The [checker](check_poisson_offset_r.py) reruns the installed package in an
isolated R library. Routine Python tests use the pinned fixture and need no R.
The R formula contains `offset(log(time))`; Python's restricted formula API
uses `exposure_time="time"` so that the time column is checked explicitly.

These two deterministic cases establish implementation agreement for this
subset. They do not establish interval coverage across repeated datasets.
They do not assess overdispersion or broader formula syntax. See the separate
[marginal offset check](POISSON_BOOTSTRAP_OFFSET_RESULTS.md).
