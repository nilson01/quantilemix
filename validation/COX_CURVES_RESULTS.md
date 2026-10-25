# Conditional Cox survival curves

`CoxResult.survival_curve()` returns conditional
survival probabilities for new complete exposure and adjustment profiles. It
uses the Cox fit's Efron baseline cumulative hazard and keeps the exposure
cutpoints learned during fitting. Rows are observed follow-up times, including
times with censoring only; columns follow the supplied profile order. Before
the first returned time, fitted survival is one. No uncertainty band or
population-average contrast is reported.

Two fixed 128-row cases match **isolated installed** R qgcomp 2.19.6 and
`survival::survfit`. They reuse the pinned shared-time Cox inputs: one uses
integer exposure scores and one groups raw exposure measurements into four
categories. Each comparison checks four profiles, their exposure scores, every
time, the Efron baseline cumulative hazard, and conditional survival. The
largest absolute cumulative-hazard difference is `4.89e-15`; the largest
survival-probability difference is `8.89e-16`. The saved
[reference result](cox-curves-reference-results.json),
[fixture](../tests/fixtures/r_cox_curves_2_19_6.json), and
[R exporter](export_cox_curves_r.R) retain the values and version details.
An independent three-person risk set gives a baseline hazard increment
`1/3 + 1/2 = 5/6` when two events share a time and a third person is
censored then. A later censor-only time leaves that hazard unchanged.

Replay the optional installed-R comparison with an isolated library containing
qgcomp 2.19.6, survival and jsonlite:

```sh
PYTHONPATH=src python validation/check_cox_curves_r.py \
  --rscript /path/to/Rscript --library /path/to/isolated-R-library \
  --output /path/to/new-report.json
```

The checker verifies the pinned input, fixture and exporter bytes, compares
fresh R values with the saved R values, then compares Python. R is not a
Python runtime dependency. Matching a fitted curve does not test proportional
hazards, independent censoring, causal assumptions or interval coverage.
See [marginal curve checks](COX_SURVCURVE_RESULTS.md) for population summaries.
These conditional curves have no uncertainty bands.
