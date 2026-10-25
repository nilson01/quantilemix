# Common questions

## Do I need R?

No. R is needed only to rerun the reference comparisons.

## What does `q=None` mean?

The exposures are already scored. For raw measurements, the default `q=4`
estimates quartile boundaries. Ties can leave fewer groups. Prediction reuses
training cutpoints; it does not estimate new ones.

## How are missing values handled?

Array fits require complete numeric data. Supported formula fits reject
missing values by default; `missing="drop"` removes incomplete training rows.
Removing rows does not by itself prevent bias. The imputation helpers have
separate assumptions; see [data tools](reference/tools.md).

## Are case weights the same as sign weights?

No. Case weights change a row's influence in supported fits. Sign weights
describe exposure coefficients within each sign group. Neither automatically
accounts for a complex survey design.

## Does a confidence interval establish a causal effect?

No. A causal interpretation needs well-defined interventions, consistency,
no interference, no unmeasured confounding, positivity and a correctly
specified model. The software cannot verify these assumptions.

## Does QuantileMix cover all of R qgcomp?

No. Each function supports a defined subset. The [coverage map](https://github.com/nilson01/quantilemix/blob/main/docs/r-api-parity.md)
lists important limits. [Validation](evidence.md) distinguishes software
agreement from statistical performance.

## How do I report a problem?

Open a [bug report](https://github.com/nilson01/quantilemix/issues/new?template=bug_report.yml)
with a small reproducible example, expected output, actual output, Python
version and package versions. Use synthetic data rather than private records.
