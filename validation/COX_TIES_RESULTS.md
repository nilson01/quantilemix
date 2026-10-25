# Cox fits with shared follow-up times

`fit_cox` accepts right-censored records with
shared follow-up times. When events share a time, it uses Efron's partial
likelihood approximation. People censored at that same time remain in the
risk set for its events. The result records the number of shared-time groups
and the maximized partial log likelihood. Positive time, complete numeric
inputs and more events than fitted coefficients remain required.

Two fixed 128-row cases agree with **isolated installed** R qgcomp 2.19.6,
which calls `survival::coxph` with its Efron default. One case has already
scored exposures; the other quantizes raw exposure measurements. They have
31 and 29 shared-time groups, respectively, including event/censor ties.
The [saved fixture](../tests/fixtures/r_cox_ties_2_19_6.json) has SHA-256
`c2cca8cb735dab35b299cbd485f5e88c86d2b3097cd0049331faa9e2ac5fa161`.
The [optional replay](cox-ties-reference-results.json) checks conditional
coefficients and covariance, joint log hazard ratio, variance, interval,
p-value, partial log likelihood, exposure scores, cutpoints and event count.
The largest difference among these numeric values is `2.85e-13`.

An independent three-person risk-set calculation checks both Efron
denominators and the corresponding score and information. The existing two
untied installed-R cases remain covered by their original fixture. Checker
tests reject changed inputs, reference values and tie methods, and verify
that a missing R executable cannot create a result file.

To replay the optional R check, supply an Rscript executable and an isolated
library containing qgcomp 2.19.6, survival and jsonlite:

```sh
PYTHONPATH=src python validation/check_cox_ties_r.py \
  --rscript /path/to/Rscript --library /path/to/isolated-R-library \
  --output /path/to/new-report.json
```

These cases compare conditional hazard ratios using Efron's approximation
to tied event ordering. They do not test model assumptions or interval
coverage. Separate reports cover [curves](COX_CURVES_RESULTS.md) and
[marginal survival fits](COX_BOOTSTRAP_RESULTS.md).
