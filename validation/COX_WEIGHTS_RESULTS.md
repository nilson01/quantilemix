# Conditional Cox case weights

`fit_cox(..., weights=w)` accepts one positive, finite case weight per row. The weights affect event
contributions and risk sets, but not exposure cutpoints. The joint effect is a
conditional log hazard ratio for increasing every exposure score by one while
holding adjustment values fixed.

Shared event times use Efron's approximation. For a group with $d$ events,
the $d$ risk denominators remove $0/d,\ldots,(d-1)/d$ of the weighted event
risk. Each denominator uses the group's average event weight. The observed
information gives the model-based covariance. As in the installed R
`survival::coxph` default, fractional case weights instead trigger a row-score
sandwich covariance; each row's score includes its weight. Positive integer
weights use the model-based covariance. The baseline cumulative hazard uses
the same weighted risk sets. The weight-only sandwich treats rows as
independent. A separate [cluster comparison](COX_CLUSTER_RESULTS.md) checks
clustered covariance.

Three fixed 128-row cases were exported from **installed R qgcomp 2.19.6** in an isolated R 4.5.1 library with survival 3.8.3: fractional weights with distinct event times, fractional weights with tied times, and integer weights with distinct times. Python agrees on coefficients, covariance, joint effect, interval, p-value, event count and weighted baseline hazard. The largest coefficient difference is **2.73 × 10⁻¹⁰** and the largest baseline-hazard difference is **8.98 × 10⁻¹⁰**, both in the tied case. The [saved check](cox-weights-reference-results.json) records each gap. The optional [checker](check_cox_weights_r.py), [R exporter](export_cox_weights_r.R) and [fixed fixture](../tests/fixtures/r_cox_weights_2_19_6.json) pin the inputs and exporter hashes. R is used only for optional validation; Python fitting has no R dependency.

Independent checks give the weighted partial likelihood and score for three-person risk sets, compare score and information with finite differences, and compare Python's weighted per-row score residuals with those returned by `survival::coxph`. Unit weights recover the original unweighted fit; invalid weights fail clearly. These fixed cases do **not** establish proportional hazards, independent censoring, case-weight interpretation for a specific sampling design, or confidence-interval coverage across repeated datasets. They do not test clusters, offsets or case-cohort sampling.
