# Marginal zero-inflated negative-binomial bootstrap checks

This restricted method estimates two degree-one population-average slopes:
one for the negative-binomial count component and one for the odds of a
structural zero. Under each joint exposure score, it samples adjustment
profiles from the observed rows, draws a structural-zero indicator and a
negative-binomial count from the conditional fit, and fits a marginal
zero-inflated negative-binomial model. It repeats both fits on independent
row-bootstrap samples while keeping the original exposure cuts. Dispersion
is estimated again in the marginal and each bootstrap fit. Neither slope
alone is a ratio of observed mean counts.

The R qgcomp 2.19.6 source uses component simulation with an NB count draw
for `dist="negbin"` in `base_zi.R`. Python follows this component design,
but NumPy and R draw different random streams. Three-resample fixed tests
check seeded reproducibility, the public conditional model's exact
standardized expected-count means, normal interval arithmetic, saved
intervention levels, immutable outputs, invalid controls, and an injected
failure reported as bootstrap replicate 1. The separate conditional NB fit
has an [installed-pscl fixed comparison](ZERO_INFLATED_NEGBIN_RESULTS.md).

These focused checks do not establish installed-R end-to-end numerical
agreement or repeated-data interval coverage. The method currently uses
complete numeric independent rows, shared adjustment in both components,
and full exposure placement in both components. Formula inputs, weights,
clusters, offsets, and alternate simulation modes remain open. A finite
`MCsize` introduces simulation noise that the row-bootstrap interval does
not separately remove. Failed replicates are named, not silently replaced.
