# Marginal negative-binomial hurdle bootstrap checks

This restricted method estimates a degree-one population-average
positive-count slope and a separate slope for the odds of any count.
Under each joint exposure score, it samples adjustment profiles from
observed rows, draws nonzero status, and draws positive counts from the
zero-truncated negative-binomial distribution. It fits a marginal hurdle
model and refits both stages, including dispersion, after independent-row
resampling with fixed exposure cuts. Neither slope alone is a ratio of
observed mean counts.

The R qgcomp 2.19.6 hurdle bootstrap is labelled experimental. Its
`base_hurdle.R` component path applies a nonzero Bernoulli gate to an
untruncated `rnbinom` draw for `dist="negbin"`. Thus the outcome can be zero
even when the gate says nonzero. Python instead draws from the positive
truncated NB distribution to preserve the stated hurdle model. Its
conditional expected count uses
`Pr(Y>0) * mu / (1 - Pr_NB(Y=0))`, where `mu` is the underlying NB mean.
The separate conditional NB fit has an
[installed-pscl fixed comparison](HURDLE_NEGBIN_RESULTS.md).

Three-resample fixed tests check seeded reproducibility, standardized
expected counts against the public conditional predictor, normal interval
arithmetic, saved intervention levels, immutable outputs, invalid controls,
and an injected failure reported as bootstrap replicate 1. A deterministic
sampler check confirms that every count remains positive when the nonzero
gate is fixed open. These checks establish the implemented arithmetic, not
end-to-end installed-R equality or repeated-data interval coverage.

If BFGS stops above the count-score tolerance, the conditional fitter refines
the candidate by solving the score equations. It accepts the result only if
the likelihood is not reduced and the score improves; otherwise it raises
an optimizer error.

The Python method uses complete numeric independent rows, shared
adjustment in both components, and exposures in both components. Formula
inputs, weights, clusters, offsets and alternate simulation modes remain
open. A finite `MCsize` adds simulation noise that the row-bootstrap
interval does not separately remove. Failed replicates are named, not
silently replaced.
