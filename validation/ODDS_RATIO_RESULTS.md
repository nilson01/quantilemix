# Marginal binary odds ratio: installed-R and implementation checks

`fit_binomial_odds_ratio` estimates a marginal odds-ratio summary. With two
joint exposure-score levels, its exponentiated slope is the odds ratio between
those levels. With more than two levels, the degree-one logistic slope is a
fitted summary across all levels; it need not equal any single adjacent-level
odds ratio. The function first fits an additive conditional logistic model. At each joint
exposure score it averages predicted event probabilities over the observed
adjustment profiles. A degree-one logistic marginal model fits those
standardized risks. Independent rows are then resampled with exposure cuts
fixed, and the standard deviation of the fitted slopes supplies the normal
interval on the log-odds scale.

The fixed two-level test independently computes both standardized risks and
the ratio of their odds. A separate four-level test checks that the fitted
marginal slope solves both logistic score equations. Another test shows that
the marginal and conditional odds ratios differ with adjustment, and checks
seeded reproducibility. The installed-wheel example uses exact binary cells:
the standardized risks are 0.25 and 0.75, giving an odds ratio of 9.

This follows the `rr=FALSE` marginal structural model in R qgcomp 2.19.6's
`qgcomp.glm.boot`. The ordinary `qgcomp(..., rr=FALSE)` wrapper still selects
the conditional odds ratio for additive terms; Python keeps that route.
The new standalone function makes the marginal estimand explicit. Two fixed
320-row, four-level cases now agree with **installed** R qgcomp 2.19.6's
exported `qgcomp.glm.boot(..., rr=FALSE)`. One case has no adjustment; the
other has a numeric covariate. Each uses the same 24 independently sampled
bootstrap row sets in R and Python. The [saved comparison](odds-ratio-reference-results.json)
checks exposure scores, standardized risks, marginal-model coefficients and
fitted risks, effect, standard error, interval, p-value and every bootstrap
slope. The largest absolute difference is `2.099875828776021e-12` in a
marginal-model coefficient. This is a fixed-case software agreement check,
not a repeated-data interval study.

The earlier status-139 R load came from selecting Homebrew R 4.5.0 with a
library prepared under framework R 4.5.1. The matching R 4.5.1 executable
loads installed qgcomp 2.19.6 and runs the [optional checker](check_odds_ratio_r.py).
Python users do not need R at runtime. A separate
[four-level known-score study](ODDS_RATIO_Q4_CALIBRATION_RESULTS.md) measures
interval coverage in four specified synthetic conditions. Coverage with
estimated cuts, case weights, clusters and other models remains open.

This subset accepts complete numeric independent rows and either saved
exposure cuts or contiguous unquantized scores. It does not yet accept case
weights, formula inputs, clusters or missing rows. A failed resample is
reported by number rather than replaced. The result is an odds ratio; it
must not be read as a risk ratio. A causal interpretation needs the usual
identification and model assumptions.
