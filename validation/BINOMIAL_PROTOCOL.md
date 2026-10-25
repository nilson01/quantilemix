# Additive Bernoulli-logit calibration protocol

Protocol `binomial-known-scores-v1`, prepared before study outcomes.
A pre-study checker repair left the grid, seeds and metrics unchanged.
The installed-R comparison is a separate software-agreement check.

## Question and target

For a correctly specified additive logistic model, how well does the existing
Python method estimate a joint one-score increase and cover its true
**conditional log-odds ratio**? In this design, the true joint coefficient is
the sum of four exposure coefficients. Coverage of this coefficient does not
establish coverage of a marginal risk difference or a causal effect.

Generate independent observations with four already scored exposures
`Q_j` independently uniform on {0,1,2,3}. Let `S = Q_1+...+Q_4` and draw
`Y | Q` from Bernoulli(`expit(a + b*S/4)`). Here `b` is the true joint
coefficient. Set `a = logit(p) - 1.5*b`, so `p` is the event probability at
the center point `Q_j=1.5` for every exposure. This is a design value, not
the exact marginal event probability. Each dataset has 500 rows. Fit an
intercept and all four scores with `q=None`, no weights or covariates.

| Factor | Fixed values |
| --- | --- |
| Center-point event probability `p` | 0.10, 0.50 |
| Joint coefficient `b` | 0 (null), 0.40 (nonzero) |
| Independent datasets per cell | pilot 5; study 1,000 |
| Total datasets | pilot 20; study 4,000 |
| RNG | NumPy PCG64 via `SeedSequence([20261023, mode, cell, replicate])` |

The cells are ordered by `p`, then `b`. The pilot and study use distinct mode
integers. Raw exposure and uniform-draw arrays determine each Bernoulli
outcome; study inputs are not selected using fit results. Keep the individual
case identities, input hashes, event counts, warnings, failures, estimates,
variances, intervals and p-values. Do not silently drop failures.

## Comparisons and planned table

The software check runs the existing two fixed cases through isolated installed
R qgcomp 2.19.6 and Python. It compares the same conditional estimand;
existing covariance tolerance reflects the R GLM iteration convention.
The repeated-data comparator independently maximizes each dataset's logistic
likelihood using SciPy BFGS or solves its score equations. It computes inverse
observed information and the four-coefficient contrast without calling the
package estimator. Its role is to detect implementation errors, not to serve
as a different statistical method.

Compare successful paired estimates,
variances, interval limits and p-values at absolute 1e-7 and relative 1e-6;
retain mismatches. The checker attempts an independent solve on every case,
including a failed package fit. It replays any recorded package failure on the
regenerated input and compares the exception class and bounded reason. The
saved case audit records independent score residuals and the smallest
information eigenvalue when the independent fit exists. A failed independent
fit remains a separate failure.

Checker status is `needs_review` and its
command exits nonzero when any independent failure, package failure or
numerical mismatch occurs; it still writes the complete case audit first.

For each cell and method, report planned, attempted, successful, failed and
missing counts; realized event fraction; mean estimate minus truth (bias), its
Monte Carlo standard error, empirical standard deviation, root mean reported
variance, interval coverage, Wilson 95% interval for coverage, rejection rate
at 0.05 and its Wilson interval. Label null rejection false-positive rate and
nonzero rejection power. Denominators for coverage and rejection are successful
fits. Also give covered/planned and best/worst all-planned coverage bounds.
At 1,000 successes, a 95% coverage fraction has approximate Monte Carlo
standard error 0.0069. A null rejection fraction of 5% has the same precision.
These expected values are benchmarks, not acceptance criteria.

Do not treat R agreement as proof of statistical calibration. Logistic Wald
intervals can have imperfect finite-sample coverage, especially at a low
event rate. Investigate a bias larger than three Monte Carlo standard errors,
a coverage Wilson interval wholly outside [0.93, 0.97], any unexpected
failure, or a paired numerical mismatch. Report all cells even if a flag is
raised; no favorable-result stopping. No adjustment for multiple cells is
planned. The study does not assess score-cut estimation, exposure dependence,
confounding, weights, missing data, clusters, model misspecification, other
binary estimands, or other outcome families.

## Execution and evidence boundary

Run the 20-case local timing pilot only after a fresh local resource check and
workspace worker reservation. Record elapsed time, peak memory, output size,
failures and exact environment. The full 4,000-case study is `SERVER_CPU` and
requires a clean committed source, reviewed protocol, server job specification,
fresh server capacity/quota check, passing pilot, output and memory guards,
separate input/summary checker, and returned file hashes. A partial run is not
resumed under the same run identity. No changes to the estimator, other
methods, or public claims follow automatically from this protocol.
