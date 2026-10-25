# Additive Bernoulli-logit calibration: known-score study

The prespecified [protocol](BINOMIAL_PROTOCOL.md) and its
synthetic protocol review preceded this study. All **4,000 planned datasets**
completed in four cells of 1,000. Every package fit and independent
likelihood fit succeeded, with zero captured warnings, missing cases or
numerical mismatches. The largest absolute difference between a saved
package result and an independently calculated value was `2.96e-11`.
These checks support the existing additive binary implementation in this
restricted setting. They do not establish full R qgcomp or paper coverage.

## Target and findings

Each dataset has 500 independent rows and four already scored exposures,
each taking values 0–3. The outcome follows a correctly specified logistic
model. The true coefficient `b` is the change in **conditional log odds**
when every exposure score increases by one, over score pairs where that
change stays within 0–3. For `b=0.40`, the conditional odds multiplier is
`exp(0.40) ≈ 1.49`. This is not a fixed change in event probability, a
marginal risk ratio or a causal effect. The design values 0.10 and 0.50 are
event probabilities at the score center, not guaranteed marginal event rates.
Realized event fractions are shown below.

Bias is mean estimated `b` minus true `b`. MCSE is the Monte Carlo standard
error of that mean across 1,000 datasets. Coverage is the fraction of nominal
95% Wald intervals containing true `b`. Wilson limits describe sampling
uncertainty in each observed coverage or rejection rate. All rates use 1,000
successful fits per cell; since there were no failures, the all-planned
coverage bounds equal the observed rates.

| Center-point event probability | True conditional log-odds `b` | Realized event fraction | Bias (MCSE) | Coverage, Wilson 95% interval | Rejection at 0.05, Wilson 95% interval |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | 0 | 0.1006 | -0.0139 (0.0087) | 0.946 [0.930, 0.958] | 0.054 [0.042, 0.070] |
| 0.10 | 0.40 | 0.1015 | +0.0067 (0.0087) | 0.955 [0.940, 0.966] | 0.318 [0.290, 0.348] |
| 0.50 | 0 | 0.5004 | +0.0028 (0.0051) | 0.952 [0.937, 0.964] | 0.048 [0.036, 0.063] |
| 0.50 | 0.40 | 0.4995 | +0.0073 (0.0052) | 0.952 [0.937, 0.964] | 0.697 [0.668, 0.725] |

The two rows with true `b=0` estimate false-positive rates. Their rejection
counts (54 and 48) are exactly the complements of their covered-interval
counts (946 and 952), because the two-sided Wald test and interval use the
same normal cutoff. They are not independent confirmation. Rejection in the
nonzero rows is descriptive power. The two event settings have different
power even with the same true coefficient because precision differs.

Empirical standard deviations of the estimates were 0.2752, 0.2747, 0.1615
and 0.1630 in table order. The corresponding square roots of mean reported
variance were 0.2718, 0.2740, 0.1618 and 0.1644. None of the four absolute
biases exceeded three MCSEs, and no coverage Wilson interval lay wholly
outside the prespecified [0.93, 0.97] investigation band. That band is a
diagnostic trigger, **not** an equivalence test. In particular, the lower
Wilson limit for the 0.10 null cell is 0.9302, just above the band edge.
Four cellwise intervals do not give a simultaneous guarantee. The observed
coverage is near 95% in these four conditions, with visible Monte Carlo
uncertainty; broader calibration is still untested.

## Independent checks and provenance

An independent score-equation solver recomputed the coefficient sum,
observed-information variance, interval and p-value on every generated
dataset. Its per-cell coverage and rejection counts exactly matched the
package's. The checker recorded 4,000 case audits, including input identity,
score residual and information eigenvalue. It would retain and replay a
reported package failure or save every paired mismatch before returning a
nonzero status. Fault-injection tests covered fabricated package failure,
independent failure, multiple mismatches and changed inputs. The software
check against isolated installed R qgcomp 2.19.6 passed its two fixed binary
cases. That same-input R comparison tests implementation agreement; it is
separate from the repeated-data coverage study.

The local 20-case pilot and its independent check passed before the full run.
Arseven Slurm pilot job `2905330` completed in seven seconds, followed by
study job `2905331` in 16 seconds. The study requested one CPU, 2 GiB RAM,
ten minutes and no GPU; one numerical thread was used. The clean source
commit was `5b43bc32abf0daa9b0192b884c4f2d74b3096088`, transferred in
a Git bundle with SHA-256
`5be77736f6ca59ecce4948f1ddb3970a2cf1a62c7b68d77ed6243bfbd6578864`.
The server used Python 3.11.5, NumPy 2.0.0 and SciPy 1.14.0. It saved 710 KiB
by `du -sk`; this disk-usage figure should not be read as a Python payload
byte count.

The saved result SHA-256 is
`1a054d5456904d6e885849b9017565f14b0408fd60c94141b67f28ef149b159a`;
the independent audit SHA-256 is
`b6ad8f290141340db6316b77c8141f413dbc8954c9effb333dfc553817475368`.
The returned manifest, all case identities and the 4,000-case audit passed
local read-only checks. The compact, machine-readable [summary](binomial-calibration-study-summary.json)
records each cell's full counts, uncertainty and provenance. Exact case-level
files, bundle and Slurm logs remain in the private project directory
`../validation-local/binomial-calibration-20261023/` for internal
reproduction. Outside analysts cannot presently retrieve that evidence.
Ordinary Python fitting needs NumPy and SciPy, not R.

## Limits

This is one correctly specified, independent, balanced and same-direction
known-score design at `n=500`. It does not test estimated quantile cuts,
correlated or mixed-sign exposures, much rarer events, model
misspecification, covariate adjustment, case weights, missing data,
clusters, or the separate marginal binary risk-ratio method. It does not
validate count, survival, censored, multinomial, zero-inflated or hurdle
families. The prior paper-study translation and its unresolved source gaps
remain separate. These limits prevent a broad R-package or paper-coverage
claim from this result.
