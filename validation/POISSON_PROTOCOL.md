# Additive Poisson-log calibration protocol

Protocol `poisson-known-scores-v1`, prepared before study outcomes.

## Question and target

For a correctly specified Poisson model with equal observation periods, how
well does the existing estimator cover the conditional log-mean ratio for a
one-score increase in all four exposures? This does not assess a marginal
mean or rate contrast, offsets, overdispersion, or a causal effect.

Generate 500 independent rows per dataset. Four scored exposures are
independent uniform draws from {0,1,2,3}. Let S be their sum and draw
Y given S from Poisson(exp(log(m) + b*(S-6)/4)). The true joint coefficient
is b. The design value m is the model mean at the hypothetical score vector
(1.5,1.5,1.5,1.5), used only to set the intercept; it is not the exact
marginal mean.

| Factor | Fixed values |
| --- | --- |
| Center-point mean m | 0.20, 2.00 |
| Joint coefficient b | 0 (null), 0.40 (nonzero) |
| Independent datasets per cell | pilot 5; study 1,000 |
| Total datasets | pilot 20; study 4,000 |
| RNG | NumPy PCG64 via SeedSequence([20261023, mode, cell, replicate]) |

Fit intercept and all four saved scores, q=None, without weights or
covariates. Cells are ordered by m, then b; mode integers separate pilot and
study draws. Keep every case identity, input hash, count total, zero count,
warning, failure, estimate, variance, interval and p-value. Do not select or
drop datasets using fit results.

## Comparisons and planned table

The separate installed-R 2.19.6 check uses two existing fixed Poisson cases.
The repeated-data check independently solves the Poisson score equations
with SciPy, computes inverse observed information and the four-coefficient
contrast, and compares estimate, variance, normal interval and p-value with
absolute tolerance 1e-7 and relative tolerance 1e-6. It audits all cases,
including package failures, and replays any failure. The runner records any
ordinary per-case exception and continues. The independent solver requires a
small score residual even if its library reports success. Retain every warning,
mismatch and independent failure; return nonzero after writing the complete
audit.
The runner writes a case journal after each attempted fit. If interrupted,
the journal preserves completed cases and no full result is claimed or resumed
under the same identity. Warning category and bounded message are retained.

For each cell, report planned, attempted, successful, failed and missing
counts; realized mean and zero fraction; bias and its Monte Carlo standard
error; empirical standard deviation; root mean reported variance; 95%
interval coverage and Wilson 95% interval; and 0.05 rejection rate and Wilson
interval. Label null rejection false-positive rate, nonzero rejection power.
Coverage denominators use successful fits; also show covered/planned and
best/worst all-planned coverage bounds. At 1,000 successful datasets,
approximate Monte Carlo standard error at 95% coverage is 0.0069.

Investigate bias above three Monte Carlo standard errors, a coverage Wilson
interval wholly outside [0.93,0.97], any fit failure, warning or numerical
mismatch. These are review flags, not pass criteria or reasons to stop early.
Report every cell. Do not treat R agreement as statistical calibration.

## Execution boundary

After a fresh local resource check, run only the 20-case timing pilot locally
with one numerical worker/thread, <2 GiB memory and <100 MiB output. Full
4,000-case evidence is SERVER_CPU and requires a clean committed source,
protocol review, successful pilot, fresh server quota/queue check, a bounded
job specification, independent audit, and returned hashes. Partial output
cannot be resumed under the same run identity. This study does not change the
estimator or authorize other outcome-family claims.
