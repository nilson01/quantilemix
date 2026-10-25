# Conditional Poisson offset calibration protocol

Protocol `poisson-offset-known-scores-v1`. This
is an internal advance plan, separate from the completed equal-period study.
Its purpose is to test interval behavior for the implemented conditional
rate-ratio estimator when observation times differ. The plan is not a result
or a formal preregistration.

## Question and target

Each dataset has 500 independent rows and four exposure scores drawn
independently and uniformly from {0, 1, 2, 3}. Let `S` be their sum. Each row
also has an independent observation time `t`, drawn uniformly from
{0.25, 0.5, 1, 2}. Given scores and time, draw the count from a Poisson model
with mean `t * exp(log(m) + b * (S - 6) / 4)`. Here `m` is the rate at the
hypothetical center score vector `(1.5, 1.5, 1.5, 1.5)`, and `b` is the true
sum of the four score coefficients. The fitted model uses the saved scores
(`q=None`) and `log(t)` as an offset with coefficient fixed at one. It fits
an intercept and all four scores; no covariates or case weights are used.

| Factor | Values fixed before the study |
| --- | --- |
| Center rate `m` | 0.20 and 2.00 per time unit |
| True joint log rate ratio `b` | 0 (null) and 0.40 (nonzero) |
| Replicates per cell | 5 for the timing pilot; 1,000 for the proposed full study |
| Datasets | 20 pilot; 4,000 proposed full study |
| Random stream | NumPy PCG64; `SeedSequence([20261024, 91 or 92, cell, replicate])` |

The pilot and study use different mode integers. Cells are ordered by rate,
then effect. Do not replace a failed dataset or select cases based on results.
The null sets all four component coefficients to zero; it does not test
cancellation of positive and negative effects. The nonzero cell has four
equal positive coefficients. Construct PCG64 explicitly and record the NumPy
version with each run.
At a true 95% coverage probability, 1,000 successful replicates give a
Monte Carlo standard error of about 0.0069 for the coverage fraction.

## Checks and planned reporting

The runner records each case identity, input hash including time, total
observation time, count total, zero count, estimate, model variance, normal
95% interval, two-sided p-value, warnings and any fit failure. An independent
SciPy score-equation solver uses `t * exp(X beta)` and the full Fisher
information. It checks the score residual, information eigenvalues, estimate,
variance, interval and p-value. Every package failure is replayed and every
case is audited. A complete result needs all planned case identities; a
partial journal is retained on interruption and is not presented as complete.
Audit its saved rows and missing identities with the read-only partial-journal
checker, and record the interruption reason separately.

For each cell, report attempted, successful, failed and missing fits; observed
mean count, zero fraction and mean observation time; bias with its Monte Carlo
standard error; empirical estimate standard deviation; root mean reported
variance; 95% interval coverage with Wilson interval; and rejection fraction
with Wilson interval. A null rejection is a false-positive rate; nonzero
rejection is descriptive power. Show both successful-fit denominators and
best/worst coverage bounds over all planned cases. Investigate bias over
three Monte Carlo standard errors, any failure or numerical mismatch, and a
coverage Wilson interval wholly outside [0.93, 0.97]. These are review flags,
not automatic acceptance rules or evidence of equivalence when absent.
Preserve adverse findings.

The separate two-case installed-R comparison checks numerical compatibility
only. This simulation tests a correctly specified Poisson rate model with
independent known scores and time. It does not test estimated cutpoints,
overdispersion, dependence, time related to exposure, missingness, marginal
effects or causal identification. The earlier equal-period findings remain
separate.

## Execution boundary

After a fresh local storage and worker check, run only the 20-case timing
pilot on the Mac: one numerical thread, under 2 GiB memory, ten minutes and
100 MiB new output. Record wall time, maximum resident memory, file size,
warnings, failures and paired-check status. Keep exact invocation, thread
controls, CPU and memory allocation, timing tool, package version, source
and output hashes, and exit status. The 4,000-case study is server
work. It requires protocol review, a passing pilot, clean committed source,
fresh server quota/queue checks and a bounded job specification. No full
study or paper-level coverage claim follows from the pilot alone.
