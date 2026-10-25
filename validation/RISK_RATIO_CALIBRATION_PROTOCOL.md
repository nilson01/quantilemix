# Marginal binary risk-ratio bootstrap calibration protocol

Protocol `binary-risk-ratio-known-scores-v1`, prepared before inspecting study outcomes. This is a private advance plan, not a
registration. The question is whether the implemented 200-resample normal
interval for the *marginal log risk ratio* covers its known population target
across repeated independent datasets. Fixed-input agreement with installed R
qgcomp 2.19.6 is a separate software check.

## Target and generating model

For each row, four known exposure scores Q1,...,Q4 are independent Bernoulli
(0.5), and an adjustment variable Z is independent Bernoulli (0.5). Set
S=Q1+...+Q4 and generate a binary outcome with

    Pr(Y=1 | Q,Z) = expit(a + b*S/4 + 0.5*Z).

Here `expit(t)=1/(1+exp(-t))`. For each planned baseline risk `r0`, choose
`a` so that the average of `expit(a)` and `expit(a+0.5)` equals `r0`.
The intervention sets all four scores to zero or all four to one while
leaving the population distribution of Z at Bernoulli (0.5). The true risks
are `r0` and `r1 = [expit(a+b)+expit(a+b+0.5)]/2`; the true log risk ratio is
`log(r1/r0)`. The model's conditional log odds contrast is `b`, which is
generally **not** the marginal log risk ratio. The fitted method standardizes
over the observed Z values; row resampling includes those values, so the
study evaluates coverage of the population target under random Z sampling.
No causal interpretation is claimed.

| Factor | Fixed values |
| --- | --- |
| Independent rows per dataset | 500 |
| All-zero intervention risk `r0` | 0.10, 0.40 |
| Joint conditional log odds contrast `b` | 0 (null), 0.80 (nonzero) |
| Bootstrap | 200 fixed-cut independent row resamples with replacement |
| Independent datasets per cell | 5 pilot; 1,000 study |
| Total study | 4,000 datasets; 800,000 bootstrap draws |
| RNG | NumPy PCG64 with separate `SeedSequence` namespaces for data and resampling |

The four cells are ordered by `r0`, then `b`. Every dataset uses the public
`fit_binomial_risk_ratio(y, Q, Z, q=None, B=200, seed=...)`, with alpha 0.05.
The score cutpoints are known; there is no empirical quantization or tie
handling. Do not replace a failed dataset or failed bootstrap draw. The
existing two installed-R fixed cases remain a separate reference check.

## Checks and planned reporting

The runner records every input identity, generated event count, bootstrap
seed, fit result or failure, warnings and all bootstrap log-ratio estimates.
A retained append-only journal records a header, each case start and terminal
record, and a completed-case count. A deterministic run identity binds the
plan, committed revision and source hashes. An active-run guard blocks a second
run with that identity in the same output directory. The final result uses an
exclusive no-replace publication; a completion receipt binds its digest to
the journal, plan, identity and source hashes.

A partial journal can be read
without fitting models and reports unattempted or started-but-unrecorded cases.
An interrupted or failed run is never treated as a completed study or resumed
under the same identity. Keep its partial journal and guard for inspection.
A separate checker regenerates each input and row schedule, solves the
Bernoulli score equations independently for the original and each resample,
then recomputes standardized risks, log ratio, bootstrap standard deviation,
normal interval and p-value.

It checks each planned case, including failures,
and independently aggregates counts from raw records. Its estimator arithmetic
is independent; generation and summary functions are shared with the runner,
so this is not a fully independent implementation of the study design.
Exact random-stream replay requires the recorded NumPy version. Bind runner,
checker and estimator source hashes to a clean committed study revision.
The checker also solves the two-state population target algebra without using
the runner's root calculation.

This checks the four stated targets, while the
generator and summary implementation still have shared code.

Report per cell: planned, attempted, successful, failed and missing datasets;
average observed event rate; bias with Monte Carlo standard error (MCSE);
empirical estimate SD; root mean reported variance; interval coverage and
rejection rates with cellwise Wilson 95% intervals; all-planned worst/best
coverage bounds. Rejection at the null is a false-positive rate; for nonzero
effects it is descriptive power. Null rejection and coverage are complements
of the same two-sided test, not independent checks. At 1,000 successful fits,
nominal 95% coverage has approximate MCSE 0.0069. Wilson intervals are
cellwise and unadjusted for four comparisons.

Investigate any warning, failure, missing case, numerical mismatch, bias above
three MCSEs, coverage Wilson interval excluding 0.95, or coverage Wilson
interval wholly outside the descriptive 0.93--0.97 review band. These flags
are not pass criteria or an equivalence test. Report adverse results and do
not revise this design after seeing them.
If a cellwise Wilson interval contains 0.95, say only that this study found
no detected departure from 0.95 in that cell. Do not call the interval
adequately calibrated without a separate tolerance and precision argument.
Display successful-fit coverage beside failed and missing counts and the
all-planned coverage bounds. In nonzero-effect cells, the observed-data event
rate is not the all-zero intervention risk `r0`.

## Execution and interpretation boundary

Only a 20-case timing pilot is admitted locally after a fresh storage and
worker check, with one numerical thread, under 2 GiB memory, ten minutes and
100 MiB new output. The full study is SERVER_CPU and requires a formal
protocol review, successful bounded pilot, clean committed source, fresh
server quota and queue checks, a bounded job specification and an inactive
duplicate-run identity. Save an append-only case journal; do not silently
resume partial work under the same identity. The package retains no R runtime
dependency. The result will not validate raw-exposure cutpoints, missingness,
weights, clusters, nonlinear exposure response, other outcome families or the
whole R interface. Prior Poisson findings remain separate.

## Audit correction after the first full run

The first full run, job `2906946`, finished all 4,000 planned datasets but
its checker returned a nonzero status. It recorded one package failure in
cell 0, repetition 190, bootstrap draw 83. The package rejected separated
data. The independent root solver incorrectly counted an extreme coefficient
vector as a successful fit because its score residual was small, even though
its coefficients were about -37 and +35 and its smallest information
eigenvalue was about `9e-15`. In this resample, all 255 rows with `Z=0`
had outcome zero; the other 245 rows had 32 events. This gives a direct
separation diagnostic, apart from the numerical threshold. A fixed-input
base R `glm` check stopped with finite coefficients around -21 and +19,
without a warning, so this check does not establish how the full R qgcomp
bootstrap handles this draw.

The checker now rejects such numerical-boundary solutions and records a
failure as verified only when the package replay and independent check
identify separation at the same draw. It continues to flag every other
failure mismatch. The study design, data streams, all 4,000 planned cases,
estimand and review flags above are unchanged. The first run and its nonzero
audit are retained privately; the corrected checker requires a new committed
run identity. This is a post-run software-audit repair, not a change to the
prespecified statistical design.
