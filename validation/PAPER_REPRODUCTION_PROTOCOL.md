# Keil et al. Gaussian simulation reproduction

Revised before the full study after a separate
protocol review. Protocol `keil-2020-gaussian-scenarios-1-4-v2`.
The [completed results](PAPER_REPRODUCTION_RESULTS.md) report the completed
server run and its limits; the design below was set before that run.
This is a separate study from [Gaussian known-score validation](GAUSSIAN_PROTOCOL.md).
The default runner command prints its plan and performs no fits. A five-replicate
pilot per condition is a timing and implementation check, not paper evidence.

## Source and question

[Keil et al. (2020)](https://doi.org/10.1289/EHP5838) studied the behavior of
quantile g-computation. Their [supplementary code archive](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7228100/supplementaryFiles)
contains `qgcomp_simdata.R` and `qgcomp_analysis.R`. The outer ZIP has SHA-256
`c7a936552eaaaf06272b2dbb6a9f5eda34d176ccf07964855a6a70a687eb636d`;
its nested code ZIP has SHA-256
`ba2a8a858a31f24593d38d94dd204db2a18a214b4aa61645cf7748f988f54ccd`.
The original scripts use GPL-licensed R code as a design reference; this runner
is a separate Python implementation. We reproduce only the **Gaussian qgcomp
arm of scenarios 1–4 at n=500 and four exposures**. We do not reproduce WQS,
the paper's other sample sizes or exposure counts, or nonlinear scenarios here.
The outer ZIP hash identifies the saved retrieval used to set this protocol;
Europe PMC may regenerate that wrapper with different ZIP metadata. A later
retrieval had a different outer hash but the **same nested code ZIP hash**.
The nested archive is the stable code identity used for the source translation.

The paper reports results from the older qgcomp 1.3. Its supplement includes
a later five-run illustration made with qgcomp 2.1.3 and warns that it does
not reproduce the paper's 1,000-run results. Our package follows tested
qgcomp 2.19.6 numerical conventions. Exact published numbers or R random
draws are therefore **not** an equality target. We will assess direction and
Monte Carlo compatibility with the published qgcomp columns, and separately
check same-input software agreement against installed R 2.19.6.

The comparison targets are the paper's [Table 3](https://pmc.ncbi.nlm.nih.gov/articles/PMC7228100/),
Q-gcomp rows with four exposures and n=500. The published values are rounded
to two decimals:

| Scenario | Published bias | Empirical SD | Root mean variance | Coverage | Rejection |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1, null | 0.00 | 0.08 | 0.08 | 0.94 | 0.06 |
| 2, cancellation | 0.00 | 0.09 | 0.09 | 0.95 | 0.05 |
| 3, single | 0.00 | 0.08 | 0.08 | 0.94 | 0.88 |
| 4, distributed | 0.00 | 0.08 | 0.08 | 0.95 | 0.86 |

These are descriptive references, not exact acceptance values. The original
paper's table calls its empirical across-dataset SD an “MCSE”; here we call
that value empirical SD and reserve **bias Monte Carlo standard error** for
SD divided by the square root of the number of successful datasets. Both
quantities are reported without changing the published data.

The primary checks are complete accounting of all 4,000 planned cases and
independent verification of every successful fit. Unexpected failures are
reported and investigated. We show both successful-case coverage/rejection
and conservative all-planned bounds: if c successes meet a criterion among
N planned cases with f fit failures, the rate lies between c/N and (c+f)/N.
If any case fails, unconditional bias cannot be estimated from successes
alone; no unconditional performance claim is made. An interruption leaves
the completed-case journal but no final study result. The study must then be
audited and, if rerun, assigned a new run identity rather than silently resumed.

For comparison with the rounded paper table, report the difference and an
approximate Monte Carlo standard error from **both** independent studies.
For bias, use the square root of the sum of our bias-MCSE squared and
(published empirical SD / √1000) squared. For coverage and rejection, use
the square root of p(1-p)/m plus p₀(1-p₀)/1000, where p and m are our
success-case rate/count and p₀ is the published rate. The paper's two-decimal
rounding adds at most 0.005 to an absolute difference. A difference larger
than three combined standard errors plus 0.005 is a **descriptive divergence
flag** for investigation. Absence of a flag is not equivalence or proof of
replication. Show Wilson 95% intervals for our binomial rates and place
empirical SD/root mean variance beside the paper values. Also compare
coverage with the Gaussian normal-interval benchmark, about 0.9494 at n=500.

## Fixed design

Each dataset has 500 rows, four exposure scores and a continuous
outcome. For each exposure, the scores 0, 1, 2 and 3 each appear exactly 125
times. The four columns are independent random permutations. This translates
the supplement's `corr=0, ncor=0` case. Outcome errors are independent standard
normal draws. For row i, the outcome is the sum of its four score values,
each multiplied by its scenario coefficient, plus one error draw. The paper's
`b0=0` is retained. The target effect is the sum of four coefficients.
The article prose describes each score as having probability 0.25. The
released R generator uses an exactly balanced score vector for each dataset;
we follow that executable supplement. This fixes each column's marginal
counts while retaining random row assignments. Because each column has fixed
counts, its rows are not independent draws from a categorical distribution;
the independently permuted columns match the executable supplement's design.

| Scenario | Four coefficients | True joint effect |
| --- | --- | ---: |
| 1, null | 0, 0, 0, 0 | 0 |
| 2, cancellation | 0.25, -0.25, 0, 0 | 0 |
| 3, single active exposure | 0.25, 0, 0, 0 | 0.25 |
| 4, distributed effect | 0.0625 each | 0.25 |

The Python fit receives the already grouped scores with `q=None`, an
intercept, all four exposure columns and no adjustment variables. The pilot
has five datasets in each scenario; the planned study has 1,000 in each.
Independent NumPy PCG64 streams are identified by master seed 20261023,
pilot/study namespace, scenario index and repetition number. Every record
includes a SHA-256 hash of the four score columns and outcome in row order.
The full dataset is reproducible from its identity and the recorded software
version. R and Python random streams are intentionally different.

## Measures and interpretation

For each scenario report planned, successful and failed fits; bias (mean
estimate minus truth); empirical standard deviation; root mean reported
variance; 95% interval coverage; and the fraction with p-value below 0.05.
For scenarios 1–2, that fraction is a false-positive rate. For scenarios 3–4,
it is power. Report bias Monte Carlo standard error, and use a binomial
interval when assessing coverage/rejection. A failed fit stays in the
denominator accounting and is never silently replaced. Retain per-repetition
estimate, variance, interval, p-value, truth, input hash and failure code.

The full run must use a server CPU job, one numerical thread per
worker, an identified source revision, a fresh output path, fixed seeds and
a read-only result checker. The runner records its and the Gaussian core's
SHA-256 hashes, source revision, package/dependency versions and platform.
On a source transfer without `.git`, `QGCOMP_SOURCE_REVISION` identifies the
verified package commit; the source-file hashes remain the actual byte checks.
Each terminal case is appended to a sibling `.json.jsonl` journal and flushed
to storage before the next case.

A complete result file records the journal
hash; the checker reconciles its lines and recomputes every successful
estimate, variance, interval and p-value by independent QR least squares.
The job record must state allocated CPU, memory, storage and time, and
preserve logs, environment versions, output hash and any interruption.
Pilot performance will size those resources. Changes after seeing full-study
results require a new protocol ID. The installed-R comparison remains a
20-case same-input pilot check; the full 4,000 cases do not have individual
R comparison results.

Software agreement and repeated-sample behavior are
distinct claims.

## Commands

From an installed development environment:

```sh
python validation/paper_reproduction.py plan
python validation/paper_reproduction.py pilot --output /path/to/fresh/pilot.json
```

`study` uses the same command shape but must run only after the review and
server resource limits described above. Neither command changes package runtime
dependencies; R is optional validation software.
