# Marginal Poisson bootstrap with unequal observation times

Protocol `poisson-bootstrap-offset-known-scores-v1`, prepared before any outcome was generated. This is an internal advance plan, not a
registration. It tests interval behavior for the existing fixed-cut row
bootstrap, separately from installed-R software comparisons.

## Question, data, and target

Does the normal 95% interval from 200 row-bootstrap draws cover the true
marginal log mean ratio when observation times differ? Each dataset has 500
independent rows. The four known exposure scores Q_j are independent uniform
integers in {0,1,2,3}; S is their sum. The adjustment variable Z is an
independent Bernoulli(0.5) draw. Observation time T is independent of Q and Z
and is uniformly sampled from {0.25,0.5,1,2}. Counts follow

    Y | Q,Z,T ~ Poisson(T * exp(log(m) + b*(S-6)/4 + 0.4*Z)).

Here m is the rate at S=6 and Z=0. The reported marginal contrast sets all
four scores to 0 versus 1 for every row, keeps that row's Z and T, averages
the predicted counts, and takes the log ratio. Its true value is b. Keeping
the same times under both interventions is essential. The additive log model
makes the ratio equal to b even though time and Z affect the two means. This
design therefore checks the offset and standardization calculation, but does
not test a noncancelling marginal contrast, informative times, causal
identification, overdispersion, clustering, or estimated quantile cuts.

| Factor | Fixed values |
| --- | --- |
| Center-point rate m | 0.20, 2.00 |
| True joint log mean ratio b | 0, 0.40 |
| Rows per dataset | 500 |
| Bootstrap draws | 200, sampling entire rows with replacement |
| Independent datasets per cell | 5 in pilot; 1,000 in full study |
| Total | 20 pilot; 4,000 full datasets and 800,000 bootstrap fits |

The four cells are ordered by m, then b. Fit `fit_poisson_bootstrap` with
`q=None`, numeric Z, `exposure_time=T`, `B=200`, and alpha 0.05. Fit
`fit_poisson` on the same data as a model-based descriptive comparator. Both
target b; their variance calculations differ. Do not pool their results.

## Reproducibility and independent checks

Use NumPy PCG64, master seed 20261024, with `SeedSequence` input namespaces
101 for pilot and 102 for study, followed by cell and repetition. Generate Q,
then Z, then time-category indices, then Y. For each bootstrap seed use
`SeedSequence([20261024,103,mode,cell,repetition])`, where mode is 1 for pilot
and 2 for study, and take the first unsigned 64-bit state word. Hash input
arrays in Q, Z, T, Y order with little-endian 64-bit integer or float bytes.
Exact stream replay requires the same NumPy version.

The checker regenerates every planned input and bootstrap schedule. It
independently solves the Poisson score equations with mean
`T*exp(X beta)`, computes inverse-information model variance, and recomputes
all 200 resample effects, standardized intervention means, bootstrap standard
error, interval, and p-value. It audits all cases, including package failures,
and compares saved summaries with the case records. The checker shares the
runner's generation, seed mapping, and summary code, so this is not a wholly
independent design audit. Any warning, failure, identity error, or numerical
mismatch produces a nonpassing audit after preserving available case records.
The full study must use a clean committed source bound to the result.

## Planned reporting and interpretation

For every cell and method report planned, attempted, successful, failed, and
missing counts; mean count and mean observation time; zero fraction; bias and
its Monte Carlo standard error; empirical estimate standard deviation; root
mean reported variance; 95% coverage and Wilson 95% interval; rejection rate
at 0.05 and its Wilson interval. A null rejection rate is a false-positive
rate; a nonzero rejection rate is descriptive power. Give covered/planned and
all-planned worst/best coverage bounds. Keep all failures and their reasons;
draw no replacement data. With 1,000 successful fits, coverage near 0.95 has
Monte Carlo standard error about 0.0069. Wilson intervals are cellwise, with
no multiple-cell adjustment. Null rejection and coverage use the same cutoff
and therefore are complementary, not independent evidence.

Flag any bias above three Monte Carlo standard errors, a coverage Wilson
interval wholly outside [0.93,0.97], nominal 0.95 outside that interval, or
any warning, fit failure, or numerical mismatch. These are investigation
flags, not acceptance criteria or an equivalence test. Report adverse results
without changing the design after seeing study outcomes. Agreement with R or
high fit success does not establish good repeated-data interval coverage.

## Compute and decision boundary

Run the 20-case timing pilot locally only after a fresh storage and worker
check. Limit it to one numerical thread, under 2 GiB memory, ten minutes, and
100 MiB new output. The full study is server CPU work. Require a formal
protocol review, passing pilot, clean committed source, fresh server quota and
queue checks, bounded job resources, and a no-duplicate run identity before
submission. Preserve a flushed case journal; never resume a partial study
under the same identity. Python fitting must remain free of R runtime needs.
