# Marginal Poisson bootstrap study with unequal observation times

This [advance protocol](POISSON_BOOTSTRAP_OFFSET_PROTOCOL.md)
tested the implemented 200-draw normal row-bootstrap interval across 4,000
independent datasets. It asks a different question from the two
[fixed-input installed-R comparisons](POISSON_BOOTSTRAP_OFFSET_RESULTS.md):
how often does the interval cover the true effect under this data-generating
model? Agreement with R on fixed inputs does not answer that question.

Each dataset had 500 independent rows, four known exposure scores, one binary
adjustment variable, and observation times sampled independently from 0.25,
0.5, 1 and 2. The Poisson mean was the row's time multiplied by an additive
log-link rate. The center rate `m` was 0.20 or 2.00; the true joint log mean
ratio `b` was 0 or 0.40. The comparison changes every score from 0 to 1,
while keeping each row's adjustment value and time fixed. It averages
predicted counts over those same rows on both sides. In this additive model,
the true marginal log expected-count ratio is `b`: time and adjustment
cancel from the ratio. The count and rate ratios coincide here. This is not a
causal estimand without further assumptions.

All 1,000 planned datasets in each of four conditions completed. Both the
bootstrap and model-based methods returned 4,000 intervals, with zero fit
failures, warnings, or missing cases. Thus successful-fit coverage equals
coverage over all planned datasets. The model-based interval is a descriptive
comparison for the same target; it uses a different variance calculation.
The table gives the mean count and observation time per row, the fraction of
zero counts, and bootstrap bias (mean estimate minus truth) with its Monte
Carlo standard error (MCSE). Brackets are cellwise 95% Wilson intervals for
the observed interval-coverage fractions. They do not adjust for looking at
four cells or two methods.

| Center rate `m` | True `b` | Mean count | Mean time | Zero fraction | Bootstrap bias (MCSE) | Bootstrap covered / 1,000 [Wilson 95%] | Model-based covered / 1,000 [Wilson 95%] |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.20 | 0 | 0.2337 | 0.9378 | 0.8032 | -0.0061 (0.0054) | 946 [0.930, 0.958] | 951 [0.936, 0.963] |
| 0.20 | 0.40 | 0.2405 | 0.9374 | 0.7998 | +0.0035 (0.0052) | 947 [0.931, 0.959] | 953 [0.938, 0.964] |
| 2.00 | 0 | 2.3361 | 0.9374 | 0.2355 | +0.0001 (0.0017) | 938 [0.921, 0.951] | 942 [0.926, 0.955] |
| 2.00 | 0.40 | 2.3974 | 0.9391 | 0.2377 | +0.0005 (0.0017) | 950 [0.935, 0.962] | 954 [0.939, 0.965] |

Bootstrap coverage ranged from 93.8% to 95.0%. The high-rate null cell had
the lowest observed coverage: 938/1,000, with Wilson interval [0.921, 0.951].
Its false-positive fraction was 62/1,000 = 0.062 [0.049, 0.079]. This is a
possible small shortfall that remains uncertain: 0.95 lies inside the
coverage interval, and the interval is not wholly outside the protocol's
0.93-0.97 review band. Neither this cell nor the four-cell pattern proves
undercoverage, exact 95% calibration, or equivalence to the nominal level.
The other null cell rejected 54/1,000 = 0.054 [0.042, 0.070]. Null rejection
and interval coverage use the same two-sided cutoff, so these are
complementary counts, not independent checks. At `b=0.40`, rejection was
0.683 [0.654, 0.711] and 1.000 [0.996, 1.000]; these are descriptive power
values for these conditions. All four absolute biases were below three MCSEs.
The [compact summary JSON](poisson-bootstrap-offset-calibration-study-summary.json)
gives full precision, reported and empirical standard deviations, every
denominator, failure count, all-planned coverage bound, and rejection interval.

The model-based coverage values are close to the bootstrap values on these
same datasets. Separate Wilson intervals do not measure uncertainty in the
paired difference, and this study does not rank the two variance methods.
The earlier [equal-time marginal bootstrap study](POISSON_BOOTSTRAP_CALIBRATION_RESULTS.md)
reported 0.959, 0.952, 0.952 and 0.960 coverage, each with its own Wilson
interval. The studies used independent random streams and had different mean
times and count distributions. Their numerical differences are unpaired
context, not evidence that unequal times caused a change in coverage. The
earlier sparse-null conditional Poisson finding concerns a different
interval and remains open.

Server job `2911570` completed `0:0` in 19m14s from clean commit
`70833fb85540bac293709f898ff3bce86abdc7b1`, with Python 3.11.5,
NumPy 2.0.0 and SciPy 1.14.0. The saved source bundle, protocol, result,
all-case audit and server output manifest passed SHA-256 checks. The all-case
audit independently solved the offset Poisson score equations for each
original and resampled fit and compared 8,000 method results: zero issues or
independent-fit failures, with maximum absolute difference `5.74e-11`. A
second local replay passed all 8,000 comparisons with matching NumPy 2.0.0
and SciPy 1.14.0, using Python 3.11.15. A separate raw-record calculation,
which does not import the study runner, matched every saved cell and method
summary. One full-study input was also generated independently from its
fixed seed and matched its returned hash, time total, and count total. The
main numerical checker still shares generation and some summary code with
the runner; these checks are not a wholly independent implementation of the
design.

Case-level evidence is not included in the source distribution. Its SHA-256
values are: result `9480c2d234148f51af35c1725f35af4684b765e873ee154a78bbff30a711ab77`,
server audit `2ef5c52f9d88712bfa94c16dee4fade84ca964dcb8690167d5643421cb5321ea`,
local replay `0c7028af03a5f36973bd5f5a8c8d9f365b929fa603927956663d79c4a12bae49`,
and separate aggregation `71ba70d7e5c0bc873077a2596725de094c1f78102b324a88f6a027d46345b964`.
All seven server-manifest files and additional job metadata are archived
privately.

These observations concern known scores, independent observation times,
independent rows, one binary covariate, a correctly specified additive
Poisson model, and the implemented 200-resample normal interval. They do not
establish behavior with estimated cutpoints, informative time, noncancelling
marginal standardization, overdispersion, weights, clustering, nonlinear
effects, or other count families. They do not reproduce the full paper or
complete R-package coverage.
